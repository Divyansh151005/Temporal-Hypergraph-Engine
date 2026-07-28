//! Binary WAL frame format.
//!
//! ```text
//! +--------+---------+-------------+-------------+--------+----------+
//! | magic  | op_code | payload_len | sequence_id | crc32  | reserved |
//! | 4B     | 2B      | 4B          | 8B          | 4B     | 2B       |
//! +--------+---------+-------------+-------------+--------+----------+
//! ```
//!
//! The CRC32 covers `op_code`, `payload_len`, `sequence_id` and the payload,
//! so a bit flip in any of them is detected.

use bytemuck::{Pod, Zeroable};

/// Frame magic bytes (format v2: hyperedge records carry a text summary).
pub const WAL_MAGIC: [u8; 4] = *b"THG2";
/// Magic of the unsupported v1 format, recognized only to refuse it.
pub const LEGACY_WAL_MAGIC_V1: [u8; 4] = *b"THG1";
/// Size of [`WalFrameHeader`] in bytes.
pub const WAL_HEADER_LEN: usize = 24;
/// Upper bound on a single payload, used to reject corrupt lengths early.
pub const MAX_PAYLOAD_LEN: u32 = 256 << 20;

/// Kind of mutation stored in a frame.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCode {
    /// Insert an entity node.
    InsertNode = 1,
    /// Insert a hyperedge.
    InsertHyperedge = 2,
    /// Soft-delete a node or hyperedge.
    SoftDelete = 3,
    /// Change a hyperedge's validity end (temporal truncation).
    UpdateValidity = 4,
    /// Marks every preceding uncommitted frame as one atomic batch.
    BatchCommit = 5,
}

impl OpCode {
    /// Decodes a raw op code.
    pub const fn from_u16(raw: u16) -> Option<Self> {
        match raw {
            1 => Some(Self::InsertNode),
            2 => Some(Self::InsertHyperedge),
            3 => Some(Self::SoftDelete),
            4 => Some(Self::UpdateValidity),
            5 => Some(Self::BatchCommit),
            _ => None,
        }
    }
}

/// Fixed 24-byte frame header.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct WalFrameHeader {
    /// Always [`WAL_MAGIC`].
    pub magic: [u8; 4],
    /// Raw [`OpCode`].
    pub op_code: u16,
    /// Payload length in bytes.
    pub payload_len: u32,
    /// Monotonic frame sequence number.
    pub sequence_id: u64,
    /// CRC32 of op code, length, sequence id and payload.
    pub crc32: u32,
    /// Reserved, always zero.
    pub reserved: u16,
}

const _: () = assert!(std::mem::size_of::<WalFrameHeader>() == WAL_HEADER_LEN);

/// Computes the frame checksum.
pub fn frame_checksum(op_code: u16, payload_len: u32, sequence_id: u64, payload: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&op_code.to_le_bytes());
    hasher.update(&payload_len.to_le_bytes());
    hasher.update(&sequence_id.to_le_bytes());
    hasher.update(payload);
    hasher.finalize()
}

/// Appends one encoded frame to `out`.
pub fn encode_frame(out: &mut Vec<u8>, op: OpCode, sequence_id: u64, payload: &[u8]) {
    let payload_len = u32::try_from(payload.len()).expect("WAL payload exceeds u32::MAX");
    debug_assert!(payload_len <= MAX_PAYLOAD_LEN);
    let header = WalFrameHeader {
        magic: WAL_MAGIC,
        op_code: op as u16,
        payload_len,
        sequence_id,
        crc32: frame_checksum(op as u16, payload_len, sequence_id, payload),
        reserved: 0,
    };
    out.extend_from_slice(bytemuck::bytes_of(&header));
    out.extend_from_slice(payload);
}

/// Why frame parsing stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptionKind {
    /// Fewer than 24 bytes remain.
    TruncatedHeader,
    /// Magic bytes differ from [`WAL_MAGIC`].
    BadMagic,
    /// Unknown op code.
    UnknownOpCode(u16),
    /// Payload length above [`MAX_PAYLOAD_LEN`].
    OversizedPayload(u32),
    /// The file ends inside the payload.
    TruncatedPayload,
    /// CRC32 mismatch.
    ChecksumMismatch,
    /// Sequence number is not the successor of the previous frame.
    SequenceGap {
        /// Expected sequence id.
        expected: u64,
        /// Sequence id found.
        found: u64,
    },
    /// A frame passed its checksum but its payload could not be decoded.
    MalformedPayload,
    /// Valid frames follow the last batch commit, but no commit closes them.
    UncommittedTail,
}

/// Location and kind of corruption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Corruption {
    /// Byte offset of the offending frame.
    pub offset: usize,
    /// What went wrong.
    pub kind: CorruptionKind,
}

/// A verified frame borrowed from the underlying buffer.
#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    /// Decoded op code.
    pub op: OpCode,
    /// Sequence id.
    pub sequence_id: u64,
    /// Payload bytes.
    pub payload: &'a [u8],
    /// Byte offset of the frame header.
    pub offset: usize,
}

/// Sequential, validating frame reader over an in-memory (or mmapped) buffer.
#[derive(Debug)]
pub struct FrameReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    expected_seq: Option<u64>,
}

impl<'a> FrameReader<'a> {
    /// Starts reading at offset 0. If `expected_first_seq` is set, the first
    /// frame must carry that sequence id.
    pub fn new(bytes: &'a [u8], expected_first_seq: Option<u64>) -> Self {
        Self {
            bytes,
            pos: 0,
            expected_seq: expected_first_seq,
        }
    }

    /// Offset just past the last frame returned.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Returns the next verified frame, `Ok(None)` at a clean end of buffer,
    /// or the corruption that stopped the scan.
    pub fn next_frame(&mut self) -> Result<Option<Frame<'a>>, Corruption> {
        let offset = self.pos;
        let rest = &self.bytes[offset..];
        if rest.is_empty() {
            return Ok(None);
        }
        let corrupt = |kind| Corruption { offset, kind };
        if rest.len() < WAL_HEADER_LEN {
            return Err(corrupt(CorruptionKind::TruncatedHeader));
        }
        let header: WalFrameHeader = bytemuck::pod_read_unaligned(&rest[..WAL_HEADER_LEN]);
        if header.magic != WAL_MAGIC {
            return Err(corrupt(CorruptionKind::BadMagic));
        }
        let op_code = header.op_code;
        let op =
            OpCode::from_u16(op_code).ok_or(corrupt(CorruptionKind::UnknownOpCode(op_code)))?;
        let payload_len = header.payload_len;
        if payload_len > MAX_PAYLOAD_LEN {
            return Err(corrupt(CorruptionKind::OversizedPayload(payload_len)));
        }
        let end = WAL_HEADER_LEN + payload_len as usize;
        if rest.len() < end {
            return Err(corrupt(CorruptionKind::TruncatedPayload));
        }
        let payload = &rest[WAL_HEADER_LEN..end];
        let sequence_id = header.sequence_id;
        if frame_checksum(op_code, payload_len, sequence_id, payload) != header.crc32 {
            return Err(corrupt(CorruptionKind::ChecksumMismatch));
        }
        if let Some(expected) = self.expected_seq {
            if sequence_id != expected {
                return Err(corrupt(CorruptionKind::SequenceGap {
                    expected,
                    found: sequence_id,
                }));
            }
        }
        self.expected_seq = Some(sequence_id + 1);
        self.pos += end;
        Ok(Some(Frame {
            op,
            sequence_id,
            payload,
            offset,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(frames: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        for i in 0..frames {
            encode_frame(&mut buf, OpCode::InsertNode, 10 + i as u64, &[i as u8; 5]);
        }
        buf
    }

    #[test]
    fn header_is_24_bytes_and_round_trips() {
        let header = WalFrameHeader {
            magic: WAL_MAGIC,
            op_code: 2,
            payload_len: 77,
            sequence_id: 123_456,
            crc32: 0xABCD,
            reserved: 0,
        };
        let bytes = bytemuck::bytes_of(&header);
        assert_eq!(bytes.len(), 24);
        assert_eq!(&bytes[..4], b"THG2");
        let back: WalFrameHeader = bytemuck::pod_read_unaligned(bytes);
        assert_eq!(back, header);
    }

    #[test]
    fn reads_back_frames() {
        let buf = sample(3);
        let mut reader = FrameReader::new(&buf, Some(10));
        for i in 0..3u8 {
            let f = reader.next_frame().unwrap().unwrap();
            assert_eq!(f.sequence_id, 10 + i as u64);
            assert_eq!(f.payload, &[i; 5]);
        }
        assert!(reader.next_frame().unwrap().is_none());
        assert_eq!(reader.position(), buf.len());
    }

    #[test]
    fn detects_each_corruption_kind() {
        let frame_len = WAL_HEADER_LEN + 5;
        let check = |buf: &[u8], kind: CorruptionKind| {
            let mut reader = FrameReader::new(buf, None);
            assert!(reader.next_frame().unwrap().is_some());
            let err = reader.next_frame().unwrap_err();
            assert_eq!(
                err,
                Corruption {
                    offset: frame_len,
                    kind
                }
            );
        };

        let mut bad_magic = sample(2);
        bad_magic[frame_len + 3] = b'9';
        check(&bad_magic, CorruptionKind::BadMagic);

        let mut flipped = sample(2);
        flipped[frame_len + WAL_HEADER_LEN + 2] ^= 0x01;
        check(&flipped, CorruptionKind::ChecksumMismatch);

        let truncated = &sample(2)[..frame_len + 10];
        check(truncated, CorruptionKind::TruncatedHeader);

        let truncated_payload = &sample(2)[..frame_len + WAL_HEADER_LEN + 2];
        check(truncated_payload, CorruptionKind::TruncatedPayload);

        let mut gap = sample(1);
        encode_frame(&mut gap, OpCode::InsertNode, 99, &[0; 5]);
        check(
            &gap,
            CorruptionKind::SequenceGap {
                expected: 11,
                found: 99,
            },
        );
    }
}

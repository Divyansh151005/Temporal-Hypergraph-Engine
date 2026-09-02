//! On-disk block framing.
//!
//! ```text
//! offset  size  field
//!      0     4  magic "V3KV"
//!      4    32  BLAKE3 digest of the payload
//!     36    10  num_layers, num_heads, head_dim, block_size, precision (u16 LE)
//!     46     2  reserved
//!     48     4  payload_len (u32 LE)
//!     52     4  CRC32 of the header with this field zeroed
//!     56     8  reserved
//!     64     -  payload, padded to a multiple of 64 bytes
//! ```

use bytemuck::{Pod, Zeroable};

use crate::error::KvError;
use crate::hash::BlockHash;

/// Record magic bytes.
pub const KV_MAGIC: [u8; 4] = *b"V3KV";
/// Size of [`KvBlockHeader`].
pub const HEADER_LEN: usize = 64;
/// Alignment of every record and therefore of every payload.
pub const RECORD_ALIGN: usize = 64;

/// Element precision of a KV block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u16)]
pub enum Precision {
    /// Unspecified geometry; any payload length is accepted.
    #[default]
    Opaque = 0,
    /// IEEE half precision.
    Fp16 = 1,
    /// bfloat16.
    Bf16 = 2,
    /// 8-bit integer quantization.
    Int8 = 3,
}

impl Precision {
    /// Parses a wire precision id.
    pub fn from_id(id: u16) -> Result<Self, KvError> {
        match id {
            0 => Ok(Self::Opaque),
            1 => Ok(Self::Fp16),
            2 => Ok(Self::Bf16),
            3 => Ok(Self::Int8),
            other => Err(KvError::InvalidArgument(format!(
                "unknown precision id {other}"
            ))),
        }
    }

    /// Wire precision id.
    pub fn id(self) -> u16 {
        self as u16
    }

    /// Bytes per element.
    pub fn element_size(self) -> usize {
        match self {
            Self::Opaque | Self::Int8 => 1,
            Self::Fp16 | Self::Bf16 => 2,
        }
    }
}

/// PagedAttention block geometry: tensor shape `[2, L, B, H, D]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvBlockLayout {
    /// Transformer layers `L`.
    pub num_layers: u16,
    /// Attention heads `H`.
    pub num_heads: u16,
    /// Head dimension `D`.
    pub head_dim: u16,
    /// Tokens per block `B`.
    pub block_size: u16,
    /// Element precision.
    pub precision: Precision,
}

impl KvBlockLayout {
    /// Layout of payloads with unspecified geometry.
    pub const OPAQUE: Self = Self::new(0, 0, 0, 0, Precision::Opaque);

    /// Llama-3-8B FP16 blocks of 16 tokens (8 MiB).
    pub const LLAMA3_8B_FP16: Self = Self::new(32, 32, 128, 16, Precision::Fp16);

    /// Creates a layout.
    pub const fn new(
        num_layers: u16,
        num_heads: u16,
        head_dim: u16,
        block_size: u16,
        precision: Precision,
    ) -> Self {
        Self {
            num_layers,
            num_heads,
            head_dim,
            block_size,
            precision,
        }
    }

    /// Whether the geometry is unspecified.
    pub fn is_opaque(&self) -> bool {
        self.precision == Precision::Opaque
    }

    /// `2 * L * B * H * D * sizeof(element)`, or `None` for opaque layouts.
    pub fn expected_bytes(&self) -> Option<usize> {
        if self.is_opaque() {
            return None;
        }
        Some(
            2 * self.num_layers as usize
                * self.block_size as usize
                * self.num_heads as usize
                * self.head_dim as usize
                * self.precision.element_size(),
        )
    }

    /// Checks that `payload_len` matches the geometry.
    pub fn validate(&self, payload_len: usize) -> Result<(), KvError> {
        match self.expected_bytes() {
            Some(expected) if expected != payload_len => Err(KvError::LayoutMismatch {
                expected,
                got: payload_len,
            }),
            _ => Ok(()),
        }
    }
}

/// Fixed 64-byte block header (see module docs).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct KvBlockHeader {
    /// [`KV_MAGIC`].
    pub magic: [u8; 4],
    /// BLAKE3 digest of the payload.
    pub hash_digest: BlockHash,
    /// Transformer layers.
    pub num_layers: u16,
    /// Attention heads.
    pub num_heads: u16,
    /// Head dimension.
    pub head_dim: u16,
    /// Tokens per block.
    pub block_size: u16,
    /// [`Precision`] id.
    pub precision: u16,
    /// Must be zero.
    pub reserved0: u16,
    /// Payload length in bytes.
    pub payload_len: u32,
    /// CRC32 of the header with this field zeroed.
    pub header_crc: u32,
    /// Must be zero.
    pub reserved1: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<KvBlockHeader>() == HEADER_LEN);
const _: () = assert!(std::mem::offset_of!(KvBlockHeader, payload_len) == 48);

impl KvBlockHeader {
    /// Builds a sealed header.
    pub fn new(hash_digest: BlockHash, layout: KvBlockLayout, payload_len: u32) -> Self {
        let mut header = Self {
            magic: KV_MAGIC,
            hash_digest,
            num_layers: layout.num_layers,
            num_heads: layout.num_heads,
            head_dim: layout.head_dim,
            block_size: layout.block_size,
            precision: layout.precision.id(),
            reserved0: 0,
            payload_len,
            header_crc: 0,
            reserved1: [0; 8],
        };
        header.header_crc = header.compute_crc();
        header
    }

    fn compute_crc(&self) -> u32 {
        let mut copy = *self;
        copy.header_crc = 0;
        crc32fast::hash(bytemuck::bytes_of(&copy))
    }

    /// Parses and validates a header; `None` if the bytes are not a sealed header.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let header: Self = bytemuck::pod_read_unaligned(bytes.get(..HEADER_LEN)?);
        (header.magic == KV_MAGIC && header.header_crc == header.compute_crc()).then_some(header)
    }

    /// Header bytes.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Block geometry recorded in the header.
    pub fn layout(&self) -> Result<KvBlockLayout, KvError> {
        Ok(KvBlockLayout::new(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.block_size,
            Precision::from_id(self.precision)?,
        ))
    }
}

/// Total aligned size of a record carrying `payload_len` bytes.
pub fn record_len(payload_len: usize) -> usize {
    HEADER_LEN + payload_len.div_ceil(RECORD_ALIGN) * RECORD_ALIGN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_block_is_8_mib() {
        assert_eq!(
            KvBlockLayout::LLAMA3_8B_FP16.expected_bytes(),
            Some(8_388_608)
        );
        assert!(KvBlockLayout::LLAMA3_8B_FP16.validate(8_388_608).is_ok());
        assert!(matches!(
            KvBlockLayout::LLAMA3_8B_FP16.validate(10),
            Err(KvError::LayoutMismatch { .. })
        ));
        assert!(KvBlockLayout::OPAQUE.validate(10).is_ok());
    }

    #[test]
    fn header_round_trips_and_detects_damage() {
        let header = KvBlockHeader::new([7; 32], KvBlockLayout::LLAMA3_8B_FP16, 8_388_608);
        let mut bytes = header.as_bytes().to_vec();
        assert_eq!(bytes.len(), HEADER_LEN);
        let parsed = KvBlockHeader::parse(&bytes).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(parsed.layout().unwrap(), KvBlockLayout::LLAMA3_8B_FP16);
        bytes[40] ^= 1;
        assert!(KvBlockHeader::parse(&bytes).is_none());
        assert!(KvBlockHeader::parse(&[0u8; HEADER_LEN]).is_none());
        assert!(KvBlockHeader::parse(&bytes[..10]).is_none());
    }

    #[test]
    fn records_are_64_byte_aligned() {
        assert_eq!(record_len(0), 64);
        assert_eq!(record_len(1), 128);
        assert_eq!(record_len(64), 128);
        assert_eq!(record_len(8_388_608) % RECORD_ALIGN, 0);
    }
}

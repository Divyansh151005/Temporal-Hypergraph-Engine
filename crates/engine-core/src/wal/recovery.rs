//! Crash recovery: replay committed batches and cut off corrupt tails.

use std::fs::{File, OpenOptions};
#[cfg(miri)]
use std::io::Read;
use std::path::Path;

#[cfg(not(miri))]
use memmap2::Mmap;
use tracing::warn;

use super::format::{Corruption, CorruptionKind, FrameReader, OpCode, LEGACY_WAL_MAGIC_V1};
use super::record::{BatchCommitRecord, Mutation};
use crate::error::EngineError;

/// Result of replaying one segment.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SegmentReplay {
    /// Sequence id of the last committed batch's commit frame.
    pub last_seq: Option<u64>,
    /// Byte length of the committed prefix.
    pub committed_len: u64,
    /// Number of committed batches replayed.
    pub batches: usize,
    /// Where replay stopped early, if it did.
    pub corruption: Option<Corruption>,
}

/// One committed batch handed to the replay callback.
#[derive(Debug)]
pub struct CommittedBatch<'a> {
    /// `(sequence_id, mutation)` pairs in log order.
    pub mutations: &'a [(u64, Mutation)],
    /// Sequence id of the commit frame.
    pub commit_seq: u64,
}

/// Replays committed batches from `bytes`, stopping cleanly at the first
/// corrupt or uncommitted frame. Errors returned by `on_batch` abort replay.
pub fn replay_bytes(
    bytes: &[u8],
    expected_first_seq: Option<u64>,
    mut on_batch: impl FnMut(CommittedBatch<'_>) -> Result<(), EngineError>,
) -> Result<SegmentReplay, EngineError> {
    let mut reader = FrameReader::new(bytes, expected_first_seq);
    let mut pending: Vec<(u64, Mutation)> = Vec::new();
    let mut outcome = SegmentReplay::default();

    loop {
        let frame = match reader.next_frame() {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                if !pending.is_empty() {
                    outcome.corruption = Some(Corruption {
                        offset: outcome.committed_len as usize,
                        kind: CorruptionKind::UncommittedTail,
                    });
                }
                break;
            }
            Err(corruption) => {
                outcome.corruption = Some(corruption);
                break;
            }
        };
        let malformed = Corruption {
            offset: frame.offset,
            kind: CorruptionKind::MalformedPayload,
        };
        if frame.op == OpCode::BatchCommit {
            let record: Option<BatchCommitRecord> = (frame.payload.len()
                == std::mem::size_of::<BatchCommitRecord>())
            .then(|| bytemuck::pod_read_unaligned(frame.payload));
            let consistent = record.is_some_and(|r| {
                r.frame_count as usize == pending.len()
                    && pending.first().map_or(true, |(seq, _)| *seq == r.first_seq)
            });
            if !consistent {
                outcome.corruption = Some(malformed);
                break;
            }
            on_batch(CommittedBatch {
                mutations: &pending,
                commit_seq: frame.sequence_id,
            })?;
            pending.clear();
            outcome.committed_len = reader.position() as u64;
            outcome.last_seq = Some(frame.sequence_id);
            outcome.batches += 1;
        } else {
            match Mutation::decode(frame.op, frame.payload) {
                Ok(mutation) => pending.push((frame.sequence_id, mutation)),
                Err(_) => {
                    outcome.corruption = Some(malformed);
                    break;
                }
            }
        }
    }
    Ok(outcome)
}

#[cfg(not(miri))]
fn map_segment(file: &File) -> std::io::Result<Mmap> {
    // SAFETY: recovery runs before any writer for this storage directory
    // is opened and the engine is the directory's only writer, so the
    // mapped bytes are not modified while the mapping is alive.
    #[allow(unsafe_code)]
    unsafe {
        Mmap::map(file)
    }
}

/// Miri cannot emulate file-backed mappings.
#[cfg(miri)]
fn map_segment(mut file: &File) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Memory-maps a segment, replays it and truncates the file to its committed
/// prefix if anything after it is corrupt or uncommitted.
pub fn replay_segment(
    path: &Path,
    expected_first_seq: Option<u64>,
    on_batch: impl FnMut(CommittedBatch<'_>) -> Result<(), EngineError>,
) -> Result<SegmentReplay, EngineError> {
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    let file_len = file.metadata()?.len();
    let outcome = if file_len == 0 {
        SegmentReplay::default()
    } else {
        let map = map_segment(&file)?;
        if map.starts_with(&LEGACY_WAL_MAGIC_V1) {
            return Err(EngineError::Recovery(format!(
                "{} was written by the unsupported WAL format v1",
                path.display()
            )));
        }
        replay_bytes(&map, expected_first_seq, on_batch)?
    };

    if outcome.committed_len < file_len {
        let reason = outcome.corruption.map_or_else(
            || "trailing bytes".to_owned(),
            |c| format!("{:?} at offset {}", c.kind, c.offset),
        );
        warn!(
            segment = %path.display(),
            committed_len = outcome.committed_len,
            file_len,
            %reason,
            "WAL replay stopped early; truncating segment to last committed batch"
        );
        file.set_len(outcome.committed_len)?;
        file.sync_all()?;
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::NodeId;
    use crate::wal::format::encode_frame;
    use std::sync::Arc;

    fn node(i: u32) -> Mutation {
        Mutation::InsertNode {
            id: NodeId::new(i, 0),
            label_hash: i as u64,
            created_at: 0,
            label: Arc::from(format!("n{i}")),
        }
    }

    fn batch(buf: &mut Vec<u8>, first_seq: u64, mutations: &[Mutation]) -> u64 {
        let mut seq = first_seq;
        let mut payload = Vec::new();
        for m in mutations {
            payload.clear();
            m.encode(&mut payload);
            encode_frame(buf, m.op_code(), seq, &payload);
            seq += 1;
        }
        let record = BatchCommitRecord {
            first_seq,
            frame_count: mutations.len() as u32,
        };
        encode_frame(buf, OpCode::BatchCommit, seq, bytemuck::bytes_of(&record));
        seq + 1
    }

    fn collect(bytes: &[u8]) -> (SegmentReplay, Vec<Mutation>) {
        let mut seen = Vec::new();
        let outcome = replay_bytes(bytes, None, |b| {
            seen.extend(b.mutations.iter().map(|(_, m)| m.clone()));
            Ok(())
        })
        .unwrap();
        (outcome, seen)
    }

    #[test]
    fn replays_committed_batches_only() {
        let mut buf = Vec::new();
        let next = batch(&mut buf, 1, &[node(0), node(1)]);
        let committed = buf.len();
        batch(&mut buf, next, &[node(2)]);
        let commit_frame_len =
            crate::wal::format::WAL_HEADER_LEN + std::mem::size_of::<BatchCommitRecord>();
        buf.truncate(buf.len() - commit_frame_len);

        let (outcome, seen) = collect(&buf);
        assert_eq!(seen, vec![node(0), node(1)]);
        assert_eq!(outcome.committed_len as usize, committed);
        assert_eq!(outcome.last_seq, Some(3));
        assert_eq!(
            outcome.corruption.unwrap().kind,
            CorruptionKind::UncommittedTail
        );
    }

    #[test]
    fn rejects_commit_with_wrong_frame_count() {
        let mut buf = Vec::new();
        let mut payload = Vec::new();
        node(0).encode(&mut payload);
        encode_frame(&mut buf, OpCode::InsertNode, 1, &payload);
        let record = BatchCommitRecord {
            first_seq: 1,
            frame_count: 2,
        };
        encode_frame(
            &mut buf,
            OpCode::BatchCommit,
            2,
            bytemuck::bytes_of(&record),
        );
        let (outcome, seen) = collect(&buf);
        assert!(seen.is_empty());
        assert_eq!(outcome.committed_len, 0);
        assert_eq!(
            outcome.corruption.unwrap().kind,
            CorruptionKind::MalformedPayload
        );
    }

    #[test]
    fn callback_errors_abort_replay() {
        let mut buf = Vec::new();
        batch(&mut buf, 1, &[node(0)]);
        let result = replay_bytes(&buf, None, |_| Err(EngineError::Recovery("boom".into())));
        assert!(matches!(result, Err(EngineError::Recovery(_))));
    }
}

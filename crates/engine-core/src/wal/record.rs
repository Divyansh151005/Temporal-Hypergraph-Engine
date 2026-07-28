//! Logical mutations and their zero-copy payload encodings.
//!
//! Each payload starts with a fixed `#[repr(C, packed)]` record cast with
//! `bytemuck`, followed by variable-length trailing data.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::format::OpCode;
use crate::ids::{HyperedgeId, NodeId};

/// Fixed part of an [`OpCode::InsertNode`] payload; followed by the UTF-8 label.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NodeRecord {
    /// Node identifier.
    pub id: NodeId,
    /// Stable label hash.
    pub label_hash: u64,
    /// First-observed timestamp.
    pub created_at: i64,
    /// Label length in bytes.
    pub label_len: u32,
}

/// Fixed part of an [`OpCode::InsertHyperedge`] payload; followed by
/// `participant_count` [`NodeId`]s, the UTF-8 event type,
/// `embedding_len` little-endian `f32`s and `summary_len` UTF-8 bytes.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct HyperedgeRecord {
    /// Hyperedge identifier.
    pub id: HyperedgeId,
    /// Stable event type hash.
    pub event_type_hash: u64,
    /// Start of validity.
    pub valid_from: i64,
    /// End of validity (meaningful only if `has_valid_to == 1`).
    pub valid_to: i64,
    /// Base confidence.
    pub decay_weight: f32,
    /// Number of participants.
    pub participant_count: u32,
    /// Event type length in bytes.
    pub event_type_len: u32,
    /// Number of embedding components.
    pub embedding_len: u32,
    /// Summary length in bytes (meaningful only if `has_summary == 1`).
    pub summary_len: u32,
    /// `1` if `valid_to` is set.
    pub has_valid_to: u8,
    /// `1` if an embedding follows.
    pub has_embedding: u8,
    /// `1` if a summary follows.
    pub has_summary: u8,
    /// Reserved, zero.
    pub reserved: u8,
}

/// [`OpCode::UpdateValidity`] payload.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ValidityRecord {
    /// Target hyperedge.
    pub id: HyperedgeId,
    /// New end of validity (meaningful only if `has_valid_to == 1`).
    pub valid_to: i64,
    /// `1` if `valid_to` is set.
    pub has_valid_to: u8,
}

/// [`OpCode::SoftDelete`] payload.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DeleteRecord {
    /// `0` = node, `1` = hyperedge.
    pub target_kind: u8,
    /// Slot index.
    pub index: u32,
    /// Slot generation.
    pub generation: u32,
}

/// [`OpCode::BatchCommit`] payload.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BatchCommitRecord {
    /// Sequence id of the first frame in the batch.
    pub first_seq: u64,
    /// Number of frames in the batch (excluding the commit frame).
    pub frame_count: u32,
}

/// Target of a soft delete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteTarget {
    /// A node without incident hyperedges.
    Node(NodeId),
    /// A hyperedge.
    Hyperedge(HyperedgeId),
}

/// A logical state change; the unit of both live application and replay.
#[derive(Debug, Clone, PartialEq)]
pub enum Mutation {
    /// Insert an entity node into slot `id`.
    InsertNode {
        /// Target slot and generation.
        id: NodeId,
        /// Stable label hash.
        label_hash: u64,
        /// First-observed timestamp.
        created_at: i64,
        /// Canonical label.
        label: Arc<str>,
    },
    /// Insert a hyperedge into slot `id` and link it to its participants.
    InsertHyperedge {
        /// Target slot and generation.
        id: HyperedgeId,
        /// Stable event type hash.
        event_type_hash: u64,
        /// Event type string.
        event_type: Arc<str>,
        /// Participants (distinct).
        participants: Vec<NodeId>,
        /// Start of validity.
        valid_from: i64,
        /// End of validity.
        valid_to: Option<i64>,
        /// Base confidence.
        decay_weight: f32,
        /// Optional embedding.
        embedding: Option<Arc<[f32]>>,
        /// Optional text payload.
        summary: Option<Arc<str>>,
    },
    /// Set a hyperedge's validity end.
    UpdateValidity {
        /// Target hyperedge.
        id: HyperedgeId,
        /// New end of validity.
        valid_to: Option<i64>,
    },
    /// Remove a node or hyperedge from the live graph.
    SoftDelete(DeleteTarget),
}

/// A payload that passed its checksum but does not decode.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("malformed {op:?} payload: {reason}")]
pub struct DecodeError {
    /// Frame op code.
    pub op: OpCode,
    /// Description.
    pub reason: &'static str,
}

fn take<'a>(buf: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    if buf.len() < n {
        return None;
    }
    let (head, tail) = buf.split_at(n);
    *buf = tail;
    Some(head)
}

fn take_pod<T: Pod>(buf: &mut &[u8]) -> Option<T> {
    take(buf, std::mem::size_of::<T>()).map(bytemuck::pod_read_unaligned)
}

fn take_str(buf: &mut &[u8], len: usize) -> Option<Arc<str>> {
    take(buf, len)
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(Arc::from)
}

const fn encode_opt(v: Option<i64>) -> (i64, u8) {
    match v {
        Some(t) => (t, 1),
        None => (0, 0),
    }
}

fn decode_opt(value: i64, flag: u8) -> Option<Option<i64>> {
    match flag {
        0 => Some(None),
        1 => Some(Some(value)),
        _ => None,
    }
}

impl Mutation {
    /// Op code of the frame carrying this mutation.
    pub fn op_code(&self) -> OpCode {
        match self {
            Mutation::InsertNode { .. } => OpCode::InsertNode,
            Mutation::InsertHyperedge { .. } => OpCode::InsertHyperedge,
            Mutation::UpdateValidity { .. } => OpCode::UpdateValidity,
            Mutation::SoftDelete(_) => OpCode::SoftDelete,
        }
    }

    /// Appends the payload encoding to `out`.
    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Mutation::InsertNode {
                id,
                label_hash,
                created_at,
                label,
            } => {
                let record = NodeRecord {
                    id: *id,
                    label_hash: *label_hash,
                    created_at: *created_at,
                    label_len: label.len() as u32,
                };
                out.extend_from_slice(bytemuck::bytes_of(&record));
                out.extend_from_slice(label.as_bytes());
            }
            Mutation::InsertHyperedge {
                id,
                event_type_hash,
                event_type,
                participants,
                valid_from,
                valid_to,
                decay_weight,
                embedding,
                summary,
            } => {
                let (valid_to, has_valid_to) = encode_opt(*valid_to);
                let record = HyperedgeRecord {
                    id: *id,
                    event_type_hash: *event_type_hash,
                    valid_from: *valid_from,
                    valid_to,
                    decay_weight: *decay_weight,
                    participant_count: participants.len() as u32,
                    event_type_len: event_type.len() as u32,
                    embedding_len: embedding.as_ref().map_or(0, |e| e.len() as u32),
                    summary_len: summary.as_ref().map_or(0, |s| s.len() as u32),
                    has_valid_to,
                    has_embedding: u8::from(embedding.is_some()),
                    has_summary: u8::from(summary.is_some()),
                    reserved: 0,
                };
                out.extend_from_slice(bytemuck::bytes_of(&record));
                out.extend_from_slice(bytemuck::cast_slice(participants));
                out.extend_from_slice(event_type.as_bytes());
                if let Some(embedding) = embedding {
                    out.extend_from_slice(bytemuck::cast_slice(embedding));
                }
                if let Some(summary) = summary {
                    out.extend_from_slice(summary.as_bytes());
                }
            }
            Mutation::UpdateValidity { id, valid_to } => {
                let (valid_to, has_valid_to) = encode_opt(*valid_to);
                let record = ValidityRecord {
                    id: *id,
                    valid_to,
                    has_valid_to,
                };
                out.extend_from_slice(bytemuck::bytes_of(&record));
            }
            Mutation::SoftDelete(target) => {
                let (target_kind, index, generation) = match *target {
                    DeleteTarget::Node(id) => (0, id.index, id.generation),
                    DeleteTarget::Hyperedge(id) => (1, id.index, id.generation),
                };
                let record = DeleteRecord {
                    target_kind,
                    index,
                    generation,
                };
                out.extend_from_slice(bytemuck::bytes_of(&record));
            }
        }
    }

    /// Decodes a payload; the whole payload must be consumed.
    pub fn decode(op: OpCode, payload: &[u8]) -> Result<Self, DecodeError> {
        let err = |reason| DecodeError { op, reason };
        let mut buf = payload;
        let mutation = match op {
            OpCode::InsertNode => {
                let record: NodeRecord = take_pod(&mut buf).ok_or(err("short node record"))?;
                let label =
                    take_str(&mut buf, record.label_len as usize).ok_or(err("bad label"))?;
                Mutation::InsertNode {
                    id: record.id,
                    label_hash: record.label_hash,
                    created_at: record.created_at,
                    label,
                }
            }
            OpCode::InsertHyperedge => {
                let record: HyperedgeRecord =
                    take_pod(&mut buf).ok_or(err("short hyperedge record"))?;
                let count = record.participant_count as usize;
                let raw = take(&mut buf, count * std::mem::size_of::<NodeId>())
                    .ok_or(err("short participant list"))?;
                let participants: Vec<NodeId> = bytemuck::cast_slice(raw).to_vec();
                let event_type = take_str(&mut buf, record.event_type_len as usize)
                    .ok_or(err("bad event type"))?;
                let embedding = match record.has_embedding {
                    0 => None,
                    1 => {
                        let len = record.embedding_len as usize;
                        let raw = take(&mut buf, len * 4).ok_or(err("short embedding"))?;
                        let values: Vec<f32> = raw
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        Some(Arc::from(values))
                    }
                    _ => return Err(err("bad embedding flag")),
                };
                let summary = match record.has_summary {
                    0 => None,
                    1 => Some(
                        take_str(&mut buf, record.summary_len as usize)
                            .ok_or(err("bad summary"))?,
                    ),
                    _ => return Err(err("bad summary flag")),
                };
                Mutation::InsertHyperedge {
                    id: record.id,
                    event_type_hash: record.event_type_hash,
                    event_type,
                    participants,
                    valid_from: record.valid_from,
                    valid_to: decode_opt(record.valid_to, record.has_valid_to)
                        .ok_or(err("bad valid_to flag"))?,
                    decay_weight: record.decay_weight,
                    embedding,
                    summary,
                }
            }
            OpCode::UpdateValidity => {
                let record: ValidityRecord =
                    take_pod(&mut buf).ok_or(err("short validity record"))?;
                Mutation::UpdateValidity {
                    id: record.id,
                    valid_to: decode_opt(record.valid_to, record.has_valid_to)
                        .ok_or(err("bad valid_to flag"))?,
                }
            }
            OpCode::SoftDelete => {
                let record: DeleteRecord = take_pod(&mut buf).ok_or(err("short delete record"))?;
                Mutation::SoftDelete(match record.target_kind {
                    0 => DeleteTarget::Node(NodeId::new(record.index, record.generation)),
                    1 => DeleteTarget::Hyperedge(HyperedgeId::new(record.index, record.generation)),
                    _ => return Err(err("bad delete target")),
                })
            }
            OpCode::BatchCommit => return Err(err("batch commit is not a mutation")),
        };
        if !buf.is_empty() {
            return Err(err("trailing bytes"));
        }
        Ok(mutation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(m: Mutation) {
        let mut buf = vec![0xAA];
        m.encode(&mut buf);
        assert_eq!(Mutation::decode(m.op_code(), &buf[1..]).unwrap(), m);
        assert!(Mutation::decode(m.op_code(), &buf[1..buf.len() - 1]).is_err());
    }

    #[test]
    fn all_mutations_round_trip_from_unaligned_buffers() {
        round_trip(Mutation::InsertNode {
            id: NodeId::new(4, 1),
            label_hash: 99,
            created_at: -5,
            label: Arc::from("Mira Murati"),
        });
        round_trip(Mutation::InsertHyperedge {
            id: HyperedgeId::new(7, 2),
            event_type_hash: 1234,
            event_type: Arc::from("CEO_OF"),
            participants: vec![NodeId::new(1, 0), NodeId::new(4, 1)],
            valid_from: 1_700_000_000,
            valid_to: Some(1_800_000_000),
            decay_weight: 0.75,
            embedding: Some(Arc::from(vec![0.5f32, -1.25, 3.0])),
            summary: Some(Arc::from("Mira Murati became CEO — interim")),
        });
        round_trip(Mutation::InsertHyperedge {
            id: HyperedgeId::new(0, 0),
            event_type_hash: 1,
            event_type: Arc::from("E"),
            participants: vec![NodeId::new(0, 0)],
            valid_from: 0,
            valid_to: None,
            decay_weight: 1.0,
            embedding: None,
            summary: None,
        });
        round_trip(Mutation::InsertHyperedge {
            id: HyperedgeId::new(2, 0),
            event_type_hash: 2,
            event_type: Arc::from("MESSAGE_HUMAN"),
            participants: vec![NodeId::new(0, 0)],
            valid_from: 5,
            valid_to: None,
            decay_weight: 1.0,
            embedding: None,
            summary: Some(Arc::from("hi")),
        });
        round_trip(Mutation::UpdateValidity {
            id: HyperedgeId::new(3, 3),
            valid_to: Some(42),
        });
        round_trip(Mutation::SoftDelete(DeleteTarget::Hyperedge(
            HyperedgeId::new(9, 9),
        )));
        round_trip(Mutation::SoftDelete(DeleteTarget::Node(NodeId::new(8, 0))));
    }

    #[test]
    fn rejects_bad_flags() {
        let mut buf = Vec::new();
        Mutation::UpdateValidity {
            id: HyperedgeId::new(1, 1),
            valid_to: None,
        }
        .encode(&mut buf);
        *buf.last_mut().unwrap() = 7;
        assert!(Mutation::decode(OpCode::UpdateValidity, &buf).is_err());
    }
}

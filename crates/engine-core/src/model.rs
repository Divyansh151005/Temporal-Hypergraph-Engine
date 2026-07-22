//! Core node and hyperedge structures of the bipartite hypergraph.

use smallvec::SmallVec;

use crate::ids::{HyperedgeId, NodeId};

/// Incident hyperedges of a node; up to 8 are stored inline.
pub type IncidentEdges = SmallVec<[HyperedgeId; 8]>;

/// Participants of a hyperedge; up to 6 are stored inline.
pub type Participants = SmallVec<[NodeId; 6]>;

/// `embedding_offset` value of hyperedges without an embedding.
pub const NO_EMBEDDING: u64 = u64::MAX;

/// `payload_offset` value of hyperedges without a text payload.
pub const NO_PAYLOAD: u32 = u32::MAX;

/// An atomic entity, concept, or token sequence (bipartite set A).
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalNode {
    /// Arena identifier.
    pub id: NodeId,
    /// Stable hash of the canonical entity label.
    pub label_hash: u64,
    /// Unix seconds at which the entity was first observed.
    pub created_at: i64,
    /// Incident hyperedges.
    pub hyperedges: IncidentEdges,
}

/// An N-ary relation binding N nodes (bipartite set B).
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalHyperedge {
    /// Arena identifier.
    pub id: HyperedgeId,
    /// Stable hash of the event type string.
    pub event_type_hash: u64,
    /// Bound nodes, in insertion order.
    pub participants: Participants,
    /// Start of validity (Unix seconds).
    pub valid_from: i64,
    /// End of validity (Unix seconds, exclusive); `None` while ongoing.
    pub valid_to: Option<i64>,
    /// `true` while `valid_to` is `None`.
    pub is_ongoing: bool,
    /// Base confidence `w0` in `[0, 1]`; time decay is applied at query time.
    pub decay_weight: f32,
    /// Offset into the embedding table, or [`NO_EMBEDDING`].
    pub embedding_offset: u64,
    /// Offset into the payload arena, or [`NO_PAYLOAD`].
    pub payload_offset: u32,
}

impl TemporalHyperedge {
    /// Whether the fact holds at `timestamp` (`valid_from <= t < valid_to`).
    #[inline]
    pub fn is_valid_at(&self, timestamp: i64) -> bool {
        self.valid_from <= timestamp && self.valid_to.map_or(true, |end| timestamp < end)
    }

    /// Whether an embedding is attached.
    #[inline]
    pub fn has_embedding(&self) -> bool {
        self.embedding_offset != NO_EMBEDDING
    }

    /// Whether a text payload is attached.
    #[inline]
    pub fn has_payload(&self) -> bool {
        self.payload_offset != NO_PAYLOAD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyperedge_fits_memory_budget() {
        assert!(std::mem::size_of::<TemporalHyperedge>() <= 128);
        assert!(std::mem::size_of::<TemporalNode>() <= 128);
    }

    #[test]
    fn validity_window_is_half_open() {
        let edge = TemporalHyperedge {
            id: HyperedgeId::new(0, 0),
            event_type_hash: 0,
            participants: Participants::new(),
            valid_from: 10,
            valid_to: Some(20),
            is_ongoing: false,
            decay_weight: 1.0,
            embedding_offset: NO_EMBEDDING,
            payload_offset: NO_PAYLOAD,
        };
        assert!(!edge.is_valid_at(9));
        assert!(edge.is_valid_at(10));
        assert!(edge.is_valid_at(19));
        assert!(!edge.is_valid_at(20));
    }
}

//! Generational bipartite arena.
//!
//! [`BipartiteArena`] is an immutable-by-convention snapshot shared with
//! readers. [`ArenaWriter`] owns the mutable working copy plus free lists and
//! applies [`Mutation`]s; publishing a snapshot is a cheap copy-on-write clone.

use std::sync::Arc;

use crate::error::EngineError;
use crate::ids::{HyperedgeId, NodeId};
use crate::model::{
    IncidentEdges, Participants, TemporalHyperedge, TemporalNode, NO_EMBEDDING, NO_PAYLOAD,
};
use crate::paged::PagedVec;
use crate::payload::PayloadArena;
use crate::wal::record::{DeleteTarget, Mutation};

/// One arena slot.
#[derive(Debug, Clone, PartialEq)]
pub enum Slot<T> {
    /// Live value; its id carries the slot generation.
    Occupied(T),
    /// Free slot remembering the generation of its last occupant.
    Vacant {
        /// Last generation used in this slot.
        generation: u32,
    },
}

/// Read-optimized snapshot of the bipartite graph.
#[derive(Debug, Clone, Default)]
pub struct BipartiteArena {
    /// Set A: entity nodes.
    pub nodes: PagedVec<Slot<TemporalNode>>,
    /// Set B: hyperedges.
    pub hyperedges: PagedVec<Slot<TemporalHyperedge>>,
    embeddings: PagedVec<Option<Arc<[f32]>>>,
    payloads: PayloadArena,
    live_nodes: u32,
    live_hyperedges: u32,
}

/// Fixed-size bitset used to mark visited slots during traversal.
struct VisitedSet(Vec<u64>);

impl VisitedSet {
    fn new(len: usize) -> Self {
        Self(vec![0; len.div_ceil(64)])
    }

    /// Marks `index`; returns `true` if it was not marked before.
    #[inline]
    fn insert(&mut self, index: u32) -> bool {
        let (word, bit) = ((index / 64) as usize, 1u64 << (index % 64));
        match self.0.get_mut(word) {
            Some(w) if *w & bit == 0 => {
                *w |= bit;
                true
            }
            _ => false,
        }
    }
}

impl BipartiteArena {
    /// Live node by id (generation must match).
    #[inline]
    pub fn node(&self, id: NodeId) -> Option<&TemporalNode> {
        match self.nodes.get(id.index as usize)? {
            Slot::Occupied(node) if node.id == id => Some(node),
            _ => None,
        }
    }

    /// Live hyperedge by id (generation must match).
    #[inline]
    pub fn hyperedge(&self, id: HyperedgeId) -> Option<&TemporalHyperedge> {
        match self.hyperedges.get(id.index as usize)? {
            Slot::Occupied(edge) if edge.id == id => Some(edge),
            _ => None,
        }
    }

    /// Embedding attached to `edge`.
    pub fn embedding(&self, edge: &TemporalHyperedge) -> Option<&[f32]> {
        self.embedding_arc(edge).map(|e| &**e)
    }

    /// Shared handle to the embedding attached to `edge`.
    pub fn embedding_arc(&self, edge: &TemporalHyperedge) -> Option<&Arc<[f32]>> {
        if edge.embedding_offset == NO_EMBEDDING {
            return None;
        }
        self.embeddings
            .get(edge.embedding_offset as usize)?
            .as_ref()
    }

    /// Text payload attached to `edge`.
    pub fn payload(&self, edge: &TemporalHyperedge) -> Option<&Arc<str>> {
        self.payloads.get(edge.payload_offset)
    }

    /// Total UTF-8 bytes of live text payloads.
    pub fn payload_bytes(&self) -> u64 {
        self.payloads.live_bytes()
    }

    /// Number of live nodes.
    pub fn live_node_count(&self) -> u32 {
        self.live_nodes
    }

    /// Number of live hyperedges.
    pub fn live_hyperedge_count(&self) -> u32 {
        self.live_hyperedges
    }

    /// Iterates live nodes in slot order.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &TemporalNode> + '_ {
        self.nodes.iter().filter_map(|slot| match slot {
            Slot::Occupied(node) => Some(node),
            Slot::Vacant { .. } => None,
        })
    }

    /// Iterates live hyperedges in slot order.
    pub fn iter_hyperedges(&self) -> impl Iterator<Item = &TemporalHyperedge> + '_ {
        self.hyperedges.iter().filter_map(|slot| match slot {
            Slot::Occupied(edge) => Some(edge),
            Slot::Vacant { .. } => None,
        })
    }

    /// Breadth-first node -> hyperedge -> node expansion for `k_hops` hops,
    /// returning every hyperedge valid at `as_of` that was reached, in
    /// discovery order. Hyperedges not valid at `as_of` are not expanded.
    pub fn traverse(
        &self,
        start_nodes: &[NodeId],
        k_hops: usize,
        as_of: i64,
        limit: usize,
    ) -> Vec<TemporalHyperedge> {
        let mut out = Vec::new();
        if k_hops == 0 || limit == 0 {
            return out;
        }
        let mut seen_nodes = VisitedSet::new(self.nodes.len());
        let mut seen_edges = VisitedSet::new(self.hyperedges.len());
        let mut frontier: Vec<NodeId> = start_nodes
            .iter()
            .copied()
            .filter(|id| self.node(*id).is_some() && seen_nodes.insert(id.index))
            .collect();
        let mut next = Vec::new();

        for hop in 0..k_hops {
            let expand = hop + 1 < k_hops;
            for &node_id in &frontier {
                let Some(node) = self.node(node_id) else {
                    continue;
                };
                for &edge_id in &node.hyperedges {
                    if !seen_edges.insert(edge_id.index) {
                        continue;
                    }
                    let Some(edge) = self.hyperedge(edge_id) else {
                        continue;
                    };
                    if !edge.is_valid_at(as_of) {
                        continue;
                    }
                    out.push(edge.clone());
                    if out.len() >= limit {
                        return out;
                    }
                    if expand {
                        next.extend(
                            edge.participants
                                .iter()
                                .copied()
                                .filter(|p| seen_nodes.insert(p.index)),
                        );
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            std::mem::swap(&mut frontier, &mut next);
            next.clear();
        }
        out
    }
}

/// Error raised when a mutation does not fit the current arena state.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ApplyError(pub String);

fn apply_err<T>(msg: impl Into<String>) -> Result<T, ApplyError> {
    Err(ApplyError(msg.into()))
}

/// Mutable working copy of the arena plus allocation state.
#[derive(Debug, Default)]
pub struct ArenaWriter {
    arena: BipartiteArena,
    free_nodes: Vec<u32>,
    free_hyperedges: Vec<u32>,
}

impl ArenaWriter {
    /// Creates an empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Wraps an existing snapshot, rebuilding free lists.
    pub fn from_arena(arena: BipartiteArena) -> Self {
        let mut writer = Self {
            arena,
            free_nodes: Vec::new(),
            free_hyperedges: Vec::new(),
        };
        writer.rebuild_free_lists();
        writer
    }

    /// Current state.
    pub fn arena(&self) -> &BipartiteArena {
        &self.arena
    }

    /// Recomputes free lists from vacant slots. Slots whose generation is
    /// exhausted are retired so stale ids can never alias a new occupant.
    pub fn rebuild_free_lists(&mut self) {
        fn vacant<T>(slots: &PagedVec<Slot<T>>) -> Vec<u32> {
            let mut free: Vec<u32> = slots
                .iter()
                .enumerate()
                .filter_map(|(i, slot)| match slot {
                    Slot::Vacant { generation } if *generation != u32::MAX => Some(i as u32),
                    _ => None,
                })
                .collect();
            free.reverse();
            free
        }
        self.free_nodes = vacant(&self.arena.nodes);
        self.free_hyperedges = vacant(&self.arena.hyperedges);
    }

    fn reserve<T>(
        slots: &PagedVec<Slot<T>>,
        free: &mut Vec<u32>,
    ) -> Result<(u32, u32), EngineError> {
        if let Some(index) = free.pop() {
            let generation = match slots.get(index as usize) {
                Some(Slot::Vacant { generation }) => generation.wrapping_add(1),
                _ => unreachable!("free list only holds vacant slots"),
            };
            return Ok((index, generation));
        }
        let index = u32::try_from(slots.len())
            .ok()
            .filter(|i| *i != u32::MAX)
            .ok_or(EngineError::CapacityExceeded)?;
        Ok((index, 0))
    }

    /// Picks the slot for the next node. The id must be consumed by an
    /// [`Mutation::InsertNode`] applied before the next reservation.
    pub fn reserve_node_id(&mut self) -> Result<NodeId, EngineError> {
        Self::reserve(&self.arena.nodes, &mut self.free_nodes).map(|(i, g)| NodeId::new(i, g))
    }

    /// Picks the slot for the next hyperedge (same contract as
    /// [`Self::reserve_node_id`]).
    pub fn reserve_hyperedge_id(&mut self) -> Result<HyperedgeId, EngineError> {
        Self::reserve(&self.arena.hyperedges, &mut self.free_hyperedges)
            .map(|(i, g)| HyperedgeId::new(i, g))
    }

    /// Restores generations of vacant slots (checkpoint load) and grows the
    /// tables to the recorded slot counts.
    pub fn restore_generations(&mut self, node_generations: &[u32], hyperedge_generations: &[u32]) {
        fn restore<T: Clone>(slots: &mut PagedVec<Slot<T>>, generations: &[u32]) {
            slots.extend_to(generations.len(), Slot::Vacant { generation: 0 });
            for (i, &generation) in generations.iter().enumerate() {
                if let Some(slot @ Slot::Vacant { .. }) = slots.get_mut(i) {
                    *slot = Slot::Vacant { generation };
                }
            }
        }
        restore(&mut self.arena.nodes, node_generations);
        restore(&mut self.arena.hyperedges, hyperedge_generations);
        self.arena
            .embeddings
            .extend_to(self.arena.hyperedges.len(), None);
    }

    /// Places `value` into `index`, which must be vacant (growing the table
    /// with vacant slots if needed).
    fn place<T: Clone>(
        slots: &mut PagedVec<Slot<T>>,
        index: u32,
        value: T,
    ) -> Result<(), ApplyError> {
        let index = index as usize;
        if index == slots.len() {
            slots.push(Slot::Occupied(value));
            return Ok(());
        }
        slots.extend_to(index + 1, Slot::Vacant { generation: 0 });
        match slots.get_mut(index) {
            Some(slot @ Slot::Vacant { .. }) => {
                *slot = Slot::Occupied(value);
                Ok(())
            }
            _ => apply_err(format!("slot {index} is already occupied")),
        }
    }

    fn release(free: &mut Vec<u32>, index: u32, generation: u32) {
        if generation != u32::MAX {
            free.push(index);
        }
    }

    /// Applies one mutation. On error the arena is unchanged.
    pub fn apply(&mut self, mutation: &Mutation) -> Result<(), ApplyError> {
        match mutation {
            Mutation::InsertNode {
                id,
                label_hash,
                created_at,
                ..
            } => {
                let node = TemporalNode {
                    id: *id,
                    label_hash: *label_hash,
                    created_at: *created_at,
                    hyperedges: IncidentEdges::new(),
                };
                Self::place(&mut self.arena.nodes, id.index, node)?;
                self.arena.live_nodes += 1;
            }
            Mutation::InsertHyperedge {
                id,
                event_type_hash,
                participants,
                valid_from,
                valid_to,
                decay_weight,
                embedding,
                summary,
                ..
            } => {
                if participants.is_empty() {
                    return apply_err(format!("hyperedge {id:?} has no participants"));
                }
                for (i, p) in participants.iter().enumerate() {
                    if self.arena.node(*p).is_none() {
                        return apply_err(format!(
                            "hyperedge {id:?} references missing node {p:?}"
                        ));
                    }
                    if participants[..i].contains(p) {
                        return apply_err(format!("hyperedge {id:?} repeats participant {p:?}"));
                    }
                }
                if valid_to.is_some_and(|end| end <= *valid_from) {
                    return apply_err(format!("hyperedge {id:?} has an empty validity window"));
                }
                let edge = TemporalHyperedge {
                    id: *id,
                    event_type_hash: *event_type_hash,
                    participants: Participants::from_slice(participants),
                    valid_from: *valid_from,
                    valid_to: *valid_to,
                    is_ongoing: valid_to.is_none(),
                    decay_weight: *decay_weight,
                    embedding_offset: if embedding.is_some() {
                        u64::from(id.index)
                    } else {
                        NO_EMBEDDING
                    },
                    payload_offset: if summary.is_some() {
                        id.index
                    } else {
                        NO_PAYLOAD
                    },
                };
                Self::place(&mut self.arena.hyperedges, id.index, edge)?;
                self.arena.payloads.put(id.index, summary.clone());
                let slot = id.index as usize;
                self.arena.embeddings.extend_to(slot + 1, None);
                *self.arena.embeddings.get_mut(slot).expect("extended above") = embedding.clone();
                for p in participants {
                    if let Some(Slot::Occupied(node)) = self.arena.nodes.get_mut(p.index as usize) {
                        node.hyperedges.push(*id);
                    }
                }
                self.arena.live_hyperedges += 1;
            }
            Mutation::UpdateValidity { id, valid_to } => {
                let Some(Slot::Occupied(edge)) = self.arena.hyperedges.get(id.index as usize)
                else {
                    return apply_err(format!("validity update for missing hyperedge {id:?}"));
                };
                if edge.id != *id {
                    return apply_err(format!("validity update for stale hyperedge {id:?}"));
                }
                if valid_to.is_some_and(|end| end <= edge.valid_from) {
                    return apply_err(format!("validity update empties hyperedge {id:?}"));
                }
                if let Some(Slot::Occupied(edge)) = self.arena.hyperedges.get_mut(id.index as usize)
                {
                    edge.valid_to = *valid_to;
                    edge.is_ongoing = valid_to.is_none();
                }
            }
            Mutation::SoftDelete(DeleteTarget::Hyperedge(id)) => {
                let Some(edge) = self.arena.hyperedge(*id) else {
                    return apply_err(format!("delete of missing hyperedge {id:?}"));
                };
                let participants = edge.participants.clone();
                let slot = id.index as usize;
                *self.arena.hyperedges.get_mut(slot).expect("checked above") = Slot::Vacant {
                    generation: id.generation,
                };
                *self
                    .arena
                    .embeddings
                    .get_mut(slot)
                    .expect("parallel to hyperedges") = None;
                self.arena.payloads.clear(id.index);
                for p in participants {
                    if let Some(Slot::Occupied(node)) = self.arena.nodes.get_mut(p.index as usize) {
                        node.hyperedges.retain(|h| *h != *id);
                    }
                }
                self.arena.live_hyperedges -= 1;
                Self::release(&mut self.free_hyperedges, id.index, id.generation);
            }
            Mutation::SoftDelete(DeleteTarget::Node(id)) => {
                let Some(node) = self.arena.node(*id) else {
                    return apply_err(format!("delete of missing node {id:?}"));
                };
                if !node.hyperedges.is_empty() {
                    return apply_err(format!("node {id:?} still has incident hyperedges"));
                }
                *self
                    .arena
                    .nodes
                    .get_mut(id.index as usize)
                    .expect("checked above") = Slot::Vacant {
                    generation: id.generation,
                };
                self.arena.live_nodes -= 1;
                Self::release(&mut self.free_nodes, id.index, id.generation);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_node(w: &mut ArenaWriter, hash: u64) -> NodeId {
        let id = w.reserve_node_id().unwrap();
        w.apply(&Mutation::InsertNode {
            id,
            label_hash: hash,
            created_at: 0,
            label: Arc::from("n"),
        })
        .unwrap();
        id
    }

    fn insert_edge(
        w: &mut ArenaWriter,
        nodes: &[NodeId],
        from: i64,
        to: Option<i64>,
    ) -> HyperedgeId {
        let id = w.reserve_hyperedge_id().unwrap();
        w.apply(&Mutation::InsertHyperedge {
            id,
            event_type_hash: 1,
            event_type: Arc::from("E"),
            participants: nodes.to_vec(),
            valid_from: from,
            valid_to: to,
            decay_weight: 1.0,
            embedding: Some(Arc::from(vec![from as f32])),
            summary: Some(Arc::from(format!("edge@{from}"))),
        })
        .unwrap();
        id
    }

    #[test]
    fn slot_reuse_bumps_generation() {
        let mut w = ArenaWriter::new();
        let a = insert_node(&mut w, 1);
        let b = insert_node(&mut w, 2);
        let e = insert_edge(&mut w, &[a, b], 0, None);
        w.apply(&Mutation::SoftDelete(DeleteTarget::Hyperedge(e)))
            .unwrap();
        assert!(w.arena().hyperedge(e).is_none());
        assert!(w.arena().node(a).unwrap().hyperedges.is_empty());

        let e2 = insert_edge(&mut w, &[a, b], 5, None);
        assert_eq!(e2, HyperedgeId::new(e.index, e.generation + 1));
        assert!(
            w.arena().hyperedge(e).is_none(),
            "stale id must not resolve"
        );
        assert_eq!(
            w.arena().embedding(w.arena().hyperedge(e2).unwrap()),
            Some(&[5.0][..])
        );
        let e2_edge = w.arena().hyperedge(e2).unwrap();
        assert_eq!(w.arena().payload(e2_edge).map(|s| &**s), Some("edge@5"));
        assert_eq!(w.arena().payload_bytes(), "edge@5".len() as u64);
    }

    #[test]
    fn rejects_inconsistent_mutations_without_side_effects() {
        let mut w = ArenaWriter::new();
        let a = insert_node(&mut w, 1);
        let ghost = NodeId::new(42, 0);
        let id = w.reserve_hyperedge_id().unwrap();
        let bad = Mutation::InsertHyperedge {
            id,
            event_type_hash: 1,
            event_type: Arc::from("E"),
            participants: vec![a, ghost],
            valid_from: 0,
            valid_to: None,
            decay_weight: 1.0,
            embedding: None,
            summary: None,
        };
        assert!(w.apply(&bad).is_err());
        assert_eq!(w.arena().live_hyperedge_count(), 0);
        assert!(w.arena().node(a).unwrap().hyperedges.is_empty());

        let e = insert_edge(&mut w, &[a], 0, None);
        assert!(w
            .apply(&Mutation::SoftDelete(DeleteTarget::Node(a)))
            .is_err());
        assert!(w
            .apply(&Mutation::UpdateValidity {
                id: e,
                valid_to: Some(0)
            })
            .is_err());
    }

    #[test]
    fn published_snapshot_is_isolated() {
        let mut w = ArenaWriter::new();
        let a = insert_node(&mut w, 1);
        let e = insert_edge(&mut w, &[a], 0, None);
        let snapshot = w.arena().clone();
        w.apply(&Mutation::UpdateValidity {
            id: e,
            valid_to: Some(10),
        })
        .unwrap();
        insert_edge(&mut w, &[a], 10, None);
        assert_eq!(snapshot.hyperedge(e).unwrap().valid_to, None);
        assert_eq!(snapshot.node(a).unwrap().hyperedges.len(), 1);
        assert_eq!(w.arena().node(a).unwrap().hyperedges.len(), 2);
    }

    #[test]
    fn three_hop_traversal_on_chain() {
        let mut w = ArenaWriter::new();
        let n: Vec<NodeId> = (0..6).map(|i| insert_node(&mut w, i)).collect();
        let e: Vec<HyperedgeId> = (0..5)
            .map(|i| insert_edge(&mut w, &[n[i], n[i + 1]], 0, None))
            .collect();
        let ids = |k| {
            w.arena()
                .traverse(&[n[0]], k, 1, usize::MAX)
                .iter()
                .map(|h| h.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(1), vec![e[0]]);
        assert_eq!(ids(2), vec![e[0], e[1]]);
        assert_eq!(ids(3), vec![e[0], e[1], e[2]]);
        assert_eq!(w.arena().traverse(&[n[0]], 3, 1, 2).len(), 2);
    }

    #[test]
    fn traversal_respects_validity_window() {
        let mut w = ArenaWriter::new();
        let a = insert_node(&mut w, 1);
        let b = insert_node(&mut w, 2);
        let c = insert_node(&mut w, 3);
        let old = insert_edge(&mut w, &[a, b], 0, Some(100));
        let new = insert_edge(&mut w, &[a, c], 100, None);
        let at = |t| {
            w.arena()
                .traverse(&[a], 1, t, usize::MAX)
                .iter()
                .map(|h| h.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(at(50), vec![old]);
        assert_eq!(at(100), vec![new]);
        assert!(at(-1).is_empty());
    }

    #[test]
    fn restore_generations_and_free_lists() {
        let mut w = ArenaWriter::new();
        let a = insert_node(&mut w, 1);
        w.restore_generations(&[0, 0, 6], &[3]);
        w.rebuild_free_lists();
        assert_eq!(w.reserve_node_id().unwrap(), NodeId::new(1, 1));
        let mut w2 = ArenaWriter::from_arena(w.arena().clone());
        assert!(w2.arena().node(a).is_some());
        assert_eq!(w2.reserve_hyperedge_id().unwrap(), HyperedgeId::new(0, 4));
    }
}

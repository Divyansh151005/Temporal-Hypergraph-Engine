//! Temporal eviction planning.

use engine_core::{ConcurrentHypergraphEngine, HyperedgeId, TemporalHyperedge};

use crate::hash::BlockHash;
use crate::store::KvBlockStore;

/// Blocks whose relevance at `now` is below `decay_threshold`, lowest first.
///
/// A block's relevance is the highest `confidence * decay(now - valid_from)`
/// among its associated live hyperedges, or `0` when all of them are gone.
/// Blocks without associations are never candidates.
pub fn plan_kv_evictions(
    engine: &ConcurrentHypergraphEngine,
    store: &KvBlockStore,
    decay_threshold: f32,
    now: i64,
) -> Vec<(BlockHash, f32)> {
    let mut candidates: Vec<(BlockHash, f32)> = store
        .associated_blocks()
        .into_iter()
        .filter_map(|(key, hyperedges)| {
            let edges: Vec<TemporalHyperedge> = hyperedges
                .iter()
                .filter_map(|&raw| engine.hyperedge(HyperedgeId::from_u64(raw)))
                .collect();
            let score = engine
                .decay_scores(&edges, now)
                .into_iter()
                .fold(0.0f32, f32::max);
            (score < decay_threshold).then_some((key, score))
        })
        .collect();
    candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    candidates
}

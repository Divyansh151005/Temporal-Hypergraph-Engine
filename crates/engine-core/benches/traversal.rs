//! 3-hop N-ary traversal over a 1,000,000-hyperedge graph (spec target < 2 ms).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine_core::{
    ConcurrentHypergraphEngine, EngineConfig, NewHyperedge, NodeId, Slot, TemporalHyperedge,
    TraversalQuery,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const EDGES: usize = 1_000_000;
const EVENT_TYPES: usize = 32;

fn build(nodes: usize) -> (ConcurrentHypergraphEngine, Vec<NodeId>) {
    let engine = ConcurrentHypergraphEngine::in_memory(EngineConfig {
        max_batch: 8_192,
        ..EngineConfig::default()
    });
    let mut rng = SmallRng::seed_from_u64(7);
    let started = std::time::Instant::now();
    let mut pending = Vec::with_capacity(8_192);
    for _ in 0..EDGES {
        let arity = rng.gen_range(2..=4);
        let participants: Vec<String> = (0..arity)
            .map(|_| format!("n{}", rng.gen_range(0..nodes)))
            .collect();
        let event = format!("EVENT_{}", rng.gen_range(0..EVENT_TYPES));
        let valid_from = rng.gen_range(0..1_000_000_000i64);
        pending.push(NewHyperedge::new(event, participants, valid_from));
        if pending.len() == 8_192 {
            engine.insert_batch(std::mem::take(&mut pending));
        }
    }
    engine.insert_batch(pending);
    let snapshot = engine.snapshot();
    eprintln!(
        "built {} hyperedges / {} nodes in {:.1?}; slot size {} B (TemporalHyperedge {} B)",
        snapshot.live_hyperedge_count(),
        snapshot.live_node_count(),
        started.elapsed(),
        std::mem::size_of::<Slot<TemporalHyperedge>>(),
        std::mem::size_of::<TemporalHyperedge>(),
    );
    let seeds: Vec<NodeId> = snapshot.iter_nodes().step_by(997).map(|n| n.id).collect();
    (engine, seeds)
}

fn traversal(c: &mut Criterion) {
    // Sparse: ~1M entities (mean degree ~3). Dense: 100k entities (mean
    // degree ~30), where a 3-hop expansion reaches tens of thousands of edges.
    for (name, nodes) in [
        ("sparse_1M_nodes", 1_000_000),
        ("dense_100k_nodes", 100_000),
    ] {
        let (engine, seeds) = build(nodes);
        let as_of = i64::MAX - 1;
        let reached: usize = seeds
            .iter()
            .take(200)
            .map(|s| engine.traverse_bipartite(&[*s], 3, as_of).len())
            .sum();
        eprintln!(
            "{name}: mean hyperedges reached by a 3-hop traversal: {}",
            reached / 200
        );

        let mut i = 0;
        c.bench_function(&format!("traverse_3hop_1M_edges/{name}"), |b| {
            b.iter(|| {
                i = (i + 1) % seeds.len();
                black_box(engine.traverse_bipartite(&[seeds[i]], 3, as_of))
            })
        });
        let mut j = 0;
        c.bench_function(
            &format!("traverse_3hop_1M_edges/{name}/as_of_midpoint"),
            |b| {
                b.iter(|| {
                    j = (j + 1) % seeds.len();
                    black_box(engine.traverse_bipartite(&[seeds[j]], 3, 500_000_000))
                })
            },
        );
        let mut l = 0;
        c.bench_function(&format!("traverse_3hop_1M_edges/{name}/limit_256"), |b| {
            b.iter(|| {
                l = (l + 1) % seeds.len();
                black_box(engine.traverse_bipartite_limited(&[seeds[l]], 3, as_of, 256))
            })
        });
        let labels: Vec<[String; 1]> = seeds
            .iter()
            .map(|s| [engine.node_label(*s).unwrap().to_string()])
            .collect();
        let mut q = 0;
        c.bench_function(
            &format!("traverse_query_3hop_1M_edges/{name}/top_16"),
            |b| {
                b.iter(|| {
                    q = (q + 1) % labels.len();
                    black_box(engine.traverse_query(&TraversalQuery {
                        top_k: 16,
                        ..TraversalQuery::new(&labels[q], 3, as_of)
                    }))
                })
            },
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(200);
    targets = traversal
}
criterion_main!(benches);

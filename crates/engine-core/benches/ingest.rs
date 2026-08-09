//! Write throughput: batched ingest into in-memory and WAL-backed engines,
//! and single-request latency under the group-commit path.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use engine_core::{ConcurrentHypergraphEngine, EngineConfig, NewHyperedge, SyncMode};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const BATCH: usize = 10_000;

fn workload(seed: u64) -> Vec<NewHyperedge> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..BATCH)
        .map(|_| {
            let participants: Vec<String> = (0..3)
                .map(|_| format!("n{}", rng.gen_range(0..200_000)))
                .collect();
            NewHyperedge::new(
                format!("E{}", rng.gen_range(0..16)),
                participants,
                rng.gen_range(0..1_000_000),
            )
        })
        .collect()
}

fn ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest");
    group.throughput(Throughput::Elements(BATCH as u64));
    group.sample_size(20);

    group.bench_function("insert_batch_in_memory", |b| {
        b.iter_batched(
            || (ConcurrentHypergraphEngine::new_test(), workload(1)),
            |(engine, edges)| engine.insert_batch(edges),
            BatchSize::LargeInput,
        )
    });

    for (name, sync_mode) in [
        ("insert_batch_wal_buffered", SyncMode::OsBuffered),
        ("insert_batch_wal_fsync", SyncMode::Fsync),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    let dir = tempfile::tempdir().unwrap();
                    let engine = ConcurrentHypergraphEngine::open_with_config(
                        dir.path(),
                        EngineConfig {
                            sync_mode,
                            ..EngineConfig::default()
                        },
                    )
                    .unwrap();
                    (dir, engine, workload(2))
                },
                |(_dir, engine, edges)| engine.insert_batch(edges),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();

    let mut single = c.benchmark_group("single_insert");
    single.throughput(Throughput::Elements(1));
    let dir = tempfile::tempdir().unwrap();
    let engine = ConcurrentHypergraphEngine::open_with_config(
        dir.path(),
        EngineConfig {
            sync_mode: SyncMode::OsBuffered,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(3);
    single.bench_function("insert_wal_buffered", |b| {
        b.iter(|| {
            let a = format!("n{}", rng.gen_range(0..200_000));
            let z = format!("n{}", rng.gen_range(0..200_000));
            engine.insert_hyperedge("E", &[&a, &z], rng.gen_range(0..1_000_000), None)
        })
    });
    single.finish();
}

criterion_group!(benches, ingest);
criterion_main!(benches);

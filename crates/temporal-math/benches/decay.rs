use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use temporal_math::{
    batch_compute_decay, compute_decay_scores_scalar, compute_decay_scores_simd, DecayConfig,
};

const EDGES: usize = 1_000_000;

fn decay_benches(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let deltas: Vec<f32> = (0..EDGES).map(|_| rng.gen_range(0.0..3_650.0)).collect();
    let now = 1_760_000_000i64;
    let timestamps: Vec<i64> = (0..EDGES)
        .map(|_| now - rng.gen_range(0..315_360_000))
        .collect();
    let config = DecayConfig::new(30.0, 1.0);
    let lambda = config.lambda();
    let mut out = vec![0.0f32; EDGES];

    let mut group = c.benchmark_group("decay_1m");
    group.throughput(Throughput::Elements(EDGES as u64));
    group.bench_function(BenchmarkId::new("scalar", EDGES), |b| {
        b.iter(|| compute_decay_scores_scalar(black_box(&deltas), lambda, black_box(&mut out)))
    });
    group.bench_function(BenchmarkId::new("simd_f32x8", EDGES), |b| {
        b.iter(|| compute_decay_scores_simd(black_box(&deltas), lambda, black_box(&mut out)))
    });
    group.bench_function(BenchmarkId::new("batch_from_timestamps", EDGES), |b| {
        b.iter(|| batch_compute_decay(now, black_box(&timestamps), &config, black_box(&mut out)))
    });
    group.finish();
}

criterion_group!(benches, decay_benches);
criterion_main!(benches);

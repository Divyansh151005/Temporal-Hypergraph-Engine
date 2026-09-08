//! 8 MiB PagedAttention block offload / prefetch latency.

use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use vllm_kv_store::{KvBlockLayout, KvBlockStore, KvStoreConfig};

const LAYOUT: KvBlockLayout = KvBlockLayout::LLAMA3_8B_FP16;

fn random_block(rng: &mut rand::rngs::SmallRng) -> Vec<u8> {
    let mut block = vec![0u8; LAYOUT.expected_bytes().unwrap()];
    rng.fill(block.as_mut_slice());
    block
}

/// Blocks per fresh store, bounding dirty page-cache growth so the benchmark
/// measures per-block offload latency rather than sustained disk write-back.
const BURST: u64 = 32;

/// Times `iters` stores of unique blocks, opening a fresh store every [`BURST`].
fn timed_offloads(
    iters: u64,
    template: &mut [u8],
    counter: &mut u64,
    mut open: impl FnMut() -> (Option<tempfile::TempDir>, KvBlockStore),
) -> Duration {
    let mut total = Duration::ZERO;
    let mut current = open();
    for i in 0..iters {
        if i > 0 && i % BURST == 0 {
            current = open();
        }
        *counter += 1;
        template[..8].copy_from_slice(&counter.to_le_bytes());
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&counter.to_le_bytes());
        let start = Instant::now();
        current.1.store(&key, template, LAYOUT, &[]).unwrap();
        total += start.elapsed();
    }
    total
}

fn bench_kv_block(c: &mut Criterion) {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    let mut template = random_block(&mut rng);
    let bytes = template.len() as u64;

    let mut group = c.benchmark_group("kv_block_8mib");
    group.throughput(Throughput::Bytes(bytes));
    group.sample_size(30);

    let mut counter = 0u64;
    group.bench_function("offload_mmap", |b| {
        b.iter_custom(|iters| {
            timed_offloads(iters, &mut template, &mut counter, || {
                let dir = tempfile::tempdir().unwrap();
                let store = KvBlockStore::open(dir.path(), KvStoreConfig::default()).unwrap();
                (Some(dir), store)
            })
        })
    });
    group.bench_function("offload_in_memory", |b| {
        b.iter_custom(|iters| {
            timed_offloads(iters, &mut template, &mut counter, || {
                (None, KvBlockStore::in_memory(KvStoreConfig::default()))
            })
        })
    });

    let dir = tempfile::tempdir().unwrap();
    let store = KvBlockStore::open(dir.path(), KvStoreConfig::default()).unwrap();
    let key = [0xEE; 32];
    store.store(&key, &template, LAYOUT, &[]).unwrap();
    let mut target = vec![0u8; template.len()];
    group.bench_function("prefetch", |b| {
        b.iter(|| store.load_into(&key, &mut target).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_kv_block);
criterion_main!(benches);

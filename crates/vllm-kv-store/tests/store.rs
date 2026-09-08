// Every test here uses file-backed mmap segments, which Miri cannot emulate;
// the unsafe segment code is exercised under Miri by the unit tests in src/raw.rs.
#![cfg(not(miri))]

use std::fs::OpenOptions;

use engine_core::{
    codec::format_hyperedge_id, ConcurrentHypergraphEngine, EngineConfig, NewHyperedge, SyncMode,
};
use rand::{Rng, SeedableRng};
use vllm_kv_store::{
    plan_kv_evictions, token_chain_hash, KvBlockLayout, KvBlockStore, KvError, KvStoreConfig,
    Precision, HEADER_LEN, RECORD_ALIGN,
};

const DAY: i64 = 86_400;
const TINY: KvBlockLayout = KvBlockLayout::new(2, 2, 4, 1, Precision::Fp16);

fn block(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut out = vec![0u8; len];
    rng.fill(out.as_mut_slice());
    out
}

fn key(n: u8) -> [u8; 32] {
    token_chain_hash(None, &(n as i32).to_le_bytes()).unwrap()
}

fn small_segments() -> KvStoreConfig {
    KvStoreConfig {
        segment_bytes: 1024,
        ..KvStoreConfig::default()
    }
}

#[test]
fn llama3_block_round_trips_bit_exact_and_aligned() {
    let dir = tempfile::tempdir().unwrap();
    let store = KvBlockStore::open(dir.path(), KvStoreConfig::default()).unwrap();
    let layout = KvBlockLayout::LLAMA3_8B_FP16;
    let payload = block(1, layout.expected_bytes().unwrap());
    let outcome = store.store(&key(1), &payload, layout, &[]).unwrap();
    assert!(!outcome.deduplicated);
    assert_eq!(outcome.bytes_written, (HEADER_LEN + payload.len()) as u64);

    let view = store.get(&key(1)).unwrap();
    assert_eq!(view.bytes().as_ptr() as usize % RECORD_ALIGN, 0);
    assert_eq!(view.layout(), layout);
    let mut out = vec![0u8; payload.len()];
    assert_eq!(store.load_into(&key(1), &mut out).unwrap(), Some(layout));
    assert_eq!(out, payload);
}

#[test]
fn layout_mismatch_is_rejected() {
    let store = KvBlockStore::in_memory(KvStoreConfig::default());
    let err = store.store(&key(1), &[0u8; 10], TINY, &[]).unwrap_err();
    assert!(matches!(
        err,
        KvError::LayoutMismatch {
            expected: 64,
            got: 10
        }
    ));
    assert!(!store.contains(&key(1)));
}

#[test]
fn identical_payloads_are_stored_once() {
    let dir = tempfile::tempdir().unwrap();
    let store = KvBlockStore::open(dir.path(), small_segments()).unwrap();
    let payload = block(2, 64);
    let first = store.store(&key(1), &payload, TINY, &[]).unwrap();
    let second = store.store(&key(2), &payload, TINY, &[]).unwrap();
    assert!(!first.deduplicated);
    assert!(second.deduplicated);
    assert_eq!(second.bytes_written, 0);
    let stats = store.stats();
    assert_eq!((stats.blocks, stats.records, stats.dedup_hits), (2, 1, 1));
    assert_eq!(stats.bytes_written, first.bytes_written);
    assert_eq!(
        store.get(&key(1)).unwrap().content_hash(),
        store.get(&key(2)).unwrap().content_hash()
    );
}

#[test]
fn reopen_restores_keys_records_and_associations() {
    let dir = tempfile::tempdir().unwrap();
    let payloads: Vec<Vec<u8>> = (0..20).map(|i| block(i, 64)).collect();
    {
        let store = KvBlockStore::open(dir.path(), small_segments()).unwrap();
        for (i, payload) in payloads.iter().enumerate() {
            store
                .store(&key(i as u8), payload, TINY, &[i as u64, 1000])
                .unwrap();
        }
        store.store(&key(100), &payloads[0], TINY, &[]).unwrap();
        assert!(store.remove(&key(19)).unwrap());
        assert!(store.stats().segments > 1);
    }
    let store = KvBlockStore::open(
        dir.path(),
        KvStoreConfig {
            verify_payload_on_open: true,
            ..small_segments()
        },
    )
    .unwrap();
    assert_eq!(store.stats().blocks, 20);
    assert!(!store.contains(&key(19)));
    for (i, payload) in payloads.iter().enumerate().take(19) {
        assert_eq!(
            store.get(&key(i as u8)).unwrap().bytes(),
            payload.as_slice()
        );
        assert_eq!(store.associations(&key(i as u8)), [i as u64, 1000]);
    }
    assert_eq!(
        store.get(&key(100)).unwrap().bytes(),
        payloads[0].as_slice()
    );

    let fresh = block(99, 64);
    store.store(&key(200), &fresh, TINY, &[]).unwrap();
    assert_eq!(store.get(&key(200)).unwrap().bytes(), fresh.as_slice());
}

#[test]
fn torn_record_is_dropped_on_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let payload = block(3, 64);
    {
        let store = KvBlockStore::open(dir.path(), small_segments()).unwrap();
        store.store(&key(1), &payload, TINY, &[]).unwrap();
        store.store(&key(2), &block(4, 64), TINY, &[]).unwrap();
    }
    // Truncate the segment in the middle of the second record's payload.
    let segment = dir.path().join("seg-000000.v3kv");
    let file = OpenOptions::new().write(true).open(&segment).unwrap();
    file.set_len((2 * (HEADER_LEN + 64) - 10) as u64).unwrap();
    drop(file);

    let store = KvBlockStore::open(dir.path(), small_segments()).unwrap();
    assert_eq!(store.get(&key(1)).unwrap().bytes(), payload.as_slice());
    assert!(!store.contains(&key(2)), "dangling binding must be dropped");
    assert_eq!(store.stats().records, 1);
}

#[test]
fn corrupt_payload_is_detected_when_verifying() {
    let dir = tempfile::tempdir().unwrap();
    {
        let store = KvBlockStore::open(dir.path(), small_segments()).unwrap();
        store.store(&key(1), &block(5, 64), TINY, &[]).unwrap();
    }
    let segment = dir.path().join("seg-000000.v3kv");
    let mut bytes = std::fs::read(&segment).unwrap();
    bytes[HEADER_LEN + 3] ^= 0xFF;
    std::fs::write(&segment, &bytes).unwrap();

    let trusting = KvBlockStore::open(dir.path(), small_segments()).unwrap();
    assert!(trusting.contains(&key(1)));
    drop(trusting);
    let verifying = KvBlockStore::open(
        dir.path(),
        KvStoreConfig {
            verify_payload_on_open: true,
            ..small_segments()
        },
    )
    .unwrap();
    assert!(!verifying.contains(&key(1)));
}

#[test]
fn eviction_plan_follows_temporal_decay() {
    let engine = ConcurrentHypergraphEngine::in_memory(EngineConfig {
        sync_mode: SyncMode::OsBuffered,
        ..EngineConfig::default()
    });
    let store = KvBlockStore::in_memory(KvStoreConfig::default());
    let now = 400 * DAY;
    let fresh = engine
        .insert(NewHyperedge::new("E", ["a", "b"], now - DAY))
        .unwrap()
        .id;
    let stale = engine
        .insert(NewHyperedge::new("E", ["c", "d"], now - 300 * DAY))
        .unwrap()
        .id;
    let doomed = engine
        .insert(NewHyperedge::new("E", ["e", "f"], now - DAY))
        .unwrap()
        .id;
    store
        .store(&key(1), &block(1, 64), TINY, &[fresh.to_u64()])
        .unwrap();
    store
        .store(&key(2), &block(2, 64), TINY, &[stale.to_u64()])
        .unwrap();
    store
        .store(&key(3), &block(3, 64), TINY, &[doomed.to_u64()])
        .unwrap();
    store
        .store(
            &key(4),
            &block(4, 64),
            TINY,
            &[stale.to_u64(), fresh.to_u64()],
        )
        .unwrap();
    store.store(&key(5), &block(5, 64), TINY, &[]).unwrap();
    engine.soft_delete_hyperedge(doomed).unwrap();

    let plan = plan_kv_evictions(&engine, &store, 0.5, now);
    let keys: Vec<[u8; 32]> = plan.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, [key(3), key(2)], "{}", format_hyperedge_id(stale));
    assert_eq!(plan[0].1, 0.0);
    assert!(plan[1].1 > 0.0 && plan[1].1 < 0.5);

    assert!(plan_kv_evictions(&engine, &store, 0.0, now).is_empty());
    assert_eq!(plan_kv_evictions(&engine, &store, 2.0, now).len(), 4);
}

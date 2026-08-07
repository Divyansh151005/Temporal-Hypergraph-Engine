//! WAL fault injection: corrupted magic, payload bit flips, torn writes and
//! uncommitted tails must recover to the last committed batch without panics.

use std::fs;
use std::path::{Path, PathBuf};

use engine_core::wal::{list_segments, FrameReader, OpCode};
use engine_core::{ConcurrentHypergraphEngine, EngineConfig, SyncMode};
use proptest::prelude::*;

const BATCHES: usize = 10;

fn config() -> EngineConfig {
    EngineConfig {
        sync_mode: SyncMode::OsBuffered,
        ..EngineConfig::default()
    }
}

/// Writes `BATCHES` single-insert batches (each: 2 node frames, 1 hyperedge
/// frame, 1 commit frame) and returns the segment path.
fn populate(dir: &Path) -> PathBuf {
    let engine = ConcurrentHypergraphEngine::open_with_config(dir, config()).unwrap();
    for i in 0..BATCHES {
        engine
            .insert_hyperedge("E", &[&format!("a{i}"), &format!("b{i}")], i as i64, None)
            .unwrap();
    }
    drop(engine);
    let segments = list_segments(dir).unwrap();
    assert_eq!(segments.len(), 1);
    segments[0].1.clone()
}

/// `(offset, op, payload_offset)` of every frame in the segment.
fn frames(path: &Path) -> Vec<(usize, OpCode, usize)> {
    let bytes = fs::read(path).unwrap();
    let mut reader = FrameReader::new(&bytes, None);
    let mut out = Vec::new();
    while let Some(f) = reader.next_frame().unwrap() {
        out.push((f.offset, f.op, f.offset + engine_core::wal::WAL_HEADER_LEN));
    }
    out
}

fn reopen_count(dir: &Path) -> u32 {
    ConcurrentHypergraphEngine::open_with_config(dir, config())
        .unwrap()
        .snapshot()
        .live_hyperedge_count()
}

fn batch_start(frames: &[(usize, OpCode, usize)], batch: usize) -> usize {
    frames[batch * 4].0
}

#[test]
fn magic_mutation_stops_replay_at_previous_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = populate(dir.path());
    let frames = frames(&path);
    let target = batch_start(&frames, 5);
    let mut bytes = fs::read(&path).unwrap();
    bytes[target..target + 4].copy_from_slice(b"THG9");
    fs::write(&path, &bytes).unwrap();

    assert_eq!(reopen_count(dir.path()), 5);
    assert_eq!(fs::metadata(&path).unwrap().len() as usize, target);
}

#[test]
fn payload_bit_flip_is_caught_by_crc() {
    let dir = tempfile::tempdir().unwrap();
    let path = populate(dir.path());
    let frames = frames(&path);
    let (_, op, payload) = frames[7 * 4 + 2];
    assert_eq!(op, OpCode::InsertHyperedge);
    let mut bytes = fs::read(&path).unwrap();
    bytes[payload + 3] ^= 0b0001_0000;
    fs::write(&path, &bytes).unwrap();

    assert_eq!(reopen_count(dir.path()), 7);
}

#[test]
fn torn_write_mid_frame_is_discarded() {
    let dir = tempfile::tempdir().unwrap();
    let path = populate(dir.path());
    let frames = frames(&path);
    let cut = frames[8 * 4 + 1].0 + 9;
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(cut as u64)
        .unwrap();
    assert_eq!(reopen_count(dir.path()), 8);
}

#[test]
fn uncommitted_tail_is_discarded_and_writes_resume() {
    let dir = tempfile::tempdir().unwrap();
    let path = populate(dir.path());
    let frames = frames(&path);
    let (commit_offset, op, _) = frames[9 * 4 + 3];
    assert_eq!(op, OpCode::BatchCommit);
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(commit_offset as u64)
        .unwrap();

    {
        let engine = ConcurrentHypergraphEngine::open_with_config(dir.path(), config()).unwrap();
        assert_eq!(engine.snapshot().live_hyperedge_count(), 9);
        assert!(
            engine.node_id("a9").is_none(),
            "no orphan nodes from the torn batch"
        );
        engine
            .insert_hyperedge("E", &["late", "arrival"], 99, None)
            .unwrap();
    }
    assert_eq!(reopen_count(dir.path()), 10);
}

#[test]
fn corrupt_segment_after_checkpoint_keeps_checkpointed_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = populate(dir.path());
    {
        let engine = ConcurrentHypergraphEngine::open_with_config(dir.path(), config()).unwrap();
        engine.checkpoint().unwrap();
        engine
            .insert_hyperedge("E", &["after", "ckpt"], 1, None)
            .unwrap();
    }
    assert!(!path.exists(), "checkpoint removes superseded segments");
    let segment = list_segments(dir.path()).unwrap().pop().unwrap().1;
    let mut bytes = fs::read(&segment).unwrap();
    bytes[0] = b'X';
    fs::write(&segment, &bytes).unwrap();
    assert_eq!(reopen_count(dir.path()), BATCHES as u32);
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(miri) { 2 } else { 48 }, ..ProptestConfig::default() })]

    #[test]
    fn random_byte_corruption_never_panics_and_keeps_a_prefix(
        flips in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..6),
        truncate in proptest::option::of(any::<prop::sample::Index>()),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = populate(dir.path());
        let mut bytes = fs::read(&path).unwrap();
        for (index, mask) in &flips {
            let i = index.index(bytes.len());
            bytes[i] ^= mask | 1;
        }
        if let Some(cut) = truncate {
            let len = cut.index(bytes.len());
            bytes.truncate(len);
        }
        fs::write(&path, &bytes).unwrap();

        let engine = ConcurrentHypergraphEngine::open_with_config(dir.path(), config()).unwrap();
        let snapshot = engine.snapshot();
        let count = snapshot.live_hyperedge_count() as usize;
        prop_assert!(count <= BATCHES);
        for i in 0..count {
            let label = format!("a{i}");
            prop_assert!(engine.node_id(&label).is_some(), "recovered state must be a prefix");
        }
        for edge in snapshot.iter_hyperedges() {
            for p in &edge.participants {
                prop_assert!(snapshot.node(*p).is_some(), "no dangling participant");
            }
        }
    }
}

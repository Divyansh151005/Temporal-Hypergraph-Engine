//! WAL chaos harness: injects every fault kind into every frame of an active
//! segment and checks recovery against an oracle derived from the pristine
//! frame layout, independently of the engine's own replay logic.

use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use engine_core::checkpoint::list_checkpoints;
use engine_core::wal::{list_segments, FrameReader, OpCode, WAL_HEADER_LEN};
use engine_core::{ConcurrentHypergraphEngine, EngineConfig, NewHyperedge, SyncMode};
use proptest::prelude::*;

const BATCHES: usize = 12;

fn config() -> EngineConfig {
    EngineConfig {
        sync_mode: SyncMode::OsBuffered,
        ..EngineConfig::default()
    }
}

fn open(dir: &Path) -> ConcurrentHypergraphEngine {
    ConcurrentHypergraphEngine::open_with_config(dir, config()).unwrap()
}

fn labels(batch: usize, edge: usize) -> (String, String) {
    (format!("s{batch}_{edge}"), format!("o{batch}_{edge}"))
}

fn batch_size(batch: usize) -> usize {
    1 + batch % 3
}

/// Writes batches `range` as one `insert_batch` call each (one WAL commit).
fn write_batches(engine: &ConcurrentHypergraphEngine, range: std::ops::Range<usize>) {
    for b in range {
        let edges = (0..batch_size(b))
            .map(|j| {
                let (s, o) = labels(b, j);
                NewHyperedge::new("CHAOS", [s, o], (b * 10 + j) as i64)
            })
            .collect();
        for result in engine.insert_batch(edges) {
            result.unwrap();
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FrameInfo {
    offset: usize,
    op: OpCode,
    payload_len: usize,
}

#[derive(Debug, Clone, Copy)]
struct BatchInfo {
    commit_end: usize,
    commit_seq: u64,
}

/// A pristine single-segment WAL plus its frame and batch layout.
struct Fixture {
    file_name: PathBuf,
    bytes: Vec<u8>,
    frames: Vec<FrameInfo>,
    batches: Vec<BatchInfo>,
}

impl Fixture {
    fn build() -> Self {
        let dir = tempfile::tempdir().unwrap();
        {
            let engine = open(dir.path());
            write_batches(&engine, 0..BATCHES);
        }
        let segments = list_segments(dir.path()).unwrap();
        assert_eq!(segments.len(), 1);
        let path = &segments[0].1;
        let bytes = fs::read(path).unwrap();

        let mut frames = Vec::new();
        let mut batches = Vec::new();
        let mut reader = FrameReader::new(&bytes, Some(1));
        while let Some(frame) = reader.next_frame().unwrap() {
            frames.push(FrameInfo {
                offset: frame.offset,
                op: frame.op,
                payload_len: frame.payload.len(),
            });
            if frame.op == OpCode::BatchCommit {
                batches.push(BatchInfo {
                    commit_end: reader.position(),
                    commit_seq: frame.sequence_id,
                });
            }
        }
        assert_eq!(reader.position(), bytes.len());
        assert_eq!(batches.len(), BATCHES, "one commit per insert_batch call");
        Self {
            file_name: path.file_name().unwrap().into(),
            bytes,
            frames,
            batches,
        }
    }

    /// Expected recovery when the first damaged byte (or truncation point)
    /// is `damage`: every batch whose commit frame ends at or before it.
    fn oracle(&self, damage: usize) -> Expected {
        let kept = self
            .batches
            .iter()
            .take_while(|b| b.commit_end <= damage)
            .count();
        let last = kept.checked_sub(1).map(|i| self.batches[i]);
        Expected {
            kept_batches: kept,
            live_edges: (0..kept).map(batch_size).sum::<usize>() as u32,
            last_sequence: last.map_or(0, |b| b.commit_seq),
            file_len: last.map_or(0, |b| b.commit_end) as u64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Expected {
    kept_batches: usize,
    live_edges: u32,
    last_sequence: u64,
    file_len: u64,
}

#[derive(Debug, Clone, Copy)]
enum HeaderField {
    OpCode,
    PayloadLen,
    SequenceId,
    Crc32,
}

#[derive(Debug, Clone, Copy)]
enum Cut {
    FrameStart,
    MidHeader,
    MidPayload,
}

#[derive(Debug, Clone, Copy)]
enum Fault {
    BadMagic,
    PayloadBitFlip { bit: u8 },
    HeaderFieldFlip(HeaderField),
    Truncate(Cut),
}

const ALL_FAULTS: [Fault; 9] = [
    Fault::BadMagic,
    Fault::PayloadBitFlip { bit: 5 },
    Fault::HeaderFieldFlip(HeaderField::OpCode),
    Fault::HeaderFieldFlip(HeaderField::PayloadLen),
    Fault::HeaderFieldFlip(HeaderField::SequenceId),
    Fault::HeaderFieldFlip(HeaderField::Crc32),
    Fault::Truncate(Cut::FrameStart),
    Fault::Truncate(Cut::MidHeader),
    Fault::Truncate(Cut::MidPayload),
];

/// Applies `fault` to `frame` and returns the offset of the first damaged
/// byte (the cut point for truncations).
fn inject(bytes: &mut Vec<u8>, frame: FrameInfo, fault: Fault) -> usize {
    let at = frame.offset;
    let flip = |bytes: &mut Vec<u8>, offset: usize, bit: u8| {
        bytes[offset] ^= 1 << (bit % 8);
        offset
    };
    match fault {
        Fault::BadMagic => {
            bytes[at..at + 4].copy_from_slice(b"THG9");
            at + 3
        }
        Fault::PayloadBitFlip { bit } => {
            if frame.payload_len == 0 {
                return flip(bytes, at + 18, bit);
            }
            flip(bytes, at + WAL_HEADER_LEN + frame.payload_len / 2, bit)
        }
        Fault::HeaderFieldFlip(field) => match field {
            HeaderField::OpCode => flip(bytes, at + 4, 1),
            HeaderField::PayloadLen => flip(bytes, at + 6, 3),
            HeaderField::SequenceId => flip(bytes, at + 10, 0),
            HeaderField::Crc32 => flip(bytes, at + 18, 7),
        },
        Fault::Truncate(cut) => {
            let len = match cut {
                Cut::FrameStart => at,
                Cut::MidHeader => at + WAL_HEADER_LEN / 2,
                Cut::MidPayload => at + WAL_HEADER_LEN + frame.payload_len / 2,
            };
            bytes.truncate(len);
            len
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Recovered {
    live_edges: u32,
    last_sequence: u64,
    file_len: u64,
}

/// Writes `bytes` as the active segment of a fresh directory, boots the
/// engine and checks the result against `expected`.
fn boot_and_check(fixture: &Fixture, bytes: &[u8], expected: Expected, context: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(&fixture.file_name);
    fs::write(&path, bytes).unwrap();

    let recovered = panic::catch_unwind(AssertUnwindSafe(|| {
        let engine = open(dir.path());
        assert_recovered_state(&engine, expected.kept_batches, context);
        Recovered {
            live_edges: engine.snapshot().live_hyperedge_count(),
            last_sequence: engine.last_sequence(),
            file_len: fs::metadata(&path).unwrap().len(),
        }
    }))
    .unwrap_or_else(|_| panic!("engine panicked during recovery: {context}"));

    assert_eq!(
        recovered,
        Recovered {
            live_edges: expected.live_edges,
            last_sequence: expected.last_sequence,
            file_len: expected.file_len,
        },
        "{context}"
    );
}

/// Batches `< kept` are fully present, later ones fully absent, and no edge
/// references a missing node.
fn assert_recovered_state(engine: &ConcurrentHypergraphEngine, kept: usize, context: &str) {
    for b in 0..BATCHES {
        for j in 0..batch_size(b) {
            let (s, o) = labels(b, j);
            let present = b < kept;
            assert_eq!(engine.node_id(&s).is_some(), present, "{s}: {context}");
            assert_eq!(engine.node_id(&o).is_some(), present, "{o}: {context}");
        }
    }
    let snapshot = engine.snapshot();
    for edge in snapshot.iter_hyperedges() {
        for p in &edge.participants {
            assert!(
                snapshot.node(*p).is_some(),
                "dangling participant: {context}"
            );
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "boots the engine once per frame x fault; hours under the interpreter"
)]
fn exhaustive_sweep_every_frame_every_fault() {
    let fixture = Fixture::build();
    let mut cases = 0;
    for (index, frame) in fixture.frames.iter().enumerate() {
        for fault in ALL_FAULTS {
            let mut bytes = fixture.bytes.clone();
            let damage = inject(&mut bytes, *frame, fault);
            let context = format!("frame {index} ({:?}) fault {fault:?}", frame.op);
            boot_and_check(&fixture, &bytes, fixture.oracle(damage), &context);
            cases += 1;
        }
    }
    assert_eq!(cases, fixture.frames.len() * ALL_FAULTS.len());
}

#[test]
fn recovery_is_idempotent_and_writable() {
    let fixture = Fixture::build();
    let target = fixture.frames[fixture.frames.len() / 2];
    let mut bytes = fixture.bytes.clone();
    let damage = inject(&mut bytes, target, Fault::PayloadBitFlip { bit: 2 });
    let expected = fixture.oracle(damage);
    assert!(expected.kept_batches > 0 && expected.kept_batches < BATCHES);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(&fixture.file_name);
    fs::write(&path, &bytes).unwrap();

    let first = {
        let engine = open(dir.path());
        (
            engine.snapshot().live_hyperedge_count(),
            engine.last_sequence(),
        )
    };
    let truncated = fs::read(&path).unwrap();
    let second = {
        let engine = open(dir.path());
        assert_recovered_state(&engine, expected.kept_batches, "second boot");
        (
            engine.snapshot().live_hyperedge_count(),
            engine.last_sequence(),
        )
    };
    assert_eq!(first, second);
    assert_eq!(first, (expected.live_edges, expected.last_sequence));
    assert_eq!(
        fs::read(&path).unwrap(),
        truncated,
        "second boot rewrote the WAL"
    );

    {
        let engine = open(dir.path());
        engine
            .insert_hyperedge("CHAOS", &["late", "arrival"], 999, None)
            .unwrap();
        // Two node frames, one hyperedge frame and one commit frame.
        assert_eq!(engine.last_sequence(), expected.last_sequence + 4);
    }

    let engine = open(dir.path());
    assert_eq!(
        engine.snapshot().live_hyperedge_count(),
        expected.live_edges + 1
    );
    let wal = fs::read(&path).unwrap();
    let mut reader = FrameReader::new(&wal, Some(1));
    let mut last = 0;
    while let Some(frame) = reader.next_frame().expect("contiguous, intact WAL") {
        last = frame.sequence_id;
    }
    assert_eq!(last, expected.last_sequence + 4);
}

/// Reproduces a crash between `checkpoint()` creating the new segment and
/// deleting the superseded one, then corrupts the superseded segment.
#[test]
fn corruption_in_superseded_segment_keeps_checkpoint_and_followers() {
    let half = BATCHES / 2;
    let dir = tempfile::tempdir().unwrap();
    let old_segment = {
        let engine = open(dir.path());
        write_batches(&engine, 0..half);
        let (_, path) = list_segments(dir.path()).unwrap().remove(0);
        let bytes = fs::read(&path).unwrap();
        engine.checkpoint().unwrap();
        write_batches(&engine, half..BATCHES);
        (path, bytes)
    };
    let (old_path, mut old_bytes) = old_segment;
    assert!(!old_path.exists(), "checkpoint deletes superseded segments");
    let mid = old_bytes.len() / 2;
    old_bytes[mid] ^= 0x40;
    fs::write(&old_path, &old_bytes).unwrap();
    assert_eq!(list_segments(dir.path()).unwrap().len(), 2);

    let engine = open(dir.path());
    assert_recovered_state(&engine, BATCHES, "superseded segment corrupted");
    let expected_edges: usize = (0..BATCHES).map(batch_size).sum();
    assert_eq!(
        engine.snapshot().live_hyperedge_count() as usize,
        expected_edges
    );
}

/// A corruption that loses committed frames not covered by a checkpoint
/// must discard every later segment: replaying them would leave a gap.
#[test]
fn corruption_with_uncovered_loss_discards_followers() {
    let half = BATCHES / 2;
    let dir = tempfile::tempdir().unwrap();
    let (first_path, first_bytes) = {
        let engine = open(dir.path());
        write_batches(&engine, 0..half);
        let (_, path) = list_segments(dir.path()).unwrap().remove(0);
        let bytes = fs::read(&path).unwrap();
        engine.checkpoint().unwrap();
        write_batches(&engine, half..BATCHES);
        (path, bytes)
    };
    let checkpoints = list_checkpoints(dir.path()).unwrap();
    assert!(!checkpoints.is_empty());
    for (_, path) in checkpoints {
        fs::remove_file(path).unwrap();
    }
    let mut bytes = first_bytes;
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x40;
    fs::write(&first_path, &bytes).unwrap();

    let engine = open(dir.path());
    let segments = list_segments(dir.path()).unwrap();
    assert_eq!(segments.len(), 1, "follower segment must be discarded");
    let live = engine.snapshot().live_hyperedge_count() as usize;
    let first_half: usize = (0..half).map(batch_size).sum();
    assert!(live < first_half, "corrupted batch must not be replayed");
    for b in half..BATCHES {
        assert!(engine.node_id(&labels(b, 0).0).is_none());
    }
}

fn fault_strategy() -> impl Strategy<Value = Fault> {
    (0..ALL_FAULTS.len(), any::<u8>()).prop_map(|(i, bit)| match ALL_FAULTS[i] {
        Fault::PayloadBitFlip { .. } => Fault::PayloadBitFlip { bit },
        other => other,
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(miri) { 2 } else { 64 }, ..ProptestConfig::default() })]

    #[test]
    fn random_fault_storm_recovers_exact_prefix(
        faults in proptest::collection::vec((any::<prop::sample::Index>(), fault_strategy()), 1..=4),
    ) {
        let fixture = Fixture::build();
        let mut bytes = fixture.bytes.clone();
        for (index, fault) in &faults {
            let frame = fixture.frames[index.index(fixture.frames.len())];
            if frame.offset + WAL_HEADER_LEN + frame.payload_len > bytes.len() {
                continue;
            }
            inject(&mut bytes, frame, *fault);
        }
        // Repeated bit flips can cancel out, so locate the damage by diffing.
        let damage = bytes
            .iter()
            .zip(&fixture.bytes)
            .position(|(a, b)| a != b)
            .unwrap_or(bytes.len());
        let expected = fixture.oracle(damage);
        boot_and_check(&fixture, &bytes, expected, &format!("{faults:?}"));
    }
}

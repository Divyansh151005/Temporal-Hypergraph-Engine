//! Property: whatever sequence of writes (and checkpoints) happens, reopening
//! the storage directory reproduces exactly the pre-shutdown state.

use engine_core::{
    BipartiteArena, ConcurrentHypergraphEngine, EngineConfig, NewHyperedge, SyncMode,
};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Insert {
        event: u8,
        participants: Vec<u8>,
        valid_from: i64,
        anchored: bool,
    },
    DeleteNth(usize),
    Checkpoint,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        8 => (0u8..3, proptest::collection::vec(0u8..12, 1..4), 0i64..40, any::<bool>())
            .prop_map(|(event, participants, valid_from, anchored)| Op::Insert { event, participants, valid_from, anchored }),
        2 => any::<usize>().prop_map(Op::DeleteNth),
        1 => Just(Op::Checkpoint),
    ]
}

type Canonical = Vec<(u64, String, Vec<String>, i64, Option<i64>, Option<String>)>;

fn canonical(engine: &ConcurrentHypergraphEngine, snapshot: &BipartiteArena) -> Canonical {
    let mut out: Canonical = snapshot
        .iter_hyperedges()
        .map(|e| {
            (
                e.id.to_u64(),
                engine.event_type(e).unwrap().to_string(),
                e.participants
                    .iter()
                    .map(|p| engine.node_label(*p).unwrap().to_string())
                    .collect(),
                e.valid_from,
                e.valid_to,
                snapshot.payload(e).map(|s| s.to_string()),
            )
        })
        .collect();
    out.sort();
    out
}

fn config() -> EngineConfig {
    EngineConfig {
        sync_mode: SyncMode::OsBuffered,
        ..EngineConfig::default()
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(miri) { 2 } else { 40 }, ..ProptestConfig::default() })]

    #[test]
    fn reopen_reproduces_state(ops in proptest::collection::vec(op(), 1..60)) {
        let dir = tempfile::tempdir().unwrap();
        let (before, nodes_before) = {
            let engine = ConcurrentHypergraphEngine::open_with_config(dir.path(), config()).unwrap();
            for op in &ops {
                match op {
                    Op::Insert { event, participants, valid_from, anchored } => {
                        let labels: Vec<String> = participants.iter().map(|p| format!("n{p}")).collect();
                        let mut edge = NewHyperedge::new(format!("EV{event}"), labels.clone(), *valid_from);
                        if valid_from % 2 == 0 {
                            edge = edge.summary(format!("summary of {labels:?} at {valid_from}"));
                        }
                        if *anchored {
                            edge = edge.anchors([labels[0].clone()]);
                        }
                        engine.insert(edge).unwrap();
                    }
                    Op::DeleteNth(n) => {
                        let snapshot = engine.snapshot();
                        let live: Vec<_> = snapshot.iter_hyperedges().map(|e| e.id).collect();
                        if !live.is_empty() {
                            engine.soft_delete_hyperedge(live[n % live.len()]).unwrap();
                        }
                    }
                    Op::Checkpoint => engine.checkpoint().unwrap(),
                }
            }
            let snapshot = engine.snapshot();
            (canonical(&engine, &snapshot), snapshot.live_node_count())
        };

        let engine = ConcurrentHypergraphEngine::open_with_config(dir.path(), config()).unwrap();
        let snapshot = engine.snapshot();
        prop_assert_eq!(canonical(&engine, &snapshot), before);
        prop_assert_eq!(snapshot.live_node_count(), nodes_before);
    }
}

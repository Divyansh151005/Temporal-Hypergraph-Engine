//! Loom model checking of the lock-free snapshot publication path.
//!
//! Run with `RUSTFLAGS="--cfg loom" cargo test -p engine-core --test loom_suite --release`.
//! Only this test target may run under `--cfg loom`: the engine's counters
//! become loom atomics, which panic when created outside `loom::model`.

#[cfg(loom)]
mod loom_tests {
    use std::sync::atomic::{AtomicBool, Ordering as StdOrdering};

    use engine_core::sync::{Arc, Ordering};
    use engine_core::{BipartiteArena, ConcurrentHypergraphEngine};
    use loom::thread;

    const DEFAULT_PREEMPTION_BOUND: usize = 3;

    /// Runs `f` under loom with a bounded preemption depth unless
    /// `LOOM_MAX_PREEMPTIONS` overrides it.
    fn model(f: impl Fn() + Sync + Send + 'static) {
        let mut builder = loom::model::Builder::new();
        if builder.preemption_bound.is_none() {
            builder.preemption_bound = Some(DEFAULT_PREEMPTION_BOUND);
        }
        builder.check(f);
    }

    /// Every visible hyperedge must be complete: all participants resolve and
    /// list the edge among their incident hyperedges.
    fn assert_consistent(snapshot: &BipartiteArena) {
        let edges = snapshot.live_hyperedge_count();
        assert_eq!(snapshot.iter_hyperedges().count() as u32, edges);
        assert_eq!(
            snapshot.iter_nodes().count() as u32,
            snapshot.live_node_count()
        );
        for edge in snapshot.iter_hyperedges() {
            for participant in &edge.participants {
                let node = snapshot
                    .node(*participant)
                    .expect("published edge references a missing node");
                assert!(node.hyperedges.contains(&edge.id));
            }
        }
    }

    // Exactly one writer thread per model: the writer path holds a
    // `parking_lot::Mutex`, which loom cannot see, while performing loom
    // atomic stores. A second writer could park the OS thread loom runs on.

    #[test]
    fn test_concurrent_snapshot_swap() {
        // Plain std atomics outlive loom's per-iteration state and prove the
        // model explored both reader-before-publish and reader-after-publish.
        static SAW_EMPTY: AtomicBool = AtomicBool::new(false);
        static SAW_PUBLISHED: AtomicBool = AtomicBool::new(false);

        model(|| {
            let engine = Arc::new(ConcurrentHypergraphEngine::new_test());

            let writer = {
                let engine = engine.clone();
                thread::spawn(move || {
                    engine
                        .insert_hyperedge_raw(
                            "TEST_EVENT",
                            &["node_a", "node_b"],
                            1000,
                            None,
                            &[0.1f32; 128],
                        )
                        .expect("insert");
                })
            };

            let reader = {
                let engine = engine.clone();
                thread::spawn(move || {
                    let published = engine.active_hyperedges_count.load(Ordering::Acquire);
                    let snapshot = engine.graph_snapshot.load_full();
                    assert!(
                        snapshot.live_hyperedge_count() >= published,
                        "counter published ahead of its snapshot"
                    );
                    match snapshot.live_hyperedge_count() {
                        0 => {
                            assert_eq!(snapshot.live_node_count(), 0);
                            SAW_EMPTY.store(true, StdOrdering::Relaxed);
                        }
                        1 => {
                            assert_eq!(snapshot.live_node_count(), 2);
                            SAW_PUBLISHED.store(true, StdOrdering::Relaxed);
                        }
                        n => panic!("impossible hyperedge count {n}"),
                    }
                    assert_consistent(&snapshot);
                })
            };

            writer.join().unwrap();
            reader.join().unwrap();

            let snapshot = engine.graph_snapshot.load_full();
            assert_eq!(snapshot.live_hyperedge_count(), 1);
            assert_eq!(snapshot.live_node_count(), 2);
            assert_eq!(engine.active_hyperedges_count.load(Ordering::Acquire), 1);
            assert_eq!(engine.active_nodes_count.load(Ordering::Acquire), 2);
            assert_consistent(&snapshot);
        });

        assert!(
            SAW_EMPTY.load(StdOrdering::Relaxed),
            "no interleaving read before publish"
        );
        assert!(
            SAW_PUBLISHED.load(StdOrdering::Relaxed),
            "no interleaving read after publish"
        );
    }

    #[test]
    fn test_snapshot_never_regresses() {
        model(|| {
            let engine = Arc::new(ConcurrentHypergraphEngine::new_test());

            let writer = {
                let engine = engine.clone();
                thread::spawn(move || {
                    for (i, pair) in [["a", "b"], ["b", "c"]].iter().enumerate() {
                        engine
                            .insert_hyperedge("E", pair, i as i64, None)
                            .expect("insert");
                    }
                })
            };

            let reader = {
                let engine = engine.clone();
                thread::spawn(move || {
                    let first_count = engine.active_hyperedges_count.load(Ordering::Acquire);
                    let first = engine.graph_snapshot.load_full();
                    let second_count = engine.active_hyperedges_count.load(Ordering::Acquire);
                    let second = engine.graph_snapshot.load_full();
                    assert!(second_count >= first_count, "counter went backwards");
                    assert!(
                        second.live_hyperedge_count() >= first.live_hyperedge_count(),
                        "snapshot went backwards"
                    );
                    assert!(second.live_hyperedge_count() >= second_count);
                    assert_consistent(&first);
                    assert_consistent(&second);
                })
            };

            writer.join().unwrap();
            reader.join().unwrap();
            assert_eq!(engine.snapshot().live_hyperedge_count(), 2);
            assert_eq!(engine.snapshot().live_node_count(), 3);
        });
    }
}

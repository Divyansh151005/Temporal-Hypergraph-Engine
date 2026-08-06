use std::sync::Arc;
use std::thread;

use engine_core::{
    ConcurrentHypergraphEngine, EngineConfig, EngineError, HyperedgeId, NewHyperedge, Resolution,
    SyncMode, TraversalOrder, TraversalQuery,
};

const YEAR: i64 = 31_536_000;

fn buffered() -> EngineConfig {
    EngineConfig {
        sync_mode: SyncMode::OsBuffered,
        ..EngineConfig::default()
    }
}

fn open(dir: &std::path::Path) -> ConcurrentHypergraphEngine {
    ConcurrentHypergraphEngine::open_with_config(dir, buffered()).unwrap()
}

fn ids(edges: &[engine_core::TemporalHyperedge]) -> Vec<HyperedgeId> {
    edges.iter().map(|e| e.id).collect()
}

#[test]
fn three_hop_traversal_through_engine() {
    let engine = ConcurrentHypergraphEngine::new_test();
    let chain = ["a", "b", "c", "d", "e"];
    let edges: Vec<HyperedgeId> = chain
        .windows(2)
        .map(|w| engine.insert_hyperedge("NEXT", w, 0, None).unwrap())
        .collect();
    let a = engine.node_id("a").unwrap();
    assert_eq!(ids(&engine.traverse_bipartite(&[a], 1, 1)), edges[..1]);
    assert_eq!(ids(&engine.traverse_bipartite(&[a], 3, 1)), edges[..3]);
    assert_eq!(
        engine
            .active_hyperedges_count
            .load(std::sync::atomic::Ordering::Acquire),
        4
    );
}

#[test]
fn n_ary_meeting_is_one_hyperedge() {
    let engine = ConcurrentHypergraphEngine::new_test();
    let meeting = engine
        .insert_hyperedge(
            "BOARD_MEETING",
            &["Alice", "Bob", "Carol", "Dan", "Eve"],
            100,
            Some(200),
        )
        .unwrap();
    let alice = engine.node_id("Alice").unwrap();
    let reached = engine.traverse_bipartite(&[alice], 1, 150);
    assert_eq!(ids(&reached), vec![meeting]);
    assert_eq!(reached[0].participants.len(), 5);
    assert!(engine.traverse_bipartite(&[alice], 1, 250).is_empty());
}

#[test]
fn ceo_succession_truncates_without_deleting() {
    let engine = ConcurrentHypergraphEngine::new_test();
    let sam = engine
        .insert(NewHyperedge::new("CEO_OF", ["Sam", "OpenAI"], 2023 * YEAR).anchors(["OpenAI"]))
        .unwrap();
    let mira = engine
        .insert(NewHyperedge::new("CEO_OF", ["Mira", "OpenAI"], 2025 * YEAR).anchors(["OpenAI"]))
        .unwrap();
    assert_eq!(mira.truncated, vec![sam.id]);

    let sam_edge = engine.hyperedge(sam.id).unwrap();
    assert_eq!(sam_edge.valid_to, Some(2025 * YEAR));
    assert!(!sam_edge.is_ongoing);

    let openai = engine.node_id("OpenAI").unwrap();
    assert_eq!(
        ids(&engine.traverse_bipartite(&[openai], 1, 2024 * YEAR)),
        vec![sam.id]
    );
    assert_eq!(
        ids(&engine.traverse_bipartite(&[openai], 1, 2026 * YEAR)),
        vec![mira.id]
    );
}

#[test]
fn out_of_order_inserts_form_contiguous_timeline() {
    let engine = ConcurrentHypergraphEngine::new_test();
    let mut outcomes = Vec::new();
    for year in [2025, 2023, 2024] {
        outcomes.push(
            engine
                .insert(NewHyperedge::new("LIVES_IN", ["Ann", "Paris"], year * YEAR))
                .unwrap(),
        );
    }
    assert_eq!(outcomes[1].resolution, Resolution::Backfilled);
    let mut edges: Vec<_> = outcomes
        .iter()
        .map(|o| engine.hyperedge(o.id).unwrap())
        .collect();
    edges.sort_by_key(|e| e.valid_from);
    for pair in edges.windows(2) {
        assert_eq!(pair[0].valid_to, Some(pair[1].valid_from));
    }
    assert_eq!(edges[2].valid_to, None);
}

#[test]
fn duplicates_and_aliases() {
    let engine = ConcurrentHypergraphEngine::new_test();
    let first = engine
        .insert(NewHyperedge::new("MET", ["x", "y"], 10).embedding(vec![1.0, 0.0]))
        .unwrap();
    let dup = engine
        .insert(NewHyperedge::new("MET", ["y", "x"], 10).embedding(vec![1.0, 0.001]))
        .unwrap();
    assert_eq!(dup.resolution, Resolution::Duplicate);
    assert_eq!(dup.id, first.id);
    let alias = engine
        .insert(NewHyperedge::new("MET", ["x", "y"], 10).embedding(vec![0.0, 1.0]))
        .unwrap();
    assert_eq!(
        alias.resolution,
        Resolution::Alias {
            canonical: first.id
        }
    );
    assert_eq!(engine.snapshot().live_hyperedge_count(), 2);
}

#[test]
fn validation_errors_leave_state_untouched() {
    let engine = ConcurrentHypergraphEngine::in_memory(EngineConfig {
        embedding_dim: 4,
        ..EngineConfig::default()
    });
    let err = engine
        .insert_hyperedge_raw("E", &["a"], 0, None, &[1.0; 3])
        .unwrap_err();
    assert!(matches!(
        err,
        EngineError::EmbeddingDimension {
            expected: 4,
            got: 3
        }
    ));
    assert!(matches!(
        engine.insert_hyperedge("E", &["a"], 10, Some(10)),
        Err(EngineError::InvalidArgument(_))
    ));
    assert!(matches!(
        engine.insert_hyperedge("E", &[], 0, None),
        Err(EngineError::InvalidArgument(_))
    ));
    assert!(matches!(
        engine.insert(NewHyperedge::new("E", ["a"], 0).anchors(["b"])),
        Err(EngineError::InvalidArgument(_))
    ));
    assert_eq!(engine.snapshot().live_node_count(), 0);
}

#[test]
fn reopen_restores_graph_labels_and_indexes() {
    let dir = tempfile::tempdir().unwrap();
    let (edge, embedding) = {
        let engine = open(dir.path());
        let edge = engine
            .insert_hyperedge_raw(
                "ACQUIRED",
                &["Acme", "Widgets"],
                500,
                Some(900),
                &[0.25, 0.5],
            )
            .unwrap();
        engine
            .insert_hyperedge("FOUNDED", &["Ada", "Acme"], 100, None)
            .unwrap();
        (edge, engine.embedding(edge).unwrap())
    };
    let engine = open(dir.path());
    assert_eq!(engine.snapshot().live_hyperedge_count(), 2);
    let restored = engine.hyperedge(edge).unwrap();
    assert_eq!(restored.valid_to, Some(900));
    assert_eq!(engine.event_type(&restored).as_deref(), Some("ACQUIRED"));
    assert_eq!(engine.embedding(edge).unwrap(), embedding);
    let acme = engine.node_id("Acme").unwrap();
    assert_eq!(engine.node_label(acme).as_deref(), Some("Acme"));
    assert_eq!(engine.traverse_bipartite(&[acme], 1, 600).len(), 2);
    assert_eq!(engine.hyperedges_in_range(0, 400).len(), 1);

    let next = engine
        .insert_hyperedge("SOLD", &["Acme", "Gizmo"], 1_000, None)
        .unwrap();
    drop(engine);
    let engine = open(dir.path());
    assert!(engine.hyperedge(next).is_some());
    assert_eq!(engine.snapshot().live_hyperedge_count(), 3);
}

#[test]
fn checkpoint_rotates_wal_and_recovers_tail() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = open(dir.path());
        for i in 0..50 {
            engine
                .insert_hyperedge("E", &[&format!("n{i}"), "hub"], i, None)
                .unwrap();
        }
        engine.checkpoint().unwrap();
        engine.checkpoint().unwrap();
        for i in 50..80 {
            engine
                .insert_hyperedge("E", &[&format!("n{i}"), "hub"], i, None)
                .unwrap();
        }
        let deleted = engine.node_id("n3").unwrap();
        let edge = engine.traverse_bipartite(&[deleted], 1, 100)[0].id;
        engine.soft_delete_hyperedge(edge).unwrap();
        engine.soft_delete_node(deleted).unwrap();
    }
    let files: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(
        files.iter().filter(|f| f.ends_with(".wal")).count(),
        1,
        "{files:?}"
    );
    assert_eq!(
        files.iter().filter(|f| f.ends_with(".ckpt")).count(),
        1,
        "{files:?}"
    );

    let engine = open(dir.path());
    assert_eq!(engine.snapshot().live_hyperedge_count(), 79);
    assert!(engine.node_id("n3").is_none());
    let hub = engine.node_id("hub").unwrap();
    assert_eq!(engine.traverse_bipartite(&[hub], 1, 1_000).len(), 79);
}

#[test]
fn soft_deleted_slot_reuse_invalidates_old_ids() {
    let dir = tempfile::tempdir().unwrap();
    let old = {
        let engine = open(dir.path());
        let old = engine.insert_hyperedge("E", &["a", "b"], 0, None).unwrap();
        engine.soft_delete_hyperedge(old).unwrap();
        assert!(matches!(
            engine.soft_delete_hyperedge(old),
            Err(EngineError::NotFound(_))
        ));
        old
    };
    let engine = open(dir.path());
    let new = engine.insert_hyperedge("E", &["a", "c"], 0, None).unwrap();
    assert_eq!({ new.index }, { old.index });
    assert_ne!(new, old);
    assert!(engine.hyperedge(old).is_none());
}

#[test]
fn fsync_mode_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = ConcurrentHypergraphEngine::open(dir.path().to_str().unwrap()).unwrap();
        engine.insert_hyperedge("E", &["a", "b"], 1, None).unwrap();
    }
    let engine = ConcurrentHypergraphEngine::open(dir.path().to_str().unwrap()).unwrap();
    assert_eq!(engine.snapshot().live_hyperedge_count(), 1);
}

#[test]
fn insert_batch_preserves_order_and_isolates_failures() {
    let engine = ConcurrentHypergraphEngine::new_test();
    let results = engine.insert_batch(vec![
        NewHyperedge::new("E", ["a", "b"], 1),
        NewHyperedge::new("", ["a"], 1),
        NewHyperedge::new("E", ["b", "c"], 2),
    ]);
    assert!(results[0].is_ok());
    assert!(matches!(results[1], Err(EngineError::InvalidArgument(_))));
    assert!(results[2].is_ok());
    assert_eq!(engine.snapshot().live_hyperedge_count(), 2);
}

#[test]
fn ranked_traversal_prefers_recent_facts() {
    let engine = ConcurrentHypergraphEngine::new_test();
    let day = 86_400;
    let old = engine
        .insert_hyperedge("SAW", &["me", "film-a"], 0, None)
        .unwrap();
    let recent = engine
        .insert_hyperedge("SAW", &["me", "film-b"], 300 * day, None)
        .unwrap();
    let me = engine.node_id("me").unwrap();
    let ranked = engine.traverse_ranked(&[me], 1, 301 * day, 301 * day);
    assert_eq!(
        ranked.iter().map(|r| r.edge.id).collect::<Vec<_>>(),
        vec![recent, old]
    );
    assert!(ranked[0].score > 0.9 && ranked[1].score < 0.01);
}

#[test]
fn concurrent_writers_and_lock_free_readers() {
    let (writes, reads) = if cfg!(miri) { (10, 20) } else { (250, 2_000) };
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(open(dir.path()));
    let writers: Vec<_> = (0..8)
        .map(|w| {
            let engine = engine.clone();
            thread::spawn(move || {
                for i in 0..writes {
                    engine
                        .insert_hyperedge(
                            "E",
                            &[&format!("w{w}-{i}"), &format!("hub{}", i % 4)],
                            i,
                            None,
                        )
                        .unwrap();
                }
            })
        })
        .collect();
    let readers: Vec<_> = (0..8)
        .map(|_| {
            let engine = engine.clone();
            thread::spawn(move || {
                let mut last = 0;
                for _ in 0..reads {
                    let snapshot = engine.graph_snapshot.load();
                    let live = snapshot.live_hyperedge_count();
                    assert!(live >= last, "snapshots must never go backwards");
                    assert_eq!(snapshot.iter_hyperedges().count() as u32, live);
                    last = live;
                    if let Some(hub) = engine.node_id("hub0") {
                        let _ = engine.traverse_bipartite(&[hub], 2, i64::MAX - 1);
                    }
                }
            })
        })
        .collect();
    for handle in writers.into_iter().chain(readers) {
        handle.join().unwrap();
    }
    let total = 8 * writes as u32;
    assert_eq!(engine.snapshot().live_hyperedge_count(), total);
    drop(engine);
    assert_eq!(open(dir.path()).snapshot().live_hyperedge_count(), total);
}

#[test]
fn query_temporal_ranks_by_similarity_and_decay() {
    let engine = ConcurrentHypergraphEngine::new_test();
    let day = 86_400;
    let insert = |label: &str, from: i64, to: Option<i64>, emb: Vec<f32>| {
        let mut edge = NewHyperedge::new("EVENT", [label, "shared"], from).embedding(emb);
        edge.valid_to = to;
        engine.insert(edge).unwrap().id
    };
    let exact_old = insert("a", 0, None, vec![1.0, 0.0]);
    let exact_new = insert("b", 60 * day, None, vec![1.0, 0.0]);
    let orthogonal = insert("c", 60 * day, None, vec![0.0, 1.0]);
    let expired = insert("d", 0, Some(day), vec![1.0, 0.0]);
    engine
        .insert(NewHyperedge::new("EVENT", ["e", "shared"], 0))
        .unwrap();

    let now = 90 * day;
    let ranked = engine.query_temporal(&[1.0, 0.0], now, None, 10);
    let ranked_ids: Vec<HyperedgeId> = ranked.iter().map(|s| s.edge.id).collect();
    assert_eq!(ranked_ids, [exact_new, exact_old, orthogonal]);
    assert!(!ranked_ids.contains(&expired));
    assert!(ranked[0].score > ranked[1].score);

    let top1 = engine.query_temporal(&[1.0, 0.0], now, None, 1);
    assert_eq!(top1.len(), 1);
    assert_eq!(top1[0].edge.id, exact_new);

    let no_decay = engine_core::temporal_math::DecayConfig::new(f32::INFINITY, 1.0);
    let flat = engine.query_temporal(&[1.0, 0.0], now, Some(no_decay), 2);
    assert!((flat[0].score - flat[1].score).abs() < 1e-6);
}

#[test]
fn summaries_survive_wal_replay_checkpoint_and_delete() {
    let dir = tempfile::tempdir().unwrap();
    let (kept, dropped, plain) = {
        let engine = open(dir.path());
        let kept = engine
            .insert(
                NewHyperedge::new("MESSAGE_HUMAN", ["session:1", "record:a"], 10).summary("hello"),
            )
            .unwrap()
            .id;
        let dropped = engine
            .insert(NewHyperedge::new("MESSAGE_AI", ["session:1", "record:b"], 11).summary("bye"))
            .unwrap()
            .id;
        let plain = engine
            .insert_hyperedge("FOUNDED", &["Ada", "Acme"], 5, None)
            .unwrap();
        assert_eq!(engine.summary(kept).as_deref(), Some("hello"));
        assert_eq!(engine.snapshot().payload_bytes(), 8);
        (kept, dropped, plain)
    };
    let engine = open(dir.path());
    assert_eq!(engine.summary(kept).as_deref(), Some("hello"));
    assert_eq!(engine.summary(dropped).as_deref(), Some("bye"));
    assert_eq!(engine.summary(plain), None);

    engine.soft_delete_hyperedge(dropped).unwrap();
    assert_eq!(engine.summary(dropped), None);
    assert_eq!(engine.snapshot().payload_bytes(), 5);
    engine.checkpoint().unwrap();
    drop(engine);

    let engine = open(dir.path());
    assert_eq!(engine.summary(kept).as_deref(), Some("hello"));
    assert_eq!(engine.summary(dropped), None);
    assert_eq!(engine.snapshot().payload_bytes(), 5);
}

#[test]
fn oversized_summaries_are_rejected() {
    let engine = ConcurrentHypergraphEngine::in_memory(EngineConfig {
        max_summary_bytes: 4,
        ..EngineConfig::default()
    });
    assert!(matches!(
        engine.insert(NewHyperedge::new("E", ["a"], 0).summary("too long")),
        Err(EngineError::InvalidArgument(_))
    ));
    assert!(engine
        .insert(NewHyperedge::new("E", ["a"], 0).summary("ok"))
        .is_ok());
}

#[test]
#[cfg_attr(miri, ignore)]
fn records_above_the_wal_frame_limit_are_rejected_not_lost() {
    let dir = tempfile::tempdir().unwrap();
    let engine = open(dir.path());
    let kept = engine.insert(NewHyperedge::new("E", ["a"], 0)).unwrap().id;
    let huge = vec![0.5f32; (256 << 20) / 4 + 1];
    assert!(matches!(
        engine.insert(NewHyperedge::new("E", ["b"], 0).embedding(huge)),
        Err(EngineError::InvalidArgument(_))
    ));
    let after = engine.insert(NewHyperedge::new("E", ["c"], 0)).unwrap().id;
    drop(engine);

    let reopened = open(dir.path());
    assert!(reopened.hyperedge(kept).is_some());
    assert!(reopened.hyperedge(after).is_some());
}

#[test]
fn legacy_v1_wal_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = b"THG1".to_vec();
    bytes.extend_from_slice(&[0; 40]);
    std::fs::write(
        dir.path().join(engine_core::wal::segment_file_name(1)),
        bytes,
    )
    .unwrap();
    let err = ConcurrentHypergraphEngine::open_with_config(dir.path(), buffered()).unwrap_err();
    assert!(
        matches!(&err, EngineError::Recovery(msg) if msg.contains("v1")),
        "{err}"
    );
}

#[test]
fn session_history_is_a_one_hop_chronological_traversal() {
    let engine = ConcurrentHypergraphEngine::new_test();
    for (i, (event, text)) in [
        ("MESSAGE_HUMAN", "hi"),
        ("MESSAGE_AI", "hello"),
        ("MESSAGE_HUMAN", "what's new?"),
    ]
    .iter()
    .enumerate()
    {
        engine
            .insert(
                NewHyperedge::new(*event, ["session:a".to_owned(), format!("record:{i}")], 100)
                    .summary(*text),
            )
            .unwrap();
    }
    engine
        .insert(NewHyperedge::new("MESSAGE_HUMAN", ["session:b", "record:x"], 100).summary("other"))
        .unwrap();
    engine
        .insert(NewHyperedge::new("NOTE", ["session:a", "Acme"], 50))
        .unwrap();

    let seeds = ["session:a".to_owned()];
    let messages = ["MESSAGE_HUMAN".to_owned(), "MESSAGE_AI".to_owned()];
    let query = TraversalQuery {
        order: TraversalOrder::Chronological,
        event_types: &messages,
        ..TraversalQuery::new(&seeds, 1, 200)
    };
    let history: Vec<String> = engine
        .traverse_query(&query)
        .iter()
        .map(|s| engine.summary(s.edge.id).unwrap().to_string())
        .collect();
    assert_eq!(history, ["what's new?", "hello", "hi"]);

    let all = engine.traverse_query(&TraversalQuery::new(&seeds, 1, 200));
    assert_eq!(all.len(), 4);
    let limited = engine.traverse_query(&TraversalQuery {
        top_k: 2,
        ..query.clone()
    });
    assert_eq!(limited.len(), 2);
    assert!(engine
        .traverse_query(&TraversalQuery::new(&["session:none".to_owned()], 1, 200))
        .is_empty());
}

#[test]
fn traverse_query_reaches_three_hops_and_reranks_by_embedding() {
    let engine = ConcurrentHypergraphEngine::new_test();
    let insert = |labels: [&str; 2], from: i64, emb: Vec<f32>| {
        engine
            .insert(NewHyperedge::new("LINK", labels, from).embedding(emb))
            .unwrap()
            .id
    };
    let ab = insert(["a", "b"], 0, vec![0.0, 1.0]);
    let bc = insert(["b", "c"], 0, vec![1.0, 0.0]);
    let cd = insert(["c", "d"], 0, vec![0.5, 0.5]);
    let de = insert(["d", "e"], 0, vec![1.0, 0.0]);

    let seeds = ["a".to_owned()];
    let three: Vec<HyperedgeId> = engine
        .traverse_query(&TraversalQuery::new(&seeds, 3, 10))
        .iter()
        .map(|s| s.edge.id)
        .collect();
    assert_eq!(three.len(), 3);
    assert!(!three.contains(&de));

    let query = [1.0, 0.0];
    let ranked: Vec<HyperedgeId> = engine
        .traverse_query(&TraversalQuery {
            query_embedding: Some(&query),
            ..TraversalQuery::new(&seeds, 3, 10)
        })
        .iter()
        .map(|s| s.edge.id)
        .collect();
    assert_eq!(ranked, [bc, cd, ab]);
}

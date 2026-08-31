mod common;

use std::collections::HashMap;
use std::time::Duration;

use bytes::Bytes;
use grpc_server::pb;
use tokio_stream::StreamExt;
use tonic::Code;

use common::{connect, start_server, Client};

fn embedding(i: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| ((i * 31 + j * 7) % 97) as f32 / 97.0 - 0.5 + i as f32 * 1e-3)
        .collect()
}

fn raw(values: &[f32]) -> Bytes {
    Bytes::copy_from_slice(engine_core::codec::encode_embedding_raw(values))
}

fn payload(label: &str, valid_from: i64, emb: &[f32]) -> pb::HyperedgePayload {
    pb::HyperedgePayload {
        event_type: "OBSERVED".into(),
        participants: vec![
            pb::Participant {
                entity_id: label.into(),
                role: "subject".into(),
            },
            pb::Participant {
                entity_id: "hub".into(),
                role: "object".into(),
            },
        ],
        temporal_boundary: Some(pb::TemporalBoundary {
            valid_from_epoch: Some(valid_from),
            is_ongoing: true,
            ..Default::default()
        }),
        confidence_score: 0.9,
        embedding_raw: raw(emb),
        ..Default::default()
    }
}

fn insert_request(label: &str, valid_from: i64, emb: &[f32]) -> pb::InsertHyperedgeRequest {
    pb::InsertHyperedgeRequest {
        session_id: "test".into(),
        hyperedge: Some(payload(label, valid_from, emb)),
    }
}

async fn insert_stream(client: &mut Client, count: usize, dim: usize) -> pb::BatchResponse {
    let requests: Vec<_> = (0..count)
        .map(|i| insert_request(&format!("e{i}"), i as i64, &embedding(i, dim)))
        .collect();
    client
        .insert_hyperedge_batch(tokio_stream::iter(requests))
        .await
        .unwrap()
        .into_inner()
}

fn query(dim: usize, top_k: u32) -> pb::QueryTemporalRequest {
    pb::QueryTemporalRequest {
        session_id: "test".into(),
        query_embedding_raw: raw(&embedding(0, dim)),
        target_timestamp_epoch: 1_000_000,
        decay_lambda: 0.0,
        top_k,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn thousand_1536d_hyperedges_round_trip_bit_exact() {
    const COUNT: usize = 1_000;
    const DIM: usize = 1536;
    let server = start_server().await;
    let mut client = connect(&server.socket).await;

    let batch = insert_stream(&mut client, COUNT, DIM).await;
    assert_eq!(batch.processed_count, COUNT as u32);
    assert!(batch.all_succeeded, "{:?}", batch.errors);

    let response = client
        .query_temporal_context(query(DIM, COUNT as u32))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.matched_edges.len(), COUNT);
    assert_eq!(response.decay_scores.len(), COUNT);
    assert!(response.decay_scores.windows(2).all(|w| w[0] >= w[1]));

    let by_label: HashMap<String, &pb::HyperedgePayload> = response
        .matched_edges
        .iter()
        .map(|e| (e.participants[0].entity_id.clone(), e))
        .collect();
    for i in 0..COUNT {
        let edge = by_label[&format!("e{i}")];
        let decoded = engine_core::codec::decode_embedding_raw(&edge.embedding_raw).unwrap();
        assert_eq!(&*decoded, embedding(i, DIM).as_slice(), "edge e{i}");
        assert_eq!(edge.event_type, "OBSERVED");
        assert_eq!(edge.participants[1].entity_id, "hub");
        let boundary = edge.temporal_boundary.as_ref().unwrap();
        assert_eq!(boundary.valid_from_epoch, Some(i as i64));
        assert!(boundary.is_ongoing);
        assert!((edge.confidence_score - 0.9).abs() < 1e-6);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unary_insert_query_and_checkpoint() {
    let server = start_server().await;
    let mut client = connect(&server.socket).await;
    let first = client
        .insert_hyperedge(insert_request("a", 10, &[1.0, 0.0]))
        .await
        .unwrap()
        .into_inner();
    let second = client
        .insert_hyperedge(insert_request("b", 20, &[0.0, 1.0]))
        .await
        .unwrap()
        .into_inner();
    assert!(first.success && second.success);
    assert!(second.sequence_id > first.sequence_id);
    assert_ne!(first.hyperedge_id, second.hyperedge_id);

    let response = client
        .query_temporal_context(pb::QueryTemporalRequest {
            query_embedding_raw: raw(&[1.0, 0.0]),
            target_timestamp_epoch: 100,
            decay_lambda: 0.1,
            top_k: 1,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.matched_edges.len(), 1);
    assert_eq!(response.matched_edges[0].id, first.hyperedge_id);

    let checkpoint = client
        .trigger_checkpoint(pb::CheckpointRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(checkpoint.success);
    assert_eq!(checkpoint.wal_sequence_id, second.sequence_id as u64);
    assert!(server.dir.path().join("graph").read_dir().unwrap().count() > 0);

    let health = client
        .health_check(pb::HealthCheckRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(health.healthy);
    assert_eq!(
        health.active_connections, 1,
        "the health check itself is in flight"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stream_respects_slow_and_departing_consumers() {
    const COUNT: usize = 500;
    let server = start_server().await;
    let mut client = connect(&server.socket).await;
    assert!(insert_stream(&mut client, COUNT, 8).await.all_succeeded);

    let mut stream = client
        .stream_temporal_context(query(8, COUNT as u32))
        .await
        .unwrap()
        .into_inner();
    let mut received = 0;
    while let Some(item) = stream.next().await {
        item.unwrap();
        received += 1;
        if received % 50 == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
    assert_eq!(received, COUNT);

    let mut partial = client
        .stream_temporal_context(query(8, COUNT as u32))
        .await
        .unwrap()
        .into_inner();
    for _ in 0..3 {
        partial.next().await.unwrap().unwrap();
    }
    drop(partial);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        client
            .health_check(pb::HealthCheckRequest {})
            .await
            .unwrap()
            .into_inner()
            .healthy
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn kv_blocks_store_load_dedup_and_plan() {
    let server = start_server().await;
    let mut client = connect(&server.socket).await;
    let edge = client
        .insert_hyperedge(insert_request("kv", 0, &[1.0]))
        .await
        .unwrap()
        .into_inner();

    let block: Bytes = (0..8 * 1024 * 1024u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect::<Vec<u8>>()
        .into();
    let layout = pb::KvBlockLayout {
        num_layers: 32,
        num_heads: 32,
        head_dim: 128,
        block_size: 16,
        precision: pb::KvPrecision::Fp16 as i32,
    };
    let hash_a = "ab".repeat(32);
    let hash_b = "cd".repeat(32);
    let stored = client
        .store_kv_blocks(pb::StoreKvBlocksRequest {
            block_hash: hash_a.clone(),
            kv_chunk_bytes: block.clone(),
            associated_hyperedge_ids: vec![edge.hyperedge_id.clone()],
            layout: Some(layout),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(stored.success && !stored.deduplicated);
    assert_eq!(stored.bytes_written, 64 + block.len() as u64);

    let dedup = client
        .store_kv_blocks(pb::StoreKvBlocksRequest {
            block_hash: hash_b.clone(),
            kv_chunk_bytes: block.clone(),
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert!(dedup.deduplicated);
    assert_eq!(dedup.bytes_written, 0);

    let loaded = client
        .load_kv_blocks(pb::LoadKvBlocksRequest {
            block_hash: hash_a.clone(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(loaded.hit);
    assert_eq!(loaded.kv_chunk_bytes, block);
    assert_eq!(loaded.layout, Some(layout));

    let miss = client
        .load_kv_blocks(pb::LoadKvBlocksRequest {
            block_hash: "00".repeat(32),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!miss.hit);

    let plan = client
        .plan_kv_evictions(pb::PlanKvEvictionsRequest {
            decay_threshold: 0.5,
            now_epoch: 365 * 86_400,
            max_candidates: 0,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(plan.block_hashes, [hash_a]);

    let bad = client
        .store_kv_blocks(pb::StoreKvBlocksRequest {
            block_hash: "ef".repeat(32),
            kv_chunk_bytes: Bytes::from_static(&[0; 10]),
            layout: Some(layout),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(bad.code(), Code::InvalidArgument);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn misaligned_embedding_is_invalid_argument() {
    let server = start_server().await;
    let mut client = connect(&server.socket).await;
    let mut request = insert_request("x", 0, &[]);
    request.hyperedge.as_mut().unwrap().embedding_raw = Bytes::from_static(&[1, 2, 3, 4, 5, 6, 7]);
    let status = client.insert_hyperedge(request).await.unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status.message().contains("alignment"),
        "{}",
        status.message()
    );

    let mut bad_query = query(2, 1);
    bad_query.query_embedding_raw = Bytes::from_static(&[0; 6]);
    let status = client.query_temporal_context(bad_query).await.unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);
}

fn record(
    event_type: &str,
    participants: &[&str],
    valid_from: i64,
    summary: &str,
) -> pb::InsertHyperedgeRequest {
    pb::InsertHyperedgeRequest {
        session_id: "s1".into(),
        hyperedge: Some(pb::HyperedgePayload {
            event_type: event_type.into(),
            summary: summary.into(),
            participants: participants
                .iter()
                .map(|p| pb::Participant {
                    entity_id: (*p).into(),
                    role: String::new(),
                })
                .collect(),
            temporal_boundary: Some(pb::TemporalBoundary {
                valid_from_epoch: Some(valid_from),
                is_ongoing: true,
                ..Default::default()
            }),
            ..Default::default()
        }),
    }
}

fn traverse(seeds: &[&str], k_hops: u32, order: pb::TraversalOrder) -> pb::TraverseTemporalRequest {
    pb::TraverseTemporalRequest {
        seed_entity_ids: seeds.iter().map(|s| (*s).into()).collect(),
        k_hops,
        as_of_epoch: 1_000,
        order: order.into(),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn summaries_session_history_traversal_and_delete() {
    let server = start_server().await;
    let mut client = connect(&server.socket).await;
    let mut ids = Vec::new();
    for (i, (event, text)) in [("MESSAGE_HUMAN", "hi"), ("MESSAGE_AI", "hello there")]
        .iter()
        .enumerate()
    {
        let record_node = format!("record:{i}");
        let response = client
            .insert_hyperedge(record(
                event,
                &["session:s1", &record_node],
                100 + i as i64,
                text,
            ))
            .await
            .unwrap()
            .into_inner();
        ids.push(response.hyperedge_id);
    }
    for (from, to) in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")] {
        client
            .insert_hyperedge(record("LINK", &[from, to], 10, &format!("{from}->{to}")))
            .await
            .unwrap();
    }

    let history = client
        .traverse_temporal_context(pb::TraverseTemporalRequest {
            event_types: vec!["MESSAGE_HUMAN".into(), "MESSAGE_AI".into()],
            ..traverse(&["session:s1"], 1, pb::TraversalOrder::Chronological)
        })
        .await
        .unwrap()
        .into_inner();
    let texts: Vec<&str> = history
        .matched_edges
        .iter()
        .map(|e| e.summary.as_str())
        .collect();
    assert_eq!(texts, ["hello there", "hi"]);
    assert_eq!(history.matched_edges[0].participants.len(), 2);

    let three_hop = client
        .traverse_temporal_context(traverse(&["a"], 3, pb::TraversalOrder::Relevance))
        .await
        .unwrap()
        .into_inner();
    let mut reached: Vec<&str> = three_hop
        .matched_edges
        .iter()
        .map(|e| e.summary.as_str())
        .collect();
    reached.sort_unstable();
    assert_eq!(reached, ["a->b", "b->c", "c->d"]);
    assert_eq!(three_hop.decay_scores.len(), 3);

    let too_deep = client
        .traverse_temporal_context(traverse(&["a"], 9, pb::TraversalOrder::Relevance))
        .await
        .unwrap_err();
    assert_eq!(too_deep.code(), Code::InvalidArgument);

    let deleted = client
        .delete_hyperedges(pb::DeleteHyperedgesRequest {
            hyperedge_ids: vec![ids[0].clone(), ids[0].clone()],
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(deleted.deleted_count, 1);
    let after = client
        .traverse_temporal_context(traverse(
            &["session:s1"],
            1,
            pb::TraversalOrder::Chronological,
        ))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(after.matched_edges.len(), 1);
    assert_eq!(after.matched_edges[0].summary, "hello there");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn api_key_is_enforced_when_configured() {
    let server = common::start_server_with_auth(grpc_server::ApiKeyCheck::new(Some("k3y"))).await;
    let mut client = connect(&server.socket).await;
    let denied = client
        .health_check(pb::HealthCheckRequest {})
        .await
        .unwrap_err();
    assert_eq!(denied.code(), Code::Unauthenticated);

    let mut wrong = tonic::Request::new(pb::HealthCheckRequest {});
    wrong
        .metadata_mut()
        .insert(grpc_server::API_KEY_HEADER, "nope".parse().unwrap());
    assert_eq!(
        client.health_check(wrong).await.unwrap_err().code(),
        Code::Unauthenticated
    );

    let mut ok = tonic::Request::new(pb::HealthCheckRequest {});
    ok.metadata_mut()
        .insert(grpc_server::API_KEY_HEADER, "k3y".parse().unwrap());
    assert!(client.health_check(ok).await.unwrap().into_inner().healthy);
}

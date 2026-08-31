//! Unary latency and streaming throughput over a Unix domain socket.

#[allow(dead_code)]
#[path = "../tests/common/mod.rs"]
mod common;

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use grpc_server::pb;
use tokio_stream::StreamExt;

const DIM: usize = 1536;
const STREAM_EDGES: usize = 2_000;

fn edge(i: usize) -> pb::InsertHyperedgeRequest {
    let embedding: Vec<f32> = (0..DIM).map(|j| ((i + j) % 13) as f32).collect();
    pb::InsertHyperedgeRequest {
        session_id: String::new(),
        hyperedge: Some(pb::HyperedgePayload {
            event_type: "BENCH".into(),
            participants: vec![
                pb::Participant {
                    entity_id: format!("n{i}"),
                    role: String::new(),
                },
                pb::Participant {
                    entity_id: "hub".into(),
                    role: String::new(),
                },
            ],
            temporal_boundary: Some(pb::TemporalBoundary {
                valid_from_epoch: Some(i as i64),
                is_ongoing: true,
                ..Default::default()
            }),
            embedding_raw: Bytes::copy_from_slice(engine_core::codec::encode_embedding_raw(
                &embedding,
            )),
            ..Default::default()
        }),
    }
}

fn bench_grpc(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let (server, client) = runtime.block_on(async {
        let server = common::start_server().await;
        let client = common::connect(&server.socket).await;
        (server, client)
    });

    let mut group = c.benchmark_group("grpc_uds");
    group.bench_function("health_check_unary", |b| {
        b.to_async(&runtime).iter(|| {
            let mut client = client.clone();
            async move {
                client
                    .health_check(pb::HealthCheckRequest {})
                    .await
                    .unwrap()
            }
        })
    });

    let mut next = 0usize;
    group.bench_function("insert_unary_1536d", |b| {
        b.to_async(&runtime).iter(|| {
            next += 1;
            let request = edge(1_000_000 + next);
            let mut client = client.clone();
            async move { client.insert_hyperedge(request).await.unwrap() }
        })
    });

    runtime.block_on(async {
        let mut client = client.clone();
        let requests: Vec<_> = (0..STREAM_EDGES).map(edge).collect();
        let response = client
            .insert_hyperedge_batch(tokio_stream::iter(requests))
            .await
            .unwrap()
            .into_inner();
        assert!(response.all_succeeded);
    });

    group.throughput(Throughput::Elements(STREAM_EDGES as u64));
    group.sample_size(20);
    group.bench_function("stream_temporal_context", |b| {
        b.to_async(&runtime).iter(|| {
            let mut client = client.clone();
            async move {
                let mut stream = client
                    .stream_temporal_context(pb::QueryTemporalRequest {
                        query_embedding_raw: Bytes::copy_from_slice(
                            engine_core::codec::encode_embedding_raw(&[1.0; DIM]),
                        ),
                        target_timestamp_epoch: i64::MAX - 1,
                        top_k: STREAM_EDGES as u32,
                        ..Default::default()
                    })
                    .await
                    .unwrap()
                    .into_inner();
                let mut count = 0;
                while let Some(item) = stream.next().await {
                    item.unwrap();
                    count += 1;
                }
                count
            }
        })
    });
    group.finish();
    drop(server);
}

criterion_group!(benches, bench_grpc);
criterion_main!(benches);

//! `HypergraphEngineService` implementation.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use engine_core::codec::{decode_embedding_raw, format_hyperedge_id, parse_hyperedge_id};
use engine_core::temporal_math::DecayConfig;
use engine_core::{
    ConcurrentHypergraphEngine, EngineError, NewHyperedge, ScoredHyperedge, TraversalOrder,
    TraversalQuery,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use vllm_kv_store::{from_hex, plan_kv_evictions, to_hex, KvBlockLayout, KvBlockStore};

use crate::convert::{
    edge_to_payload, engine_status, kv_status, layout_from_pb, layout_to_pb,
    payload_to_new_hyperedge,
};
use crate::pb;
use crate::pb::hypergraph_engine_service_server::HypergraphEngineService;

/// Results returned when a query leaves `top_k` unset.
const DEFAULT_TOP_K: usize = 10;
/// Buffered messages per server stream before the producer blocks.
const STREAM_BUFFER: usize = 64;
/// Deepest traversal a client may request.
const MAX_HOPS: u32 = 4;

/// The service state shared by every RPC.
#[derive(Clone)]
pub struct HypergraphService {
    engine: Arc<ConcurrentHypergraphEngine>,
    kv: Arc<KvBlockStore>,
    started: Instant,
    in_flight: Arc<AtomicU32>,
}

impl std::fmt::Debug for HypergraphService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HypergraphService")
            .field("engine", &self.engine)
            .field("kv", &self.kv)
            .finish_non_exhaustive()
    }
}

async fn blocking<T, F>(work: F) -> Result<T, Status>
where
    F: FnOnce() -> Result<T, Status> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| Status::internal(format!("blocking task failed: {e}")))?
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

fn block_key(hex: &str) -> Result<[u8; 32], Status> {
    from_hex(hex).map_err(kv_status)
}

impl HypergraphService {
    /// Wraps an engine and its KV store.
    pub fn new(engine: Arc<ConcurrentHypergraphEngine>, kv: Arc<KvBlockStore>) -> Self {
        Self {
            engine,
            kv,
            started: Instant::now(),
            in_flight: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Tower layer feeding this service's in-flight request gauge.
    pub fn in_flight_layer(&self) -> InFlightLayer {
        InFlightLayer(self.in_flight.clone())
    }

    fn decay_for(&self, lambda: f32) -> Option<DecayConfig> {
        (lambda > 0.0 && lambda.is_finite()).then(|| {
            DecayConfig::new(
                std::f32::consts::LN_2 / lambda,
                self.engine.config().decay.base_weight,
            )
        })
    }

    async fn run_query(
        &self,
        request: pb::QueryTemporalRequest,
    ) -> Result<Vec<ScoredHyperedge>, Status> {
        let decay = self.decay_for(request.decay_lambda);
        let top_k = match request.top_k {
            0 => DEFAULT_TOP_K,
            k => k as usize,
        };
        let engine = self.engine.clone();
        blocking(move || {
            let query =
                decode_embedding_raw(&request.query_embedding_raw).map_err(engine_status)?;
            Ok(engine.query_temporal(&query, request.target_timestamp_epoch, decay, top_k))
        })
        .await
    }

    async fn run_traversal(
        &self,
        request: pb::TraverseTemporalRequest,
    ) -> Result<Vec<ScoredHyperedge>, Status> {
        let order = pb::TraversalOrder::try_from(request.order)
            .map_err(|_| Status::invalid_argument(format!("unknown order {}", request.order)))?;
        let k_hops = match request.k_hops {
            0 => 1,
            k @ 1..=MAX_HOPS => k as usize,
            k => {
                return Err(Status::invalid_argument(format!(
                    "k_hops {k} exceeds {MAX_HOPS}"
                )))
            }
        };
        let decay = self.decay_for(request.decay_lambda);
        let engine = self.engine.clone();
        blocking(move || {
            let embedding =
                decode_embedding_raw(&request.query_embedding_raw).map_err(engine_status)?;
            let query = TraversalQuery {
                seeds: &request.seed_entity_ids,
                k_hops,
                as_of: match request.as_of_epoch {
                    0 => unix_now(),
                    t => t,
                },
                query_embedding: (!embedding.is_empty()).then_some(&*embedding),
                decay,
                top_k: match request.top_k {
                    0 => usize::MAX,
                    k => k as usize,
                },
                order: match order {
                    pb::TraversalOrder::Relevance => TraversalOrder::Relevance,
                    pb::TraversalOrder::Chronological => TraversalOrder::Chronological,
                },
                event_types: &request.event_types,
            };
            Ok(engine.traverse_query(&query))
        })
        .await
    }

    async fn render(
        &self,
        ranked: Vec<ScoredHyperedge>,
        started: Instant,
    ) -> Result<pb::QueryTemporalResponse, Status> {
        let engine = self.engine.clone();
        let (matched_edges, decay_scores) = blocking(move || {
            Ok(ranked
                .iter()
                .map(|s| (edge_to_payload(&engine, &s.edge), s.score))
                .unzip())
        })
        .await?;
        Ok(pb::QueryTemporalResponse {
            matched_edges,
            decay_scores,
            query_latency_us: started.elapsed().as_micros() as u64,
        })
    }
}

#[tonic::async_trait]
impl HypergraphEngineService for HypergraphService {
    async fn insert_hyperedge(
        &self,
        request: Request<pb::InsertHyperedgeRequest>,
    ) -> Result<Response<pb::InsertHyperedgeResponse>, Status> {
        let payload = request
            .into_inner()
            .hyperedge
            .ok_or_else(|| Status::invalid_argument("hyperedge is required"))?;
        let edge = payload_to_new_hyperedge(&payload)?;
        let engine = self.engine.clone();
        let (id, seq) = blocking(move || {
            let outcome = engine.insert(edge).map_err(engine_status)?;
            Ok((outcome.id, engine.last_sequence()))
        })
        .await?;
        Ok(Response::new(pb::InsertHyperedgeResponse {
            success: true,
            hyperedge_id: format_hyperedge_id(id),
            sequence_id: seq as i64,
        }))
    }

    async fn insert_hyperedge_batch(
        &self,
        request: Request<Streaming<pb::InsertHyperedgeRequest>>,
    ) -> Result<Response<pb::BatchResponse>, Status> {
        let mut stream = request.into_inner();
        let chunk_size = self.engine.config().max_batch.max(1);
        let mut processed: u32 = 0;
        let mut errors = Vec::new();
        let mut chunk: Vec<(u32, NewHyperedge)> = Vec::with_capacity(chunk_size);
        loop {
            let next = stream.message().await?;
            let finished = next.is_none();
            if let Some(message) = next {
                let index = processed;
                processed = processed
                    .checked_add(1)
                    .ok_or_else(|| Status::invalid_argument("too many items in one batch stream"))?;
                match message
                    .hyperedge
                    .ok_or_else(|| Status::invalid_argument("hyperedge is required"))
                    .and_then(|payload| payload_to_new_hyperedge(&payload))
                {
                    Ok(edge) => chunk.push((index, edge)),
                    Err(status) => errors.push(format!("item {index}: {}", status.message())),
                }
            }
            if chunk.len() >= chunk_size || (finished && !chunk.is_empty()) {
                let batch = std::mem::take(&mut chunk);
                let engine = self.engine.clone();
                let failures = blocking(move || {
                    let (indices, edges): (Vec<u32>, Vec<NewHyperedge>) = batch.into_iter().unzip();
                    Ok(indices
                        .into_iter()
                        .zip(engine.insert_batch(edges))
                        .filter_map(|(i, r)| r.err().map(|e| format!("item {i}: {e}")))
                        .collect::<Vec<_>>())
                })
                .await?;
                errors.extend(failures);
            }
            if finished {
                break;
            }
        }
        Ok(Response::new(pb::BatchResponse {
            processed_count: processed,
            all_succeeded: errors.is_empty(),
            errors,
        }))
    }

    async fn query_temporal_context(
        &self,
        request: Request<pb::QueryTemporalRequest>,
    ) -> Result<Response<pb::QueryTemporalResponse>, Status> {
        let started = Instant::now();
        let ranked = self.run_query(request.into_inner()).await?;
        Ok(Response::new(self.render(ranked, started).await?))
    }

    async fn traverse_temporal_context(
        &self,
        request: Request<pb::TraverseTemporalRequest>,
    ) -> Result<Response<pb::QueryTemporalResponse>, Status> {
        let started = Instant::now();
        let ranked = self.run_traversal(request.into_inner()).await?;
        Ok(Response::new(self.render(ranked, started).await?))
    }

    async fn delete_hyperedges(
        &self,
        request: Request<pb::DeleteHyperedgesRequest>,
    ) -> Result<Response<pb::DeleteHyperedgesResponse>, Status> {
        let ids = request
            .into_inner()
            .hyperedge_ids
            .iter()
            .map(|id| parse_hyperedge_id(id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(engine_status)?;
        let engine = self.engine.clone();
        let deleted_count = blocking(move || {
            let mut deleted = 0u32;
            for id in ids {
                match engine.soft_delete_hyperedge(id) {
                    Ok(()) => deleted += 1,
                    Err(EngineError::NotFound(_)) => {}
                    Err(err) => return Err(engine_status(err)),
                }
            }
            Ok(deleted)
        })
        .await?;
        Ok(Response::new(pb::DeleteHyperedgesResponse {
            deleted_count,
        }))
    }

    type StreamTemporalContextStream = ReceiverStream<Result<pb::HyperedgePayload, Status>>;

    async fn stream_temporal_context(
        &self,
        request: Request<pb::QueryTemporalRequest>,
    ) -> Result<Response<Self::StreamTemporalContextStream>, Status> {
        let ranked = self.run_query(request.into_inner()).await?;
        let (tx, rx) = mpsc::channel(STREAM_BUFFER);
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || {
            for scored in &ranked {
                if tx
                    .blocking_send(Ok(edge_to_payload(&engine, &scored.edge)))
                    .is_err()
                {
                    break;
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn store_kv_blocks(
        &self,
        request: Request<pb::StoreKvBlocksRequest>,
    ) -> Result<Response<pb::StoreKvBlocksResponse>, Status> {
        let request = request.into_inner();
        let key = block_key(&request.block_hash)?;
        let layout = request
            .layout
            .as_ref()
            .map(layout_from_pb)
            .transpose()?
            .unwrap_or(KvBlockLayout::OPAQUE);
        let hyperedges = request
            .associated_hyperedge_ids
            .iter()
            .map(|id| parse_hyperedge_id(id).map(|id| id.to_u64()))
            .collect::<Result<Vec<u64>, _>>()
            .map_err(engine_status)?;
        let kv = self.kv.clone();
        let outcome = blocking(move || {
            kv.store(&key, &request.kv_chunk_bytes, layout, &hyperedges)
                .map_err(kv_status)
        })
        .await?;
        Ok(Response::new(pb::StoreKvBlocksResponse {
            success: true,
            bytes_written: outcome.bytes_written,
            deduplicated: outcome.deduplicated,
        }))
    }

    async fn load_kv_blocks(
        &self,
        request: Request<pb::LoadKvBlocksRequest>,
    ) -> Result<Response<pb::LoadKvBlocksResponse>, Status> {
        let key = block_key(&request.into_inner().block_hash)?;
        let kv = self.kv.clone();
        let response = blocking(move || {
            Ok(match kv.get(&key) {
                Some(block) => pb::LoadKvBlocksResponse {
                    hit: true,
                    kv_chunk_bytes: Bytes::copy_from_slice(block.bytes()),
                    layout: layout_to_pb(block.layout()),
                },
                None => pb::LoadKvBlocksResponse::default(),
            })
        })
        .await?;
        Ok(Response::new(response))
    }

    async fn plan_kv_evictions(
        &self,
        request: Request<pb::PlanKvEvictionsRequest>,
    ) -> Result<Response<pb::PlanKvEvictionsResponse>, Status> {
        let request = request.into_inner();
        let now = match request.now_epoch {
            0 => unix_now(),
            t => t,
        };
        let engine = self.engine.clone();
        let kv = self.kv.clone();
        let mut plan = blocking(move || {
            Ok(plan_kv_evictions(
                &engine,
                &kv,
                request.decay_threshold,
                now,
            ))
        })
        .await?;
        if request.max_candidates > 0 {
            plan.truncate(request.max_candidates as usize);
        }
        let (block_hashes, scores) = plan.iter().map(|(key, s)| (to_hex(key), *s)).unzip();
        Ok(Response::new(pb::PlanKvEvictionsResponse {
            block_hashes,
            scores,
        }))
    }

    async fn health_check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        Ok(Response::new(pb::HealthCheckResponse {
            healthy: true,
            uptime_seconds: self.started.elapsed().as_secs(),
            active_connections: self.in_flight.load(Ordering::Relaxed),
        }))
    }

    async fn trigger_checkpoint(
        &self,
        _request: Request<pb::CheckpointRequest>,
    ) -> Result<Response<pb::CheckpointResponse>, Status> {
        let engine = self.engine.clone();
        let seq = blocking(move || {
            engine.checkpoint().map_err(engine_status)?;
            Ok(engine.last_sequence())
        })
        .await?;
        Ok(Response::new(pb::CheckpointResponse {
            success: true,
            wal_sequence_id: seq,
        }))
    }
}

/// Counts requests between arrival and response headers.
#[derive(Debug, Clone)]
pub struct InFlightLayer(Arc<AtomicU32>);

impl<S> tower::Layer<S> for InFlightLayer {
    type Service = InFlight<S>;

    fn layer(&self, inner: S) -> Self::Service {
        InFlight {
            inner,
            gauge: self.0.clone(),
        }
    }
}

/// Service produced by [`InFlightLayer`].
#[derive(Debug, Clone)]
pub struct InFlight<S> {
    inner: S,
    gauge: Arc<AtomicU32>,
}

struct GaugeGuard(Arc<AtomicU32>);

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl<S, R> tower::Service<R> for InFlight<S>
where
    S: tower::Service<R>,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<S::Response, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: R) -> Self::Future {
        self.gauge.fetch_add(1, Ordering::Relaxed);
        let guard = GaugeGuard(self.gauge.clone());
        let future = self.inner.call(request);
        Box::pin(async move {
            let _guard = guard;
            future.await
        })
    }
}

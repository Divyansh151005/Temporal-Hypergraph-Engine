//! `hypergraph_engine_native`: in-process Python bindings.
//!
//! Embeddings cross the boundary as packed little-endian `f32` bytes and are
//! reinterpreted in place. KV blocks cross as raw pointers
//! (`torch.Tensor.data_ptr()`), so an 8 MiB block is copied exactly once,
//! straight into (or out of) its memory-mapped segment. The GIL is released
//! for all storage work.
#![deny(unsafe_code)]
// PyO3 0.22's `#[pymethods]` expansion wraps every `PyResult` in a no-op `.into()`.
#![allow(clippy::useless_conversion)]

use std::sync::Arc;

use engine_core::codec::{
    decode_embedding_raw, encode_embedding_raw, format_hyperedge_id, parse_hyperedge_id,
};
use engine_core::temporal_math::DecayConfig;
use engine_core::{
    ConcurrentHypergraphEngine, EngineConfig, EngineError, NewHyperedge, ScoredHyperedge, SyncMode,
    TraversalOrder, TraversalQuery,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use vllm_kv_store::{
    token_chain_hash, BlockHash, KvBlockLayout, KvBlockStore, KvError, KvStoreConfig, Precision,
    HEADER_LEN,
};

/// `(num_layers, num_heads, head_dim, block_size, precision_id)`.
type LayoutTuple = (u16, u16, u16, u16, u16);

fn engine_err(err: EngineError) -> PyErr {
    let message = err.to_string();
    match err {
        EngineError::InvalidArgument(_)
        | EngineError::EmbeddingDimension { .. }
        | EngineError::InvalidAlignment(_) => PyValueError::new_err(message),
        EngineError::NotFound(_) => PyKeyError::new_err(message),
        EngineError::Io(_)
        | EngineError::CapacityExceeded
        | EngineError::BatchAborted(_)
        | EngineError::Recovery(_)
        | EngineError::CheckpointCorrupt(_)
        | EngineError::Closed => PyRuntimeError::new_err(message),
    }
}

fn kv_err(err: KvError) -> PyErr {
    let message = err.to_string();
    match err {
        KvError::InvalidArgument(_)
        | KvError::LayoutMismatch { .. }
        | KvError::DestinationTooSmall { .. } => PyValueError::new_err(message),
        KvError::Io(_) | KvError::Corrupt(_) => PyRuntimeError::new_err(message),
    }
}

fn block_key(block_hash: &[u8]) -> PyResult<BlockHash> {
    block_hash.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "block_hash must be 32 bytes, got {}",
            block_hash.len()
        ))
    })
}

fn layout_from_tuple(layout: Option<LayoutTuple>) -> PyResult<KvBlockLayout> {
    match layout {
        None => Ok(KvBlockLayout::OPAQUE),
        Some((layers, heads, head_dim, block_size, precision)) => Ok(KvBlockLayout::new(
            layers,
            heads,
            head_dim,
            block_size,
            Precision::from_id(precision).map_err(kv_err)?,
        )),
    }
}

fn hyperedge_ids(hyperedge_id: Option<&str>) -> PyResult<Vec<u64>> {
    hyperedge_id
        .map(|id| parse_hyperedge_id(id).map(|id| id.to_u64()))
        .transpose()
        .map(|id| id.into_iter().collect())
        .map_err(engine_err)
}

/// In-process handle to a hypergraph engine and its KV block store.
#[pyclass(module = "hypergraph_engine_native", frozen)]
pub struct NativeHypergraphEngine {
    engine: Arc<ConcurrentHypergraphEngine>,
    kv: Arc<KvBlockStore>,
}

impl NativeHypergraphEngine {
    fn decay_for(&self, decay_lambda: f32) -> Option<DecayConfig> {
        (decay_lambda > 0.0 && decay_lambda.is_finite()).then(|| {
            DecayConfig::new(
                std::f32::consts::LN_2 / decay_lambda,
                self.engine.config().decay.base_weight,
            )
        })
    }

    fn scored_to_dict<'py>(
        &self,
        py: Python<'py>,
        scored: &ScoredHyperedge,
        with_embedding: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let edge = &scored.edge;
        let dict = PyDict::new_bound(py);
        dict.set_item("id", format_hyperedge_id(edge.id))?;
        dict.set_item(
            "event_type",
            self.engine.event_type(edge).map(|t| t.to_string()),
        )?;
        let participants: Vec<String> = edge
            .participants
            .iter()
            .filter_map(|&node| self.engine.node_label(node).map(|l| l.to_string()))
            .collect();
        dict.set_item("participants", participants)?;
        dict.set_item("valid_from", edge.valid_from)?;
        dict.set_item("valid_to", edge.valid_to)?;
        dict.set_item("is_ongoing", edge.is_ongoing)?;
        dict.set_item("confidence", edge.decay_weight)?;
        dict.set_item("score", scored.score)?;
        dict.set_item(
            "summary",
            self.engine.summary(edge.id).map(|s| s.to_string()),
        )?;
        let embedding = with_embedding
            .then(|| self.engine.embedding(edge.id))
            .flatten()
            .map(|e| PyBytes::new_bound(py, encode_embedding_raw(&e)));
        dict.set_item("embedding", embedding)?;
        Ok(dict)
    }
}

#[pymethods]
impl NativeHypergraphEngine {
    /// Opens (or creates) an engine at `storage_path`; KV blocks live in `storage_path/kv`.
    #[new]
    #[pyo3(signature = (storage_path, embedding_dim = 0, sync = "fsync"))]
    fn new(py: Python<'_>, storage_path: &str, embedding_dim: usize, sync: &str) -> PyResult<Self> {
        let sync_mode = match sync {
            "fsync" => SyncMode::Fsync,
            "buffered" => SyncMode::OsBuffered,
            other => {
                return Err(PyValueError::new_err(format!(
                    "sync must be 'fsync' or 'buffered', got {other:?}"
                )))
            }
        };
        let config = EngineConfig {
            embedding_dim,
            sync_mode,
            ..EngineConfig::default()
        };
        let path = std::path::PathBuf::from(storage_path);
        py.allow_threads(|| {
            let engine =
                ConcurrentHypergraphEngine::open_with_config(&path, config).map_err(engine_err)?;
            let kv =
                KvBlockStore::open(path.join("kv"), KvStoreConfig::default()).map_err(kv_err)?;
            Ok(Self {
                engine: Arc::new(engine),
                kv: Arc::new(kv),
            })
        })
    }

    /// Volatile engine and KV store (nothing touches disk).
    #[staticmethod]
    #[pyo3(signature = (embedding_dim = 0))]
    fn in_memory(embedding_dim: usize) -> Self {
        Self {
            engine: Arc::new(ConcurrentHypergraphEngine::in_memory(EngineConfig {
                embedding_dim,
                sync_mode: SyncMode::OsBuffered,
                ..EngineConfig::default()
            })),
            kv: Arc::new(KvBlockStore::in_memory(KvStoreConfig {
                segment_bytes: 256 << 20,
                ..KvStoreConfig::default()
            })),
        }
    }

    /// Inserts a hyperedge; `embedding_buffer` holds packed little-endian
    /// float32 values (empty for no embedding). Returns the hyperedge id.
    #[pyo3(signature = (event_type, node_labels, valid_from, valid_to, embedding_buffer, confidence = 1.0, summary = None))]
    #[allow(clippy::too_many_arguments)]
    fn insert_hyperedge_fast(
        &self,
        py: Python<'_>,
        event_type: &str,
        node_labels: Vec<String>,
        valid_from: i64,
        valid_to: Option<i64>,
        embedding_buffer: &Bound<'_, PyBytes>,
        confidence: f32,
        summary: Option<String>,
    ) -> PyResult<String> {
        let raw = embedding_buffer.as_bytes();
        py.allow_threads(|| {
            let mut edge =
                NewHyperedge::new(event_type, node_labels, valid_from).confidence(confidence);
            edge.valid_to = valid_to;
            edge.summary = summary;
            if !raw.is_empty() {
                edge.embedding = Some(decode_embedding_raw(raw).map_err(engine_err)?.into_owned());
            }
            let outcome = self.engine.insert(edge).map_err(engine_err)?;
            Ok(format_hyperedge_id(outcome.id))
        })
    }

    /// The `top_k` hyperedges valid at `target_ts`, ranked by
    /// `cosine * confidence * decay`. `decay_lambda` is per day; `0` uses the
    /// engine default.
    #[pyo3(signature = (query_bytes, target_ts, decay_lambda = 0.0, top_k = 10))]
    fn query_temporal<'py>(
        &self,
        py: Python<'py>,
        query_bytes: &[u8],
        target_ts: i64,
        decay_lambda: f32,
        top_k: usize,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let decay = self.decay_for(decay_lambda);
        let ranked = py.allow_threads(|| {
            let query = decode_embedding_raw(query_bytes).map_err(engine_err)?;
            Ok::<_, PyErr>(self.engine.query_temporal(&query, target_ts, decay, top_k))
        })?;
        ranked
            .iter()
            .map(|scored| self.scored_to_dict(py, scored, true))
            .collect()
    }

    /// Hyperedges within `k_hops` of the seed entities that are valid at
    /// `as_of`, ranked by `confidence * decay` (times cosine similarity when
    /// `query_bytes` is given) or, with `chronological`, newest first.
    /// `top_k = 0` returns every match. Unknown seeds are ignored.
    #[pyo3(signature = (
        seed_labels, k_hops, as_of, query_bytes = None, decay_lambda = 0.0, top_k = 10,
        chronological = false, event_types = None, *, with_embeddings = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn traverse<'py>(
        &self,
        py: Python<'py>,
        seed_labels: Vec<String>,
        k_hops: usize,
        as_of: i64,
        query_bytes: Option<&[u8]>,
        decay_lambda: f32,
        top_k: usize,
        chronological: bool,
        event_types: Option<Vec<String>>,
        with_embeddings: bool,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let decay = self.decay_for(decay_lambda);
        let event_types = event_types.unwrap_or_default();
        let ranked = py.allow_threads(|| {
            let embedding = query_bytes
                .map(decode_embedding_raw)
                .transpose()
                .map_err(engine_err)?;
            let query = TraversalQuery {
                query_embedding: embedding.as_deref().filter(|e| !e.is_empty()),
                decay,
                top_k: if top_k == 0 { usize::MAX } else { top_k },
                order: if chronological {
                    TraversalOrder::Chronological
                } else {
                    TraversalOrder::Relevance
                },
                event_types: &event_types,
                ..TraversalQuery::new(&seed_labels, k_hops, as_of)
            };
            Ok::<_, PyErr>(self.engine.traverse_query(&query))
        })?;
        ranked
            .iter()
            .map(|scored| self.scored_to_dict(py, scored, with_embeddings))
            .collect()
    }

    /// Removes a hyperedge from the live graph.
    fn soft_delete_hyperedge(&self, py: Python<'_>, hyperedge_id: &str) -> PyResult<()> {
        let id = parse_hyperedge_id(hyperedge_id).map_err(engine_err)?;
        py.allow_threads(|| self.engine.soft_delete_hyperedge(id).map_err(engine_err))
    }

    /// Writes a checkpoint; returns the WAL sequence it covers.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<u64> {
        py.allow_threads(|| {
            self.engine.checkpoint().map_err(engine_err)?;
            Ok(self.engine.last_sequence())
        })
    }

    /// Engine and KV store counters.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = self.engine.snapshot();
        let kv = self.kv.stats();
        let dict = PyDict::new_bound(py);
        dict.set_item("nodes", snapshot.live_node_count())?;
        dict.set_item("hyperedges", snapshot.live_hyperedge_count())?;
        dict.set_item("last_sequence", self.engine.last_sequence())?;
        dict.set_item("kv_blocks", kv.blocks)?;
        dict.set_item("kv_records", kv.records)?;
        dict.set_item("kv_segments", kv.segments)?;
        dict.set_item("kv_bytes_written", kv.bytes_written)?;
        dict.set_item("kv_dedup_hits", kv.dedup_hits)?;
        Ok(dict)
    }

    /// Stores `num_bytes` read from `data_ptr` under `block_hash` (32 bytes).
    ///
    /// The caller must keep the memory alive, contiguous and unmodified for
    /// the duration of the call (e.g. a synchronized contiguous CPU tensor).
    #[pyo3(signature = (block_hash, data_ptr, num_bytes, hyperedge_id = None, *, layout = None))]
    #[allow(unsafe_code)]
    fn store_kv_block_raw(
        &self,
        py: Python<'_>,
        block_hash: &[u8],
        data_ptr: usize,
        num_bytes: usize,
        hyperedge_id: Option<&str>,
        layout: Option<LayoutTuple>,
    ) -> PyResult<bool> {
        let key = block_key(block_hash)?;
        let layout = layout_from_tuple(layout)?;
        let hyperedges = hyperedge_ids(hyperedge_id)?;
        py.allow_threads(|| {
            // SAFETY: the caller guarantees `data_ptr..data_ptr + num_bytes`
            // is live, readable and not mutated during the call.
            unsafe {
                self.kv
                    .store_raw(&key, data_ptr as *const u8, num_bytes, layout, &hyperedges)
            }
            .map_err(kv_err)
        })?;
        Ok(true)
    }

    /// Copies the block stored under `block_hash` to `target_ptr`. Returns
    /// `False` on a miss.
    ///
    /// The caller must keep `target_len` writable bytes alive at `target_ptr`
    /// and not touch them during the call.
    #[allow(unsafe_code)]
    fn load_kv_block_raw(
        &self,
        py: Python<'_>,
        block_hash: &[u8],
        target_ptr: usize,
        target_len: usize,
    ) -> PyResult<bool> {
        let key = block_key(block_hash)?;
        py.allow_threads(|| {
            // SAFETY: the caller guarantees `target_ptr..target_ptr + target_len`
            // is live, writable and unaliased during the call.
            unsafe { self.kv.load_raw(&key, target_ptr as *mut u8, target_len) }.map_err(kv_err)
        })
    }

    /// Safe variant of `store_kv_block_raw` for any contiguous byte buffer
    /// (`bytes`, `bytearray`, uint8 NumPy arrays, `memoryview.cast('B')`).
    #[pyo3(signature = (block_hash, buffer, hyperedge_id = None, *, layout = None))]
    fn store_kv_block(
        &self,
        py: Python<'_>,
        block_hash: &[u8],
        buffer: PyBuffer<u8>,
        hyperedge_id: Option<&str>,
        layout: Option<LayoutTuple>,
    ) -> PyResult<bool> {
        let key = block_key(block_hash)?;
        let layout = layout_from_tuple(layout)?;
        let hyperedges = hyperedge_ids(hyperedge_id)?;
        let payload = buffer.to_vec(py)?;
        py.allow_threads(|| {
            self.kv
                .store(&key, &payload, layout, &hyperedges)
                .map_err(kv_err)
        })?;
        Ok(true)
    }

    /// The block stored under `block_hash` as `bytes`, or `None`.
    fn load_kv_block<'py>(
        &self,
        py: Python<'py>,
        block_hash: &[u8],
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let key = block_key(block_hash)?;
        let Some(block) = self.kv.get(&key) else {
            return Ok(None);
        };
        let bytes = block.bytes();
        PyBytes::new_bound_with(py, bytes.len(), |dst| {
            dst.copy_from_slice(bytes);
            Ok(())
        })
        .map(Some)
    }

    /// Whether a block is stored under `block_hash`.
    fn has_kv_block(&self, block_hash: &[u8]) -> PyResult<bool> {
        Ok(self.kv.contains(&block_key(block_hash)?))
    }

    /// Layout tuple of a stored block, or `None`.
    fn kv_block_layout(&self, block_hash: &[u8]) -> PyResult<Option<LayoutTuple>> {
        Ok(self.kv.get(&block_key(block_hash)?).map(|block| {
            let l = block.layout();
            (
                l.num_layers,
                l.num_heads,
                l.head_dim,
                l.block_size,
                l.precision.id(),
            )
        }))
    }

    /// Drops a block key (and its associations) from the store.
    fn remove_kv_block(&self, py: Python<'_>, block_hash: &[u8]) -> PyResult<bool> {
        let key = block_key(block_hash)?;
        py.allow_threads(|| self.kv.remove(&key).map_err(kv_err))
    }

    /// Block hashes whose associated hyperedges' decayed relevance at `now`
    /// is below `decay_threshold`, least relevant first.
    fn plan_kv_evictions<'py>(
        &self,
        py: Python<'py>,
        decay_threshold: f32,
        now: i64,
    ) -> Vec<Bound<'py, PyBytes>> {
        let plan = py.allow_threads(|| {
            vllm_kv_store::plan_kv_evictions(&self.engine, &self.kv, decay_threshold, now)
        });
        plan.iter()
            .map(|(key, _)| PyBytes::new_bound(py, key))
            .collect()
    }
}

/// `BLAKE3(parent || token_ids)` where `token_ids` are packed int32 values.
#[pyfunction]
#[pyo3(signature = (token_ids, parent = None))]
fn blake3_token_hash<'py>(
    py: Python<'py>,
    token_ids: &[u8],
    parent: Option<&[u8]>,
) -> PyResult<Bound<'py, PyBytes>> {
    let hash = token_chain_hash(parent, token_ids).map_err(kv_err)?;
    Ok(PyBytes::new_bound(py, &hash))
}

/// BLAKE3 digest of arbitrary bytes.
#[pyfunction]
fn blake3_content_hash<'py>(py: Python<'py>, data: &[u8]) -> Bound<'py, PyBytes> {
    PyBytes::new_bound(py, &vllm_kv_store::content_hash(data))
}

#[pymodule]
fn hypergraph_engine_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeHypergraphEngine>()?;
    m.add_function(wrap_pyfunction!(blake3_token_hash, m)?)?;
    m.add_function(wrap_pyfunction!(blake3_content_hash, m)?)?;
    m.add("KV_HEADER_LEN", HEADER_LEN)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

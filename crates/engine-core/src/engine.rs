//! The concurrent hypergraph engine.
//!
//! Readers load an immutable [`BipartiteArena`] snapshot from an [`ArcSwap`]
//! and never block. Writers enqueue requests on a bounded channel; whichever
//! writer holds the writer lock drains the queue as one batch (group commit):
//! it resolves temporal conflicts, applies mutations to a copy-on-write
//! working arena, appends one WAL batch, syncs, and only then publishes the
//! new snapshot and index entries.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use crossbeam_channel::{Receiver, Sender};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use parking_lot::Mutex;
use smallvec::SmallVec;
use temporal_math::{
    batch_compute_decay, cosine_similarity, decay_weight, resolve_conflicts_scoped, ConflictScope,
    DecayConfig, EdgeView, WalMutationCommand,
};
use tracing::warn;

use crate::arena::{ArenaWriter, BipartiteArena};
use crate::checkpoint;
use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::hash::stable_hash;
use crate::ids::{HyperedgeId, NodeId};
use crate::model::TemporalHyperedge;
use crate::sync::{AtomicU32, Ordering};
use crate::wal::{self, CommittedBatch, DeleteTarget, Mutation, WalWriter};

/// A hyperedge to insert.
#[derive(Debug, Clone, PartialEq)]
pub struct NewHyperedge {
    /// Semantic event type, e.g. `"CEO_OF"`.
    pub event_type: String,
    /// Participant labels; duplicates are ignored.
    pub participants: Vec<String>,
    /// Start of validity (Unix seconds).
    pub valid_from: i64,
    /// End of validity (Unix seconds, exclusive).
    pub valid_to: Option<i64>,
    /// Base extraction confidence in `[0, 1]`.
    pub confidence: f32,
    /// Optional embedding.
    pub embedding: Option<Vec<f32>>,
    /// Optional text payload (message content, document chunk, ...).
    pub summary: Option<String>,
    /// Participants identifying the relation for conflict resolution. `None`
    /// means the exact participant set (see [`ConflictScope`]).
    pub anchors: Option<Vec<String>>,
}

impl NewHyperedge {
    /// Creates an ongoing hyperedge with confidence `1.0`.
    pub fn new(
        event_type: impl Into<String>,
        participants: impl IntoIterator<Item = impl Into<String>>,
        valid_from: i64,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            participants: participants.into_iter().map(Into::into).collect(),
            valid_from,
            valid_to: None,
            confidence: 1.0,
            embedding: None,
            summary: None,
            anchors: None,
        }
    }

    /// Sets the end of validity.
    pub fn valid_to(mut self, valid_to: i64) -> Self {
        self.valid_to = Some(valid_to);
        self
    }

    /// Sets the base confidence.
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Attaches an embedding.
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Attaches a text payload.
    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Sets the conflict anchors.
    pub fn anchors(mut self, anchors: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.anchors = Some(anchors.into_iter().map(Into::into).collect());
        self
    }
}

/// How an insert was reconciled with existing facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// Inserted as the newest fact (possibly truncating its predecessor).
    Inserted,
    /// Inserted into the past and bounded by the next known fact.
    Backfilled,
    /// Already known; nothing was written and `id` is the existing edge.
    Duplicate,
    /// Same relation and instant as `canonical` but a distinct fact.
    Alias {
        /// The pre-existing edge it aliases.
        canonical: HyperedgeId,
    },
}

/// Result of an insert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertOutcome {
    /// The inserted (or, for duplicates, existing) hyperedge.
    pub id: HyperedgeId,
    /// How conflicts were resolved.
    pub resolution: Resolution,
    /// Existing hyperedges whose validity was truncated.
    pub truncated: Vec<HyperedgeId>,
}

/// A hyperedge with its decayed relevance score.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredHyperedge {
    /// The hyperedge.
    pub edge: TemporalHyperedge,
    /// `confidence * decay(now - valid_from)`.
    pub score: f32,
}

enum WriteOp {
    Insert(NewHyperedge),
    DeleteHyperedge(HyperedgeId),
    DeleteNode(NodeId),
}

enum WriteOutput {
    Inserted(InsertOutcome),
    Deleted,
}

struct WriteRequest {
    op: WriteOp,
    ack: Sender<Result<WriteOutput, EngineError>>,
}

struct WriterState {
    arena: ArenaWriter,
    wal: Option<WalWriter>,
    next_seq: u64,
}

enum IndexDelta {
    NewNode { label_hash: u64, id: NodeId },
    RemovedNode { label_hash: u64, id: NodeId },
    NewEdge { valid_from: i64, id: HyperedgeId },
    RemovedEdge { valid_from: i64, id: HyperedgeId },
}

struct BatchContext {
    first_seq: u64,
    frames: u32,
    deltas: Vec<IndexDelta>,
    batch_nodes: HashMap<u64, NodeId>,
    payload: Vec<u8>,
}

/// Concurrent temporal hypergraph engine (see module docs).
pub struct ConcurrentHypergraphEngine {
    /// `stable_hash(label) -> NodeId`.
    pub node_index: DashMap<u64, NodeId>,
    /// `(valid_from, slot index) -> HyperedgeId`, ordered chronologically.
    pub temporal_index: SkipMap<(i64, u32), HyperedgeId>,
    /// Read-optimized snapshot of the bipartite arena.
    pub graph_snapshot: ArcSwap<BipartiteArena>,
    /// Live node count of the published snapshot.
    pub active_nodes_count: AtomicU32,
    /// Live hyperedge count of the published snapshot.
    pub active_hyperedges_count: AtomicU32,
    labels: DashMap<u64, Arc<str>>,
    event_types: DashMap<u64, Arc<str>>,
    config: EngineConfig,
    storage_dir: Option<PathBuf>,
    queue_tx: Sender<WriteRequest>,
    queue_rx: Receiver<WriteRequest>,
    writer: Mutex<WriterState>,
}

impl std::fmt::Debug for ConcurrentHypergraphEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConcurrentHypergraphEngine")
            .field("storage_dir", &self.storage_dir)
            .field("nodes", &self.active_nodes_count.load(Ordering::Relaxed))
            .field(
                "hyperedges",
                &self.active_hyperedges_count.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

fn invalid(msg: impl Into<String>) -> EngineError {
    EngineError::InvalidArgument(msg.into())
}

impl ConcurrentHypergraphEngine {
    /// Creates or restores an engine from `storage_path` with default config.
    pub fn open(storage_path: &str) -> Result<Self, EngineError> {
        Self::open_with_config(storage_path, EngineConfig::default())
    }

    /// Creates or restores an engine from `storage_path`.
    ///
    /// Recovery loads the newest checkpoint, replays committed WAL batches
    /// after it, truncates corrupt or uncommitted tails and discards any
    /// segment that follows a corruption.
    pub fn open_with_config(
        storage_path: impl AsRef<Path>,
        config: EngineConfig,
    ) -> Result<Self, EngineError> {
        let dir = storage_path.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "tmp") {
                fs::remove_file(&path)?;
            }
        }

        let mut labels = DashMap::new();
        let mut event_types = DashMap::new();
        let (mut arena, base_seq) = match checkpoint::load_latest(&dir)? {
            Some(loaded) => {
                labels.extend(loaded.labels);
                event_types.extend(loaded.event_types);
                (loaded.writer, loaded.last_seq)
            }
            None => (ArenaWriter::new(), 0),
        };

        let mut last_applied = base_seq;
        let mut active: Option<(PathBuf, u64, u64, Option<u64>)> = None;
        let mut corrupted = false;
        for (start_seq, path) in wal::list_segments(&dir)? {
            // A corruption confined to frames a checkpoint already covers
            // leaves later segments contiguous; anything else would replay
            // past a gap of lost committed frames.
            if corrupted && start_seq != last_applied + 1 {
                warn!(segment = %path.display(), "discarding WAL segment that follows a corruption");
                fs::remove_file(&path)?;
                continue;
            }
            let replay = wal::replay_segment(&path, None, |batch: CommittedBatch<'_>| {
                if batch.commit_seq <= base_seq {
                    return Ok(());
                }
                let first = batch
                    .mutations
                    .first()
                    .map_or(batch.commit_seq, |(seq, _)| *seq);
                if first != last_applied + 1 {
                    return Err(EngineError::Recovery(format!(
                        "WAL sequence gap: expected {}, found {first}",
                        last_applied + 1
                    )));
                }
                for (seq, mutation) in batch.mutations {
                    arena
                        .apply(mutation)
                        .map_err(|e| EngineError::Recovery(format!("frame {seq}: {e}")))?;
                    match mutation {
                        Mutation::InsertNode {
                            label_hash, label, ..
                        } => {
                            labels.insert(*label_hash, label.clone());
                        }
                        Mutation::InsertHyperedge {
                            event_type_hash,
                            event_type,
                            ..
                        } => {
                            event_types.insert(*event_type_hash, event_type.clone());
                        }
                        Mutation::UpdateValidity { .. } | Mutation::SoftDelete(_) => {}
                    }
                }
                last_applied = batch.commit_seq;
                Ok(())
            })?;
            corrupted |= replay.corruption.is_some();
            active = Some((path, start_seq, replay.committed_len, replay.last_seq));
        }
        arena.rebuild_free_lists();

        let next_seq = last_applied + 1;
        let wal = match active {
            Some((path, start_seq, committed_len, last_seq))
                if last_seq == Some(last_applied)
                    || (last_seq.is_none() && start_seq == next_seq) =>
            {
                WalWriter::open_existing(&path, start_seq, committed_len, config.sync_mode)?
            }
            _ => WalWriter::create(&dir, next_seq, config.sync_mode)?,
        };

        let mut engine = Self::assemble(config, Some(dir), arena, Some(wal), next_seq);
        engine.labels = labels;
        engine.event_types = event_types;
        Ok(engine)
    }

    /// Creates a volatile engine without WAL or checkpoints.
    pub fn in_memory(config: EngineConfig) -> Self {
        Self::assemble(config, None, ArenaWriter::new(), None, 1)
    }

    /// Volatile engine with default configuration, for tests.
    pub fn new_test() -> Self {
        Self::in_memory(EngineConfig::default())
    }

    fn assemble(
        config: EngineConfig,
        storage_dir: Option<PathBuf>,
        arena: ArenaWriter,
        wal: Option<WalWriter>,
        next_seq: u64,
    ) -> Self {
        let (queue_tx, queue_rx) = crossbeam_channel::bounded(config.queue_capacity.max(1));
        let snapshot = arena.arena().clone();
        let node_index = DashMap::new();
        for node in snapshot.iter_nodes() {
            node_index.insert(node.label_hash, node.id);
        }
        let temporal_index = SkipMap::new();
        for edge in snapshot.iter_hyperedges() {
            temporal_index.insert((edge.valid_from, edge.id.index), edge.id);
        }
        Self {
            node_index,
            temporal_index,
            active_nodes_count: AtomicU32::new(snapshot.live_node_count()),
            active_hyperedges_count: AtomicU32::new(snapshot.live_hyperedge_count()),
            graph_snapshot: ArcSwap::from_pointee(snapshot),
            labels: DashMap::new(),
            event_types: DashMap::new(),
            config,
            storage_dir,
            queue_tx,
            queue_rx,
            writer: Mutex::new(WriterState {
                arena,
                wal,
                next_seq,
            }),
        }
    }

    /// Engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Sequence number of the last committed WAL batch (`0` before any write).
    pub fn last_sequence(&self) -> u64 {
        let mut writer = self.writer.lock();
        while self.drain_queue(&mut writer) > 0 {}
        writer.next_seq.saturating_sub(1)
    }

    /// Storage directory, or `None` for in-memory engines.
    pub fn storage_dir(&self) -> Option<&Path> {
        self.storage_dir.as_deref()
    }

    // ----------------------------------------------------------------- writes

    /// Inserts an N-ary hyperedge, durably and atomically with any temporal
    /// truncation it causes.
    pub fn insert(&self, edge: NewHyperedge) -> Result<InsertOutcome, EngineError> {
        match self.submit(WriteOp::Insert(edge))? {
            WriteOutput::Inserted(outcome) => Ok(outcome),
            WriteOutput::Deleted => Err(EngineError::Recovery("unexpected write output".into())),
        }
    }

    /// Inserts an N-ary hyperedge, updating nodes, indexes and WAL atomically.
    pub fn insert_hyperedge(
        &self,
        event_type: &str,
        node_labels: &[&str],
        valid_from: i64,
        valid_to: Option<i64>,
    ) -> Result<HyperedgeId, EngineError> {
        let mut edge = NewHyperedge::new(event_type, node_labels.iter().copied(), valid_from);
        edge.valid_to = valid_to;
        self.insert(edge).map(|o| o.id)
    }

    /// [`Self::insert_hyperedge`] with an embedding.
    pub fn insert_hyperedge_raw(
        &self,
        event_type: &str,
        node_labels: &[&str],
        valid_from: i64,
        valid_to: Option<i64>,
        embedding: &[f32],
    ) -> Result<HyperedgeId, EngineError> {
        let mut edge = NewHyperedge::new(event_type, node_labels.iter().copied(), valid_from)
            .embedding(embedding.to_vec());
        edge.valid_to = valid_to;
        self.insert(edge).map(|o| o.id)
    }

    /// Inserts many hyperedges with one WAL commit per `max_batch` edges.
    /// Results are returned in input order.
    pub fn insert_batch(
        &self,
        edges: Vec<NewHyperedge>,
    ) -> Vec<Result<InsertOutcome, EngineError>> {
        let ops: Vec<WriteOp> = edges.into_iter().map(WriteOp::Insert).collect();
        let mut writer = self.writer.lock();
        while self.drain_queue(&mut writer) > 0 {}
        let mut results = Vec::with_capacity(ops.len());
        for chunk in ops.chunks(self.config.max_batch.max(1)) {
            let refs: Vec<&WriteOp> = chunk.iter().collect();
            results.extend(self.process_ops(&mut writer, &refs).into_iter().map(|r| {
                r.and_then(|out| match out {
                    WriteOutput::Inserted(outcome) => Ok(outcome),
                    WriteOutput::Deleted => {
                        Err(EngineError::Recovery("unexpected write output".into()))
                    }
                })
            }));
        }
        results
    }

    /// Removes a hyperedge from the live graph. The WAL keeps its history.
    pub fn soft_delete_hyperedge(&self, id: HyperedgeId) -> Result<(), EngineError> {
        self.submit(WriteOp::DeleteHyperedge(id)).map(|_| ())
    }

    /// Removes a node that has no incident hyperedges.
    pub fn soft_delete_node(&self, id: NodeId) -> Result<(), EngineError> {
        self.submit(WriteOp::DeleteNode(id)).map(|_| ())
    }

    /// Writes a checkpoint, starts a fresh WAL segment and deletes the
    /// segments and checkpoints it supersedes. No-op for in-memory engines.
    pub fn checkpoint(&self) -> Result<(), EngineError> {
        let mut writer = self.writer.lock();
        while self.drain_queue(&mut writer) > 0 {}
        let Some(dir) = self.storage_dir.as_deref() else {
            return Ok(());
        };
        let last_seq = writer.next_seq - 1;
        checkpoint::write_checkpoint(
            dir,
            last_seq,
            writer.arena.arena(),
            |h| self.labels.get(&h).map(|v| v.clone()),
            |h| self.event_types.get(&h).map(|v| v.clone()),
        )?;
        if writer
            .wal
            .as_ref()
            .map_or(true, |w| w.start_seq() != writer.next_seq)
        {
            writer.wal = Some(WalWriter::create(
                dir,
                writer.next_seq,
                self.config.sync_mode,
            )?);
        }
        for (start_seq, path) in wal::list_segments(dir)? {
            if start_seq < writer.next_seq {
                fs::remove_file(&path)?;
            }
        }
        checkpoint::remove_older_than(dir, last_seq)?;
        Ok(())
    }

    fn submit(&self, op: WriteOp) -> Result<WriteOutput, EngineError> {
        let (ack, done) = crossbeam_channel::bounded(1);
        self.queue_tx
            .send(WriteRequest { op, ack })
            .map_err(|_| EngineError::Closed)?;
        {
            let mut writer = self.writer.lock();
            loop {
                if let Ok(result) = done.try_recv() {
                    return result;
                }
                if self.drain_queue(&mut writer) == 0 {
                    break;
                }
            }
        }
        // A previous lock holder drained our request and acked it before
        // releasing the lock, so this cannot block indefinitely.
        done.recv().map_err(|_| EngineError::Closed)?
    }

    fn drain_queue(&self, writer: &mut WriterState) -> usize {
        let requests: Vec<WriteRequest> = self
            .queue_rx
            .try_iter()
            .take(self.config.max_batch.max(1))
            .collect();
        if requests.is_empty() {
            return 0;
        }
        let ops: Vec<&WriteOp> = requests.iter().map(|r| &r.op).collect();
        let results = self.process_ops(writer, &ops);
        let n = requests.len();
        for (request, result) in requests.into_iter().zip(results) {
            let _ = request.ack.send(result);
        }
        n
    }

    fn process_ops(
        &self,
        writer: &mut WriterState,
        ops: &[&WriteOp],
    ) -> Vec<Result<WriteOutput, EngineError>> {
        let mut ctx = BatchContext {
            first_seq: writer.next_seq,
            frames: 0,
            deltas: Vec::new(),
            batch_nodes: HashMap::new(),
            payload: Vec::new(),
        };
        let mut results = Vec::with_capacity(ops.len());
        let mut failure = None;
        for op in ops {
            let frames_before = ctx.frames;
            let result = self.execute(writer, &mut ctx, op);
            if let Err(e) = &result {
                if ctx.frames != frames_before {
                    failure = Some(format!("request failed after partial application: {e}"));
                    break;
                }
            }
            results.push(result);
        }

        if failure.is_none() && ctx.frames > 0 {
            let commit_seq = writer.next_seq;
            let committed = writer
                .wal
                .as_mut()
                .map_or(Ok(()), |wal| wal.commit(commit_seq));
            match committed {
                Ok(()) => writer.next_seq += 1,
                Err(e) => failure = Some(format!("WAL commit failed: {e}")),
            }
        }

        if let Some(reason) = failure {
            warn!(%reason, "rolling back write batch");
            if let Some(wal) = writer.wal.as_mut() {
                if let Err(e) = wal.rollback() {
                    warn!(error = %e, "WAL rollback failed; further writes will be rejected");
                }
            }
            writer.next_seq = ctx.first_seq;
            writer.arena = ArenaWriter::from_arena((**self.graph_snapshot.load()).clone());
            return ops
                .iter()
                .map(|_| Err(EngineError::BatchAborted(reason.clone())))
                .collect();
        }

        if ctx.frames > 0 {
            self.publish(writer, ctx.deltas);
        }
        results
    }

    fn publish(&self, writer: &WriterState, deltas: Vec<IndexDelta>) {
        let snapshot = writer.arena.arena().clone();
        let (live_nodes, live_edges) =
            (snapshot.live_node_count(), snapshot.live_hyperedge_count());
        self.graph_snapshot.store(Arc::new(snapshot));
        for delta in deltas {
            match delta {
                IndexDelta::NewNode { label_hash, id } => {
                    self.node_index.insert(label_hash, id);
                }
                IndexDelta::RemovedNode { label_hash, id } => {
                    self.node_index
                        .remove_if(&label_hash, |_, current| *current == id);
                }
                IndexDelta::NewEdge { valid_from, id } => {
                    self.temporal_index.insert((valid_from, id.index), id);
                }
                IndexDelta::RemovedEdge { valid_from, id } => {
                    self.temporal_index.remove(&(valid_from, id.index));
                }
            }
        }
        self.active_nodes_count.store(live_nodes, Ordering::Release);
        self.active_hyperedges_count
            .store(live_edges, Ordering::Release);
    }

    fn emit(
        &self,
        writer: &mut WriterState,
        ctx: &mut BatchContext,
        mutation: Mutation,
    ) -> Result<(), EngineError> {
        writer
            .arena
            .apply(&mutation)
            .map_err(|e| EngineError::Recovery(format!("internal invariant violated: {e}")))?;
        if let Some(wal) = writer.wal.as_mut() {
            ctx.payload.clear();
            mutation.encode(&mut ctx.payload);
            wal.append(mutation.op_code(), writer.next_seq, &ctx.payload);
        }
        writer.next_seq += 1;
        ctx.frames += 1;
        Ok(())
    }

    fn execute(
        &self,
        writer: &mut WriterState,
        ctx: &mut BatchContext,
        op: &WriteOp,
    ) -> Result<WriteOutput, EngineError> {
        match op {
            WriteOp::Insert(edge) => self
                .execute_insert(writer, ctx, edge)
                .map(WriteOutput::Inserted),
            WriteOp::DeleteHyperedge(id) => {
                let valid_from = writer
                    .arena
                    .arena()
                    .hyperedge(*id)
                    .ok_or_else(|| EngineError::NotFound(format!("hyperedge {id:?}")))?
                    .valid_from;
                self.emit(
                    writer,
                    ctx,
                    Mutation::SoftDelete(DeleteTarget::Hyperedge(*id)),
                )?;
                ctx.deltas.push(IndexDelta::RemovedEdge {
                    valid_from,
                    id: *id,
                });
                Ok(WriteOutput::Deleted)
            }
            WriteOp::DeleteNode(id) => {
                let node = writer
                    .arena
                    .arena()
                    .node(*id)
                    .ok_or_else(|| EngineError::NotFound(format!("node {id:?}")))?;
                if !node.hyperedges.is_empty() {
                    return Err(invalid("node still has incident hyperedges"));
                }
                let label_hash = node.label_hash;
                self.emit(writer, ctx, Mutation::SoftDelete(DeleteTarget::Node(*id)))?;
                ctx.batch_nodes.remove(&label_hash);
                ctx.deltas.push(IndexDelta::RemovedNode {
                    label_hash,
                    id: *id,
                });
                Ok(WriteOutput::Deleted)
            }
        }
    }

    fn validate(&self, edge: &NewHyperedge) -> Result<(), EngineError> {
        if edge.event_type.is_empty() {
            return Err(invalid("event_type must not be empty"));
        }
        if edge.participants.is_empty() {
            return Err(invalid("a hyperedge needs at least one participant"));
        }
        if edge.participants.iter().any(String::is_empty) {
            return Err(invalid("participant labels must not be empty"));
        }
        if edge.valid_to.is_some_and(|end| end <= edge.valid_from) {
            return Err(invalid("valid_to must be after valid_from"));
        }
        if !(0.0..=1.0).contains(&edge.confidence) {
            return Err(invalid("confidence must be within [0, 1]"));
        }
        if let Some(embedding) = &edge.embedding {
            if embedding.is_empty() || embedding.iter().any(|v| !v.is_finite()) {
                return Err(invalid("embedding must be non-empty and finite"));
            }
            let dim = self.config.embedding_dim;
            if dim != 0 && embedding.len() != dim {
                return Err(EngineError::EmbeddingDimension {
                    expected: dim,
                    got: embedding.len(),
                });
            }
        }
        if let Some(summary) = &edge.summary {
            let max = self.config.max_summary_bytes;
            if summary.len() > max {
                return Err(invalid(format!(
                    "summary is {} bytes, above the {max}-byte limit",
                    summary.len()
                )));
            }
        }
        if let Some(anchors) = &edge.anchors {
            if anchors.is_empty() || anchors.iter().any(|a| !edge.participants.contains(a)) {
                return Err(invalid(
                    "anchors must be a non-empty subset of participants",
                ));
            }
        }
        // Recovery rejects frames above MAX_PAYLOAD_LEN, so an oversized record
        // would be committed and then truncated away with everything after it.
        const RECORD_OVERHEAD: usize = 256;
        let hyperedge_bytes = RECORD_OVERHEAD
            + edge.participants.len() * std::mem::size_of::<NodeId>()
            + edge.event_type.len()
            + edge.embedding.as_ref().map_or(0, |e| e.len() * 4)
            + edge.summary.as_ref().map_or(0, |s| s.len());
        let node_bytes =
            RECORD_OVERHEAD + edge.participants.iter().map(String::len).max().unwrap_or(0);
        let limit = wal::format::MAX_PAYLOAD_LEN as usize;
        if hyperedge_bytes.max(node_bytes) > limit {
            return Err(invalid(format!(
                "hyperedge record is about {} bytes, above the {limit}-byte WAL frame limit",
                hyperedge_bytes.max(node_bytes)
            )));
        }
        Ok(())
    }

    fn resolve_node(
        &self,
        writer: &mut WriterState,
        ctx: &mut BatchContext,
        label: &str,
        label_hash: u64,
        created_at: i64,
    ) -> Result<NodeId, EngineError> {
        let known = ctx
            .batch_nodes
            .get(&label_hash)
            .copied()
            .or_else(|| self.node_index.get(&label_hash).map(|r| *r));
        if let Some(id) = known.filter(|id| writer.arena.arena().node(*id).is_some()) {
            return Ok(id);
        }
        let id = writer.arena.reserve_node_id()?;
        let label: Arc<str> = Arc::from(label);
        self.labels
            .entry(label_hash)
            .or_insert_with(|| label.clone());
        self.emit(
            writer,
            ctx,
            Mutation::InsertNode {
                id,
                label_hash,
                created_at,
                label,
            },
        )?;
        ctx.batch_nodes.insert(label_hash, id);
        ctx.deltas.push(IndexDelta::NewNode { label_hash, id });
        Ok(id)
    }

    fn conflict_commands(
        arena: &BipartiteArena,
        event_type_hash: u64,
        participants: &[u64],
        scope_nodes: &[u64],
        anchored: bool,
        edge: &NewHyperedge,
    ) -> Vec<WalMutationCommand> {
        let Some(pivot) = scope_nodes
            .iter()
            .filter_map(|raw| arena.node(NodeId::from_u64(*raw)))
            .min_by_key(|node| node.hyperedges.len())
        else {
            return Vec::new();
        };
        let candidates: Vec<(&TemporalHyperedge, Vec<u64>)> = pivot
            .hyperedges
            .iter()
            .filter_map(|id| arena.hyperedge(*id))
            .filter(|e| e.event_type_hash == event_type_hash)
            .map(|e| {
                let mut ps: Vec<u64> = e.participants.iter().map(|p| p.to_u64()).collect();
                ps.sort_unstable();
                (e, ps)
            })
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }
        let views: Vec<EdgeView<'_>> = candidates
            .iter()
            .map(|(e, ps)| EdgeView {
                handle: e.id.to_u64(),
                event_type_hash,
                participants: ps,
                valid_from: e.valid_from,
                valid_to: e.valid_to,
                embedding: arena.embedding(e),
            })
            .collect();
        let new_view = EdgeView {
            handle: u64::MAX,
            event_type_hash,
            participants,
            valid_from: edge.valid_from,
            valid_to: edge.valid_to,
            embedding: edge.embedding.as_deref(),
        };
        let scope = if anchored {
            ConflictScope::Anchors(scope_nodes)
        } else {
            ConflictScope::ExactParticipants
        };
        resolve_conflicts_scoped(&new_view, &views, scope)
    }

    fn execute_insert(
        &self,
        writer: &mut WriterState,
        ctx: &mut BatchContext,
        edge: &NewHyperedge,
    ) -> Result<InsertOutcome, EngineError> {
        self.validate(edge)?;
        let event_type_hash = stable_hash(&edge.event_type);

        let mut hashes: SmallVec<[u64; 8]> = SmallVec::new();
        let mut participant_ids: Vec<NodeId> = Vec::with_capacity(edge.participants.len());
        for label in &edge.participants {
            let label_hash = stable_hash(label);
            if hashes.contains(&label_hash) {
                continue;
            }
            hashes.push(label_hash);
            participant_ids.push(self.resolve_node(
                writer,
                ctx,
                label,
                label_hash,
                edge.valid_from,
            )?);
        }

        let mut sorted_participants: Vec<u64> =
            participant_ids.iter().map(|p| p.to_u64()).collect();
        sorted_participants.sort_unstable();
        let anchor_nodes: Option<Vec<u64>> = edge.anchors.as_ref().map(|anchors| {
            let mut ids: Vec<u64> = anchors
                .iter()
                .filter_map(|a| {
                    let h = stable_hash(a);
                    hashes
                        .iter()
                        .position(|x| *x == h)
                        .map(|i| participant_ids[i].to_u64())
                })
                .collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        });
        let commands = Self::conflict_commands(
            writer.arena.arena(),
            event_type_hash,
            &sorted_participants,
            anchor_nodes.as_deref().unwrap_or(&sorted_participants),
            anchor_nodes.is_some(),
            edge,
        );

        let mut valid_to = edge.valid_to;
        let mut resolution = Resolution::Inserted;
        let mut truncated = Vec::new();
        for command in commands {
            match command {
                WalMutationCommand::DropDuplicate { duplicate_of } => {
                    return Ok(InsertOutcome {
                        id: HyperedgeId::from_u64(duplicate_of),
                        resolution: Resolution::Duplicate,
                        truncated: Vec::new(),
                    });
                }
                WalMutationCommand::MergeAlias {
                    canonical,
                    valid_to: end,
                } => {
                    valid_to = end;
                    resolution = Resolution::Alias {
                        canonical: HyperedgeId::from_u64(canonical),
                    };
                }
                WalMutationCommand::BoundNew { valid_to: end } => {
                    valid_to = Some(valid_to.map_or(end, |v| v.min(end)));
                    resolution = Resolution::Backfilled;
                }
                WalMutationCommand::TruncateExisting {
                    handle,
                    valid_to: end,
                } => {
                    truncated.push((HyperedgeId::from_u64(handle), end));
                }
            }
        }

        for &(id, end) in &truncated {
            self.emit(
                writer,
                ctx,
                Mutation::UpdateValidity {
                    id,
                    valid_to: Some(end),
                },
            )?;
        }
        let event_type = self
            .event_types
            .entry(event_type_hash)
            .or_insert_with(|| Arc::from(edge.event_type.as_str()))
            .clone();
        let id = writer.arena.reserve_hyperedge_id()?;
        self.emit(
            writer,
            ctx,
            Mutation::InsertHyperedge {
                id,
                event_type_hash,
                event_type,
                participants: participant_ids,
                valid_from: edge.valid_from,
                valid_to,
                decay_weight: edge.confidence,
                embedding: edge.embedding.as_deref().map(Arc::from),
                summary: edge.summary.as_deref().map(Arc::from),
            },
        )?;
        ctx.deltas.push(IndexDelta::NewEdge {
            valid_from: edge.valid_from,
            id,
        });
        Ok(InsertOutcome {
            id,
            resolution,
            truncated: truncated.into_iter().map(|(id, _)| id).collect(),
        })
    }

    // ------------------------------------------------------------------ reads

    /// Current read snapshot.
    pub fn snapshot(&self) -> Arc<BipartiteArena> {
        self.graph_snapshot.load_full()
    }

    /// Node id for a label.
    pub fn node_id(&self, label: &str) -> Option<NodeId> {
        self.node_index.get(&stable_hash(label)).map(|r| *r)
    }

    /// Label of a node.
    pub fn node_label(&self, id: NodeId) -> Option<Arc<str>> {
        let hash = self.graph_snapshot.load().node(id)?.label_hash;
        self.labels.get(&hash).map(|v| v.clone())
    }

    /// Event type string of a hyperedge.
    pub fn event_type(&self, edge: &TemporalHyperedge) -> Option<Arc<str>> {
        self.event_types
            .get(&edge.event_type_hash)
            .map(|v| v.clone())
    }

    /// Live hyperedge by id.
    pub fn hyperedge(&self, id: HyperedgeId) -> Option<TemporalHyperedge> {
        self.graph_snapshot.load().hyperedge(id).cloned()
    }

    /// Embedding of a hyperedge.
    pub fn embedding(&self, id: HyperedgeId) -> Option<Arc<[f32]>> {
        let snapshot = self.graph_snapshot.load();
        let edge = snapshot.hyperedge(id)?;
        snapshot.embedding_arc(edge).cloned()
    }

    /// Hyperedges whose `valid_from` lies in `[from, to)`, chronologically.
    pub fn hyperedges_in_range(&self, from: i64, to: i64) -> Vec<HyperedgeId> {
        self.temporal_index
            .range((from, 0)..(to, 0))
            .map(|entry| *entry.value())
            .collect()
    }

    /// Executes a k-hop temporal traversal starting from seed nodes.
    pub fn traverse_bipartite(
        &self,
        start_nodes: &[NodeId],
        k_hops: usize,
        as_of_timestamp: i64,
    ) -> Vec<TemporalHyperedge> {
        self.traverse_bipartite_limited(start_nodes, k_hops, as_of_timestamp, usize::MAX)
    }

    /// [`Self::traverse_bipartite`] returning at most `limit` hyperedges.
    pub fn traverse_bipartite_limited(
        &self,
        start_nodes: &[NodeId],
        k_hops: usize,
        as_of_timestamp: i64,
        limit: usize,
    ) -> Vec<TemporalHyperedge> {
        self.graph_snapshot
            .load()
            .traverse(start_nodes, k_hops, as_of_timestamp, limit)
    }

    /// `confidence * decay(now - valid_from)` for each edge.
    pub fn decay_scores(&self, edges: &[TemporalHyperedge], now: i64) -> Vec<f32> {
        let timestamps: Vec<i64> = edges.iter().map(|e| e.valid_from).collect();
        let mut scores = vec![0.0; edges.len()];
        batch_compute_decay(now, &timestamps, &self.config.decay, &mut scores);
        for (score, edge) in scores.iter_mut().zip(edges) {
            *score *= edge.decay_weight;
        }
        scores
    }

    /// Traversal results ranked by decayed relevance, highest first.
    pub fn traverse_ranked(
        &self,
        start_nodes: &[NodeId],
        k_hops: usize,
        as_of_timestamp: i64,
        now: i64,
    ) -> Vec<ScoredHyperedge> {
        let edges = self.traverse_bipartite(start_nodes, k_hops, as_of_timestamp);
        let scores = self.decay_scores(&edges, now);
        let mut ranked: Vec<ScoredHyperedge> = edges
            .into_iter()
            .zip(scores)
            .map(|(edge, score)| ScoredHyperedge { edge, score })
            .collect();
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
        ranked
    }

    /// The `top_k` hyperedges valid at `target_ts` ranked by
    /// `cosine(query, embedding) * confidence * decay(target_ts - valid_from)`,
    /// highest first. Edges without embeddings are skipped. `decay` overrides
    /// the configured curve.
    pub fn query_temporal(
        &self,
        query: &[f32],
        target_ts: i64,
        decay: Option<DecayConfig>,
        top_k: usize,
    ) -> Vec<ScoredHyperedge> {
        if top_k == 0 || query.is_empty() {
            return Vec::new();
        }
        let decay = decay.unwrap_or(self.config.decay);
        let snapshot = self.graph_snapshot.load();
        let mut scored: Vec<(f32, &TemporalHyperedge)> = snapshot
            .iter_hyperedges()
            .filter(|edge| edge.is_valid_at(target_ts))
            .filter_map(|edge| {
                let embedding = snapshot.embedding(edge)?;
                let similarity = cosine_similarity(query, embedding);
                let score = similarity
                    * edge.decay_weight
                    * decay_weight(target_ts, edge.valid_from, &decay);
                Some((score, edge))
            })
            .collect();
        let by_score_desc =
            |a: &(f32, &TemporalHyperedge), b: &(f32, &TemporalHyperedge)| b.0.total_cmp(&a.0);
        if scored.len() > top_k {
            scored.select_nth_unstable_by(top_k - 1, by_score_desc);
            scored.truncate(top_k);
        }
        scored.sort_by(by_score_desc);
        scored
            .into_iter()
            .map(|(score, edge)| ScoredHyperedge {
                edge: edge.clone(),
                score,
            })
            .collect()
    }

    /// Text payload of a live hyperedge.
    pub fn summary(&self, id: HyperedgeId) -> Option<Arc<str>> {
        let snapshot = self.graph_snapshot.load();
        let edge = snapshot.hyperedge(id)?;
        snapshot.payload(edge).cloned()
    }

    /// Hyperedges reachable within `k_hops` of the seed entities and valid at
    /// `as_of`, filtered by event type and ordered per [`TraversalOrder`].
    ///
    /// Relevance scores are `confidence * decay(as_of - valid_from)`,
    /// multiplied by `cosine(query, embedding)` when a query is given (edges
    /// without an embedding then score `0`). Unknown seeds are ignored.
    pub fn traverse_query(&self, query: &TraversalQuery<'_>) -> Vec<ScoredHyperedge> {
        if query.top_k == 0 || query.k_hops == 0 {
            return Vec::new();
        }
        let seeds: Vec<NodeId> = query
            .seeds
            .iter()
            .filter_map(|label| self.node_id(label))
            .collect();
        if seeds.is_empty() {
            return Vec::new();
        }
        let wanted: SmallVec<[u64; 8]> = query.event_types.iter().map(|t| stable_hash(t)).collect();
        let decay = query.decay.unwrap_or(self.config.decay);
        let snapshot = self.graph_snapshot.load();
        let mut scored: Vec<ScoredHyperedge> = snapshot
            .traverse(&seeds, query.k_hops, query.as_of, usize::MAX)
            .into_iter()
            .filter(|edge| wanted.is_empty() || wanted.contains(&edge.event_type_hash))
            .map(|edge| {
                let mut score =
                    edge.decay_weight * decay_weight(query.as_of, edge.valid_from, &decay);
                if let Some(vector) = query.query_embedding {
                    score *= snapshot
                        .embedding(&edge)
                        .map_or(0.0, |e| cosine_similarity(vector, e));
                }
                ScoredHyperedge { edge, score }
            })
            .collect();
        let order: fn(&ScoredHyperedge, &ScoredHyperedge) -> std::cmp::Ordering = match query.order
        {
            TraversalOrder::Relevance => |a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then(b.edge.valid_from.cmp(&a.edge.valid_from))
            },
            TraversalOrder::Chronological => |a, b| {
                (b.edge.valid_from, b.edge.id.index).cmp(&(a.edge.valid_from, a.edge.id.index))
            },
        };
        if scored.len() > query.top_k {
            scored.select_nth_unstable_by(query.top_k - 1, order);
            scored.truncate(query.top_k);
        }
        scored.sort_by(order);
        scored
    }
}

/// Result ordering of [`ConcurrentHypergraphEngine::traverse_query`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraversalOrder {
    /// Highest relevance score first.
    #[default]
    Relevance,
    /// Newest `valid_from` first.
    Chronological,
}

/// Parameters of [`ConcurrentHypergraphEngine::traverse_query`].
#[derive(Debug, Clone)]
pub struct TraversalQuery<'a> {
    /// Entity labels to start from.
    pub seeds: &'a [String],
    /// Node -> hyperedge -> node expansions (`1` = edges touching a seed).
    pub k_hops: usize,
    /// Only hyperedges valid at this instant are returned or expanded.
    pub as_of: i64,
    /// Optional query embedding for similarity re-ranking.
    pub query_embedding: Option<&'a [f32]>,
    /// Decay curve override.
    pub decay: Option<DecayConfig>,
    /// Maximum results (`usize::MAX` for all).
    pub top_k: usize,
    /// Result ordering.
    pub order: TraversalOrder,
    /// Event types to keep; empty keeps all.
    pub event_types: &'a [String],
}

impl<'a> TraversalQuery<'a> {
    /// A relevance-ordered query over every event type with the configured
    /// decay and no result limit.
    pub fn new(seeds: &'a [String], k_hops: usize, as_of: i64) -> Self {
        Self {
            seeds,
            k_hops,
            as_of,
            query_embedding: None,
            decay: None,
            top_k: usize::MAX,
            order: TraversalOrder::Relevance,
            event_types: &[],
        }
    }
}

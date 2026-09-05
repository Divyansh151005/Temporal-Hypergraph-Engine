//! The content-addressed KV block store.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};

use crate::error::KvError;
use crate::hash::{content_hash, BlockHash, PARALLEL_HASH_THRESHOLD};
use crate::header::{record_len, KvBlockHeader, KvBlockLayout, HEADER_LEN};
use crate::index_log::{IndexLog, LogEntry};
use crate::raw::Segment;

const INDEX_LOG_FILE: &str = "index.log";
const SEGMENT_PREFIX: &str = "seg-";
const SEGMENT_SUFFIX: &str = ".v3kv";

/// Tunables for [`KvBlockStore`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvStoreConfig {
    /// Capacity of each segment; records larger than this get a dedicated segment.
    pub segment_bytes: usize,
    /// Flush records and the index log to stable storage on every store.
    pub sync: bool,
    /// Re-hash every payload while recovering (detects torn payloads).
    pub verify_payload_on_open: bool,
}

impl Default for KvStoreConfig {
    fn default() -> Self {
        Self {
            segment_bytes: 1 << 30,
            sync: false,
            verify_payload_on_open: false,
        }
    }
}

/// Result of [`KvBlockStore::store`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreOutcome {
    /// Record bytes appended (header and padding included); `0` when deduplicated.
    pub bytes_written: u64,
    /// The payload was already stored, possibly under another key.
    pub deduplicated: bool,
}

/// Point-in-time counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvStoreStats {
    /// Keys currently resolvable.
    pub blocks: usize,
    /// Distinct payload records.
    pub records: usize,
    /// Segments allocated.
    pub segments: usize,
    /// Record bytes appended since open.
    pub bytes_written: u64,
    /// Stores satisfied by an existing record since open.
    pub dedup_hits: u64,
}

#[derive(Debug, Clone, Copy)]
struct RecordLoc {
    segment: u32,
    offset: usize,
    payload_len: usize,
    layout: KvBlockLayout,
}

/// A stored block borrowed directly from its (memory-mapped) segment.
pub struct KvBlockRef {
    segment: Arc<Segment>,
    loc: RecordLoc,
    content_hash: BlockHash,
}

impl KvBlockRef {
    /// Payload bytes; 64-byte aligned.
    pub fn bytes(&self) -> &[u8] {
        self.segment
            .read(self.loc.offset + HEADER_LEN, self.loc.payload_len)
    }

    /// Block geometry.
    pub fn layout(&self) -> KvBlockLayout {
        self.loc.layout
    }

    /// BLAKE3 digest of the payload.
    pub fn content_hash(&self) -> &BlockHash {
        &self.content_hash
    }
}

impl std::fmt::Debug for KvBlockRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvBlockRef")
            .field("len", &self.loc.payload_len)
            .field("layout", &self.loc.layout)
            .finish_non_exhaustive()
    }
}

struct WriterState {
    active: Option<u32>,
    cursor: usize,
    log: IndexLog,
}

/// Content-addressed KV block store (see crate docs).
pub struct KvBlockStore {
    dir: Option<PathBuf>,
    config: KvStoreConfig,
    segments: RwLock<Vec<Arc<Segment>>>,
    writer: Mutex<WriterState>,
    keys: DashMap<BlockHash, BlockHash>,
    records: DashMap<BlockHash, RecordLoc>,
    associations: DashMap<BlockHash, Vec<u64>>,
    bytes_written: AtomicU64,
    dedup_hits: AtomicU64,
}

impl std::fmt::Debug for KvBlockStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvBlockStore")
            .field("dir", &self.dir)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

fn segment_path(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("{SEGMENT_PREFIX}{id:06}{SEGMENT_SUFFIX}"))
}

fn list_segments(dir: &Path) -> Result<Vec<u32>, KvError> {
    let mut ids = Vec::new();
    for entry in fs::read_dir(dir)? {
        let name = entry?.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(id) = name
            .strip_prefix(SEGMENT_PREFIX)
            .and_then(|rest| rest.strip_suffix(SEGMENT_SUFFIX))
            .and_then(|id| id.parse::<u32>().ok())
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    if ids.iter().enumerate().any(|(i, &id)| i as u32 != id) {
        return Err(KvError::Corrupt(format!(
            "segment files are not numbered contiguously from 0: {ids:?}"
        )));
    }
    Ok(ids)
}

impl KvBlockStore {
    fn empty(dir: Option<PathBuf>, config: KvStoreConfig, log: IndexLog) -> Self {
        Self {
            dir,
            config,
            segments: RwLock::new(Vec::new()),
            writer: Mutex::new(WriterState {
                active: None,
                cursor: 0,
                log,
            }),
            keys: DashMap::new(),
            records: DashMap::new(),
            associations: DashMap::new(),
            bytes_written: AtomicU64::new(0),
            dedup_hits: AtomicU64::new(0),
        }
    }

    /// Volatile store backed by heap segments.
    pub fn in_memory(config: KvStoreConfig) -> Self {
        Self::empty(None, config, IndexLog::in_memory())
    }

    /// Opens or creates a store in `dir`, recovering every intact record and
    /// index entry. Scanning a segment stops at its first invalid record.
    pub fn open(dir: impl AsRef<Path>, config: KvStoreConfig) -> Result<Self, KvError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let (log, entries) = IndexLog::open(&dir.join(INDEX_LOG_FILE))?;
        let store = Self::empty(Some(dir.clone()), config, log);

        let mut tail = 0;
        let ids = list_segments(&dir)?;
        for &id in &ids {
            let segment = Segment::open_file(&segment_path(&dir, id))?;
            tail = store.scan_segment(id, &segment);
            store.segments.write().push(Arc::new(segment));
        }
        {
            let mut writer = store.writer.lock();
            writer.active = ids.last().copied();
            writer.cursor = tail;
        }

        for entry in entries {
            match entry {
                LogEntry::Bind { key, content } => {
                    store.keys.insert(key, content);
                }
                LogEntry::Associate { key, hyperedge } => {
                    let mut ids = store.associations.entry(key).or_default();
                    if !ids.contains(&hyperedge) {
                        ids.push(hyperedge);
                    }
                }
                LogEntry::Remove { key } => {
                    store.keys.remove(&key);
                    store.associations.remove(&key);
                }
            }
        }
        store
            .keys
            .retain(|_, content| store.records.contains_key(content));
        store
            .associations
            .retain(|key, _| store.keys.contains_key(key));
        Ok(store)
    }

    /// Indexes the valid records of `segment`; returns the offset after the last one.
    fn scan_segment(&self, id: u32, segment: &Segment) -> usize {
        let mut offset = 0;
        while offset + HEADER_LEN <= segment.len() {
            let Some(header) = KvBlockHeader::parse(segment.read(offset, HEADER_LEN)) else {
                break;
            };
            let payload_len = header.payload_len as usize;
            let len = record_len(payload_len);
            let Ok(layout) = header.layout() else { break };
            if offset + len > segment.len() {
                break;
            }
            if self.config.verify_payload_on_open
                && content_hash(segment.read(offset + HEADER_LEN, payload_len))
                    != header.hash_digest
            {
                tracing::warn!(
                    segment = id,
                    offset,
                    "KV payload digest mismatch; truncating"
                );
                break;
            }
            self.records.insert(
                header.hash_digest,
                RecordLoc {
                    segment: id,
                    offset,
                    payload_len,
                    layout,
                },
            );
            offset += len;
        }
        offset
    }

    /// Store configuration.
    pub fn config(&self) -> &KvStoreConfig {
        &self.config
    }

    /// Storage directory, or `None` for in-memory stores.
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Stores `payload` under `key` and associates it with `hyperedges`
    /// (packed `HyperedgeId`s). Identical payloads are stored once.
    pub fn store(
        &self,
        key: &BlockHash,
        payload: &[u8],
        layout: KvBlockLayout,
        hyperedges: &[u64],
    ) -> Result<StoreOutcome, KvError> {
        layout.validate(payload.len())?;
        let payload_len = u32::try_from(payload.len()).map_err(|_| {
            KvError::InvalidArgument(format!("payload of {} bytes exceeds 4 GiB", payload.len()))
        })?;
        let mut writer = self.writer.lock();
        let (segment_id, segment, offset) = self.reserve(&mut writer, record_len(payload.len()))?;
        // The payload lands past the write cursor (unpublished) while it is
        // hashed; it only becomes a record if the digest is new.
        let write_payload = || segment.write(offset + HEADER_LEN, payload);
        let content = if payload.len() >= PARALLEL_HASH_THRESHOLD {
            rayon::join(|| content_hash(payload), write_payload).0
        } else {
            write_payload();
            content_hash(payload)
        };
        let deduplicated = self.records.contains_key(&content);
        let mut bytes_written = 0;
        if !deduplicated {
            let loc = self.seal_record(
                &mut writer,
                segment_id,
                &segment,
                offset,
                &content,
                layout,
                payload_len,
            )?;
            bytes_written = record_len(payload.len()) as u64;
            self.records.insert(content, loc);
        }

        let mut entries = Vec::with_capacity(1 + hyperedges.len());
        if self.keys.get(key).map(|c| *c) != Some(content) {
            entries.push(LogEntry::Bind { key: *key, content });
        }
        let known = self
            .associations
            .get(key)
            .map(|ids| ids.clone())
            .unwrap_or_default();
        let mut fresh: Vec<u64> = Vec::new();
        for &hyperedge in hyperedges {
            if !known.contains(&hyperedge) && !fresh.contains(&hyperedge) {
                fresh.push(hyperedge);
                entries.push(LogEntry::Associate {
                    key: *key,
                    hyperedge,
                });
            }
        }
        if !entries.is_empty() {
            writer.log.append(&entries, self.config.sync)?;
        }
        self.keys.insert(*key, content);
        if !fresh.is_empty() {
            self.associations.entry(*key).or_default().extend(fresh);
        }
        drop(writer);

        if deduplicated {
            self.dedup_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.bytes_written
                .fetch_add(bytes_written, Ordering::Relaxed);
        }
        Ok(StoreOutcome {
            bytes_written,
            deduplicated,
        })
    }

    /// Ensures the active segment has `len` free bytes at the cursor and
    /// returns `(segment id, segment, offset)` without advancing the cursor.
    fn reserve(
        &self,
        writer: &mut WriterState,
        len: usize,
    ) -> Result<(u32, Arc<Segment>, usize), KvError> {
        let fits = writer
            .active
            .is_some_and(|id| writer.cursor + len <= self.segments.read()[id as usize].len());
        if !fits {
            let id = self.segments.read().len() as u32;
            let size = self.config.segment_bytes.max(len);
            let segment = match &self.dir {
                Some(dir) => Segment::create_file(&segment_path(dir, id), size)?,
                None => Segment::heap(size),
            };
            self.segments.write().push(Arc::new(segment));
            writer.active = Some(id);
            writer.cursor = 0;
        }
        let id = writer.active.expect("active segment was just ensured");
        let segment = self.segments.read()[id as usize].clone();
        Ok((id, segment, writer.cursor))
    }

    /// Seals a payload already written at `offset + HEADER_LEN`: writes the
    /// chain terminator and finally the header, so an interrupted append
    /// never leaves a valid-looking record.
    #[allow(clippy::too_many_arguments)]
    fn seal_record(
        &self,
        writer: &mut WriterState,
        segment_id: u32,
        segment: &Segment,
        offset: usize,
        content: &BlockHash,
        layout: KvBlockLayout,
        payload_len: u32,
    ) -> Result<RecordLoc, KvError> {
        let len = record_len(payload_len as usize);
        let next = offset + len;
        if next + HEADER_LEN <= segment.len() {
            segment.write(next, &[0u8; HEADER_LEN]);
        }
        if self.config.sync {
            segment.flush(offset + HEADER_LEN, len - HEADER_LEN)?;
        }
        segment.write(
            offset,
            KvBlockHeader::new(*content, layout, payload_len).as_bytes(),
        );
        if self.config.sync {
            segment.flush(offset, HEADER_LEN)?;
        }
        writer.cursor = next;
        Ok(RecordLoc {
            segment: segment_id,
            offset,
            payload_len: payload_len as usize,
            layout,
        })
    }

    /// Borrows the block stored under `key`.
    pub fn get(&self, key: &BlockHash) -> Option<KvBlockRef> {
        let content = *self.keys.get(key)?;
        let loc = *self.records.get(&content)?;
        let segment = self.segments.read()[loc.segment as usize].clone();
        Some(KvBlockRef {
            segment,
            loc,
            content_hash: content,
        })
    }

    /// Copies the block stored under `key` into the front of `dst`. Returns
    /// its layout, or `None` on a miss.
    pub fn load_into(
        &self,
        key: &BlockHash,
        dst: &mut [u8],
    ) -> Result<Option<KvBlockLayout>, KvError> {
        let Some(block) = self.get(key) else {
            return Ok(None);
        };
        let bytes = block.bytes();
        if dst.len() < bytes.len() {
            return Err(KvError::DestinationTooSmall {
                needed: bytes.len(),
                got: dst.len(),
            });
        }
        dst[..bytes.len()].copy_from_slice(bytes);
        Ok(Some(block.layout()))
    }

    /// Whether `key` resolves to a stored block.
    pub fn contains(&self, key: &BlockHash) -> bool {
        self.keys.contains_key(key)
    }

    /// Drops `key` and its associations. Record space is not reclaimed.
    pub fn remove(&self, key: &BlockHash) -> Result<bool, KvError> {
        let mut writer = self.writer.lock();
        if !self.keys.contains_key(key) {
            return Ok(false);
        }
        writer
            .log
            .append(&[LogEntry::Remove { key: *key }], self.config.sync)?;
        self.keys.remove(key);
        self.associations.remove(key);
        Ok(true)
    }

    /// Hyperedges associated with `key`.
    pub fn associations(&self, key: &BlockHash) -> Vec<u64> {
        self.associations
            .get(key)
            .map(|ids| ids.clone())
            .unwrap_or_default()
    }

    /// Every key that has at least one hyperedge association.
    pub fn associated_blocks(&self) -> Vec<(BlockHash, Vec<u64>)> {
        self.associations
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    /// Current counters.
    pub fn stats(&self) -> KvStoreStats {
        KvStoreStats {
            blocks: self.keys.len(),
            records: self.records.len(),
            segments: self.segments.read().len(),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            dedup_hits: self.dedup_hits.load(Ordering::Relaxed),
        }
    }
}

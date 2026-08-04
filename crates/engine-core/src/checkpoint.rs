//! Atomic on-disk snapshots of the arena.
//!
//! Layout: a [`CheckpointHeader`], then the generation of every node slot and
//! every hyperedge slot (`u32` each), then one `(op u16, len u32, payload)`
//! record per live node followed by one per live hyperedge, using the WAL
//! payload encodings.

use std::fs::{self, File};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use tracing::warn;

use crate::arena::{ArenaWriter, BipartiteArena, Slot};
use crate::error::EngineError;
use crate::wal::record::Mutation;
use crate::wal::writer::sync_dir;
use crate::wal::OpCode;

/// Checkpoint magic bytes.
pub const CHECKPOINT_MAGIC: [u8; 4] = *b"THGC";
const CHECKPOINT_VERSION: u32 = 2;
const PREFIX: &str = "checkpoint-";
const SUFFIX: &str = ".ckpt";

/// Fixed checkpoint header.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CheckpointHeader {
    /// Always [`CHECKPOINT_MAGIC`].
    pub magic: [u8; 4],
    /// Format version.
    pub version: u32,
    /// Sequence id of the last WAL frame folded into this checkpoint.
    pub last_seq: u64,
    /// Number of node slots.
    pub node_slots: u32,
    /// Number of hyperedge slots.
    pub hyperedge_slots: u32,
    /// Number of records in the body.
    pub record_count: u64,
    /// Body length in bytes.
    pub body_len: u64,
    /// CRC32 of the body.
    pub body_crc: u32,
    /// Reserved, zero.
    pub reserved: u32,
}

const HEADER_LEN: usize = std::mem::size_of::<CheckpointHeader>();

/// File name of the checkpoint covering frames up to `last_seq`.
pub fn checkpoint_file_name(last_seq: u64) -> String {
    format!("{PREFIX}{last_seq:020}{SUFFIX}")
}

/// Lists checkpoints in `dir`, sorted by sequence id.
pub fn list_checkpoints(dir: &Path) -> io::Result<Vec<(u64, PathBuf)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(seq) = name
            .to_str()
            .and_then(|n| n.strip_prefix(PREFIX)?.strip_suffix(SUFFIX)?.parse().ok())
        else {
            continue;
        };
        out.push((seq, entry.path()));
    }
    out.sort_unstable_by_key(|(seq, _)| *seq);
    Ok(out)
}

struct CrcWriter<W> {
    inner: W,
    hasher: crc32fast::Hasher,
    len: u64,
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.len += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn slot_generations<'a, T>(
    slots: &'a crate::paged::PagedVec<Slot<T>>,
    generation_of: impl Fn(&T) -> u32 + 'a,
) -> impl Iterator<Item = u32> + 'a {
    slots.iter().map(move |slot| match slot {
        Slot::Occupied(value) => generation_of(value),
        Slot::Vacant { generation } => *generation,
    })
}

/// Writes a checkpoint of `arena` atomically (temp file, fsync, rename,
/// directory fsync) and returns its path.
pub fn write_checkpoint(
    dir: &Path,
    last_seq: u64,
    arena: &BipartiteArena,
    label_of: impl Fn(u64) -> Option<Arc<str>>,
    event_type_of: impl Fn(u64) -> Option<Arc<str>>,
) -> Result<PathBuf, EngineError> {
    let final_path = dir.join(checkpoint_file_name(last_seq));
    let tmp_path = dir.join(format!("{}.tmp", checkpoint_file_name(last_seq)));
    let mut file = File::create(&tmp_path)?;
    file.write_all(&[0u8; HEADER_LEN])?;

    let mut body = CrcWriter {
        inner: BufWriter::with_capacity(1 << 20, file.try_clone()?),
        hasher: crc32fast::Hasher::new(),
        len: 0,
    };
    for generation in slot_generations(&arena.nodes, |n| n.id.generation) {
        body.write_all(&generation.to_le_bytes())?;
    }
    for generation in slot_generations(&arena.hyperedges, |e| e.id.generation) {
        body.write_all(&generation.to_le_bytes())?;
    }

    let mut record_count = 0u64;
    let mut payload = Vec::new();
    let mut write_record = |body: &mut CrcWriter<_>, mutation: Mutation| -> io::Result<()> {
        payload.clear();
        mutation.encode(&mut payload);
        body.write_all(&(mutation.op_code() as u16).to_le_bytes())?;
        body.write_all(&(payload.len() as u32).to_le_bytes())?;
        body.write_all(&payload)?;
        record_count += 1;
        Ok(())
    };
    for node in arena.iter_nodes() {
        let label = label_of(node.label_hash).ok_or_else(|| {
            EngineError::Recovery(format!("no label recorded for node {:?}", node.id))
        })?;
        write_record(
            &mut body,
            Mutation::InsertNode {
                id: node.id,
                label_hash: node.label_hash,
                created_at: node.created_at,
                label,
            },
        )?;
    }
    for edge in arena.iter_hyperedges() {
        let event_type = event_type_of(edge.event_type_hash).ok_or_else(|| {
            EngineError::Recovery(format!(
                "no event type recorded for hyperedge {:?}",
                edge.id
            ))
        })?;
        write_record(
            &mut body,
            Mutation::InsertHyperedge {
                id: edge.id,
                event_type_hash: edge.event_type_hash,
                event_type,
                participants: edge.participants.to_vec(),
                valid_from: edge.valid_from,
                valid_to: edge.valid_to,
                decay_weight: edge.decay_weight,
                embedding: arena.embedding_arc(edge).cloned(),
                summary: arena.payload(edge).cloned(),
            },
        )?;
    }
    body.flush()?;
    let CrcWriter { inner, hasher, len } = body;
    drop(inner.into_inner().map_err(|e| e.into_error())?);

    let header = CheckpointHeader {
        magic: CHECKPOINT_MAGIC,
        version: CHECKPOINT_VERSION,
        last_seq,
        node_slots: arena.nodes.len() as u32,
        hyperedge_slots: arena.hyperedges.len() as u32,
        record_count,
        body_len: len,
        body_crc: hasher.finalize(),
        reserved: 0,
    };
    file.seek(SeekFrom::Start(0))?;
    file.write_all(bytemuck::bytes_of(&header))?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp_path, &final_path)?;
    sync_dir(dir)?;
    Ok(final_path)
}

/// A checkpoint loaded into a fresh arena.
#[derive(Debug)]
pub struct LoadedCheckpoint {
    /// Sequence id of the last WAL frame folded into the checkpoint.
    pub last_seq: u64,
    /// Restored arena with rebuilt free lists.
    pub writer: ArenaWriter,
    /// `(label_hash, label)` pairs.
    pub labels: Vec<(u64, Arc<str>)>,
    /// `(event_type_hash, event_type)` pairs.
    pub event_types: Vec<(u64, Arc<str>)>,
}

fn corrupt(path: &Path, reason: impl std::fmt::Display) -> EngineError {
    EngineError::CheckpointCorrupt(format!("{}: {reason}", path.display()))
}

fn read_u32s(bytes: &[u8], count: usize) -> Option<(Vec<u32>, &[u8])> {
    let len = count.checked_mul(4)?;
    (bytes.len() >= len).then(|| {
        let (head, tail) = bytes.split_at(len);
        let values = head
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (values, tail)
    })
}

/// Loads and validates one checkpoint file.
pub fn load_checkpoint(path: &Path) -> Result<LoadedCheckpoint, EngineError> {
    let bytes = fs::read(path)?;
    if bytes.len() < HEADER_LEN {
        return Err(corrupt(path, "file shorter than header"));
    }
    let header: CheckpointHeader = bytemuck::pod_read_unaligned(&bytes[..HEADER_LEN]);
    let (version, body_len, body_crc) = (header.version, header.body_len, header.body_crc);
    if header.magic != CHECKPOINT_MAGIC {
        return Err(corrupt(path, "bad magic"));
    }
    if version != CHECKPOINT_VERSION {
        return Err(corrupt(
            path,
            format!("unsupported version {version} (expected {CHECKPOINT_VERSION})"),
        ));
    }
    let body = &bytes[HEADER_LEN..];
    if body.len() as u64 != body_len || crc32fast::hash(body) != body_crc {
        return Err(corrupt(path, "body length or checksum mismatch"));
    }

    let (node_generations, rest) = read_u32s(body, header.node_slots as usize)
        .ok_or_else(|| corrupt(path, "short node table"))?;
    let (edge_generations, mut rest) = read_u32s(rest, header.hyperedge_slots as usize)
        .ok_or_else(|| corrupt(path, "short hyperedge table"))?;

    let mut writer = ArenaWriter::new();
    let mut labels = Vec::new();
    let mut event_types = Vec::new();
    let mut records = 0u64;
    while !rest.is_empty() {
        if rest.len() < 6 {
            return Err(corrupt(path, "truncated record header"));
        }
        let op = u16::from_le_bytes([rest[0], rest[1]]);
        let len = u32::from_le_bytes([rest[2], rest[3], rest[4], rest[5]]) as usize;
        rest = &rest[6..];
        if rest.len() < len {
            return Err(corrupt(path, "truncated record"));
        }
        let (payload, tail) = rest.split_at(len);
        rest = tail;
        let op = OpCode::from_u16(op).ok_or_else(|| corrupt(path, "unknown record op"))?;
        let mutation = Mutation::decode(op, payload).map_err(|e| corrupt(path, e))?;
        match &mutation {
            Mutation::InsertNode {
                label_hash, label, ..
            } => labels.push((*label_hash, label.clone())),
            Mutation::InsertHyperedge {
                event_type_hash,
                event_type,
                ..
            } => event_types.push((*event_type_hash, event_type.clone())),
            Mutation::UpdateValidity { .. } | Mutation::SoftDelete(_) => {
                return Err(corrupt(path, "unexpected record kind"))
            }
        }
        writer.apply(&mutation).map_err(|e| corrupt(path, e))?;
        records += 1;
    }
    if records != header.record_count {
        return Err(corrupt(path, "record count mismatch"));
    }
    writer.restore_generations(&node_generations, &edge_generations);
    writer.rebuild_free_lists();
    Ok(LoadedCheckpoint {
        last_seq: header.last_seq,
        writer,
        labels,
        event_types,
    })
}

/// Loads the newest checkpoint in `dir`, if any.
///
/// Older checkpoints are not used as a fallback: the WAL they would need has
/// already been discarded, so silently loading one would lose data.
pub fn load_latest(dir: &Path) -> Result<Option<LoadedCheckpoint>, EngineError> {
    match list_checkpoints(dir)?.pop() {
        Some((_, path)) => load_checkpoint(&path).map(Some),
        None => Ok(None),
    }
}

/// Removes checkpoints older than `keep_seq`.
pub fn remove_older_than(dir: &Path, keep_seq: u64) -> io::Result<()> {
    for (seq, path) in list_checkpoints(dir)? {
        if seq < keep_seq {
            if let Err(e) = fs::remove_file(&path) {
                warn!(path = %path.display(), error = %e, "failed to remove stale checkpoint");
            }
        }
    }
    Ok(())
}

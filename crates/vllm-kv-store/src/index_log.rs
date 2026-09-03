//! Append-only, CRC-checked log of key bindings and hyperedge associations.
//!
//! Each entry is 72 bytes: `tag: u32 | crc: u32 | key: [u8; 32] | value: [u8; 32]`,
//! with the CRC covering tag, key and value. Replay stops at the first
//! damaged entry and truncates the file there.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use crate::error::KvError;
use crate::hash::BlockHash;

const ENTRY_LEN: usize = 72;
const TAG_BIND: u32 = 1;
const TAG_ASSOCIATE: u32 = 2;
const TAG_REMOVE: u32 = 3;

/// A logged index mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogEntry {
    /// `key` now resolves to the record with digest `content`.
    Bind { key: BlockHash, content: BlockHash },
    /// `key` is associated with a hyperedge (packed `HyperedgeId`).
    Associate { key: BlockHash, hyperedge: u64 },
    /// `key` and its associations were dropped.
    Remove { key: BlockHash },
}

impl LogEntry {
    fn encode(&self, out: &mut Vec<u8>) {
        let (tag, key, value) = match *self {
            Self::Bind { key, content } => (TAG_BIND, key, content),
            Self::Associate { key, hyperedge } => {
                let mut value = [0u8; 32];
                value[..8].copy_from_slice(&hyperedge.to_le_bytes());
                (TAG_ASSOCIATE, key, value)
            }
            Self::Remove { key } => (TAG_REMOVE, key, [0u8; 32]),
        };
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&tag.to_le_bytes());
        hasher.update(&key);
        hasher.update(&value);
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&hasher.finalize().to_le_bytes());
        out.extend_from_slice(&key);
        out.extend_from_slice(&value);
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let tag = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let crc = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let key: BlockHash = bytes[8..40].try_into().ok()?;
        let value: BlockHash = bytes[40..72].try_into().ok()?;
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&bytes[0..4]);
        hasher.update(&key);
        hasher.update(&value);
        if hasher.finalize() != crc {
            return None;
        }
        match tag {
            TAG_BIND => Some(Self::Bind {
                key,
                content: value,
            }),
            TAG_ASSOCIATE => Some(Self::Associate {
                key,
                hyperedge: u64::from_le_bytes(value[..8].try_into().ok()?),
            }),
            TAG_REMOVE => Some(Self::Remove { key }),
            _ => None,
        }
    }
}

/// The index log; `file` is `None` for in-memory stores.
pub(crate) struct IndexLog {
    file: Option<File>,
    buf: Vec<u8>,
}

impl IndexLog {
    pub(crate) fn in_memory() -> Self {
        Self {
            file: None,
            buf: Vec::new(),
        }
    }

    /// Opens (or creates) the log and returns its valid entries.
    pub(crate) fn open(path: &Path) -> Result<(Self, Vec<LogEntry>), KvError> {
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut entries = Vec::with_capacity(bytes.len() / ENTRY_LEN);
        let mut valid_len = 0;
        for chunk in bytes.chunks_exact(ENTRY_LEN) {
            let Some(entry) = LogEntry::decode(chunk) else {
                break;
            };
            entries.push(entry);
            valid_len += ENTRY_LEN;
        }
        if valid_len != bytes.len() {
            tracing::warn!(
                path = %path.display(),
                dropped = bytes.len() - valid_len,
                "truncating damaged KV index log tail"
            );
            file.set_len(valid_len as u64)?;
        }
        Ok((
            Self {
                file: Some(file),
                buf: Vec::new(),
            },
            entries,
        ))
    }

    pub(crate) fn append(&mut self, entries: &[LogEntry], sync: bool) -> Result<(), KvError> {
        let Some(file) = self.file.as_mut() else {
            return Ok(());
        };
        self.buf.clear();
        for entry in entries {
            entry.encode(&mut self.buf);
        }
        file.write_all(&self.buf)?;
        if sync {
            file.sync_data()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_stops_at_damage_and_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.log");
        let entries = [
            LogEntry::Bind {
                key: [1; 32],
                content: [2; 32],
            },
            LogEntry::Associate {
                key: [1; 32],
                hyperedge: 0xDEAD_BEEF_0000_0001,
            },
            LogEntry::Remove { key: [3; 32] },
        ];
        {
            let (mut log, replayed) = IndexLog::open(&path).unwrap();
            assert!(replayed.is_empty());
            log.append(&entries, false).unwrap();
        }
        let (_, replayed) = IndexLog::open(&path).unwrap();
        assert_eq!(replayed, entries);

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[ENTRY_LEN + 20] ^= 0xff;
        bytes.extend_from_slice(&[0u8; 5]);
        std::fs::write(&path, &bytes).unwrap();
        let (_, replayed) = IndexLog::open(&path).unwrap();
        assert_eq!(replayed, entries[..1]);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), ENTRY_LEN as u64);
    }
}

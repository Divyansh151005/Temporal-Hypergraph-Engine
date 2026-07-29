//! Append-only, group-committing WAL segment writer.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use super::format::{encode_frame, OpCode};
use super::record::BatchCommitRecord;
use super::segment_file_name;
use crate::config::SyncMode;

/// Writes frames to one segment file. Frames are buffered until
/// [`WalWriter::commit`], which appends a [`OpCode::BatchCommit`] frame and
/// writes the whole batch with a single `write` (+ optional `fdatasync`).
#[derive(Debug)]
pub struct WalWriter {
    file: File,
    path: PathBuf,
    start_seq: u64,
    buf: Vec<u8>,
    committed_len: u64,
    sync_mode: SyncMode,
    pending_frames: u32,
    pending_first_seq: u64,
    poisoned: bool,
}

/// Fsyncs a directory so that file creations and renames inside it are durable.
pub(crate) fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(())
    }
}

impl WalWriter {
    /// Creates segment `wal-{start_seq}.wal` in `dir`. An existing empty file
    /// with that name is reused; a non-empty one is an error.
    pub fn create(dir: &Path, start_seq: u64, sync_mode: SyncMode) -> io::Result<Self> {
        let path = dir.join(segment_file_name(start_seq));
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        if file.metadata()?.len() != 0 {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("WAL segment {} already contains data", path.display()),
            ));
        }
        file.sync_all()?;
        sync_dir(dir)?;
        Ok(Self::with_file(file, path, start_seq, 0, sync_mode))
    }

    /// Re-opens an existing segment for appending after recovery, discarding
    /// anything past `committed_len`.
    pub fn open_existing(
        path: &Path,
        start_seq: u64,
        committed_len: u64,
        sync_mode: SyncMode,
    ) -> io::Result<Self> {
        let file = OpenOptions::new().append(true).open(path)?;
        if file.metadata()?.len() != committed_len {
            file.set_len(committed_len)?;
            file.sync_all()?;
        }
        Ok(Self::with_file(
            file,
            path.to_path_buf(),
            start_seq,
            committed_len,
            sync_mode,
        ))
    }

    fn with_file(
        file: File,
        path: PathBuf,
        start_seq: u64,
        committed_len: u64,
        sync_mode: SyncMode,
    ) -> Self {
        Self {
            file,
            path,
            start_seq,
            buf: Vec::with_capacity(64 * 1024),
            committed_len,
            sync_mode,
            pending_frames: 0,
            pending_first_seq: 0,
            poisoned: false,
        }
    }

    /// Path of the segment file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Sequence id the segment starts at (from its file name).
    pub fn start_seq(&self) -> u64 {
        self.start_seq
    }

    /// Bytes durably committed to the segment.
    pub fn committed_len(&self) -> u64 {
        self.committed_len
    }

    /// Buffers one frame for the current batch.
    pub fn append(&mut self, op: OpCode, sequence_id: u64, payload: &[u8]) {
        if self.pending_frames == 0 {
            self.pending_first_seq = sequence_id;
        }
        encode_frame(&mut self.buf, op, sequence_id, payload);
        self.pending_frames += 1;
    }

    /// Seals the pending frames with a commit frame at `commit_seq`, writes
    /// them and syncs according to the configured [`SyncMode`]. On failure
    /// the segment is truncated back to its last committed length.
    pub fn commit(&mut self, commit_seq: u64) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other(
                "WAL segment is unusable after a failed rollback; reopen the engine",
            ));
        }
        if self.pending_frames == 0 {
            return Ok(());
        }
        let record = BatchCommitRecord {
            first_seq: self.pending_first_seq,
            frame_count: self.pending_frames,
        };
        encode_frame(
            &mut self.buf,
            OpCode::BatchCommit,
            commit_seq,
            bytemuck::bytes_of(&record),
        );

        let result = self
            .file
            .write_all(&self.buf)
            .and_then(|()| match self.sync_mode {
                SyncMode::Fsync => self.file.sync_data(),
                SyncMode::OsBuffered => Ok(()),
            });
        match result {
            Ok(()) => {
                self.committed_len += self.buf.len() as u64;
                self.buf.clear();
                self.pending_frames = 0;
                Ok(())
            }
            Err(e) => {
                let _ = self.rollback();
                Err(e)
            }
        }
    }

    /// Drops the pending batch and truncates any partially written bytes.
    pub fn rollback(&mut self) -> io::Result<()> {
        self.buf.clear();
        self.pending_frames = 0;
        let result = self.file.set_len(self.committed_len);
        if result.is_err() {
            // Torn bytes may remain at the tail; appending after them would
            // hide later batches from recovery.
            self.poisoned = true;
        }
        result
    }
}

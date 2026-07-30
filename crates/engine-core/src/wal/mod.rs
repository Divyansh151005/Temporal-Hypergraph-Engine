//! Write-ahead log: frame format, payload records, writer and recovery.

pub mod format;
pub mod record;
pub mod recovery;
pub mod writer;

use std::io;
use std::path::{Path, PathBuf};

pub use format::{
    Corruption, CorruptionKind, Frame, FrameReader, OpCode, WalFrameHeader, LEGACY_WAL_MAGIC_V1,
    WAL_HEADER_LEN, WAL_MAGIC,
};
pub use record::{DeleteTarget, Mutation};
pub use recovery::{replay_bytes, replay_segment, CommittedBatch, SegmentReplay};
pub use writer::WalWriter;

const SEGMENT_PREFIX: &str = "wal-";
const SEGMENT_SUFFIX: &str = ".wal";

/// File name of the segment whose first frame has sequence id `start_seq`.
pub fn segment_file_name(start_seq: u64) -> String {
    format!("{SEGMENT_PREFIX}{start_seq:020}{SEGMENT_SUFFIX}")
}

/// Parses a segment file name back into its start sequence id.
pub fn parse_segment_name(name: &str) -> Option<u64> {
    name.strip_prefix(SEGMENT_PREFIX)?
        .strip_suffix(SEGMENT_SUFFIX)?
        .parse()
        .ok()
}

/// Lists WAL segments in `dir`, sorted by start sequence id.
pub fn list_segments(dir: &Path) -> io::Result<Vec<(u64, PathBuf)>> {
    let mut segments = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if let Some(seq) = entry.file_name().to_str().and_then(parse_segment_name) {
            segments.push((seq, entry.path()));
        }
    }
    segments.sort_unstable_by_key(|(seq, _)| *seq);
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_names_round_trip_and_sort() {
        let name = segment_file_name(42);
        assert_eq!(name, "wal-00000000000000000042.wal");
        assert_eq!(parse_segment_name(&name), Some(42));
        assert_eq!(parse_segment_name("wal-x.wal"), None);
        assert!(segment_file_name(9) < segment_file_name(10));
    }
}

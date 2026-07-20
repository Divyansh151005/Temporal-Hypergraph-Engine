//! Engine configuration.

use temporal_math::DecayConfig;

/// Durability of a committed write batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// `fdatasync` after every batch: committed writes survive power loss.
    #[default]
    Fsync,
    /// Leave flushing to the OS page cache: committed writes survive a
    /// process crash but not a kernel crash or power loss.
    OsBuffered,
}

/// Tunables for [`crate::ConcurrentHypergraphEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    /// Required embedding length; `0` accepts any length.
    pub embedding_dim: usize,
    /// WAL durability mode.
    pub sync_mode: SyncMode,
    /// Maximum requests folded into one WAL commit.
    pub max_batch: usize,
    /// Capacity of the ingestion queue.
    pub queue_capacity: usize,
    /// Decay curve used for ranking.
    pub decay: DecayConfig,
    /// Largest accepted hyperedge summary, in UTF-8 bytes.
    pub max_summary_bytes: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 0,
            sync_mode: SyncMode::Fsync,
            max_batch: 1024,
            queue_capacity: 16 * 1024,
            decay: DecayConfig::default(),
            max_summary_bytes: 64 * 1024,
        }
    }
}

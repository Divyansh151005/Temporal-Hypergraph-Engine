//! Engine error type.

use std::io;

/// Errors returned by the storage engine.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Underlying filesystem failure.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// The caller supplied an invalid request.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// Embedding length does not match the configured dimension.
    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDimension {
        /// Configured dimension.
        expected: usize,
        /// Supplied dimension.
        got: usize,
    },
    /// A packed `f32` buffer whose byte length is not a multiple of four.
    #[error("invalid alignment: {0} bytes is not a whole number of f32 values")]
    InvalidAlignment(usize),
    /// A referenced node or hyperedge does not exist (or was reused).
    #[error("not found: {0}")]
    NotFound(String),
    /// The arena ran out of addressable slots.
    #[error("arena capacity exceeded")]
    CapacityExceeded,
    /// The batch containing this request could not be committed and was
    /// rolled back; nothing from it is visible or durable.
    #[error("write batch aborted: {0}")]
    BatchAborted(String),
    /// Persistent state is inconsistent and cannot be recovered automatically.
    #[error("recovery failed: {0}")]
    Recovery(String),
    /// A checkpoint file failed validation.
    #[error("checkpoint corrupt: {0}")]
    CheckpointCorrupt(String),
    /// The engine can no longer accept writes.
    #[error("engine closed")]
    Closed,
}

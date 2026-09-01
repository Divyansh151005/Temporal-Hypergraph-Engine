//! KV store error type.

use std::io;

/// Errors returned by the KV block store.
#[derive(Debug, thiserror::Error)]
pub enum KvError {
    /// Underlying filesystem failure.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// The caller supplied an invalid request.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// Payload length disagrees with the declared block layout.
    #[error("payload is {got} bytes but the layout requires {expected}")]
    LayoutMismatch {
        /// Bytes implied by the layout.
        expected: usize,
        /// Bytes supplied.
        got: usize,
    },
    /// The destination buffer cannot hold the stored block.
    #[error("destination buffer holds {got} bytes but the block needs {needed}")]
    DestinationTooSmall {
        /// Stored payload length.
        needed: usize,
        /// Destination capacity.
        got: usize,
    },
    /// Persistent state failed validation.
    #[error("corrupt KV store: {0}")]
    Corrupt(String),
}

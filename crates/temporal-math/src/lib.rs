//! Temporal mathematics for the Temporal Hypergraph Engine.
//!
//! * [`decay`]: SIMD-accelerated exponential time decay, `w(t) = w0 * exp(-lambda * dt)`.
//! * [`fourier`]: sinusoidal (Fourier) time encodings for periodic context.
//! * [`conflict`]: deterministic temporal conflict resolution between hyperedges.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod conflict;
pub mod decay;
pub mod fourier;

pub use conflict::{
    cosine_similarity, resolve_conflicts, resolve_conflicts_scoped, ConflictScope, EdgeView,
    WalMutationCommand, DUPLICATE_COSINE_THRESHOLD,
};
pub use decay::{
    batch_compute_decay, compute_decay_scores, compute_decay_scores_scalar,
    compute_decay_scores_simd, decay_weight, DecayConfig,
};
pub use fourier::{add_time_encoding, fourier_time_encoding, time_encoding, TimeScale};

/// The `temporal` facade named in the core specification.
pub mod temporal {
    pub use crate::conflict::{resolve_conflicts, WalMutationCommand};
    pub use crate::decay::{batch_compute_decay, DecayConfig};
}

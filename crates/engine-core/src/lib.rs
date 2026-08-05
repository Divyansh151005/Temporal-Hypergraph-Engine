//! Core storage engine of the Temporal Hypergraph Engine.
//!
//! The hypergraph is stored as a bipartite graph (entity nodes on one side,
//! N-ary hyperedges on the other) in generational, copy-on-write arenas.
//! Readers traverse immutable snapshots lock-free; writers group-commit
//! through a CRC32-checked write-ahead log with atomic batches.
#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(target_endian = "big")]
compile_error!("engine-core's on-disk format is little-endian only");

pub mod arena;
pub mod checkpoint;
pub mod codec;
pub mod config;
pub mod engine;
pub mod error;
pub mod hash;
pub mod ids;
pub mod model;
pub mod paged;
pub mod payload;
pub mod sync;
pub mod wal;

pub use arena::{ArenaWriter, BipartiteArena, Slot};
pub use config::{EngineConfig, SyncMode};
pub use engine::{
    ConcurrentHypergraphEngine, InsertOutcome, NewHyperedge, Resolution, ScoredHyperedge,
    TraversalOrder, TraversalQuery,
};
pub use error::EngineError;
pub use hash::stable_hash;
pub use ids::{HyperedgeId, NodeId};
pub use model::{TemporalHyperedge, TemporalNode, NO_EMBEDDING, NO_PAYLOAD};
pub use payload::PayloadArena;
pub use temporal_math;

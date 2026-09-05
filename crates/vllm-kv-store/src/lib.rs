//! Secondary storage tier for vLLM PagedAttention KV-cache blocks.
//!
//! Blocks are framed as a 64-byte [`KvBlockHeader`] followed by the raw
//! tensor bytes, appended to 64-byte-aligned slots of memory-mapped segment
//! files. Each payload is content-addressed by its BLAKE3 digest, so blocks
//! stored under different keys but with identical bytes share one record.
//! Caller keys (typically vLLM token-chain hashes) and hyperedge associations
//! are persisted in a CRC-checked index log. Temporal eviction planning ranks
//! blocks by the decayed relevance of their associated hyperedges.
#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod eviction;
pub mod hash;
pub mod header;
mod index_log;
pub mod raw;
pub mod store;

pub use error::KvError;
pub use eviction::plan_kv_evictions;
pub use hash::{content_hash, from_hex, to_hex, token_chain_hash, BlockHash};
pub use header::{KvBlockHeader, KvBlockLayout, Precision, HEADER_LEN, KV_MAGIC, RECORD_ALIGN};
pub use store::{KvBlockRef, KvBlockStore, KvStoreConfig, KvStoreStats, StoreOutcome};

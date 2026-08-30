//! gRPC front-end (`hypergraph.v1.HypergraphEngineService`) for the Temporal
//! Hypergraph Engine and its vLLM KV block tier.
//!
//! All `bytes` fields decode as [`bytes::Bytes`] slices of the receive
//! buffer; embeddings are reinterpreted in place with `bytemuck` when
//! aligned. Blocking engine work runs on Tokio's blocking pool.
#![deny(unsafe_code)]
#![warn(missing_docs)]
// Tonic's generated service trait fixes `tonic::Status` as the error type.
#![allow(clippy::result_large_err)]

pub mod auth;
pub mod convert;
pub mod server;
pub mod service;

/// Generated protobuf and Tonic types.
#[allow(missing_docs, clippy::all, clippy::pedantic)]
pub mod pb {
    tonic::include_proto!("hypergraph.v1");
}

pub use auth::{ApiKeyCheck, API_KEY_HEADER};
pub use server::{serve, serve_with_auth, Listen, MAX_MESSAGE_BYTES};
pub use service::HypergraphService;

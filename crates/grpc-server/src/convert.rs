//! Conversions between protobuf messages and engine / KV store types.

use bytes::Bytes;
use engine_core::codec::{decode_embedding_raw, encode_embedding_raw, format_hyperedge_id};
use engine_core::{ConcurrentHypergraphEngine, EngineError, NewHyperedge, TemporalHyperedge};
use tonic::Status;
use vllm_kv_store::{KvBlockLayout, KvError, Precision};

use crate::pb;

/// Maps an engine error onto a gRPC status.
pub fn engine_status(err: EngineError) -> Status {
    let message = err.to_string();
    match err {
        EngineError::InvalidArgument(_)
        | EngineError::EmbeddingDimension { .. }
        | EngineError::InvalidAlignment(_) => Status::invalid_argument(message),
        EngineError::NotFound(_) => Status::not_found(message),
        EngineError::CapacityExceeded => Status::resource_exhausted(message),
        EngineError::Closed => Status::unavailable(message),
        EngineError::BatchAborted(_) => Status::aborted(message),
        EngineError::Io(_) | EngineError::Recovery(_) | EngineError::CheckpointCorrupt(_) => {
            Status::internal(message)
        }
    }
}

/// Maps a KV store error onto a gRPC status.
pub fn kv_status(err: KvError) -> Status {
    let message = err.to_string();
    match err {
        KvError::InvalidArgument(_)
        | KvError::LayoutMismatch { .. }
        | KvError::DestinationTooSmall { .. } => Status::invalid_argument(message),
        KvError::Io(_) | KvError::Corrupt(_) => Status::internal(message),
    }
}

/// Builds an insertable hyperedge from a payload.
pub fn payload_to_new_hyperedge(payload: &pb::HyperedgePayload) -> Result<NewHyperedge, Status> {
    let boundary = payload
        .temporal_boundary
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("temporal_boundary is required"))?;
    let valid_from = boundary.valid_from_epoch.ok_or_else(|| {
        Status::invalid_argument("temporal_boundary.valid_from_epoch is required")
    })?;
    let mut edge = NewHyperedge::new(
        payload.event_type.clone(),
        payload.participants.iter().map(|p| p.entity_id.clone()),
        valid_from,
    );
    if !boundary.is_ongoing {
        edge.valid_to = boundary.valid_to_epoch;
    }
    if payload.confidence_score != 0.0 {
        edge.confidence = payload.confidence_score;
    }
    if !payload.embedding_raw.is_empty() {
        let embedding = decode_embedding_raw(&payload.embedding_raw).map_err(engine_status)?;
        edge.embedding = Some(embedding.into_owned());
    }
    if !payload.summary.is_empty() {
        edge.summary = Some(payload.summary.clone());
    }
    Ok(edge)
}

/// Renders a stored hyperedge (labels, embedding) as a payload.
pub fn edge_to_payload(
    engine: &ConcurrentHypergraphEngine,
    edge: &TemporalHyperedge,
) -> pb::HyperedgePayload {
    let participants = edge
        .participants
        .iter()
        .map(|&node| pb::Participant {
            entity_id: engine
                .node_label(node)
                .map(|label| label.to_string())
                .unwrap_or_default(),
            role: String::new(),
        })
        .collect();
    let embedding_raw = engine
        .embedding(edge.id)
        .map(|e| Bytes::copy_from_slice(encode_embedding_raw(&e)))
        .unwrap_or_default();
    pb::HyperedgePayload {
        id: format_hyperedge_id(edge.id),
        event_type: engine
            .event_type(edge)
            .map(|t| t.to_string())
            .unwrap_or_default(),
        summary: engine
            .summary(edge.id)
            .map(|s| s.to_string())
            .unwrap_or_default(),
        participants,
        temporal_boundary: Some(pb::TemporalBoundary {
            valid_from_epoch: Some(edge.valid_from),
            valid_to_epoch: edge.valid_to,
            is_ongoing: edge.is_ongoing,
            raw_expression: String::new(),
        }),
        confidence_score: edge.decay_weight,
        embedding_raw,
    }
}

/// Parses a protobuf block layout.
pub fn layout_from_pb(layout: &pb::KvBlockLayout) -> Result<KvBlockLayout, Status> {
    let dim = |value: u32, name: &str| {
        u16::try_from(value)
            .map_err(|_| Status::invalid_argument(format!("layout.{name} {value} exceeds u16")))
    };
    let precision = u16::try_from(layout.precision)
        .ok()
        .and_then(|id| Precision::from_id(id).ok())
        .ok_or_else(|| {
            Status::invalid_argument(format!("unknown precision {}", layout.precision))
        })?;
    Ok(KvBlockLayout::new(
        dim(layout.num_layers, "num_layers")?,
        dim(layout.num_heads, "num_heads")?,
        dim(layout.head_dim, "head_dim")?,
        dim(layout.block_size, "block_size")?,
        precision,
    ))
}

/// Renders a block layout; `None` for opaque payloads.
pub fn layout_to_pb(layout: KvBlockLayout) -> Option<pb::KvBlockLayout> {
    (!layout.is_opaque()).then(|| pb::KvBlockLayout {
        num_layers: layout.num_layers.into(),
        num_heads: layout.num_heads.into(),
        head_dim: layout.head_dim.into(),
        block_size: layout.block_size.into(),
        precision: i32::from(layout.precision.id()),
    })
}

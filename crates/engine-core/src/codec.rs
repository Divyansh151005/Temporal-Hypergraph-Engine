//! Wire codecs shared by the gRPC and PyO3 front-ends.
//!
//! Embeddings travel as packed little-endian `f32` bytes rather than
//! per-element encodings, so decoding is a reinterpretation of the buffer.

use std::borrow::Cow;

use crate::error::EngineError;
use crate::ids::HyperedgeId;

/// Reinterprets packed little-endian `f32` bytes as a float slice.
///
/// Borrows the input when it is 4-byte aligned; otherwise copies it into an
/// aligned vector. Fails with [`EngineError::InvalidAlignment`] when the
/// length is not a multiple of four.
pub fn decode_embedding_raw(raw: &[u8]) -> Result<Cow<'_, [f32]>, EngineError> {
    if raw.len() % std::mem::size_of::<f32>() != 0 {
        return Err(EngineError::InvalidAlignment(raw.len()));
    }
    match bytemuck::try_cast_slice(raw) {
        Ok(floats) => Ok(Cow::Borrowed(floats)),
        Err(_) => Ok(Cow::Owned(
            raw.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        )),
    }
}

/// Views a float slice as packed little-endian bytes.
pub fn encode_embedding_raw(embedding: &[f32]) -> &[u8] {
    bytemuck::cast_slice(embedding)
}

/// Canonical string form of a hyperedge id (decimal of [`HyperedgeId::to_u64`]).
pub fn format_hyperedge_id(id: HyperedgeId) -> String {
    id.to_u64().to_string()
}

/// Inverse of [`format_hyperedge_id`].
pub fn parse_hyperedge_id(text: &str) -> Result<HyperedgeId, EngineError> {
    text.parse::<u64>()
        .map(HyperedgeId::from_u64)
        .map_err(|_| EngineError::InvalidArgument(format!("malformed hyperedge id {text:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_lengths_not_divisible_by_four() {
        assert!(matches!(
            decode_embedding_raw(&[0u8; 7]),
            Err(EngineError::InvalidAlignment(7))
        ));
    }

    #[test]
    fn borrows_aligned_and_copies_unaligned() {
        let floats = [1.5f32, -2.25, 3.0];
        let bytes = encode_embedding_raw(&floats);
        assert!(matches!(
            decode_embedding_raw(bytes).unwrap(),
            Cow::Borrowed(_)
        ));

        let mut shifted = vec![0u8];
        shifted.extend_from_slice(bytes);
        let decoded = decode_embedding_raw(&shifted[1..]).unwrap();
        assert!(matches!(decoded, Cow::Owned(_)));
        assert_eq!(&*decoded, &floats);
    }

    #[test]
    fn hyperedge_id_round_trips() {
        let id = HyperedgeId::new(42, 7);
        assert_eq!(parse_hyperedge_id(&format_hyperedge_id(id)).unwrap(), id);
        assert!(parse_hyperedge_id("nope").is_err());
    }
}

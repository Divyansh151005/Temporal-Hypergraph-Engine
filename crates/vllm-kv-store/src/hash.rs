//! BLAKE3 block addressing.

use crate::error::KvError;

/// A 256-bit BLAKE3 digest.
pub type BlockHash = [u8; 32];

/// Payloads at least this large are hashed on the rayon pool.
pub(crate) const PARALLEL_HASH_THRESHOLD: usize = 1 << 20;

/// BLAKE3 digest of a block payload (its content address).
pub fn content_hash(payload: &[u8]) -> BlockHash {
    let mut hasher = blake3::Hasher::new();
    if payload.len() >= PARALLEL_HASH_THRESHOLD {
        hasher.update_rayon(payload);
    } else {
        hasher.update(payload);
    }
    *hasher.finalize().as_bytes()
}

/// Deterministic vLLM-style block key: `BLAKE3(parent_hash || token_ids)`,
/// where `token_ids` are packed little-endian `i32` values.
pub fn token_chain_hash(parent: Option<&[u8]>, token_ids: &[u8]) -> Result<BlockHash, KvError> {
    if token_ids.len() % 4 != 0 {
        return Err(KvError::InvalidArgument(format!(
            "token buffer of {} bytes is not a whole number of int32 values",
            token_ids.len()
        )));
    }
    let mut hasher = blake3::Hasher::new();
    if let Some(parent) = parent {
        hasher.update(parent);
    }
    hasher.update(token_ids);
    Ok(*hasher.finalize().as_bytes())
}

/// Lowercase hex encoding of a digest.
pub fn to_hex(hash: &BlockHash) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in hash {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    out
}

/// Parses a 64-character hex digest (either case).
pub fn from_hex(text: &str) -> Result<BlockHash, KvError> {
    let bytes = text.as_bytes();
    if bytes.len() != 64 {
        return Err(KvError::InvalidArgument(format!(
            "block hash must be 64 hex characters, got {}",
            bytes.len()
        )));
    }
    let nibble = |c: u8| -> Result<u8, KvError> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(KvError::InvalidArgument(format!(
                "invalid hex character {:?} in block hash",
                c as char
            ))),
        }
    };
    let mut out = [0u8; 32];
    for (i, pair) in bytes.chunks_exact(2).enumerate() {
        out[i] = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips() {
        let hash = content_hash(b"hello");
        assert_eq!(from_hex(&to_hex(&hash)).unwrap(), hash);
        assert_eq!(from_hex(&to_hex(&hash).to_uppercase()).unwrap(), hash);
        assert!(from_hex("zz").is_err());
        assert!(from_hex(&"g".repeat(64)).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore = "hashes 2 MiB; hours under the interpreter")]
    fn parallel_and_serial_hashing_agree() {
        let payload: Vec<u8> = (0..PARALLEL_HASH_THRESHOLD * 2).map(|i| i as u8).collect();
        assert_eq!(content_hash(&payload), *blake3::hash(&payload).as_bytes());
    }

    #[test]
    fn token_chain_depends_on_parent() {
        let tokens: Vec<u8> = [1i32, 2, 3].iter().flat_map(|t| t.to_le_bytes()).collect();
        let root = token_chain_hash(None, &tokens).unwrap();
        let child = token_chain_hash(Some(&root), &tokens).unwrap();
        assert_ne!(root, child);
        assert!(token_chain_hash(None, &[0u8; 3]).is_err());
    }
}

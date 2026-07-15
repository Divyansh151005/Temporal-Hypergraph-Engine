//! Stable string hashing for labels and event types.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// 64-bit FNV-1a followed by a splitmix64 finalizer.
///
/// The output is persisted in the WAL and checkpoints, so this function must
/// never change.
pub fn stable_hash(text: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for byte in text.as_bytes() {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

#[cfg(test)]
mod tests {
    use super::stable_hash;

    #[test]
    fn is_stable_and_discriminating() {
        assert_eq!(stable_hash("OpenAI"), stable_hash("OpenAI"));
        assert_ne!(stable_hash("OpenAI"), stable_hash("openai"));
        assert_ne!(stable_hash("ab"), stable_hash("ba"));
    }
}

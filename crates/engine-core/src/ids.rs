//! Compact generational identifiers for arena slots.

use bytemuck::{Pod, Zeroable};

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[repr(C, packed)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
        pub struct $name {
            /// Slot index inside the arena.
            pub index: u32,
            /// Generation of the slot; bumped every time the slot is reused.
            pub generation: u32,
        }

        impl $name {
            /// Creates an identifier from its parts.
            pub const fn new(index: u32, generation: u32) -> Self {
                Self { index, generation }
            }

            /// Packs the identifier into a single `u64` (`index << 32 | generation`).
            pub const fn to_u64(self) -> u64 {
                ((self.index as u64) << 32) | self.generation as u64
            }

            /// Inverse of [`Self::to_u64`].
            pub const fn from_u64(raw: u64) -> Self {
                Self {
                    index: (raw >> 32) as u32,
                    generation: raw as u32,
                }
            }
        }

        const _: () = assert!(
            std::mem::size_of::<$name>() == 8 && std::mem::align_of::<$name>() == 1
        );
    };
}

define_id!(
    /// Identifier of an atomic entity node (bipartite set A).
    NodeId
);
define_id!(
    /// Identifier of an N-ary hyperedge (bipartite set B).
    HyperedgeId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_round_trip() {
        let id = NodeId::new(0xDEAD_BEEF, 7);
        assert_eq!(NodeId::from_u64(id.to_u64()), id);
        let raw = HyperedgeId::new(3, u32::MAX).to_u64();
        assert_eq!(raw, (3u64 << 32) | u32::MAX as u64);
    }

    #[test]
    fn casts_from_unaligned_bytes() {
        let ids = [NodeId::new(1, 2), NodeId::new(3, 4)];
        let mut bytes = vec![0u8];
        bytes.extend_from_slice(bytemuck::cast_slice(&ids));
        let back: &[NodeId] = bytemuck::cast_slice(&bytes[1..]);
        assert_eq!(back, ids);
    }
}

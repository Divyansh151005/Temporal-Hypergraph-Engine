//! Payload arena: the text summaries attached to hyperedges.
//!
//! Text is kept out of [`crate::TemporalHyperedge`] so the hot struct stays
//! cache-dense; the hyperedge only carries a 4-byte `payload_offset`. Entries
//! are immutable `Arc<str>`s in a copy-on-write [`PagedVec`] indexed by
//! hyperedge slot, so publishing a snapshot never copies text.

use std::sync::Arc;

use crate::model::NO_PAYLOAD;
use crate::paged::PagedVec;

/// Copy-on-write side table of hyperedge text payloads.
#[derive(Debug, Clone, Default)]
pub struct PayloadArena {
    texts: PagedVec<Option<Arc<str>>>,
    live_bytes: u64,
}

impl PayloadArena {
    /// Payload stored at `offset`.
    #[inline]
    pub fn get(&self, offset: u32) -> Option<&Arc<str>> {
        if offset == NO_PAYLOAD {
            return None;
        }
        self.texts.get(offset as usize)?.as_ref()
    }

    /// Total UTF-8 bytes of live payloads.
    pub fn live_bytes(&self) -> u64 {
        self.live_bytes
    }

    /// Stores `text` at `slot` and returns the offset to record on the
    /// hyperedge ([`NO_PAYLOAD`] when `text` is `None`).
    pub(crate) fn put(&mut self, slot: u32, text: Option<Arc<str>>) -> u32 {
        self.clear(slot);
        let Some(text) = text else {
            return NO_PAYLOAD;
        };
        let index = slot as usize;
        self.texts.extend_to(index + 1, None);
        self.live_bytes += text.len() as u64;
        *self.texts.get_mut(index).expect("extended above") = Some(text);
        slot
    }

    /// Drops the payload stored at `slot`, if any.
    pub(crate) fn clear(&mut self, slot: u32) {
        if let Some(entry) = self.texts.get_mut(slot as usize) {
            if let Some(old) = entry.take() {
                self.live_bytes -= old.len() as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_clear_track_live_bytes() {
        let mut arena = PayloadArena::default();
        assert_eq!(arena.put(3, None), NO_PAYLOAD);
        let offset = arena.put(3, Some(Arc::from("hello")));
        assert_eq!(offset, 3);
        assert_eq!(arena.get(offset).map(|s| &**s), Some("hello"));
        assert_eq!(arena.get(NO_PAYLOAD), None);
        assert_eq!(arena.live_bytes(), 5);

        let snapshot = arena.clone();
        arena.clear(3);
        assert_eq!(arena.get(3), None);
        assert_eq!(arena.live_bytes(), 0);
        assert_eq!(snapshot.get(3).map(|s| &**s), Some("hello"));
    }
}

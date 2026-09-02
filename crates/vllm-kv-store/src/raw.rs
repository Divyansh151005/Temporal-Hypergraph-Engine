//! The crate's only `unsafe` code: segment memory backing and raw-pointer
//! entry points for foreign callers (PyO3 / `torch.Tensor.data_ptr()`).
//!
//! Segments are append-only. Bytes are written only past the store's write
//! cursor, under the store's writer lock, and a range becomes readable only
//! after it is published to the index. Readers and the writer therefore
//! never touch the same bytes concurrently.
#![allow(unsafe_code)]

use std::alloc::{self, Layout};
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::ptr::NonNull;

use memmap2::MmapRaw;

use crate::error::KvError;
use crate::hash::BlockHash;
use crate::header::{KvBlockLayout, RECORD_ALIGN};
use crate::store::{KvBlockStore, StoreOutcome};

enum Backing {
    /// File-backed or anonymous mapping (page aligned, lazily zeroed).
    Mapped { map: MmapRaw, _file: Option<File> },
    /// Global-allocator memory; used under Miri, which cannot emulate `mmap`.
    #[cfg_attr(not(miri), allow(dead_code))]
    Heap { ptr: NonNull<u8>, layout: Layout },
}

/// A fixed-size, 64-byte-aligned region holding block records.
pub(crate) struct Segment {
    backing: Backing,
    len: usize,
}

// SAFETY: all access goes through bounds-checked methods, and the store only
// writes ranges that no reader can observe yet (see module docs).
unsafe impl Send for Segment {}
// SAFETY: as above.
unsafe impl Sync for Segment {}

impl Segment {
    /// Zeroed volatile segment for in-memory stores.
    #[cfg(not(miri))]
    pub(crate) fn heap(len: usize) -> Self {
        let map = memmap2::MmapOptions::new()
            .len(len.max(RECORD_ALIGN))
            .map_anon()
            .unwrap_or_else(|err| panic!("anonymous mapping of {len} bytes failed: {err}"));
        Self {
            backing: Backing::Mapped {
                map: map.into(),
                _file: None,
            },
            len,
        }
    }

    /// Zeroed volatile segment for in-memory stores.
    #[cfg(miri)]
    pub(crate) fn heap(len: usize) -> Self {
        let layout = Layout::from_size_align(len.max(RECORD_ALIGN), RECORD_ALIGN)
            .expect("segment size overflows isize");
        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self {
            backing: Backing::Heap { ptr, layout },
            len,
        }
    }

    /// Creates a preallocated (sparse) segment file of `len` bytes.
    pub(crate) fn create_file(path: &Path, len: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        file.set_len(len as u64)?;
        Self::map(file, len)
    }

    /// Maps an existing segment file.
    pub(crate) fn open_file(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "segment too large"))?;
        Self::map(file, len)
    }

    fn map(file: File, len: usize) -> io::Result<Self> {
        let map = MmapRaw::map_raw(&file)?;
        Ok(Self {
            backing: Backing::Mapped {
                map,
                _file: Some(file),
            },
            len,
        })
    }

    fn base(&self) -> *mut u8 {
        match &self.backing {
            Backing::Mapped { map, .. } => map.as_mut_ptr(),
            Backing::Heap { ptr, .. } => ptr.as_ptr(),
        }
    }

    /// Segment capacity in bytes.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn check_range(&self, offset: usize, len: usize) {
        let end = offset.checked_add(len).expect("segment range overflows");
        assert!(
            end <= self.len,
            "segment range {offset}..{end} exceeds {}",
            self.len
        );
    }

    /// Copies `data` to `offset`. Callers must hold the store's writer lock
    /// and target only bytes not yet published to readers.
    pub(crate) fn write(&self, offset: usize, data: &[u8]) {
        self.check_range(offset, data.len());
        // SAFETY: the range is in bounds, and by the caller contract no
        // reference to it exists while it is written.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.base().add(offset), data.len());
        }
    }

    /// Borrows `len` bytes at `offset`. Only published (immutable) ranges may
    /// be read.
    pub(crate) fn read(&self, offset: usize, len: usize) -> &[u8] {
        self.check_range(offset, len);
        // SAFETY: the range is in bounds and, once published, never written
        // again for the lifetime of the segment.
        unsafe { std::slice::from_raw_parts(self.base().add(offset), len) }
    }

    /// Flushes a written range to stable storage (no-op for heap segments).
    pub(crate) fn flush(&self, offset: usize, len: usize) -> io::Result<()> {
        match &self.backing {
            Backing::Mapped { map, .. } => map.flush_range(offset, len),
            Backing::Heap { .. } => Ok(()),
        }
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        if let Backing::Heap { ptr, layout } = self.backing {
            // SAFETY: allocated in `Segment::heap` with this exact layout.
            unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }
}

impl KvBlockStore {
    /// [`KvBlockStore::store`] reading the payload from a raw pointer.
    ///
    /// # Safety
    ///
    /// `data` must be valid for reads of `len` bytes for the whole call and
    /// must not be mutated concurrently (e.g. a contiguous CPU tensor whose
    /// producing CUDA stream has been synchronized).
    pub unsafe fn store_raw(
        &self,
        key: &BlockHash,
        data: *const u8,
        len: usize,
        layout: KvBlockLayout,
        hyperedges: &[u64],
    ) -> Result<StoreOutcome, KvError> {
        if len == 0 {
            return self.store(key, &[], layout, hyperedges);
        }
        if data.is_null() {
            return Err(KvError::InvalidArgument("null source pointer".into()));
        }
        // SAFETY: guaranteed by the caller.
        let payload = unsafe { std::slice::from_raw_parts(data, len) };
        self.store(key, payload, layout, hyperedges)
    }

    /// [`KvBlockStore::load_into`] writing into a raw pointer. Returns whether
    /// the key was present.
    ///
    /// # Safety
    ///
    /// `target` must be valid for writes of `target_len` bytes for the whole
    /// call, and no other reference to that memory may be used concurrently.
    pub unsafe fn load_raw(
        &self,
        key: &BlockHash,
        target: *mut u8,
        target_len: usize,
    ) -> Result<bool, KvError> {
        if target.is_null() {
            return Err(KvError::InvalidArgument("null target pointer".into()));
        }
        // SAFETY: guaranteed by the caller.
        let dst = unsafe { std::slice::from_raw_parts_mut(target, target_len) };
        self.load_into(key, dst).map(|layout| layout.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::KvStoreConfig;

    fn small_store() -> KvBlockStore {
        KvBlockStore::in_memory(KvStoreConfig {
            segment_bytes: 512,
            ..KvStoreConfig::default()
        })
    }

    #[test]
    fn heap_segment_is_aligned_and_zeroed() {
        let segment = Segment::heap(256);
        assert_eq!(segment.base() as usize % RECORD_ALIGN, 0);
        assert!(segment.read(0, 256).iter().all(|&b| b == 0));
        segment.write(10, &[1, 2, 3]);
        assert_eq!(segment.read(10, 3), [1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn out_of_bounds_write_panics() {
        Segment::heap(64).write(60, &[0; 8]);
    }

    #[test]
    fn raw_round_trip_across_segments() {
        let store = small_store();
        let blocks: Vec<Vec<u8>> = (0..5u8).map(|i| vec![i; 200 + i as usize]).collect();
        for (i, block) in blocks.iter().enumerate() {
            let key = [i as u8; 32];
            // SAFETY: `block` is a live, unaliased vector.
            unsafe {
                store
                    .store_raw(
                        &key,
                        block.as_ptr(),
                        block.len(),
                        KvBlockLayout::OPAQUE,
                        &[],
                    )
                    .unwrap();
            }
        }
        for (i, block) in blocks.iter().enumerate() {
            let mut out = vec![0xAAu8; block.len()];
            // SAFETY: `out` is a live, unaliased vector of `out.len()` bytes.
            let hit = unsafe { store.load_raw(&[i as u8; 32], out.as_mut_ptr(), out.len()) };
            assert!(hit.unwrap());
            assert_eq!(&out, block);
        }
        assert!(store.stats().segments > 1);
    }

    #[test]
    fn raw_rejects_null_and_short_targets() {
        let store = small_store();
        let data = [9u8; 100];
        // SAFETY: `data` is a live array.
        unsafe {
            store
                .store_raw(
                    &[1; 32],
                    data.as_ptr(),
                    data.len(),
                    KvBlockLayout::OPAQUE,
                    &[],
                )
                .unwrap();
        }
        // SAFETY: null pointers are rejected before any access.
        let null = unsafe { store.load_raw(&[1; 32], std::ptr::null_mut(), 100) };
        assert!(matches!(null, Err(KvError::InvalidArgument(_))));
        let mut short = [0u8; 10];
        // SAFETY: `short` is a live array of `short.len()` bytes.
        let result = unsafe { store.load_raw(&[1; 32], short.as_mut_ptr(), short.len()) };
        assert!(matches!(
            result,
            Err(KvError::DestinationTooSmall {
                needed: 100,
                got: 10
            })
        ));
        let mut miss = [0u8; 4];
        // SAFETY: as above.
        let miss = unsafe { store.load_raw(&[2; 32], miss.as_mut_ptr(), miss.len()) };
        assert!(!miss.unwrap());
    }
}

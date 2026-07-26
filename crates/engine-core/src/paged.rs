//! A persistent, copy-on-write paged vector.
//!
//! Elements live in fixed-size pages grouped into directories, both behind
//! `Arc`s. Cloning the vector (publishing a read snapshot) copies only the
//! directory pointers; mutating an element copies just the touched directory
//! and page if they are still shared with an older snapshot.

use std::sync::Arc;

const PAGE_BITS: u32 = 6;
const DIR_BITS: u32 = 7;
/// Elements per page.
pub const PAGE_LEN: usize = 1 << PAGE_BITS;
/// Pages per directory.
pub const DIR_LEN: usize = 1 << DIR_BITS;

type Page<T> = Vec<T>;

#[derive(Debug, Clone)]
struct Directory<T> {
    pages: Vec<Arc<Page<T>>>,
}

/// Copy-on-write paged vector (see module docs).
#[derive(Debug)]
pub struct PagedVec<T> {
    dirs: Vec<Arc<Directory<T>>>,
    len: usize,
}

impl<T> Clone for PagedVec<T> {
    fn clone(&self) -> Self {
        Self {
            dirs: self.dirs.clone(),
            len: self.len,
        }
    }
}

impl<T> Default for PagedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
const fn split(index: usize) -> (usize, usize, usize) {
    (
        index >> (PAGE_BITS + DIR_BITS),
        (index >> PAGE_BITS) & (DIR_LEN - 1),
        index & (PAGE_LEN - 1),
    )
}

impl<T> PagedVec<T> {
    /// Creates an empty vector.
    pub const fn new() -> Self {
        Self {
            dirs: Vec::new(),
            len: 0,
        }
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the element at `index`.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let (d, p, o) = split(index);
        Some(&self.dirs[d].pages[p][o])
    }

    /// Iterates over all elements in index order.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.dirs
            .iter()
            .flat_map(|dir| dir.pages.iter().flat_map(|page| page.iter()))
    }
}

impl<T: Clone> PagedVec<T> {
    /// Mutable access to the element at `index`, copying its page (and
    /// directory) first if they are shared with another snapshot.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        let (d, p, o) = split(index);
        let dir = Arc::make_mut(&mut self.dirs[d]);
        let page = Arc::make_mut(&mut dir.pages[p]);
        Some(&mut page[o])
    }

    /// Appends an element and returns its index.
    pub fn push(&mut self, value: T) -> usize {
        let index = self.len;
        let (d, p, _) = split(index);
        if d == self.dirs.len() {
            self.dirs.push(Arc::new(Directory {
                pages: Vec::with_capacity(DIR_LEN),
            }));
        }
        let dir = Arc::make_mut(&mut self.dirs[d]);
        if p == dir.pages.len() {
            dir.pages.push(Arc::new(Vec::with_capacity(PAGE_LEN)));
        }
        Arc::make_mut(&mut dir.pages[p]).push(value);
        self.len += 1;
        index
    }

    /// Grows the vector to `new_len` by appending clones of `value`.
    pub fn extend_to(&mut self, new_len: usize, value: T) {
        while self.len < new_len {
            self.push(value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_get_across_directories() {
        let mut v = PagedVec::new();
        let n = PAGE_LEN * DIR_LEN * 2 + 17;
        for i in 0..n {
            assert_eq!(v.push(i), i);
        }
        assert_eq!(v.len(), n);
        for i in (0..n).step_by(97) {
            assert_eq!(v.get(i), Some(&i));
        }
        assert_eq!(v.get(n), None);
        assert!(v.iter().copied().eq(0..n));
    }

    #[test]
    fn snapshots_are_isolated_from_later_writes() {
        let mut v: PagedVec<u32> = PagedVec::new();
        for i in 0..1_000 {
            v.push(i);
        }
        let snapshot = v.clone();
        *v.get_mut(500).unwrap() = 9_999;
        v.push(1_000);
        assert_eq!(snapshot.get(500), Some(&500));
        assert_eq!(snapshot.len(), 1_000);
        assert_eq!(v.get(500), Some(&9_999));
        assert_eq!(v.len(), 1_001);
    }

    #[test]
    fn untouched_pages_stay_shared() {
        let mut v: PagedVec<u64> = PagedVec::new();
        for i in 0..(PAGE_LEN as u64 * 4) {
            v.push(i);
        }
        let snapshot = v.clone();
        *v.get_mut(0).unwrap() = 42;
        assert!(Arc::ptr_eq(&v.dirs[0].pages[1], &snapshot.dirs[0].pages[1]));
        assert!(!Arc::ptr_eq(
            &v.dirs[0].pages[0],
            &snapshot.dirs[0].pages[0]
        ));
    }
}

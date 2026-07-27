//! Synchronization primitives, swapped for loom's instrumented versions when
//! compiled with `RUSTFLAGS="--cfg loom"`.

#[cfg(loom)]
pub use loom::sync::atomic::{AtomicU32, AtomicU64, Ordering};
#[cfg(loom)]
pub use loom::sync::Arc;

#[cfg(not(loom))]
pub use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
#[cfg(not(loom))]
pub use std::sync::Arc;

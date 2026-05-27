//! Shared types and traits for the `dragonwing-edge` framework.
//!
//! This crate is intentionally minimal: it only defines types that need to be
//! visible to every other crate in the workspace (capability descriptors, the
//! [`Backend`] trait, the [`Error`] type). It must not depend on any heavy
//! runtime crate.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod capabilities;
pub mod error;

pub use capabilities::HardwareCapabilities;
pub use error::{Error, Result};

/// Marker trait for inference backends. Intentionally empty at this stage —
/// the concrete method surface is deferred to implementation task 002, once
/// the backend path is chosen on the basis of the Phase-6 probe results.
pub trait Backend {
    /// Human-readable backend name, e.g. `"vulkan-turnip"`, `"cpu-neon"`.
    fn name(&self) -> &'static str;
}

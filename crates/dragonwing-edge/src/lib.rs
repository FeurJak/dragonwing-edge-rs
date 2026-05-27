//! `dragonwing-edge` — top-level façade for the workspace.
//!
//! This crate currently just re-exports `dragonwing-core` and `dragonwing-hal`
//! so downstream consumers (e.g. `cortex-edge-rs`) have a single dep entry.

pub use dragonwing_core as core;
pub use dragonwing_hal as hal;

pub use dragonwing_core::{Backend, Error, HardwareCapabilities, Result};

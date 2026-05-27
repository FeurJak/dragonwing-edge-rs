//! Hardware abstraction layer for the QRB2210-based Arduino UNO Q.
//!
//! Each submodule probes one subsystem and returns a fragment of the
//! [`HardwareCapabilities`](dragonwing_core::HardwareCapabilities) struct.
//! Probing is non-fatal: a missing subsystem yields a default-valued fragment
//! with `present = false`, never a panic.
//!
//! This crate intentionally uses **no external dependencies**. Every probe is
//! a `std::fs` read of a procfs/sysfs/devfs path or a `dlopen` of a system
//! library. This is the most portable approach on a Debian/aarch64 host and
//! keeps the dependency surface flat (per `project.md`).

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod cpu;
pub mod dsp;
pub mod gpu;
pub mod kernel;
pub mod mcu;
pub mod memory;
pub mod thermal;

use dragonwing_core::HardwareCapabilities;

/// Run every probe and assemble a full capability snapshot.
///
/// `host` and `captured_at` are caller-supplied because they aren't strictly
/// hardware properties and we want them deterministic in tests.
#[must_use]
pub fn probe_all(host: String, captured_at: String) -> HardwareCapabilities {
    HardwareCapabilities {
        schema_version: HardwareCapabilities::SCHEMA_VERSION,
        captured_at,
        host,
        kernel: kernel::probe(),
        cpu: cpu::probe(),
        memory: memory::probe(),
        thermal: thermal::probe(),
        gpu: gpu::probe(),
        dsp: dsp::probe(),
        mcu_link: mcu::probe(),
    }
}

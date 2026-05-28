//! Shared types and traits for the `dragonwing-edge` framework.
//!
//! # What this crate provides
//!
//! * [`HardwareCapabilities`] and its supporting types — the structured
//!   description of a device produced by `dragonwing-hal` and consumed by
//!   anything that needs to make a runtime backend choice.
//! * The [`Backend`] trait and [`BackendBuffer`] handle — the minimum
//!   surface every inference backend in the workspace must implement.
//! * [`Error`] / [`Result`] — the small workspace-wide error type.
//! * [`Dtype`] — element data type enumeration (F32, F16, I8, I32).
//! * Quantization types for INT8 inference ([`quantization`] module).
//!
//! # What this crate deliberately does **not** provide
//!
//! * Compute ops (`axpy`, `gemm`, `conv2d`, …). Those live in each backend
//!   crate. See the [`backend`] module documentation for the rationale.
//! * `std`-flavoured I/O. The crate is `no_std` with `extern crate alloc`,
//!   so it can be reused on the STM32U585 side via the sibling
//!   `DragonWing-rs` repository if useful in the future.
//!
//! # Dependency policy
//!
//! Zero mandatory dependencies. The optional `serde` feature only adds the
//! `serde` crate (no derive macro chain at runtime). This is enforced via
//! review — any PR adding a hard dep to `dragonwing-core` should be
//! refused.

#![cfg_attr(not(test), no_std)]
// F16 conversion requires unsafe for NEON intrinsics and bit manipulation
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;

pub mod backend;
pub mod capabilities;
pub mod dtype;
pub mod error;
pub mod quantization;

pub use backend::{Backend, BackendBuffer, BufferKind};
pub use capabilities::HardwareCapabilities;
pub use dtype::{Dtype, F16};
pub use error::{Error, Result};
pub use quantization::{QuantScale, PerChannelScale, QuantizationParams};

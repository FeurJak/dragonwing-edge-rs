//! Cross-backend parity testing library.
//!
//! This crate provides utilities to verify that the CPU (NEON) and Vulkan
//! backends produce numerically equivalent results for all ops.
//!
//! # Usage
//!
//! ```no_run
//! use dragonwing_test::{ParityTest, TestConfig};
//! use dragonwing_cpu::CpuBackend;
//! use dragonwing_vulkan::{VulkanBackend, VulkanConfig};
//!
//! let cpu = CpuBackend::new();
//! let vulkan = VulkanBackend::new(VulkanConfig::default()).unwrap();
//!
//! let config = TestConfig::default();
//! let results = ParityTest::run_all(&cpu, &vulkan, &config);
//! results.print_summary();
//! ```
//!
//! # Tolerance
//!
//! Floating-point results may differ slightly between CPU and GPU due to:
//!
//! * Different FMA (fused multiply-add) behavior
//! * Different rounding modes
//! * Different instruction scheduling
//!
//! The default tolerance is `1e-5` for element-wise ops and `1e-3` for GEMM
//! (which accumulates many FMA operations).

#![warn(missing_docs)]

pub mod harness;
pub mod generators;
pub mod micro_graph;

pub use harness::{ParityTest, TestConfig, TestResult, TestResults};
pub use micro_graph::{run_cpu, run_cpu_mt, compare_results, MicroGraphResult};

//! `dragonwing-cpu` — CPU reference backend.
//!
//! This crate implements [`dragonwing_core::Backend`] over plain host
//! memory. It serves two roles in the framework:
//!
//! 1. **Correctness oracle.** Every op in this crate is the *reference
//!    implementation* against which GPU backends are validated by the
//!    `dragonwing-test` parity harness. When the Vulkan backend and the
//!    CPU backend disagree, the CPU result is taken as correct unless
//!    proven otherwise.
//! 2. **Fallback execution path.** If no GPU is reachable (`probe_all`
//!    reports `gpu.present == false`), user code can drop in this backend
//!    and continue running, albeit at much lower throughput.
//!
//! # Implementation strategy
//!
//! * `#[cfg(target_arch = "aarch64")]`: hand-written NEON intrinsics from
//!   `std::arch::aarch64`. Targets the QRB2210's 4× Kryo (Cortex-A53
//!   derivative) cluster. **Single-threaded** for now — task 003 will add a
//!   work-stealing scheduler.
//! * Other architectures: portable scalar implementations. These exist so
//!   the workspace builds and tests on a developer laptop (macOS aarch64
//!   uses the NEON path too; x86_64 hosts use the scalar path).
//!
//! Op surface in task 002:
//!
//! | Op | Signature | NEON intrinsics used |
//! |----|-----------|----------------------|
//! | [`ops::fill_f32`] | `(y: &mut [f32], v: f32)` | `vdupq_n_f32`, `vst1q_f32` |
//! | [`ops::axpy_f32`] | `(y: &mut [f32], a: f32, x: &[f32])` | `vld1q_f32`, `vdupq_n_f32`, `vfmaq_f32`, `vst1q_f32` |
//! | [`ops::relu_f32`] | `(y: &mut [f32], x: &[f32])` | `vld1q_f32`, `vmaxq_f32`, `vst1q_f32` |
//! | [`ops::gemm_f32_naive`] | `(c, a, b, m, n, k)` | scalar inner; NEON not yet used |
//!
//! # Why `unsafe` is unavoidable here
//!
//! The NEON intrinsics in `std::arch::aarch64` are all `unsafe fn` because
//! the compiler does not verify that the target CPU supports them. On the
//! UNO Q we have already verified (`probe_all` → `cpu.features` contains
//! `"asimd"`) that NEON is universally available, but the language doesn't
//! know that. Every `unsafe` block in this crate is annotated with a
//! `// SAFETY:` comment explaining what invariant makes the call sound.
//!
//! # Example
//!
//! ```
//! use dragonwing_core::{Backend, BackendBuffer, BufferKind};
//! use dragonwing_cpu::{CpuBackend, ops};
//!
//! let b = CpuBackend::new();
//! let n = 16;
//! let mut buf = b.alloc(n * std::mem::size_of::<f32>(), BufferKind::Storage).unwrap();
//!
//! // Stage data on the host, write it into the buffer, then read it back.
//! let mut host = vec![0.0_f32; n];
//! ops::fill_f32(&mut host, 1.5);
//!
//! let host_bytes: &[u8] = unsafe {
//!     std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), host.len() * 4)
//! };
//! b.upload(&mut buf, host_bytes).unwrap();
//!
//! let mut readback = vec![0u8; buf.len_bytes()];
//! b.download(&buf, &mut readback).unwrap();
//! assert_eq!(host_bytes, &readback[..]);
//! ```

// Note on `std` vs `no_std`:
// This crate uses `std` because the CPU ops rely on `f32::mul_add` (which
// routes through libm) to match the rounding of the NEON `vfmaq_f32`
// intrinsic. `dragonwing-core` remains `no_std`; only this CPU
// implementation crate pulls in `std`.
#![warn(missing_docs)]

pub mod ops;
pub mod pool;

pub use pool::{num_cpus, parallel_for_scoped, SendPtr, ThreadPool};

use dragonwing_core::error::Error;
use dragonwing_core::{Backend, BackendBuffer, BufferKind, Result};

/// CPU-resident byte buffer. Owns its storage; dropped when the buffer is
/// dropped.
#[derive(Debug)]
pub struct CpuBuffer {
    data: Vec<u8>,
}

impl CpuBuffer {
    /// View the buffer as raw bytes. Op functions that need typed access
    /// (e.g. `&mut [f32]`) reinterpret the bytes after a length check.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Mutable raw-byte view of the buffer.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// View the buffer as a `&[f32]`. Panics if `len_bytes` is not a
    /// multiple of 4. The pointer is naturally aligned because `Vec<u8>`
    /// reserves with the platform allocator, which on every supported
    /// target returns at least 4-byte-aligned memory.
    ///
    /// # Panics
    ///
    /// Panics if `len_bytes` is not a multiple of 4.
    #[must_use]
    pub fn as_f32(&self) -> &[f32] {
        assert!(
            self.data.len() % core::mem::size_of::<f32>() == 0,
            "CpuBuffer::as_f32: length {} is not a multiple of 4",
            self.data.len()
        );
        // SAFETY: We've checked the length is a multiple of 4, and the
        // global allocator guarantees alignment of at least 4 bytes for
        // any reservation. `f32` has no validity invariant — any bit
        // pattern is a valid `f32` (including NaN).
        unsafe {
            core::slice::from_raw_parts(
                self.data.as_ptr().cast::<f32>(),
                self.data.len() / core::mem::size_of::<f32>(),
            )
        }
    }

    /// Mutable `&mut [f32]` view. Same alignment / length contract as
    /// [`as_f32`](Self::as_f32).
    ///
    /// # Panics
    ///
    /// Panics if `len_bytes` is not a multiple of 4.
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        assert!(
            self.data.len() % core::mem::size_of::<f32>() == 0,
            "CpuBuffer::as_f32_mut: length {} is not a multiple of 4",
            self.data.len()
        );
        // SAFETY: see `as_f32` — same reasoning.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().cast::<f32>(),
                self.data.len() / core::mem::size_of::<f32>(),
            )
        }
    }

    /// View the buffer as a `&[F16]`. Panics if `len_bytes` is not a
    /// multiple of 2.
    ///
    /// # Panics
    ///
    /// Panics if `len_bytes` is not a multiple of 2.
    #[must_use]
    pub fn as_f16(&self) -> &[dragonwing_core::F16] {
        assert!(
            self.data.len() % core::mem::size_of::<dragonwing_core::F16>() == 0,
            "CpuBuffer::as_f16: length {} is not a multiple of 2",
            self.data.len()
        );
        // SAFETY: F16 is repr(transparent) over u16, which has alignment 2.
        // Vec<u8> is at least 4-byte aligned, satisfying the requirement.
        // Any bit pattern is valid for F16 (it's just stored bits).
        unsafe {
            core::slice::from_raw_parts(
                self.data.as_ptr().cast::<dragonwing_core::F16>(),
                self.data.len() / core::mem::size_of::<dragonwing_core::F16>(),
            )
        }
    }

    /// Mutable `&mut [F16]` view. Same alignment / length contract as
    /// [`as_f16`](Self::as_f16).
    ///
    /// # Panics
    ///
    /// Panics if `len_bytes` is not a multiple of 2.
    pub fn as_f16_mut(&mut self) -> &mut [dragonwing_core::F16] {
        assert!(
            self.data.len() % core::mem::size_of::<dragonwing_core::F16>() == 0,
            "CpuBuffer::as_f16_mut: length {} is not a multiple of 2",
            self.data.len()
        );
        // SAFETY: see `as_f16` — same reasoning.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().cast::<dragonwing_core::F16>(),
                self.data.len() / core::mem::size_of::<dragonwing_core::F16>(),
            )
        }
    }
}

impl BackendBuffer for CpuBuffer {
    fn len_bytes(&self) -> usize {
        self.data.len()
    }
}

/// The CPU reference backend. Construct with [`CpuBackend::new`].
///
/// Cheap to clone — internally just a zero-sized marker. Multiple clones
/// share no state because the CPU backend has no state to share.
#[derive(Debug, Clone, Default)]
pub struct CpuBackend {
    _private: (),
}

impl CpuBackend {
    /// Construct a new CPU backend. Always succeeds — the CPU backend
    /// performs no resource acquisition at construction time.
    #[must_use]
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Backend for CpuBackend {
    type Buffer = CpuBuffer;

    fn name(&self) -> &'static str {
        #[cfg(target_arch = "aarch64")]
        {
            "cpu-neon"
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            "cpu-scalar"
        }
    }

    fn alloc(&self, len_bytes: usize, _kind: BufferKind) -> Result<Self::Buffer> {
        // `Vec::with_capacity` allocates uninitialised memory, but
        // `vec![0u8; n]` zero-initialises which makes downstream bugs more
        // deterministic. Since the trait contract says contents are
        // unspecified, callers must not rely on zeroing, but zeroing here
        // makes accidental over-reads benign.
        let data = vec![0u8; len_bytes];
        Ok(CpuBuffer { data })
    }

    fn upload(&self, dst: &mut Self::Buffer, src: &[u8]) -> Result<()> {
        if dst.data.len() != src.len() {
            return Err(Error::Backend(format!(
                "cpu upload size mismatch: dst={} src={}",
                dst.data.len(),
                src.len()
            )));
        }
        dst.data.copy_from_slice(src);
        Ok(())
    }

    fn download(&self, src: &Self::Buffer, dst: &mut [u8]) -> Result<()> {
        if src.data.len() != dst.len() {
            return Err(Error::Backend(format!(
                "cpu download size mismatch: src={} dst={}",
                src.data.len(),
                dst.len()
            )));
        }
        dst.copy_from_slice(&src.data);
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        // The CPU backend executes synchronously; there is no queued work
        // to wait on. This is intentionally a `Result` so the trait stays
        // uniform across backends.
        Ok(())
    }
}

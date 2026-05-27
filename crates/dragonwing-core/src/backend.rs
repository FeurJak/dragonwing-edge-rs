//! The [`Backend`] trait and its associated [`BackendBuffer`] type.
//!
//! # Design overview for future contributors and agents
//!
//! `dragonwing-edge` follows a **closed-trait, open-ops** design:
//!
//! * The [`Backend`] trait defines the *minimum* surface every backend must
//!   implement: name, buffer allocation, host↔device copy, and a barrier. It
//!   intentionally does **not** declare any compute op (no `gemm`, no `conv`).
//!   This keeps the trait small and stable.
//! * Compute ops are free functions in each backend crate (e.g.
//!   `dragonwing_vulkan::ops::axpy` and `dragonwing_cpu::ops::axpy`). Two
//!   backends with the same op name are expected to produce numerically
//!   equivalent results, enforced by the cross-backend parity harness in
//!   `dragonwing-test`.
//! * Op dispatch from generic code is done by writing a function that takes a
//!   concrete backend handle, not via dynamic dispatch through the trait.
//!   Trait-based op dispatch was considered and rejected because it would
//!   force every op signature into a god-trait that all backends pay for.
//!
//! # Memory model
//!
//! All transfers are byte-oriented. The framework deliberately does **not**
//! carry tensor element types through the [`Backend`] trait — a buffer is a
//! contiguous span of bytes, and op functions reinterpret those bytes
//! according to the op's contract (e.g. `axpy_f32` treats them as `[f32]`).
//! This keeps the trait `no_std`-compatible and avoids a generic-type
//! explosion across backends.
//!
//! Backends are free to back a [`BackendBuffer`] with any storage: a
//! `Vec<u8>` (CPU), a `VkBuffer` + device memory (Vulkan), or a `cl_mem`
//! (future OpenCL backend). The handle is opaque; ops never reach inside it
//! except via backend-specific methods.
//!
//! # Synchronisation
//!
//! Every backend operation is conceptually queued. [`Backend::synchronize`]
//! must block until all previously-submitted work has completed. Op functions
//! are responsible for inserting per-dispatch synchronisation as needed (the
//! Vulkan backend uses timeline semaphores; the CPU backend executes
//! synchronously and [`Backend::synchronize`] is a no-op).
//!
//! # Lifetime and `Send` / `Sync`
//!
//! Backends are typically held inside an `Arc<B>` by user code so they can
//! be shared between threads. Implementations should be `Send + Sync` unless
//! the underlying hardware genuinely forbids it (none of the targeted
//! backends do at the time of writing).
//!
//! # Example integration sketch
//!
//! ```ignore
//! use dragonwing_edge::{Backend, hal};
//!
//! let caps = hal::probe_all("dev".into(), "now".into());
//! let backend = if caps.gpu.vulkan.is_some() {
//!     dragonwing_vulkan::VulkanBackend::new(Default::default())?
//! } else {
//!     // unreachable in normal flow; here for illustration of the dispatch
//!     // pattern. The CPU backend is a separate type, so a real selector
//!     // returns `Box<dyn Backend<...>>` or uses enum dispatch.
//!     return Err("no GPU".into());
//! };
//! ```

use crate::error::Result;

/// A handle to a contiguous, backend-owned byte buffer.
///
/// The buffer may live in host memory (CPU backend), unified host/device
/// memory (Vulkan on the QRB2210), or pure device memory (a discrete GPU).
/// Callers must not assume any particular storage location; use the
/// [`Backend`]'s `upload` / `download` methods for transfers.
///
/// Buffers are dropped through their backend's `Drop` impl, which is
/// responsible for releasing any underlying handles (`VkBuffer`,
/// `VkDeviceMemory`, etc.). A buffer outlives its backend only by reference
/// — backends typically use lifetime parameters or an `Arc<Inner>` to
/// guarantee correct ordering.
pub trait BackendBuffer {
    /// Length of the buffer in bytes, as requested at allocation time.
    ///
    /// The backend may have allocated more than this internally
    /// (e.g. rounded up to a page boundary), but only `len_bytes()` are
    /// addressable through this handle.
    fn len_bytes(&self) -> usize;
}

/// Marker for the kind of memory access an allocation needs.
///
/// All backends in the workspace currently treat every allocation as
/// `Storage`. The enum exists so that, when an image/texture-backed op is
/// added in a later task, we can introduce a new variant without breaking
/// the trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferKind {
    /// A linear storage buffer (SSBO in Vulkan terms; `Vec<u8>` on CPU).
    /// Suitable for tensor activations, weights, and biases.
    Storage,
}

/// The contract every inference backend must satisfy.
///
/// See the module-level documentation for the design rationale. In short:
/// this trait is deliberately minimal so that adding a new backend (e.g.
/// OpenCL via Rusticl, or a future direct-Adreno backend) is a small,
/// well-defined surface to implement.
///
/// # Implementing this trait
///
/// At minimum an implementor must:
///
/// 1. Pick an [`BackendBuffer`] type. CPU backends use `CpuBuffer(Vec<u8>)`;
///    GPU backends usually wrap a device handle plus an `Arc` to the
///    owning context.
/// 2. Implement [`alloc`](Backend::alloc) — allocate `len_bytes` of storage.
/// 3. Implement [`upload`](Backend::upload) and [`download`](Backend::download)
///    — byte-oriented host↔device copy. On unified-memory backends these
///    may reduce to `memcpy` into a mapped pointer.
/// 4. Implement [`synchronize`](Backend::synchronize) — block until queued
///    work completes. On CPU backends this is a no-op.
///
/// Op functions (`axpy`, `gemm`, …) are **not** part of this trait. They
/// are added as inherent methods on the concrete backend type or as free
/// functions in the backend crate.
pub trait Backend {
    /// The opaque buffer handle type this backend issues from
    /// [`alloc`](Backend::alloc).
    type Buffer: BackendBuffer;

    /// Human-readable identifier used in logs and parity-test output.
    ///
    /// Convention: lowercase, hyphen-separated, indicates both the access
    /// path and the hardware target. Examples:
    ///
    /// * `"cpu-neon"` — CPU reference backend with AArch64 NEON intrinsics.
    /// * `"cpu-scalar"` — CPU reference backend on non-AArch64 hosts.
    /// * `"vulkan-turnip"` — Vulkan compute via Mesa/Turnip on Adreno.
    /// * `"vulkan-llvmpipe"` — Vulkan compute via Mesa's CPU fallback (for
    ///   host development).
    fn name(&self) -> &'static str;

    /// Allocate `len_bytes` of backend-owned storage.
    ///
    /// The buffer's contents are **unspecified** on return — callers must
    /// not assume zero-initialisation. Use a `fill` op if zeroing is
    /// required.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`](crate::error::Error::Backend) if the
    /// backend cannot satisfy the allocation (out of memory, descriptor
    /// pool exhausted, etc.). Backends should produce errors with a
    /// concrete reason string, not silently allocate a smaller buffer.
    fn alloc(&self, len_bytes: usize, kind: BufferKind) -> Result<Self::Buffer>;

    /// Copy bytes from `src` (host) into `dst` (backend).
    ///
    /// `src.len()` must equal `dst.len_bytes()`; the backend should return
    /// [`Error::Backend`](crate::error::Error::Backend) on mismatch rather
    /// than truncate or panic.
    ///
    /// On unified-memory backends this is a `memcpy` into a mapped pointer.
    /// On backends with a separate device heap, the implementation may
    /// transparently use a staging buffer + `vkCmdCopyBuffer`-style
    /// transfer; in that case the copy is **complete** by the time this
    /// function returns (i.e. uploads behave synchronously).
    ///
    /// # Errors
    ///
    /// Returns an error on length mismatch or transfer failure.
    fn upload(&self, dst: &mut Self::Buffer, src: &[u8]) -> Result<()>;

    /// Copy bytes from `src` (backend) into `dst` (host).
    ///
    /// `dst.len()` must equal `src.len_bytes()`. Implicitly calls
    /// [`synchronize`](Backend::synchronize) before the read, so any
    /// previously-dispatched ops that wrote to `src` are visible.
    ///
    /// # Errors
    ///
    /// Returns an error on length mismatch, transfer failure, or if a
    /// preceding dispatch reported a device-lost condition.
    fn download(&self, src: &Self::Buffer, dst: &mut [u8]) -> Result<()>;

    /// Block until every operation previously submitted on this backend
    /// has completed.
    ///
    /// On the CPU backend this is a no-op (ops execute synchronously). On
    /// the Vulkan backend this waits on the latest timeline-semaphore
    /// value submitted by an op dispatch.
    ///
    /// # Errors
    ///
    /// Returns an error only if the underlying API reports a device-lost
    /// or unrecoverable timeout condition.
    fn synchronize(&self) -> Result<()>;
}

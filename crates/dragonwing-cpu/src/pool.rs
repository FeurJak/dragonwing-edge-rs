//! Thread pool for CPU backend parallelism.
//!
//! A lightweight, persistent thread pool for multi-threading GEMM and Conv2D
//! operations across the available CPU cores. Uses `std::thread::scope` with
//! `Mutex<VecDeque>` + `Condvar` for work distribution.
//!
//! # Design
//!
//! - Workers are spawned at pool construction and persist until drop.
//! - Work is submitted as boxed closures via an MPSC-like channel.
//! - Pool size defaults to `num_cpus() - 1` (leave one core for the OS).
//! - No external dependencies — reads `/sys/devices/system/cpu/online` on Linux,
//!   falls back to `std::thread::available_parallelism()` elsewhere.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// A job is a boxed closure that takes no arguments and returns nothing.
/// The closure must be `Send` because it crosses thread boundaries.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// Message sent to workers: either a job to execute or a shutdown signal.
enum Message {
    Job(Job),
    Shutdown,
}

impl std::fmt::Debug for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Message::Job(_) => f.write_str("Job(...)"),
            Message::Shutdown => f.write_str("Shutdown"),
        }
    }
}

/// Shared state between the pool and its workers.
#[derive(Debug)]
struct SharedState {
    queue: VecDeque<Message>,
    shutdown: bool,
}

/// A persistent thread pool for parallel work execution.
///
/// # Example
///
/// ```ignore
/// let pool = ThreadPool::new(4);
/// pool.execute(|| println!("Hello from worker!"));
/// pool.join(); // Wait for all jobs to complete
/// ```
#[derive(Debug)]
pub struct ThreadPool {
    workers: Vec<JoinHandle<()>>,
    state: Arc<(Mutex<SharedState>, Condvar)>,
    pending: Arc<(Mutex<usize>, Condvar)>,
}

impl ThreadPool {
    /// Create a new thread pool with the specified number of workers.
    ///
    /// # Panics
    ///
    /// Panics if `num_workers` is 0.
    #[must_use]
    pub fn new(num_workers: usize) -> Self {
        assert!(num_workers > 0, "ThreadPool requires at least 1 worker");

        let state = Arc::new((
            Mutex::new(SharedState {
                queue: VecDeque::new(),
                shutdown: false,
            }),
            Condvar::new(),
        ));

        let pending = Arc::new((Mutex::new(0_usize), Condvar::new()));

        let mut workers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let state_clone = Arc::clone(&state);
            let pending_clone = Arc::clone(&pending);

            let handle = thread::spawn(move || {
                worker_loop(state_clone, pending_clone);
            });

            workers.push(handle);
        }

        Self {
            workers,
            state,
            pending,
        }
    }

    /// Create a thread pool with the default number of workers.
    ///
    /// Default is `max(1, num_cpus() - 1)` to leave one core for the OS.
    #[must_use]
    pub fn with_default_size() -> Self {
        let cpus = num_cpus();
        let workers = cpus.saturating_sub(1).max(1);
        Self::new(workers)
    }

    /// Returns the number of workers in the pool.
    #[must_use]
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Submit a job to the pool for execution.
    ///
    /// The job will be picked up by one of the available workers.
    pub fn execute<F>(&self, job: F)
    where
        F: FnOnce() + Send + 'static,
    {
        // Increment pending count
        {
            let (lock, _) = &*self.pending;
            let mut count = lock.lock().unwrap();
            *count += 1;
        }

        // Push the job to the queue
        {
            let (lock, cvar) = &*self.state;
            let mut state = lock.lock().unwrap();
            state.queue.push_back(Message::Job(Box::new(job)));
            cvar.notify_one();
        }
    }

    /// Wait for all submitted jobs to complete.
    ///
    /// This blocks the calling thread until the pending job count reaches zero.
    pub fn join(&self) {
        let (lock, cvar) = &*self.pending;
        let mut count = lock.lock().unwrap();
        while *count > 0 {
            count = cvar.wait(count).unwrap();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Send shutdown messages to all workers
        {
            let (lock, cvar) = &*self.state;
            let mut state = lock.lock().unwrap();
            state.shutdown = true;
            for _ in &self.workers {
                state.queue.push_back(Message::Shutdown);
            }
            cvar.notify_all();
        }

        // Join all worker threads (with a timeout to prevent deadlocks)
        for handle in self.workers.drain(..) {
            // We can't easily timeout with std::thread, so just join.
            // In practice, workers should exit quickly on shutdown.
            let _ = handle.join();
        }
    }
}

/// Worker loop: wait for jobs and execute them until shutdown.
fn worker_loop(
    state: Arc<(Mutex<SharedState>, Condvar)>,
    pending: Arc<(Mutex<usize>, Condvar)>,
) {
    loop {
        let message = {
            let (lock, cvar) = &*state;
            let mut state = lock.lock().unwrap();

            // Wait for a message
            while state.queue.is_empty() && !state.shutdown {
                state = cvar.wait(state).unwrap();
            }

            // Check if we should exit
            if state.shutdown && state.queue.is_empty() {
                return;
            }

            state.queue.pop_front()
        };

        match message {
            Some(Message::Job(job)) => {
                // Execute the job
                job();

                // Decrement pending count and notify waiters
                let (lock, cvar) = &*pending;
                let mut count = lock.lock().unwrap();
                *count -= 1;
                if *count == 0 {
                    cvar.notify_all();
                }
            }
            Some(Message::Shutdown) | None => {
                return;
            }
        }
    }
}

/// Detect the number of CPUs available.
///
/// On Linux, reads `/sys/devices/system/cpu/online` for accurate online CPU count.
/// On other platforms, falls back to `std::thread::available_parallelism()`.
#[must_use]
pub fn num_cpus() -> usize {
    // Try Linux-specific method first
    #[cfg(target_os = "linux")]
    {
        if let Ok(online) = std::fs::read_to_string("/sys/devices/system/cpu/online") {
            if let Some(count) = parse_cpu_online(&online) {
                return count;
            }
        }
    }

    // Fallback to std
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Parse the `/sys/devices/system/cpu/online` format.
///
/// Format is like "0-3" or "0,2-3" or "0" — a comma-separated list of
/// ranges where each range is either "N" or "N-M".
#[cfg(target_os = "linux")]
fn parse_cpu_online(s: &str) -> Option<usize> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let mut count = 0;
    for part in s.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            let start: usize = start.trim().parse().ok()?;
            let end: usize = end.trim().parse().ok()?;
            count += end - start + 1;
        } else {
            let _: usize = part.parse().ok()?;
            count += 1;
        }
    }

    Some(count)
}

// ============================================================================
// Parallel iteration helpers
// ============================================================================

/// Execute a function in parallel over chunks of a range.
///
/// Divides `0..total` into `num_chunks` roughly equal parts and executes
/// `f(chunk_start, chunk_end)` for each chunk, potentially in parallel.
///
/// If `pool` is `Some`, work is distributed across the thread pool.
/// If `pool` is `None` or `num_chunks <= 1`, executes sequentially.
pub fn parallel_for<F>(pool: Option<&ThreadPool>, total: usize, num_chunks: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync + 'static,
{
    if total == 0 {
        return;
    }

    let num_chunks = num_chunks.min(total).max(1);

    // Single-threaded path
    if pool.is_none() || num_chunks <= 1 {
        f(0, total);
        return;
    }

    let pool = pool.unwrap();
    let chunk_size = (total + num_chunks - 1) / num_chunks;
    let f = Arc::new(f);

    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(total);
        if start >= total {
            break;
        }

        let f_clone = Arc::clone(&f);
        pool.execute(move || {
            f_clone(start, end);
        });
    }

    pool.join();
}

/// A scoped parallel for loop that doesn't require 'static lifetimes.
///
/// Uses `std::thread::scope` internally to allow borrowing from the caller's
/// stack frame. This is the preferred API for most parallel ops.
pub fn parallel_for_scoped<F>(num_threads: usize, total: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    if total == 0 {
        return;
    }

    let num_threads = num_threads.min(total).max(1);

    // Single-threaded path
    if num_threads <= 1 {
        f(0, total);
        return;
    }

    let chunk_size = (total + num_threads - 1) / num_threads;

    std::thread::scope(|s| {
        for i in 0..num_threads {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(total);
            if start >= total {
                break;
            }

            // Reference f instead of moving it so all threads can use it
            let f_ref = &f;
            s.spawn(move || {
                f_ref(start, end);
            });
        }
    });
}

// ============================================================================
// Raw pointer wrappers for thread-safe sharing
// ============================================================================

/// Wrapper around a raw pointer that asserts it's safe to send/share.
///
/// # Safety
///
/// The caller must ensure that:
/// 1. The pointee lives for the duration of all threads using this wrapper
/// 2. For mutable pointers: threads access non-overlapping regions
/// 3. For const pointers: the pointee is not mutated during parallel access
#[derive(Clone, Copy, Debug)]
pub struct SendPtr<T>(*mut T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// Create a new `SendPtr` from a mutable pointer.
    ///
    /// # Safety
    ///
    /// Caller must ensure thread-safe access patterns.
    #[inline]
    pub unsafe fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// Create a new `SendPtr` from a const pointer.
    ///
    /// # Safety
    ///
    /// Caller must ensure the pointee is not mutated during parallel access.
    #[inline]
    pub unsafe fn from_const(ptr: *const T) -> Self {
        Self(ptr as *mut T)
    }

    /// Get the inner pointer.
    #[inline]
    pub fn as_ptr(self) -> *mut T {
        self.0
    }

    /// Get the inner pointer as const.
    #[inline]
    pub fn as_const_ptr(self) -> *const T {
        self.0 as *const T
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn pool_basic_execution() {
        let pool = ThreadPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..10 {
            let c = Arc::clone(&counter);
            pool.execute(move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
        }

        pool.join();
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn pool_default_size() {
        let pool = ThreadPool::with_default_size();
        assert!(pool.num_workers() >= 1);
    }

    #[test]
    fn parallel_for_scoped_basic() {
        let data = vec![0_u32; 100];
        let sum = AtomicUsize::new(0);

        // Fill with values
        let mut data = data;
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as u32;
        }

        // Parallel sum
        parallel_for_scoped(4, data.len(), |start, end| {
            let local_sum: usize = data[start..end].iter().map(|&x| x as usize).sum();
            sum.fetch_add(local_sum, Ordering::SeqCst);
        });

        // Expected: 0 + 1 + 2 + ... + 99 = 4950
        assert_eq!(sum.load(Ordering::SeqCst), 4950);
    }

    #[test]
    fn num_cpus_returns_positive() {
        assert!(num_cpus() >= 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_cpu_online_formats() {
        assert_eq!(parse_cpu_online("0-3"), Some(4));
        assert_eq!(parse_cpu_online("0"), Some(1));
        assert_eq!(parse_cpu_online("0,2-3"), Some(3));
        assert_eq!(parse_cpu_online("0-1,3-5"), Some(5));
        assert_eq!(parse_cpu_online(""), None);
    }
}

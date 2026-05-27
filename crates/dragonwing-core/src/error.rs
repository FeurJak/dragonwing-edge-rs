//! Crate-wide error type.

use alloc::string::String;
use core::fmt;

/// Result alias used across the workspace.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Top-level error type. Kept deliberately small; backends can carry their
/// own error context inside [`Error::Backend`].
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An I/O operation failed while probing or running.
    Io(String),
    /// A required hardware feature was not present.
    Unsupported(&'static str),
    /// Backend-specific error message.
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(m) => write!(f, "io error: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::Backend(m) => write!(f, "backend error: {m}"),
        }
    }
}

// `std::error::Error` impl deliberately omitted: the crate is `no_std` and
// we don't currently expose a `std` feature. Add it in a later task when the
// concrete backend surface lands.

// SPDX-License-Identifier: Apache-2.0
//! What one metastore call may cost (design P3-lakehouse §5.6): a deadline for the whole call, a budget of bytes
//! read, a budget of decoded elements and a longest string. `thrift` bounds none of these as a total: it checks one
//! container at a time against a count times a minimum element size, and a socket timeout restarts on every read,
//! so a server that sends one byte at a time would hold a call forever and a message of empty structs would
//! allocate far more than its size on the wire. The guarded socket spends the deadline and the bytes on every
//! read; the type readers spend the elements before they allocate. A tripped limit is remembered, so the error that
//! comes out of the protocol layer can be named for what it was.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Which limit a call ran into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    /// More bytes than `max_message_bytes` were read for one call.
    Bytes,
    /// More decoded elements (list items, map entries) than `max_elements`.
    Elements,
    /// A string or container longer than the protocol's configured size.
    Size,
    /// Nested deeper than `max_depth`.
    Depth,
}

impl Limit {
    fn code(self) -> u8 {
        match self {
            Self::Bytes => 1,
            Self::Elements => 2,
            Self::Size => 3,
            Self::Depth => 4,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Bytes),
            2 => Some(Self::Elements),
            3 => Some(Self::Size),
            4 => Some(Self::Depth),
            _ => None,
        }
    }
}

/// The limits of one client, and what is left of them for the call in progress.
#[derive(Debug)]
pub struct Budget {
    request_timeout: Duration,
    max_bytes: usize,
    max_elements: usize,
    deadline: Mutex<Instant>,
    bytes_left: AtomicUsize,
    elements_left: AtomicUsize,
    tripped: AtomicU8,
}

impl Budget {
    pub(crate) fn new(request_timeout: Duration, max_bytes: usize, max_elements: usize) -> Self {
        Self {
            request_timeout,
            max_bytes,
            max_elements,
            deadline: Mutex::new(Instant::now() + request_timeout),
            bytes_left: AtomicUsize::new(max_bytes),
            elements_left: AtomicUsize::new(max_elements),
            tripped: AtomicU8::new(0),
        }
    }

    /// Starts a call: a fresh deadline and fresh budgets, nothing tripped.
    pub(crate) fn begin(&self) {
        self.begin_until(Instant::now() + self.request_timeout);
    }

    /// Starts a call that must end by `deadline`: one established before the call waited for a connection, so the
    /// wait and the call share it.
    pub(crate) fn begin_until(&self, deadline: Instant) {
        *self
            .deadline
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = deadline;
        self.bytes_left.store(self.max_bytes, Ordering::Release);
        self.elements_left
            .store(self.max_elements, Ordering::Release);
        self.tripped.store(0, Ordering::Release);
    }

    /// Time left for the call; `None` when it is over.
    pub(crate) fn remaining(&self) -> Option<Duration> {
        let deadline = *self
            .deadline
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
    }

    /// How many more bytes this call may read; a read that finds none left trips [`Limit::Bytes`].
    pub(crate) fn bytes_allowed(&self) -> usize {
        let left = self.bytes_left.load(Ordering::Acquire);
        if left == 0 {
            self.trip(Limit::Bytes);
        }
        left
    }

    pub(crate) fn spend_bytes(&self, n: usize) {
        self.bytes_left
            .try_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                Some(left.saturating_sub(n))
            })
            .ok();
    }

    /// Spends `n` decoded elements before they are allocated.
    ///
    /// # Errors
    /// A protocol error, remembered as [`Limit::Elements`], when the call has decoded too many.
    pub(crate) fn take_elements(&self, n: usize) -> thrift::Result<()> {
        let taken = self
            .elements_left
            .try_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                left.checked_sub(n)
            })
            .is_ok();
        if taken {
            Ok(())
        } else {
            self.trip(Limit::Elements);
            Err(thrift::Error::Protocol(thrift::ProtocolError::new(
                thrift::ProtocolErrorKind::SizeLimit,
                "too many elements in one reply",
            )))
        }
    }

    pub(crate) fn trip(&self, limit: Limit) {
        self.tripped.store(limit.code(), Ordering::Release);
    }

    /// The limit this call ran into, if any.
    pub(crate) fn tripped(&self) -> Option<Limit> {
        Limit::from_code(self.tripped.load(Ordering::Acquire))
    }
}

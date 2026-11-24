// SPDX-License-Identifier: Apache-2.0
//! Regex results by input value, kept per thread across batches, so a repeated value is matched
//! once (perf C-1a, `krishiv-internal/research/perf-parity/c1-step0/README.md`).
//!
//! The regex family's kernels cost about a microsecond a row (the capture engine walks the
//! whole match), while hashing a value costs tens of nanoseconds. ClickBench's non-empty
//! `Referer` is 35% distinct within a 8,192-row batch and about 14% within a row group, so a
//! cache that outlives the batch evaluates a fraction of the rows. It is used only where the
//! result depends on the value alone (pattern, replacement, index and position are constants of
//! the batch, part of the cache's key). Results are never errors: the first row of a value is
//! evaluated in row order and its error raised, exactly as without the cache.
//!
//! Bounds: a thread keeps [`CACHES`] keys, each at most [`CAP_BYTES`] of keys and results
//! (cleared when full). Input that keeps missing (over four lookups in five, over a window of
//! [`WINDOW`] lookups) turns a cache off for [`OFF_LOOKUPS`] lookups, so distinct input pays only
//! a window's probes now and then. Prior art: Velox evaluates over a dictionary's distinct base
//! values; ClickHouse runs functions on a `LowCardinality` dictionary.

use std::cell::RefCell;
use std::collections::HashMap;

use krishiv_common::error::KrishivError;

/// Lookups per adaptivity window.
pub(super) const WINDOW: u32 = 4096;

/// Lookups a cache stays off once a window found the input mostly distinct.
pub(super) const OFF_LOOKUPS: u64 = 1 << 20;

/// Bytes of keys and results one cache holds before it is cleared.
pub(super) const CAP_BYTES: usize = 4 << 20;

/// Caches (keys) one thread keeps, newest first.
const CACHES: usize = 2;

/// What an entry costs beyond its bytes (the boxed key, the slot, the result's header).
const ENTRY_OVERHEAD: usize = 48;

/// A cached result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Value {
    /// A boolean result (`regexp_like`).
    Bool(bool),
    /// A string result (`regexp_replace`, `regexp_extract`).
    Text(Box<str>),
}

impl Value {
    fn bytes(&self) -> usize {
        match self {
            Self::Bool(_) => 0,
            Self::Text(t) => t.len(),
        }
    }
}

/// One cache: the results of one function call shape (function, pattern, constants) by value.
pub(super) struct Memo {
    /// The function and its constants (pattern, replacement, ...), joined.
    key: (&'static str, String),
    map: HashMap<Box<str>, Value, foldhash::fast::RandomState>,
    bytes: usize,
    /// Lookups and misses of the current window.
    window: (u32, u32),
    /// Lookups left while off.
    off: u64,
    /// The last result evaluated while off.
    scratch: Option<Value>,
}

impl Memo {
    fn new(key: (&'static str, String)) -> Self {
        Self {
            key,
            map: HashMap::default(),
            bytes: 0,
            window: (0, 0),
            off: 0,
            scratch: None,
        }
    }

    /// The result for `value`: `eval` runs only when it is not cached.
    ///
    /// # Errors
    /// What `eval` returns for the value (never cached).
    pub(super) fn get(
        &mut self,
        value: &str,
        eval: impl FnOnce(&str) -> Result<Value, KrishivError>,
    ) -> Result<&Value, KrishivError> {
        if self.off > 0 {
            self.off -= 1;
            return Ok(self.scratch.insert(eval(value)?));
        }
        self.window.0 += 1;
        let hit = self.map.contains_key(value);
        if !hit {
            self.window.1 += 1;
        }
        if self.window.0 == WINDOW {
            let (lookups, misses) = std::mem::take(&mut self.window);
            if misses * 5 > lookups * 4 {
                self.off = OFF_LOOKUPS;
                self.map = HashMap::default();
                self.bytes = 0;
                if !hit {
                    return Ok(self.scratch.insert(eval(value)?));
                }
            }
        }
        if hit {
            return self.map.get(value).ok_or_else(|| {
                super::internal("regex memo: a cached value went missing".to_owned())
            });
        }
        let computed = eval(value)?;
        let cost = value.len() + computed.bytes() + ENTRY_OVERHEAD;
        if self.bytes + cost > CAP_BYTES {
            self.map.clear();
            self.bytes = 0;
        }
        self.bytes += cost;
        Ok(self.map.entry(Box::from(value)).or_insert(computed))
    }

    /// Whether the cache is off (for tests).
    #[cfg(test)]
    pub(super) fn is_off(&self) -> bool {
        self.off > 0
    }

    /// Values cached (for tests).
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.map.len()
    }
}

thread_local! {
    static MEMOS: RefCell<Vec<Memo>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with this thread's cache for `function` with the constants `key`, or with a fresh
/// cache of its own when this thread's caches are in use (a nested call).
pub(super) fn with_memo<R>(function: &'static str, key: &str, f: impl FnOnce(&mut Memo) -> R) -> R {
    MEMOS.with(|memos| {
        let Ok(mut memos) = memos.try_borrow_mut() else {
            return f(&mut Memo::new((function, key.to_owned())));
        };
        let at = memos
            .iter()
            .position(|m| m.key.0 == function && m.key.1 == key);
        let mut memo = match at {
            Some(at) => memos.remove(at),
            None => Memo::new((function, key.to_owned())),
        };
        let out = f(&mut memo);
        memos.truncate(CACHES - 1);
        memos.insert(0, memo);
        out
    })
}

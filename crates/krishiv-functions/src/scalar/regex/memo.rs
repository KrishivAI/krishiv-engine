// SPDX-License-Identifier: Apache-2.0
//! A batch's regex results by input value, so a value repeated in the batch is matched once
//! (perf C-1a, `krishiv-internal/research/perf-parity/c1-step0/README.md`).
//!
//! The regex family's kernels cost about a microsecond a row (the capture engine walks the
//! whole match), while hashing a value costs tens of nanoseconds. ClickBench's `Referer` repeats
//! a value about five times in a 8,192-row batch, so the memo evaluates a fifth of the rows. It
//! is used only where the result depends on the value alone (pattern, replacement, index and
//! position are constants of the batch). The first row of each value is evaluated in row order,
//! so the first failing row raises exactly as without the memo.
//!
//! A batch whose first [`PROBE_ROWS`] values are mostly distinct stops memoising: after that it
//! pays nothing but those rows' probes. Prior art: Velox evaluates over a dictionary's distinct
//! base values; ClickHouse runs functions on a `LowCardinality` dictionary.

use std::collections::HashMap;

use krishiv_common::error::KrishivError;

/// Rows looked up before the memo decides whether the batch repeats enough.
pub(super) const PROBE_ROWS: usize = 1024;

/// The memo of one batch: each distinct value's result, in first-seen order.
pub(super) struct Memo<'a, T> {
    /// Where each value's result is in `values`.
    index: HashMap<&'a str, u32, foldhash::fast::RandomState>,
    /// The results, by first appearance.
    values: Vec<T>,
    /// Rows looked up so far.
    rows: usize,
    /// Off once the probe found the batch mostly distinct: every row is evaluated.
    off: bool,
    /// The last result evaluated while off.
    scratch: Option<T>,
}

impl<'a, T> Memo<'a, T> {
    /// An empty memo for a batch of `rows` rows.
    pub(super) fn new(rows: usize) -> Self {
        let off = rows < 2;
        let cap = if off { 0 } else { rows.min(PROBE_ROWS) };
        Self {
            index: HashMap::with_capacity_and_hasher(cap, foldhash::fast::RandomState::default()),
            values: Vec::with_capacity(cap),
            rows: 0,
            off,
            scratch: None,
        }
    }

    /// The result for `value`: `eval` runs on the value's first row only.
    ///
    /// # Errors
    /// What `eval` returns for the value.
    pub(super) fn get(
        &mut self,
        value: &'a str,
        eval: impl FnOnce(&str) -> Result<T, KrishivError>,
    ) -> Result<&T, KrishivError> {
        if self.off {
            return Ok(self.scratch.insert(eval(value)?));
        }
        self.rows += 1;
        if self.rows == PROBE_ROWS && self.index.len() * 5 > PROBE_ROWS * 4 {
            // Over four values in five distinct: the memo would cost more than it saves.
            self.off = true;
            self.index = HashMap::default();
            self.values = Vec::new();
            return Ok(self.scratch.insert(eval(value)?));
        }
        let slot = match self.index.get(value) {
            Some(&at) => at,
            None => {
                let computed = eval(value)?;
                let at = u32::try_from(self.values.len()).map_err(|_| internal_rows())?;
                self.values.push(computed);
                self.index.insert(value, at);
                at
            }
        };
        self.values.get(slot as usize).ok_or_else(internal_rows)
    }

    /// Whether the memo stopped (for tests).
    #[cfg(test)]
    pub(super) fn is_off(&self) -> bool {
        self.off
    }
}

fn internal_rows() -> KrishivError {
    super::internal("regex memo: result slot out of range".to_owned())
}

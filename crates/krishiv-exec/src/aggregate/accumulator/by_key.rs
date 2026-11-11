// SPDX-License-Identifier: Apache-2.0
//! Which row wins a `max_by`/`min_by` key comparison, ties included.

use super::{Col, Values, float_lt};
use crate::error::{Result, internal};

impl Values {
    /// Whether row `i` of `src` replaces group `g`'s value: Spark's
    /// `MaxBy`/`MinBy` keep the held row only while it orders strictly after
    /// (`max`) or before the new one, so a tie takes the new row (the last
    /// of equal keys; `-0.0` equals `0.0`, NaN equals NaN).
    pub(super) fn beats(
        &self,
        g: usize,
        src: &Col<'_>,
        i: usize,
        max: bool,
        first_wins: bool,
    ) -> Result<bool> {
        let takes = |o: std::cmp::Ordering| match (max, first_wins) {
            (true, false) => o.is_ge(),
            (false, false) => o.is_le(),
            (true, true) => o.is_gt(),
            (false, true) => o.is_lt(),
        };
        Ok(match (self, src) {
            (Self::Int(v), Col::Int(c)) => takes(c[i].cmp(&v[g])),
            (Self::Float(v), Col::Float(c)) => {
                !float_lt(if max { c[i] } else { v[g] }, if max { v[g] } else { c[i] })
            }
            (Self::Bytes(v), Col::Bytes(c)) => takes(c[i].cmp(&v[g][..])),
            _ => {
                return Err(internal(
                    "an ordering key of another kind than its state".to_owned(),
                ));
            }
        })
    }
}

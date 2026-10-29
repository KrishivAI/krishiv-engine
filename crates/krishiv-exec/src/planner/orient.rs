// SPDX-License-Identifier: Apache-2.0
//! Which input of a hash join builds the table (design P2-T166).
//!
//! The planner builds the right input by default, as Spark's hash joins
//! do. A semi, anti or left outer join whose right input is much larger
//! than its left is better built on the left input, as a right semi, right
//! anti or right outer join: the probe marks the build rows it matches and
//! the join emits the build rows by their marks at the end. The planner
//! then plans that join, and `EXPLAIN` shows the statistics that chose it
//! as `builds left (left: ...; right: ...)`: `left` and `right` name the
//! join's logical inputs (the left one is now built), not its probe and
//! build sides. The decision is
//! a pure function of the two inputs' statistics ([`InputStats`]), so the
//! same function serves plan-time estimates now and statistics merged from
//! materialized exchanges later; it never reads process state, so every
//! task of a stage that calls it with the same statistics agrees.
//!
//! - A null-aware anti join (`NOT IN`) is never reoriented: it reads
//!   whether the whole subquery side holds a NULL key.
//! - Existence joins are never reoriented: their output is one row per
//!   probe row (DuckDB's mark join has no inverse either).
//! - A single-match left outer join (a scalar subquery) is never
//!   reoriented: its at-most-one-match check counts the matches of each
//!   probe row, which a build-side outer join does not see.
//! - Inner joins keep the join orderer's choice.
//! - A build the planner would broadcast (at most
//!   [`BROADCAST_MAX_BYTES`](super::join::BROADCAST_MAX_BYTES)) stays.
//! - Otherwise the left input is built when the right one costs more than
//!   [`STATS_FACTOR`] times as much (both sides estimated from table
//!   statistics), or [`DEFAULT_FACTOR`] times (either side from default
//!   estimates). A tie keeps the right input.

use crate::join::JoinType;

use super::join::BROADCAST_MAX_BYTES;

/// How much larger the right input's table must be to build the left one,
/// when both estimates rest on table statistics.
pub(crate) const STATS_FACTOR: f64 = 2.0;

/// The same factor when either estimate rests on defaults.
pub(crate) const DEFAULT_FACTOR: f64 = 8.0;

/// Where an input's estimate comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StatsSource {
    /// Some table below has no statistics: default row counts.
    Default,
    /// Every table below has statistics (row counts, column statistics).
    Statistics,
}

/// What is known of one join input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InputStats {
    /// Estimated rows.
    pub(crate) rows: f64,
    /// Estimated bytes of the hash table built from this input.
    pub(crate) bytes: f64,
    /// Where the estimate comes from.
    pub(crate) source: StatsSource,
}

impl InputStats {
    /// `rows / MiB, source` for `EXPLAIN`.
    pub(crate) fn describe(&self) -> String {
        let source = match self.source {
            StatsSource::Default => "default",
            StatsSource::Statistics => "statistics",
        };
        format!(
            "{:.0} rows / {:.1} MiB, {source}",
            self.rows,
            self.bytes / (1024.0 * 1024.0)
        )
    }
}

/// The join a hash join with a right build becomes when it builds its left
/// input instead, if it can; `single_match` is
/// [`HashJoinPlan::single_match`](crate::physical::HashJoinPlan::single_match).
pub(crate) fn flipped(t: JoinType, single_match: bool) -> Option<JoinType> {
    if single_match {
        return None;
    }
    match t {
        JoinType::LeftSemi => Some(JoinType::RightSemi),
        JoinType::LeftAnti => Some(JoinType::RightAnti),
        JoinType::LeftOuter => Some(JoinType::RightOuter),
        JoinType::Inner
        | JoinType::NullAwareAnti
        | JoinType::Existence
        | JoinType::RightSemi
        | JoinType::RightAnti
        | JoinType::RightOuter => None,
    }
}

/// The join type of a hash join whose right input builds by default, given
/// its inputs: `join_type` itself (build right), or the variant that builds
/// the left input.
pub(crate) fn orient(
    join_type: JoinType,
    single_match: bool,
    left: &InputStats,
    right: &InputStats,
) -> JoinType {
    let Some(other) = flipped(join_type, single_match) else {
        return join_type;
    };
    if right.bytes <= BROADCAST_MAX_BYTES {
        return join_type;
    }
    let factor =
        if left.source == StatsSource::Statistics && right.source == StatsSource::Statistics {
            STATS_FACTOR
        } else {
            DEFAULT_FACTOR
        };
    // NaN compares false: no flip.
    if right.bytes > factor * left.bytes {
        other
    } else {
        join_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: f64 = 1024.0 * 1024.0;

    fn stats(bytes: f64, source: StatsSource) -> InputStats {
        InputStats {
            rows: bytes / 32.0,
            bytes,
            source,
        }
    }

    fn known(bytes: f64) -> InputStats {
        stats(bytes, StatsSource::Statistics)
    }

    #[test]
    fn a_much_larger_right_input_flips_semi_anti_and_left_outer_joins() {
        let (small, large) = (known(20.0 * MIB), known(500.0 * MIB));
        for (t, want) in [
            (JoinType::LeftSemi, JoinType::RightSemi),
            (JoinType::LeftAnti, JoinType::RightAnti),
            (JoinType::LeftOuter, JoinType::RightOuter),
        ] {
            assert_eq!(orient(t, false, &small, &large), want, "{t:?}");
            // The other way round keeps the right build.
            assert_eq!(orient(t, false, &large, &small), t, "{t:?}");
        }
    }

    #[test]
    fn inner_null_aware_and_flipped_joins_keep_their_build() {
        let (small, large) = (known(1.0 * MIB), known(5000.0 * MIB));
        for t in [
            JoinType::Inner,
            JoinType::NullAwareAnti,
            JoinType::Existence,
            JoinType::RightSemi,
            JoinType::RightAnti,
            JoinType::RightOuter,
        ] {
            assert_eq!(orient(t, false, &small, &large), t, "{t:?}");
            assert_eq!(flipped(t, false), None, "{t:?}");
        }
    }

    /// A single-match left outer join (a scalar subquery) keeps its right
    /// build however lopsided its inputs: a right outer join cannot check
    /// that each probe row matches at most once.
    #[test]
    fn a_single_match_left_outer_join_keeps_its_build() {
        let (small, large) = (known(1.0 * MIB), known(5000.0 * MIB));
        assert_eq!(
            orient(JoinType::LeftOuter, false, &small, &large),
            JoinType::RightOuter
        );
        assert_eq!(
            orient(JoinType::LeftOuter, true, &small, &large),
            JoinType::LeftOuter
        );
        assert_eq!(flipped(JoinType::LeftOuter, true), None);
    }

    #[test]
    fn a_broadcastable_right_input_stays_the_build() {
        let left = known(1.0);
        assert_eq!(
            orient(
                JoinType::LeftSemi,
                false,
                &left,
                &known(BROADCAST_MAX_BYTES)
            ),
            JoinType::LeftSemi
        );
        assert_eq!(
            orient(
                JoinType::LeftSemi,
                false,
                &left,
                &known(BROADCAST_MAX_BYTES + 1.0)
            ),
            JoinType::RightSemi
        );
    }

    #[test]
    fn the_factor_depends_on_where_the_estimates_come_from() {
        let right = known(100.0 * MIB);
        // Exactly the factor is a tie: the right input stays the build.
        assert_eq!(
            orient(JoinType::LeftAnti, false, &known(50.0 * MIB), &right),
            JoinType::LeftAnti
        );
        assert_eq!(
            orient(JoinType::LeftAnti, false, &known(49.0 * MIB), &right),
            JoinType::RightAnti
        );
        // A default estimate on either side needs a ratio above 8.
        for (l, r) in [
            (StatsSource::Default, StatsSource::Statistics),
            (StatsSource::Statistics, StatsSource::Default),
        ] {
            let right = stats(100.0 * MIB, r);
            assert_eq!(
                orient(JoinType::LeftSemi, false, &stats(20.0 * MIB, l), &right),
                JoinType::LeftSemi
            );
            assert_eq!(
                orient(JoinType::LeftSemi, false, &stats(12.0 * MIB, l), &right),
                JoinType::RightSemi
            );
        }
    }

    #[test]
    fn unusable_estimates_keep_the_right_build() {
        for bad in [f64::NAN, f64::INFINITY] {
            let left = known(bad);
            assert_eq!(
                orient(JoinType::LeftSemi, false, &left, &known(100.0 * MIB)),
                JoinType::LeftSemi
            );
        }
        assert_eq!(
            orient(JoinType::LeftSemi, false, &known(1.0), &known(f64::NAN)),
            JoinType::LeftSemi
        );
    }

    /// The decision is a function of its inputs only: any order of calls
    /// over the same inputs gives the same answers.
    #[test]
    fn orientation_is_deterministic() {
        let sources = [StatsSource::Default, StatsSource::Statistics];
        let types = [
            JoinType::Inner,
            JoinType::LeftOuter,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::NullAwareAnti,
        ];
        let mut cases = Vec::new();
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        for i in 0..512 {
            // A fixed xorshift sequence of sizes from 1 B to ~1 TiB.
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            #[expect(clippy::cast_precision_loss, reason = "test sizes")]
            let (a, b) = ((x % (1 << 40)) as f64, ((x >> 20) % (1 << 40)) as f64);
            let t = types[i % types.len()];
            let l = stats(a, sources[i % 2]);
            let r = stats(b, sources[(i / 2) % 2]);
            cases.push((t, l, r));
        }
        let first: Vec<_> = cases
            .iter()
            .map(|(t, l, r)| orient(*t, false, l, r))
            .collect();
        let reversed: Vec<_> = cases
            .iter()
            .rev()
            .map(|(t, l, r)| orient(*t, false, l, r))
            .collect();
        assert!(first.iter().eq(reversed.iter().rev()));
        assert!(first.iter().any(|t| matches!(
            t,
            JoinType::RightSemi | JoinType::RightAnti | JoinType::RightOuter
        )));
        assert!(first.iter().any(|t| matches!(
            t,
            JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftOuter
        )));
    }
}

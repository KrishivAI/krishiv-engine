// SPDX-License-Identifier: Apache-2.0
//! [`PlanStats`]: the estimates of every relation of an optimized plan,
//! computed in one pass.
//!
//! The table is keyed by relation (pointer identity, so a shared subtree has
//! one entry) and holds what [`Estimator`] says of it: rows, and for each
//! output attribute its distinct count, null fraction, bounds and the
//! [`StatSource`](crate::StatSource) of the count. Subquery plans and `WITH`
//! definitions have entries too. Planning cost is linear in the plan: each
//! relation is estimated once, however many consumers ask. The consumers (the
//! physical planner's join and runtime-filter costing, `EXPLAIN COST`) look
//! relations up and never estimate again.
//!
//! The numbers are estimates only: they steer choices that are already plan
//! data (a build side, a broadcast), and never change what a query returns.
//! They are planning-time data: no plan fragment carries them.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use krishiv_plan::{ExprNode, LogicalPlan};
use krishiv_types::DataType;

use crate::Options;
use crate::estimate::Estimator;
use crate::stats::{DEFAULT_ROWS, RelStats, Statistics, to_f64};

fn addr(plan: &LogicalPlan) -> usize {
    std::ptr::from_ref(plan).addr()
}

/// The estimates of the relations of one plan.
#[derive(Debug, Default)]
pub struct PlanStats {
    /// The relation is kept alive so its address stays unique.
    nodes: HashMap<usize, (Arc<LogicalPlan>, Arc<RelStats>)>,
    computed: usize,
}

impl PlanStats {
    /// The estimates of `root` and every relation below it, subquery plans
    /// included, from the tables' statistics.
    #[must_use]
    pub fn compute(root: &Arc<LogicalPlan>, stats: &dyn Statistics) -> Self {
        Self::compute_with_options(root, stats, &Options::default())
    }

    /// As [`Self::compute`], with the optimizer's per-rule switches
    /// (`stats_propagation: false` scales row counts only: the estimates
    /// before propagation, which the q-error report compares against).
    #[must_use]
    pub fn compute_with_options(
        root: &Arc<LogicalPlan>,
        stats: &dyn Statistics,
        options: &Options,
    ) -> Self {
        let propagate = options.stats_propagation;
        // Estimation recurses once per level of the plan.
        krishiv_common::stack::on_big_stack(|| Self::build(root, stats, propagate))
            .unwrap_or_else(|| Self::build(root, stats, propagate))
    }

    fn build(root: &Arc<LogicalPlan>, stats: &dyn Statistics, propagate: bool) -> Self {
        let mut estimator = Estimator::with_propagation(stats, propagate);
        let mut seen = HashSet::new();
        let mut plans = vec![Arc::clone(root)];
        let mut nodes = HashMap::new();
        while let Some(p) = plans.pop() {
            if !seen.insert(addr(&p)) {
                continue;
            }
            let est = estimator.plan(&p);
            p.for_each_child(&mut |c| plans.push(Arc::clone(c)));
            subquery_plans(&p, &mut |s| plans.push(Arc::clone(s)));
            nodes.insert(addr(&p), (p, est));
        }
        Self {
            nodes,
            computed: estimator.computed(),
        }
    }

    /// The estimate of `plan`, when it is a relation of the plan this was
    /// computed for.
    #[must_use]
    pub fn get(&self, plan: &LogicalPlan) -> Option<&RelStats> {
        self.nodes.get(&addr(plan)).map(|(_, e)| &**e)
    }

    /// The rows cost decisions read for `plan`: at least 1;
    /// [`DEFAULT_ROWS`] for a relation this table does not hold.
    #[must_use]
    pub fn rows(&self, plan: &LogicalPlan) -> f64 {
        self.get(plan)
            .map_or(DEFAULT_ROWS, RelStats::rows_for_costing)
    }

    /// Whether every table `plan` reads reports statistics, so its estimate
    /// does not rest on default row counts. A relation this table does not
    /// hold has none.
    #[must_use]
    pub fn has_statistics(&self, plan: &LogicalPlan) -> bool {
        self.get(plan).is_some_and(|e| e.from_statistics)
    }

    /// How many relations were estimated: one per relation of the plan.
    #[must_use]
    pub fn computed(&self) -> usize {
        self.computed
    }

    /// The number of relations the table holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the table holds no relation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// `Statistics(sizeInBytes=.., rowCount=..)` of `plan` as Spark's
    /// `EXPLAIN COST` prints it: the size is the rows times 8 bytes plus
    /// the default size of each output column (at least 1 byte); the row
    /// count has three significant digits.
    #[must_use]
    pub fn describe(&self, plan: &LogicalPlan) -> Option<String> {
        let est = self.get(plan)?;
        let width = plan
            .resolved_output()
            .unwrap_or_default()
            .iter()
            .map(|c| default_size(&c.data_type))
            .fold(8, u64::saturating_add);
        let rows = est.rows.max(0.0).ceil();
        let size = (rows * to_f64(width)).max(1.0);
        let mut out = String::from("Statistics(sizeInBytes=");
        out.push_str(&bytes_to_string(size));
        let _ = write!(out, ", rowCount={})", row_count(rows));
        Some(out)
    }
}

/// The plans of the subquery expressions in `p`'s own expressions.
fn subquery_plans<'a>(p: &'a LogicalPlan, f: &mut impl FnMut(&'a Arc<LogicalPlan>)) {
    let mut stack = Vec::new();
    p.for_each_expr(&mut |e| stack.push(e));
    let mut seen = HashSet::new();
    while let Some(e) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(&**e)) {
            continue;
        }
        if let ExprNode::Subquery { plan, .. } = &e.node {
            f(plan);
        }
        e.for_each_child(&mut |c| stack.push(c));
    }
}

/// Spark's `DataType.defaultSize`.
fn default_size(t: &DataType) -> u64 {
    match t {
        DataType::Null | DataType::Boolean | DataType::Byte => 1,
        DataType::Short => 2,
        DataType::Integer | DataType::Float | DataType::Date => 4,
        DataType::Decimal(d) if d.precision() > 18 => 16,
        DataType::String => 20,
        DataType::Binary => 100,
        DataType::Array(a) => default_size(a.element_type()),
        DataType::Map(m) => default_size(m.key_type()).saturating_add(default_size(m.value_type())),
        DataType::Struct(s) => s
            .fields()
            .iter()
            .map(|f| default_size(f.data_type()))
            .fold(0, u64::saturating_add),
        // Long, Double, timestamps and decimals of up to 18 digits.
        _ => 8,
    }
}

/// Spark's `Utils.bytesToString`: one decimal, in the largest unit the size
/// is at least twice of.
fn bytes_to_string(size: f64) -> String {
    const UNITS: [(&str, f64); 6] = [
        ("EiB", 1_152_921_504_606_846_976.0),
        ("PiB", 1_125_899_906_842_624.0),
        ("TiB", 1_099_511_627_776.0),
        ("GiB", 1_073_741_824.0),
        ("MiB", 1_048_576.0),
        ("KiB", 1024.0),
    ];
    for (unit, bytes) in UNITS {
        if size >= 2.0 * bytes {
            // Java's `%.1f` rounds half up.
            return format!("{:.1} {unit}", (size / bytes * 10.0).round() / 10.0);
        }
    }
    format!("{size:.1} B")
}

/// A row count as Spark prints it: `BigDecimal(n, MathContext(3,
/// HALF_UP)).toString`, plain up to three digits, else `d.ddE+n`.
fn row_count(rows: f64) -> String {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a non-negative, whole estimate"
    )]
    let n = rows as u128;
    let digits = n.to_string();
    if digits.len() <= 3 {
        return digits;
    }
    let (head, rest) = digits.split_at(3);
    let mut mantissa: u32 = head.parse().unwrap_or(0);
    let mut exponent = digits.len() - 1;
    if rest.as_bytes().first().is_some_and(|d| *d >= b'5') {
        mantissa += 1;
        if mantissa == 1000 {
            mantissa = 100;
            exponent += 1;
        }
    }
    format!("{}.{:02}E+{exponent}", mantissa / 100, mantissa % 100)
}

#[cfg(test)]
mod tests {
    use super::{bytes_to_string, row_count};

    #[test]
    fn sizes_print_as_spark_does() {
        for (size, want) in [
            (1.0, "1.0 B"),
            (16.0, "16.0 B"),
            (2047.0, "2047.0 B"),
            (2048.0, "2.0 KiB"),
            (3_145_728.0, "3.0 MiB"),
            (1_048_576.0 * 1_024.0 * 5.0, "5.0 GiB"),
        ] {
            assert_eq!(bytes_to_string(size), want);
        }
    }

    #[test]
    fn row_counts_keep_three_digits() {
        for (rows, want) in [
            (0.0, "0"),
            (1.0, "1"),
            (999.0, "999"),
            (1000.0, "1.00E+3"),
            (1500.0, "1.50E+3"),
            (12_345.0, "1.23E+4"),
            (999_500.0, "1.00E+6"),
            (6_001_215.0, "6.00E+6"),
        ] {
            assert_eq!(row_count(rows), want, "{rows}");
        }
    }
}

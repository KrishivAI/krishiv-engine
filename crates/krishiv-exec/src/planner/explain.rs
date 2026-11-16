// SPDX-License-Identifier: Apache-2.0
//! The printable tree of a physical plan (`EXPLAIN`'s physical section).

use std::fmt;

/// One node of a physical plan as `EXPLAIN` prints it: a name, its
/// details and its inputs. Printed in the style of the logical plan
/// (Spark's `treeString`): the first inputs under `:- `, the last under
/// `+- `.
///
/// Two nodes are equal when they print the same: the planner's
/// [`estimate`](Self::estimate) is not printed, so it is not compared (a
/// plan decoded from the plan codec, which does not carry it, still equals
/// the plan it was encoded from).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExplainNode {
    /// The operator, e.g. `HashJoin`.
    pub name: String,
    /// What it does, e.g. `Inner, [l_orderkey#0] = [o_orderkey#16]`.
    pub detail: String,
    /// Its inputs.
    pub children: Vec<ExplainNode>,
    /// The planner's estimate of the node's output rows: set on the node
    /// that produces a logical relation's rows, `None` on nodes the
    /// planner adds below it (exchanges, a partial aggregate) and on a plan
    /// decoded from the plan codec. The profile's `plan[].estimate`.
    pub estimate: Option<RowEstimate>,
}

impl PartialEq for ExplainNode {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.detail == other.detail && self.children == other.children
    }
}

impl Eq for ExplainNode {}

/// The planner's estimate of a node's output rows (T164's statistics, as
/// join ordering and the runtime-filter cost read them).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RowEstimate {
    /// Estimated output rows, rounded.
    pub rows: u64,
    /// Where the estimate comes from: [`RowEstimate::STATISTICS`] when every
    /// table under the node reports statistics, [`RowEstimate::DEFAULT`]
    /// when some table's row count is a default. A profile read from JSON
    /// keeps a string it does not know.
    pub source: String,
}

impl RowEstimate {
    /// An estimate of `rows` from `source`.
    pub fn new(rows: u64, source: impl Into<String>) -> Self {
        Self {
            rows,
            source: source.into(),
        }
    }

    /// Every table below the node reports statistics.
    pub const STATISTICS: &'static str = "statistics";
    /// Some table below the node has no statistics: its row count is a
    /// default.
    pub const DEFAULT: &'static str = "default";
}

impl ExplainNode {
    /// A node without inputs.
    pub fn leaf(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            detail: detail.into(),
            children: Vec::new(),
            estimate: None,
        }
    }

    /// A node over `children`.
    pub fn new(
        name: impl Into<String>,
        detail: impl Into<String>,
        children: Vec<ExplainNode>,
    ) -> Self {
        Self {
            name: name.into(),
            detail: detail.into(),
            children,
            estimate: None,
        }
    }

    fn write(&self, f: &mut fmt::Formatter<'_>, prefix: &mut String) -> fmt::Result {
        f.write_str(&self.name)?;
        if !self.detail.is_empty() {
            write!(f, " {}", self.detail)?;
        }
        f.write_str("\n")?;
        let last = self.children.len().saturating_sub(1);
        for (i, child) in self.children.iter().enumerate() {
            let (head, tail) = if i == last {
                ("+- ", "   ")
            } else {
                (":- ", ":  ")
            };
            write!(f, "{prefix}{head}")?;
            let len = prefix.len();
            prefix.push_str(tail);
            child.write(f, prefix)?;
            prefix.truncate(len);
        }
        Ok(())
    }
}

impl fmt::Display for ExplainNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write(f, &mut String::new())
    }
}

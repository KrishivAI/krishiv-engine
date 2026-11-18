// SPDX-License-Identifier: Apache-2.0
//! The printable tree of a physical plan (`EXPLAIN`'s physical section).

use std::fmt;
use std::sync::Arc;

/// What the planner knew about a node when it planned it, kept for the advisor (P3-T073): data about the
/// plan, never printed in `EXPLAIN` text and never sent through the plan codec (a decoded plan has none).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeFacts {
    /// The planner's estimate of the node's output rows, where it computed one.
    pub estimated_rows: Option<f64>,
    /// A join's inputs, probe first, build second.
    pub inputs: Vec<InputFacts>,
    /// Hash-join keys that got no runtime filter, with the reason.
    pub filter_gaps: Vec<FilterGap>,
    /// A scan: the table it reads when the provider reports no statistics.
    pub tables_without_stats: Vec<String>,
}

/// One join input's planner facts.
#[derive(Clone, Debug, PartialEq)]
pub struct InputFacts {
    /// Estimated rows.
    pub estimated_rows: f64,
    /// Whether every table below has statistics.
    pub statistics: bool,
    /// The tables below with none.
    pub tables_without_stats: Vec<String>,
}

/// A hash-join key that publishes no runtime filter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterGap {
    /// The key's index.
    pub key: usize,
    /// The probe key expression, as `EXPLAIN` prints it.
    pub probe_key: String,
    /// Why there is no filter.
    pub reason: FilterGapReason,
}

/// Why a join key has no runtime filter.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FilterGapReason {
    /// `krishiv.runtimeFilter.enabled` is false.
    Disabled,
    /// The key is computed: no scan column is its source.
    NoScanColumn,
    /// The key is a scan column, but an operator between the join and the scan stops the filter.
    Blocked,
}

/// One node of a physical plan as `EXPLAIN` prints it: a name, its
/// details and its inputs. Printed in the style of the logical plan
/// (Spark's `treeString`): the first inputs under `:- `, the last under
/// `+- `.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExplainNode {
    /// The operator, e.g. `HashJoin`.
    pub name: String,
    /// What it does, e.g. `Inner, [l_orderkey#0] = [o_orderkey#16]`.
    pub detail: String,
    /// Its inputs.
    pub children: Vec<ExplainNode>,
    /// What the planner knew (ignored by equality: a decoded plan has none).
    pub facts: Option<Arc<NodeFacts>>,
}

impl PartialEq for ExplainNode {
    fn eq(&self, o: &Self) -> bool {
        self.name == o.name && self.detail == o.detail && self.children == o.children
    }
}

impl Eq for ExplainNode {}

impl ExplainNode {
    /// A node without inputs.
    pub fn leaf(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            detail: detail.into(),
            children: Vec::new(),
            facts: None,
        }
    }

    /// This node with planner `facts`.
    #[must_use]
    pub fn with_facts(mut self, facts: NodeFacts) -> Self {
        self.facts = Some(Arc::new(facts));
        self
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
            facts: None,
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

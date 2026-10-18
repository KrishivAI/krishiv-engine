// SPDX-License-Identifier: Apache-2.0
//! The printable tree of a physical plan (`EXPLAIN`'s physical section).

use std::fmt;

/// One node of a physical plan as `EXPLAIN` prints it: a name, its
/// details and its inputs. Printed in the style of the logical plan
/// (Spark's `treeString`): the first inputs under `:- `, the last under
/// `+- `.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExplainNode {
    /// The operator, e.g. `HashJoin`.
    pub name: String,
    /// What it does, e.g. `Inner, [l_orderkey#0] = [o_orderkey#16]`.
    pub detail: String,
    /// Its inputs.
    pub children: Vec<ExplainNode>,
}

impl ExplainNode {
    /// A node without inputs.
    pub fn leaf(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            detail: detail.into(),
            children: Vec::new(),
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

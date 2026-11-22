// SPDX-License-Identifier: Apache-2.0
//! Generators in DataFrame select lists: the lifting itself is [`krishiv_plan::select_with_generators`]
//! and [`krishiv_plan::with_column_generator`], shared with the Connect translation. It also holds
//! the set operations, `limit`, `offset`, `sort` and `hint`.

use std::sync::Arc;

use krishiv_plan::{Expr, Literal, Origin, PlanNode, SetOperation};

use super::DataFrame;
use crate::column::Column;

impl DataFrame {
    fn next_origin(&self) -> Origin {
        Origin::plan(self.session.next_plan_id())
    }

    /// `SELECT items`, the generator calls among them lifted into `Generate` nodes.
    pub(super) fn select_lifting_generators(&self, items: Vec<Arc<Expr>>) -> Self {
        self.derive(krishiv_plan::select_with_generators(
            Arc::clone(&self.plan),
            items,
            || self.next_origin(),
        ))
    }

    /// `withColumn(name, value)` for a generator `value`, `None` for any other.
    pub(super) fn with_column_generator(&self, name: &str, value: &Arc<Expr>) -> Option<Self> {
        let node: PlanNode =
            krishiv_plan::with_column_generator(&self.plan, name, value, || self.next_origin())?;
        Some(self.derive(node))
    }
}

impl DataFrame {
    /// Sorts each partition by `keys` without ordering the partitions (PySpark's
    /// `sortWithinPartitions`, SQL's `SORT BY`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").sort_within_partitions(["a"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn sort_within_partitions(
        &self,
        keys: impl IntoIterator<Item = impl Into<Column>>,
    ) -> Self {
        self.derive(PlanNode::Sort {
            input: Arc::clone(&self.plan),
            order: keys.into_iter().map(|k| k.into().sort_order()).collect(),
            global: false,
        })
    }

    /// The same as [`Self::order_by`] (PySpark's `sort`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").sort(["a"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn sort(&self, keys: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        self.order_by(keys)
    }

    /// The first `n` rows (PySpark's `limit`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").limit(10);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn limit(&self, n: i32) -> Self {
        self.derive(PlanNode::Limit {
            input: Arc::clone(&self.plan),
            limit: Arc::new(Expr::literal(Literal::Integer(n))),
        })
    }

    /// This DataFrame without its first `n` rows (PySpark's `offset`);
    /// `df.offset(m).limit(n)` is `LIMIT n OFFSET m`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").offset(10).limit(5);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn offset(&self, n: i32) -> Self {
        self.derive(PlanNode::Offset {
            input: Arc::clone(&self.plan),
            offset: Arc::new(Expr::literal(Literal::Integer(n))),
        })
    }

    /// The distinct rows (PySpark's `distinct`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").distinct();
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn distinct(&self) -> Self {
        self.derive(PlanNode::Distinct {
            input: Arc::clone(&self.plan),
        })
    }

    /// Every row of this DataFrame and `other`, columns matched by
    /// position (PySpark's `union` and `unionAll`; SQL's `UNION ALL`).
    /// Column types widen per position; a different column count fails
    /// analysis.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").union(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        self.pair(
            other,
            PlanNode::Union {
                inputs: vec![Arc::clone(&self.plan), Arc::clone(&other.plan)],
                by_name: false,
                allow_missing_columns: false,
            },
        )
    }

    /// Every row of this DataFrame and `other`, columns matched by name
    /// (PySpark's `unionByName`): `other`'s columns are reordered to this
    /// frame's, and with `allow_missing_columns` a column one side lacks
    /// reads NULL there.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").union_by_name(&s.table("r"), true);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn union_by_name(&self, other: &Self, allow_missing_columns: bool) -> Self {
        self.pair(
            other,
            PlanNode::Union {
                inputs: vec![Arc::clone(&self.plan), Arc::clone(&other.plan)],
                by_name: true,
                allow_missing_columns,
            },
        )
    }

    /// The rows of this DataFrame the `other` one has too, without
    /// duplicates (PySpark's `intersect`; SQL's `INTERSECT`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").intersect(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        self.set_op(other, SetOperation::Intersect, false)
    }

    /// The rows of this DataFrame the `other` one has too, duplicates kept
    /// (PySpark's `intersectAll`; SQL's `INTERSECT ALL`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").intersect_all(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn intersect_all(&self, other: &Self) -> Self {
        self.set_op(other, SetOperation::Intersect, true)
    }

    /// The rows of this DataFrame the `other` one lacks, without
    /// duplicates (PySpark's `subtract`; SQL's `EXCEPT`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").subtract(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn subtract(&self, other: &Self) -> Self {
        self.set_op(other, SetOperation::Except, false)
    }

    /// The rows of this DataFrame the `other` one lacks, duplicates kept
    /// (PySpark's `exceptAll`; SQL's `EXCEPT ALL`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").except_all(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn except_all(&self, other: &Self) -> Self {
        self.set_op(other, SetOperation::Except, true)
    }

    /// This DataFrame with the hint `name` and `params` (PySpark's
    /// `hint`): a join strategy hint wraps the relations `params` name
    /// (columns and string literals name relations); an unknown hint and
    /// a hint naming no relation come back as warnings, and a
    /// partitioning hint (`COALESCE`, `REPARTITION`, ...) is checked and
    /// otherwise dropped: a single-node plan has no partitions to set.
    /// The hint never changes the rows.
    ///
    /// ```
    /// # use krishiv_session::functions::lit;
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").hint("BROADCAST", [lit("t")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn hint(&self, name: &str, params: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        self.derive(PlanNode::Hint {
            input: Arc::clone(&self.plan),
            name: name.to_owned(),
            params: params
                .into_iter()
                .map(|c| Arc::new(c.into().expr))
                .collect(),
        })
    }

    /// This DataFrame with its columns renamed to `names`, in order
    /// (PySpark's `toDF`; SQL's `AS (columns)`). A different count fails
    /// analysis.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").to_df(["x", "y"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn to_df(&self, names: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        self.derive(PlanNode::SubqueryAlias {
            input: Arc::clone(&self.plan),
            alias: String::new(),
            columns: names.into_iter().map(|n| n.as_ref().to_owned()).collect(),
        })
    }
}

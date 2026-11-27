// SPDX-License-Identifier: Apache-2.0
//! Connect relations to IR relations, after Spark's
//! `SparkConnectPlanner.transformRelation`.
//!
//! Supported: `Read` (named tables and data sources), `SQL`, `Project`,
//! `Filter`, `Join`, `Sort`, `Limit`, `Aggregate`, `SubqueryAlias`,
//! `WithColumns`, `Range` and the Python UDF relations (`relation_udf`).
//! The `plan_id` of a relation's `RelationCommon` becomes the `Origin` of
//! the node it translates to, as Spark tags the plan it builds for the
//! relation; `df["a"]` references name that id.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::secret::OptionValue;
use krishiv_plan::{
    ALIAS_PART_SEP, AsOfDirection, AsOfMatch, Builtin, CastMode, ColumnNamesOp, Expr, ExprNode,
    Grouping, JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PivotValue, PlanId, PlanNode,
    ReadSource, SampleBounds, SetOperation, SqlArgs, UnpivotColumnSet,
};

use super::errors::{
    assertion, expression_not_set, invalid_enum, not_set, relation_not_set, unsupported,
    unsupported_field, unsupported_variant,
};
use super::names::{parse_attribute_name, parse_multipart_identifier};
use super::{Result, Translator};
use crate::extension::{Extension, Slot, resolve};
use crate::proto::v4_3::{self as pb, Expression};
use crate::proto::v4_3::{
    Relation, aggregate::GroupType, join::JoinType, read::ReadType, relation::RelType,
};

impl Translator<'_> {
    /// Translates a relation and everything below it.
    pub(crate) fn relation(&mut self, rel: &Relation) -> Result<Arc<LogicalPlan>> {
        self.count()?;
        self.nested(|t| t.relation_inner(rel))
    }

    fn relation_inner(&mut self, rel: &Relation) -> Result<Arc<LogicalPlan>> {
        let Some(kind) = &rel.rel_type else {
            return Err(relation_not_set());
        };
        let node = match kind {
            RelType::Read(r) => self.read(r)?,
            RelType::Sql(s) => self.sql(s)?,
            RelType::Project(p) => {
                // Spark reads a missing input as `OneRowRelation`.
                let input = match &p.input {
                    Some(input) => self.relation(input)?,
                    None => Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
                };
                // Generators in the list are lifted as the native DataFrame API does.
                krishiv_plan::select_with_generators(
                    input,
                    self.expressions(&p.expressions)?,
                    || Origin::UNKNOWN,
                )
            }
            RelType::Filter(f) => PlanNode::Filter {
                input: self.input("Filter", f.input.as_deref())?,
                predicate: self.required_expression(f.condition.as_ref())?,
            },
            RelType::Join(j) => self.join(j)?,
            RelType::AsOfJoin(j) => self.as_of_join(j)?,
            RelType::Sort(s) => {
                if s.order.is_empty() {
                    return Err(assertion("'order' must be present and contain elements."));
                }
                PlanNode::Sort {
                    input: self.input("Sort", s.input.as_deref())?,
                    order: s
                        .order
                        .iter()
                        .map(|o| self.sort_order(o))
                        .collect::<Result<_>>()?,
                    global: s.is_global.unwrap_or(false),
                }
            }
            RelType::Limit(l) => PlanNode::Limit {
                input: self.input("Limit", l.input.as_deref())?,
                limit: Arc::new(Expr::literal(Literal::Integer(l.limit))),
            },
            RelType::Offset(o) => PlanNode::Offset {
                input: self.input("Offset", o.input.as_deref())?,
                offset: Arc::new(Expr::literal(Literal::Integer(o.offset))),
            },
            // A negative count is the analyzer's `INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE`.
            RelType::Tail(t) => PlanNode::Tail {
                input: self.input("Tail", t.input.as_deref())?,
                limit: Arc::new(Expr::literal(Literal::Integer(t.limit))),
            },
            RelType::Aggregate(a) => self.aggregate(a)?,
            RelType::SubqueryAlias(a) => PlanNode::SubqueryAlias {
                input: self.input("SubqueryAlias", a.input.as_deref())?,
                alias: alias_name(&a.qualifier, &a.alias)?,
                columns: Vec::new(),
            },
            RelType::WithColumns(w) => self.with_columns(w)?,
            RelType::Range(r) => self.range(r)?,
            RelType::SetOp(s) => self.set_op(s, origin_of(rel))?,
            RelType::MapPartitions(m) => self.map_partitions(m)?,
            RelType::GroupMap(g) => self.group_map(g)?,
            RelType::CoGroupMap(g) => self.cogroup_map(g)?,
            RelType::LocalRelation(l) => {
                let opts = self.conf.parser;
                PlanNode::Read(ReadSource::Local(
                    self.local_relation(|| super::local::inline_request(l, &opts))?,
                ))
            }
            RelType::CachedLocalRelation(c) => {
                PlanNode::Read(ReadSource::Local(self.local_relation(|| {
                    Ok(super::LocalRequest::Cached {
                        hash: c.hash.clone(),
                    })
                })?))
            }
            RelType::Catalog(c) => {
                let call = super::catalog::catalog_call(c)?;
                PlanNode::Read(ReadSource::Local(
                    self.local_relation(|| Ok(super::LocalRequest::Catalog(call)))?,
                ))
            }
            RelType::CachedRemoteRelation(c) => {
                self.conf.cached_remote.record(&c.relation_id);
                PlanNode::Read(ReadSource::Table {
                    name: super::cached_remote_table(&c.relation_id),
                    options: std::collections::BTreeMap::new(),
                })
            }
            RelType::CollectMetrics(c) => PlanNode::CollectMetrics {
                input: self.input("CollectMetrics", c.input.as_deref())?,
                name: c.name.clone(),
                metrics: self.expressions(&c.metrics)?,
            },
            RelType::Sample(s) => self.sample(s)?,
            RelType::Deduplicate(d) => self.deduplicate(d)?,
            RelType::Hint(h) => PlanNode::Hint {
                input: self.input("Hint", h.input.as_deref())?,
                name: h.name.clone(),
                params: self.expressions(&h.parameters)?,
            },
            RelType::Cov(c) => {
                self.stat_pair("StatCov", c.input.as_deref(), &c.col1, &c.col2, "cov")?
            }
            RelType::Corr(c) => {
                if let Some(m) = c.method.as_deref()
                    && m != "pearson"
                {
                    return Err(unsupported_field(
                        "StatCorr",
                        "method",
                        "a correlation method other than pearson",
                    ));
                }
                self.stat_pair("StatCorr", c.input.as_deref(), &c.col1, &c.col2, "corr")?
            }
            RelType::SampleBy(s) => self.sample_by(s)?,
            RelType::FillNa(n) => self.na_fill(n)?,
            RelType::Replace(n) => self.na_replace(n)?,
            RelType::DropNa(n) => {
                // A count of 0 or less keeps every row, as `AtLeastNNonNulls(0 or less)` does.
                let min = n.min_non_nulls.map(|m| m.max(0).unsigned_abs());
                PlanNode::ColumnNames {
                    input: self.input("NaDrop", n.input.as_deref())?,
                    op: ColumnNamesOp::DropNa {
                        subset: n.cols.clone(),
                        min_non_nulls: min,
                    },
                }
            }
            RelType::Repartition(r) => PlanNode::Hint {
                input: self.input("Repartition", r.input.as_deref())?,
                // Spark's own hint forms: `repartition(n)` is `REPARTITION(n)`, and
                // `coalesce(n)` (shuffle false, or unset: the field is `optional bool`, default false,
                // read with `getShuffle`) is `COALESCE(n)`. Rows pass through.
                name: if r.shuffle.unwrap_or(false) {
                    "REPARTITION"
                } else {
                    "COALESCE"
                }
                .to_owned(),
                params: vec![Arc::new(Expr::literal(Literal::Integer(r.num_partitions)))],
            },
            RelType::RepartitionByExpression(r) => self.repartition_by_expression(r)?,
            RelType::Unpivot(u) => self.unpivot(u)?,
            RelType::ToDf(t) => PlanNode::ColumnNames {
                input: self.input("ToDF", t.input.as_deref())?,
                op: ColumnNamesOp::ToDf(t.column_names.clone()),
            },
            RelType::WithColumnsRenamed(w) => self.with_columns_renamed(w)?,
            RelType::Drop(d) => self.drop_columns(d)?,
            // Spark 4.3's `Zip`: the columns of two DataFrames side by side (the analyzer merges the two
            // projection chains over their shared base, or refuses with `ZIP_PLANS_NOT_MERGEABLE`).
            RelType::Zip(z) => self.zip(z)?,
            RelType::Extension(any) => match resolve(any, Slot::Relation)? {
                // One row, one string column: `SELECT '<message>' AS message`.
                Extension::Echo(echo) => PlanNode::Project {
                    input: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
                    exprs: vec![Arc::new(
                        Expr::literal(Literal::String(echo.message)).alias("message"),
                    )],
                },
                // Registered for the command slot only: `resolve` has refused it here.
                Extension::PersistPythonFunction(_) => {
                    return Err(super::errors::unsupported(
                        "the PersistPythonFunction extension in a relation",
                    ));
                }
            },
            other => {
                return Err(unsupported_variant(
                    "spark.connect.Relation",
                    other.variant(),
                ));
            }
        };
        Ok(Arc::new(LogicalPlan::new(node).with_origin(origin_of(rel))))
    }

    /// `Sample`, as Spark's `transformSample`: Bernoulli sampling without replacement, the seed
    /// drawn per plan when the client sends none. Sampling with replacement (a Poisson draw) and
    /// `deterministic_order` (`randomSplit`'s per-partition sort) have no IR node.
    fn sample(&mut self, s: &pb::Sample) -> Result<PlanNode> {
        if s.with_replacement.unwrap_or(false) {
            return Err(unsupported_field(
                "Sample",
                "with_replacement",
                "sampling with replacement",
            ));
        }
        if s.deterministic_order {
            return Err(unsupported_field(
                "Sample",
                "deterministic_order",
                "randomSplit's deterministic order",
            ));
        }
        // Spark translates the input first; then the `Sample` constructor requires `upper - lower`
        // in [0 - 1e-6, 1 + 1e-6] (NaN fails it too). Its classless `IllegalArgumentException` is
        // `KRISHIV_SAMPLE_FRACTION_OUT_OF_RANGE` here, a recorded difference.
        let input = self.input("Sample", s.input.as_deref())?;
        let fraction = s.upper_bound - s.lower_bound;
        if !(-1e-6..=1.0 + 1e-6).contains(&fraction) {
            return Err(krishiv_common::error::KrishivSampleFractionOutOfRange {
                fraction: krishiv_kernels::text::java_double_string(fraction),
            }
            .into());
        }
        Ok(PlanNode::Sample {
            input,
            bounds: SampleBounds {
                lower: s.lower_bound,
                upper: s.upper_bound,
            },
            seed: s.seed,
        })
    }

    /// `StatSampleBy`: Spark's `DataFrameStatFunctions.sampleBy` keeps a row when
    /// `rand(seed) < fractions.getOrElse(stratum, 0.0)`, drawing the random number for every row.
    /// Lowered to `Filter(rand(seed) < CASE WHEN col = k1 THEN f1 ... ELSE 0.0 END)` so the draw
    /// is not skipped for unmatched strata. A NULL stratum matches nothing (fraction 0.0). A
    /// fraction outside [0, 1] is refused (Spark's classless `require`). A stratum listed twice
    /// keeps the last fraction (Spark's `Map`). The stratum is compared with SQL `=` (numeric
    /// coercion); whether Spark's boxed lookup matches an Int stratum to a Long column the same
    /// way is not verified against a real Spark run.
    fn sample_by(&mut self, s: &pb::StatSampleBy) -> Result<PlanNode> {
        let input = self.input("StatSampleBy", s.input.as_deref())?;
        let column = self.required_expression(s.col.as_ref())?;
        if s.fractions
            .iter()
            .any(|f| !(0.0..=1.0).contains(&f.fraction))
        {
            return Err(assertion("Fractions must be in [0, 1]"));
        }
        let mut branches: Vec<Expr> = Vec::new();
        // `CASE` takes the first match, so walking the pairs backwards keeps the last of a repeated stratum.
        for f in s.fractions.iter().rev() {
            let stratum = f.stratum.as_ref().ok_or_else(expression_not_set)?;
            let key = self.literal_expression(stratum)?;
            branches.push(Expr::function(
                Builtin::Eq.name(),
                vec![(*column).clone(), (*key).clone()],
            ));
            branches.push(Expr::literal(Literal::Double(f.fraction)));
        }
        branches.push(Expr::literal(Literal::Double(0.0)));
        let fraction = Expr::function(Builtin::When.name(), branches);
        let draw = match s.seed {
            Some(seed) => Expr::function("rand", vec![Expr::literal(Literal::Long(seed))]),
            None => Expr::function("rand", Vec::new()),
        };
        Ok(PlanNode::Filter {
            input,
            predicate: Arc::new(Expr::function(Builtin::Lt.name(), vec![draw, fraction])),
        })
    }

    /// `StatCov` and `StatCorr`: Spark's `StatFunctions.calculateCov` / `pearsonCorrelation`, which
    /// read each NULL as 0.0 (a Spark 4.0 run: `(1,2),(NULL,3),(3,7),(4,NULL)` gives cov -1/3), and
    /// answer one never-NULL double row: `cov` 0.0 and `corr` NaN below two rows. Lowered to
    /// `coalesce(covar_samp | corr (coalesce(cast(c as double), 0.0)), 0.0 | NaN)`. The cast is
    /// ANSI, so a value that is not a number is an error, never 0.0. Recorded differences: Spark
    /// refuses every non-numeric column by type (`require(NumericType)`), the translator sees no
    /// types, so a STRING column of numbers is answered here; and a zero-variance `corr` is NaN
    /// where Spark 4.0 under ANSI raises DIVIDE_BY_ZERO.
    fn stat_pair(
        &mut self,
        message: &str,
        input: Option<&Relation>,
        col1: &str,
        col2: &str,
        name: &str,
    ) -> Result<PlanNode> {
        let input = self.input(message, input)?;
        // Each argument is an attribute reference (`t.x`, `` `a.b` ``), parsed as `UnresolvedAttribute`.
        let zero_filled = |c: &str| -> Result<Expr> {
            Ok(Expr::function(
                "coalesce",
                vec![
                    Expr::new(ExprNode::Cast {
                        expr: Arc::new(Expr::column(parse_attribute_name(c)?)),
                        to: krishiv_types::DataType::Double,
                        mode: CastMode::Ansi,
                    }),
                    Expr::literal(Literal::Double(0.0)),
                ],
            ))
        };
        let (left, right) = (zero_filled(col1)?, zero_filled(col2)?);
        let (func, default) = if name == "cov" {
            ("covar_samp", 0.0)
        } else {
            ("corr", f64::NAN)
        };
        let value = Expr::function(
            "coalesce",
            vec![
                Expr::function(func, vec![left, right]),
                Expr::literal(Literal::Double(default)),
            ],
        )
        .alias(name);
        Ok(PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(Vec::new()),
            aggregates: vec![Arc::new(value)],
        })
    }

    /// `NAFill`: Spark's `transformFillNa`. No values, or several values that don't pair with
    /// the columns, are `CONNECT_INVALID_PLAN` errors; the type each value fills is decided by
    /// the analyzer, which knows the columns' types.
    fn na_fill(&mut self, n: &pb::NaFill) -> Result<PlanNode> {
        use krishiv_common::error::{
            ConnectInvalidPlanNaFillValuesEmpty, ConnectInvalidPlanNaFillValuesLengthMismatch,
        };
        if n.values.is_empty() {
            return Err(ConnectInvalidPlanNaFillValuesEmpty {}.into());
        }
        if n.values.len() > 1 && n.values.len() != n.cols.len() {
            return Err(ConnectInvalidPlanNaFillValuesLengthMismatch {}.into());
        }
        let values = n
            .values
            .iter()
            .map(|l| self.literal_expression(l))
            .collect::<Result<Vec<_>>>()?;
        Ok(PlanNode::ColumnNames {
            input: self.input("NaFill", n.input.as_deref())?,
            op: ColumnNamesOp::FillNa {
                cols: n.cols.clone(),
                values,
            },
        })
    }

    /// `NAReplace`: Spark's `transformReplace`; an empty replacement list leaves the input as it is.
    /// The columns each pair reaches are decided by the analyzer, which knows their types.
    fn na_replace(&mut self, n: &pb::NaReplace) -> Result<PlanNode> {
        let mut pairs = Vec::with_capacity(n.replacements.len() * 2);
        for r in &n.replacements {
            let (Some(old), Some(new)) = (&r.old_value, &r.new_value) else {
                return Err(assertion(
                    "NAReplace.Replacement needs old_value and new_value",
                ));
            };
            pairs.push(self.literal_expression(old)?);
            pairs.push(self.literal_expression(new)?);
        }
        Ok(PlanNode::ColumnNames {
            input: self.input("NaReplace", n.input.as_deref())?,
            op: ColumnNamesOp::Replace {
                cols: n.cols.clone(),
                pairs,
            },
        })
    }

    /// `Deduplicate`, as Spark's `transformDeduplicate`: over every column it is `distinct()`;
    /// over a subset it keeps the first row per key, which needs the input's columns and so
    /// waits for an IR node. `within_watermark` is a streaming operator.
    fn deduplicate(&mut self, d: &pb::Deduplicate) -> Result<PlanNode> {
        use krishiv_common::error::{
            ConnectInvalidPlanDeduplicateAllColumnsAndSubset,
            ConnectInvalidPlanDeduplicateNeedsInput,
            ConnectInvalidPlanDeduplicateRequiresColumnsOrAll,
        };
        // Spark checks that the input is present, then the two column fields, and only then
        // transforms the input.
        let Some(rel) = d.input.as_deref() else {
            return Err(ConnectInvalidPlanDeduplicateNeedsInput {}.into());
        };
        let all = d.all_columns_as_keys.unwrap_or(false);
        if !all && d.column_names.is_empty() {
            return Err(ConnectInvalidPlanDeduplicateRequiresColumnsOrAll {}.into());
        }
        if all && !d.column_names.is_empty() {
            return Err(ConnectInvalidPlanDeduplicateAllColumnsAndSubset {}.into());
        }
        if d.within_watermark.unwrap_or(false) {
            return Err(unsupported_field(
                "Deduplicate",
                "within_watermark",
                "dropDuplicatesWithinWatermark",
            ));
        }
        if !all {
            return Err(unsupported_field(
                "Deduplicate",
                "column_names",
                "deduplicating on a subset of columns",
            ));
        }
        Ok(PlanNode::Distinct {
            input: self.relation(rel)?,
        })
    }

    /// `RepartitionByExpression` without a partition count is `DISTRIBUTE BY exprs`; a count has
    /// no IR field yet.
    fn repartition_by_expression(&mut self, r: &pb::RepartitionByExpression) -> Result<PlanNode> {
        if r.num_partitions.is_some() {
            return Err(unsupported_field(
                "RepartitionByExpression",
                "num_partitions",
                "repartition(n, exprs)",
            ));
        }
        if r.partition_exprs.is_empty() {
            return Err(unsupported_field(
                "RepartitionByExpression",
                "partition_exprs",
                "a repartition with no expressions (round robin)",
            ));
        }
        Ok(PlanNode::Repartition {
            input: self.input("RepartitionByExpression", r.input.as_deref())?,
            exprs: self.expressions(&r.partition_exprs)?,
        })
    }

    /// `WithColumnsRenamed`: the `renames` list, or the older `rename_columns_map` when the list is
    /// empty (Spark's own fallback). Names that match no column are ignored, as Spark does.
    /// Spark applies the map in the order of the decoded Java map, which is wire order and is
    /// lost once the message is decoded, so a map whose result depends on the order is refused.
    fn with_columns_renamed(&mut self, w: &pb::WithColumnsRenamed) -> Result<PlanNode> {
        let pairs = if w.renames.is_empty() {
            let keys: Vec<String> = w
                .rename_columns_map
                .keys()
                .map(|k| k.to_lowercase())
                .collect();
            // A new name equal to *another* entry's key makes the result depend on order; equal
            // to its own key (`a -> a`, `name -> Name`) it cannot.
            let ordered = w.rename_columns_map.iter().all(|(from, to)| {
                let (from, to) = (from.to_lowercase(), to.to_lowercase());
                from == to || !keys.contains(&to)
            }) && keys.iter().enumerate().all(|(i, k)| !keys[..i].contains(k));
            if !ordered {
                return Err(unsupported_field(
                    "WithColumnsRenamed",
                    "rename_columns_map",
                    "a map whose result depends on the order of its entries (a new name equal to \
                     another key, or two keys equal ignoring case); use the renames list",
                ));
            }
            w.rename_columns_map
                .iter()
                .map(|(from, to)| (from.clone(), to.clone()))
                .collect()
        } else {
            w.renames
                .iter()
                .map(|r| (r.col_name.clone(), r.new_col_name.clone()))
                .collect()
        };
        Ok(PlanNode::ColumnNames {
            input: self.input("WithColumnsRenamed", w.input.as_deref())?,
            op: ColumnNamesOp::Rename(pairs),
        })
    }

    /// Its own function, not inline in the relation match: that frame is recursed through, and a
    /// deep relation tree must still translate on a small stack.
    #[inline(never)]
    fn zip(&mut self, z: &pb::Zip) -> Result<PlanNode> {
        Ok(PlanNode::Zip {
            left: self.input("Zip", z.left.as_deref())?,
            right: self.input("Zip", z.right.as_deref())?,
        })
    }

    /// `Drop` by name, or by column expressions that are plain one-part column names (resolved
    /// over the input, so an ambiguous one is an error as in Spark). A qualified column or one
    /// tied to a DataFrame (`df["a"]`) drops by attribute in Spark and is refused.
    fn drop_columns(&mut self, d: &pb::Drop) -> Result<PlanNode> {
        let names = d.column_names.clone();
        let mut columns = Vec::new();
        for e in self.expressions(&d.columns)? {
            match &e.node {
                ExprNode::Column(c) if c.name.len() == 1 && c.plan_id.is_none() => {
                    columns.push(c.name[0].clone());
                }
                _ => {
                    return Err(unsupported_field(
                        "Drop",
                        "columns",
                        "dropping a column expression that is not a plain column name",
                    ));
                }
            }
        }
        Ok(PlanNode::ColumnNames {
            input: self.input("Drop", d.input.as_deref())?,
            op: ColumnNamesOp::Drop { names, columns },
        })
    }

    /// `Unpivot` (`DataFrame.unpivot`, `melt`): Spark's output is the id columns, then the name
    /// and value columns, with every null value row kept (the DataFrame form has no `EXCLUDE
    /// NULLS`). The IR node keeps every other column as an id, so the input is first projected
    /// to the ids and the value columns. A value that is not a plain column, and an absent
    /// `values` (every non-id column, known only after analysis), are refused.
    fn unpivot(&mut self, u: &pb::Unpivot) -> Result<PlanNode> {
        let Some(values) = &u.values else {
            return Err(unsupported_field(
                "Unpivot",
                "values",
                "unpivot without explicit value columns",
            ));
        };
        // An empty list reaches the analyzer, which raises UNPIVOT_REQUIRES_VALUE_COLUMNS as Spark does.
        let input = self.input("Unpivot", u.input.as_deref())?;
        let ids = self.expressions(&u.ids)?;
        let values = self.expressions(&values.values)?;
        let column_sets = values
            .iter()
            .map(|v| {
                let column = match &v.node {
                    ExprNode::Column(_) => Arc::clone(v),
                    _ => {
                        return Err(unsupported_field(
                            "Unpivot",
                            "values",
                            "unpivot of a value that is not a column",
                        ));
                    }
                };
                Ok(UnpivotColumnSet {
                    columns: vec![column],
                    alias: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // The analyzer keeps every input column no column set references, so a column that is
        // both an id and a value (Spark keeps it and unpivots it) or a value listed twice would
        // lose its id copy and make the projection ambiguous. Refuse it by the last name part, ignoring case
        // (a qualified spelling must not slip past; over-refusing is safe), until the IR can keep a referenced column.
        let mut seen = std::collections::HashSet::new();
        for e in ids.iter().chain(values.iter()) {
            // The output name: an alias, else the last part of a column's name.
            let key = match &e.node {
                ExprNode::Alias { name, .. } => Some(name.to_lowercase()),
                ExprNode::Column(c) => c.name.last().map(|p| p.to_lowercase()),
                _ => None,
            };
            if let Some(key) = key
                && !seen.insert(key)
            {
                return Err(unsupported_field(
                    "Unpivot",
                    "values",
                    "unpivot of a column that is also an id or listed twice",
                ));
            }
        }
        let projected = Arc::new(
            LogicalPlan::new(PlanNode::Project {
                input,
                exprs: ids.into_iter().chain(values).collect(),
            })
            .with_origin(Origin::UNKNOWN),
        );
        Ok(PlanNode::Unpivot {
            input: projected,
            value_columns: vec![u.value_column_name.clone()],
            name_column: u.variable_column_name.clone(),
            column_sets,
            include_nulls: true,
        })
    }

    /// `SetOperation`, as Spark's `transformSetOperation` (v4.2.0): `is_all` defaults to false,
    /// `by_name` is for `UNION` only, `allow_missing_columns` needs `by_name`, and a `UNION` that
    /// is not `ALL` is a `Distinct` over the `Union`.
    fn set_op(&mut self, s: &pb::SetOperation, origin: Origin) -> Result<PlanNode> {
        let (Some(left), Some(right)) = (&s.left_input, &s.right_input) else {
            return Err(
                krishiv_common::error::ConnectInvalidPlanSetOperationMustHaveTwoInputs {}.into(),
            );
        };
        let left = self.relation(left)?;
        let right = self.relation(right)?;
        let all = s.is_all.unwrap_or(false);
        let by_name = s.by_name.unwrap_or(false);
        let allow_missing_columns = s.allow_missing_columns.unwrap_or(false);
        Ok(
            match pb::set_operation::SetOpType::try_from(s.set_op_type) {
                Ok(pb::set_operation::SetOpType::Except) => {
                    if by_name {
                        return Err(krishiv_common::error::ConnectInvalidPlanExceptDoesNotSupportUnionByName {}.into());
                    }
                    PlanNode::SetOp {
                        left,
                        right,
                        op: SetOperation::Except,
                        all,
                    }
                }
                Ok(pb::set_operation::SetOpType::Intersect) => {
                    if by_name {
                        return Err(krishiv_common::error::ConnectInvalidPlanIntersectDoesNotSupportUnionByName {}.into());
                    }
                    PlanNode::SetOp {
                        left,
                        right,
                        op: SetOperation::Intersect,
                        all,
                    }
                }
                Ok(pb::set_operation::SetOpType::Union) => {
                    if !by_name && allow_missing_columns {
                        return Err(krishiv_common::error::ConnectInvalidPlanUnionByNameAllowMissingColRequiresByName {}.into());
                    }
                    let union = PlanNode::Union {
                        inputs: vec![left, right],
                        by_name,
                        allow_missing_columns,
                    };
                    if all {
                        union
                    } else {
                        PlanNode::Distinct {
                            input: Arc::new(LogicalPlan::new(union).with_origin(origin)),
                        }
                    }
                }
                Ok(pb::set_operation::SetOpType::Unspecified) => {
                    return Err(invalid_enum(
                        "spark.connect.SetOperation.SetOpType",
                        "SET_OP_TYPE_UNSPECIFIED",
                        s.set_op_type,
                    ));
                }
                Err(_) => {
                    return Err(invalid_enum(
                        "spark.connect.SetOperation.SetOpType",
                        "UNRECOGNIZED",
                        s.set_op_type,
                    ));
                }
            },
        )
    }

    /// The next local relation of the plan, from the host (see [`super::LocalRelations`]).
    fn local_relation(
        &mut self,
        request: impl FnOnce() -> Result<super::LocalRequest>,
    ) -> Result<krishiv_plan::LocalRelation> {
        let ordinal = self.locals;
        self.locals += 1;
        self.conf.local_relations.get(ordinal, request)
    }

    /// The input relation of `message`, which must be set.
    fn input(&mut self, message: &str, input: Option<&Relation>) -> Result<Arc<LogicalPlan>> {
        match input {
            Some(rel) => self.relation(rel),
            None => Err(assertion(format!("{message} needs an input relation"))),
        }
    }

    fn expressions(&mut self, list: &[Expression]) -> Result<Vec<Arc<Expr>>> {
        list.iter().map(|e| self.expression(e)).collect()
    }

    /// `Read`. Option values and paths are classified once, here, by
    /// [`OptionValue::from_user`] (ADR 0038): a credential or a signed URL
    /// never enters the IR as plain text.
    fn read(&mut self, r: &pb::Read) -> Result<PlanNode> {
        if r.is_streaming {
            return Err(unsupported_field("Read", "is_streaming", "streaming reads"));
        }
        let typed = |options: &BTreeMap<String, String>| -> BTreeMap<String, OptionValue> {
            options
                .iter()
                .map(|(k, v)| (k.clone(), OptionValue::from_user(k, v.as_str())))
                .collect()
        };
        let Some(read_type) = &r.read_type else {
            return Err(not_set("spark.connect.Read", "READ_TYPE_NOT_SET"));
        };
        Ok(PlanNode::Read(match read_type {
            ReadType::NamedTable(t) => ReadSource::Table {
                name: parse_multipart_identifier(&t.unparsed_identifier)?,
                options: typed(&t.options),
            },
            ReadType::DataSource(d) => {
                // Spark: a set and non-empty schema is the reader's schema.
                let schema = match d.schema.as_deref() {
                    Some(s) if !s.is_empty() => {
                        Some(super::schema::parse_table_schema(s, &self.conf.parser)?)
                    }
                    _ => None,
                };
                if !d.predicates.is_empty() {
                    return Err(unsupported_field(
                        "Read.DataSource",
                        "predicates",
                        "reads with predicates",
                    ));
                }
                let source = ReadSource::files(
                    d.format
                        .clone()
                        .unwrap_or_else(|| self.conf.default_data_source.clone()),
                    d.paths
                        .iter()
                        .map(|p| OptionValue::from_user("path", p.as_str()))
                        .collect(),
                    typed(&d.options),
                );
                match schema {
                    Some(schema) => source.with_schema(schema),
                    None => source,
                }
            }
        }))
    }

    /// `SQL`. The parameters are, in Spark's order of precedence, the named
    /// expressions, the positional expressions, and the deprecated literal
    /// map and list.
    fn sql(&mut self, s: &pb::Sql) -> Result<PlanNode> {
        #[expect(
            deprecated,
            reason = "Spark still accepts the literal arguments of older clients"
        )]
        let (literal_map, literal_list) = (&s.args, &s.pos_args);
        let args = if !s.named_arguments.is_empty() {
            SqlArgs::Named(
                s.named_arguments
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.expression(v)?)))
                    .collect::<Result<_>>()?,
            )
        } else if !s.pos_arguments.is_empty() {
            SqlArgs::Positional(self.expressions(&s.pos_arguments)?)
        } else if !literal_map.is_empty() {
            SqlArgs::Named(
                literal_map
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.literal_expression(v)?)))
                    .collect::<Result<_>>()?,
            )
        } else if !literal_list.is_empty() {
            SqlArgs::Positional(
                literal_list
                    .iter()
                    .map(|v| self.literal_expression(v))
                    .collect::<Result<_>>()?,
            )
        } else {
            SqlArgs::None
        };
        Ok(PlanNode::Sql {
            query: s.query.clone(),
            args,
        })
    }

    fn literal_expression(&mut self, l: &pb::expression::Literal) -> Result<Arc<Expr>> {
        self.count()?;
        Ok(Arc::new(Expr::new(ExprNode::Literal(
            super::literal::literal(l)?,
        ))))
    }

    fn join(&mut self, j: &pb::Join) -> Result<PlanNode> {
        let (Some(left), Some(right)) = (&j.left, &j.right) else {
            return Err(assertion("Both join sides must be present"));
        };
        if j.join_condition.is_some() && !j.using_columns.is_empty() {
            return Err(
                krishiv_common::error::ConnectInvalidPlanUsingColumnsOrJoinConditionSetInJoin {}
                    .into(),
            );
        }
        if j.join_data_type.is_some() {
            return Err(unsupported_field(
                "Join",
                "join_data_type",
                "joinWith (typed joins)",
            ));
        }
        let kind = match JoinType::try_from(j.join_type) {
            Ok(JoinType::Inner) => JoinKind::Inner,
            Ok(JoinType::FullOuter) => JoinKind::FullOuter,
            Ok(JoinType::LeftOuter) => JoinKind::LeftOuter,
            Ok(JoinType::RightOuter) => JoinKind::RightOuter,
            Ok(JoinType::LeftAnti) => JoinKind::LeftAnti,
            Ok(JoinType::LeftSemi) => JoinKind::LeftSemi,
            Ok(JoinType::Cross) => JoinKind::Cross,
            Ok(JoinType::Unspecified) => {
                return Err(invalid_enum(
                    "spark.connect.Join.JoinType",
                    "JOIN_TYPE_UNSPECIFIED",
                    0,
                ));
            }
            Err(_) => {
                return Err(invalid_enum(
                    "spark.connect.Join.JoinType",
                    "UNRECOGNIZED",
                    j.join_type,
                ));
            }
        };
        let condition = match &j.join_condition {
            Some(c) => JoinCondition::On(self.expression(c)?),
            None if !j.using_columns.is_empty() => JoinCondition::Using(j.using_columns.clone()),
            None => JoinCondition::None,
        };
        Ok(PlanNode::Join {
            left: self.relation(left)?,
            right: self.relation(right)?,
            kind,
            condition,
        })
    }

    /// `AsOfJoin`, as Spark's `transformAsOfJoin` and `Dataset.joinAsOf` (v4.3.0): an inner or
    /// left join (`JoinType`'s names, case and `_` ignored), `backward`, `forward` or `nearest`
    /// (any case), the as-of values per side as given (no reordering, unlike the SQL match), the
    /// tolerance kept for the analyzer. `using_columns` (Spark: `left[c] === right[c]` conditions)
    /// is not translated yet.
    fn as_of_join(&mut self, j: &pb::AsOfJoin) -> Result<PlanNode> {
        let (Some(left), Some(right)) = (&j.left, &j.right) else {
            return Err(assertion("Both join sides must be present"));
        };
        let kind = match j.join_type.to_ascii_lowercase().replace('_', "").as_str() {
            "inner" => JoinKind::Inner,
            "left" | "leftouter" => JoinKind::LeftOuter,
            _ => {
                return Err(unsupported(format!(
                    "an as-of join of type '{}'",
                    j.join_type
                )));
            }
        };
        let direction = match j.direction.to_ascii_lowercase().as_str() {
            "backward" => AsOfDirection::Backward,
            "forward" => AsOfDirection::Forward,
            "nearest" => AsOfDirection::Nearest,
            _ => {
                return Err(unsupported(format!(
                    "the as-of join direction '{}'",
                    j.direction
                )));
            }
        };
        if !j.using_columns.is_empty() {
            return Err(unsupported_field(
                "AsOfJoin",
                "using_columns",
                "an as-of join on using columns",
            ));
        }
        let (Some(left_as_of), Some(right_as_of)) = (&j.left_as_of, &j.right_as_of) else {
            return Err(assertion("Both as-of values must be present"));
        };
        let condition = match &j.join_expr {
            Some(c) => JoinCondition::On(self.expression(c)?),
            None => JoinCondition::None,
        };
        let as_of = AsOfMatch {
            left: self.expression(left_as_of)?,
            right: self.expression(right_as_of)?,
            direction,
            allow_exact: j.allow_exact_matches,
            tolerance: match &j.tolerance {
                Some(t) => Some(self.expression(t)?),
                None => None,
            },
            distance: None,
            // Spark runs a Connect `AsOfJoin` as `RewriteAsOfJoin`'s `min_by` over every
            // passing right row, which evaluates the distance on each.
            distance_on_every_pair: true,
            as_written: false,
        };
        Ok(PlanNode::AsOfJoin {
            left: self.relation(left)?,
            right: self.relation(right)?,
            kind,
            condition,
            as_of: Arc::new(as_of),
        })
    }

    /// `Aggregate`. The output is the grouping expressions followed by the
    /// aggregate expressions, as Spark builds it.
    fn aggregate(&mut self, a: &pb::Aggregate) -> Result<PlanNode> {
        let Some(input) = &a.input else {
            return Err(krishiv_common::error::ConnectInvalidPlanAggregateNeedsPlanInput {}.into());
        };
        let input = self.relation(input)?;
        let keys = self.expressions(&a.grouping_expressions)?;
        let aggregates = self.expressions(&a.aggregate_expressions)?;
        // The output keeps `expr AS name`; the grouping expressions are the
        // bare `expr`: an alias is not an input column (PY-009).
        let bare = |k: &Arc<Expr>| match &k.node {
            ExprNode::Alias { expr, .. } => Arc::clone(expr),
            _ => Arc::clone(k),
        };
        let group_keys: Vec<Arc<Expr>> = keys.iter().map(bare).collect();
        let grouping = match GroupType::try_from(a.group_type) {
            Ok(GroupType::Groupby) => Grouping::GroupBy(group_keys),
            Ok(GroupType::Rollup) => Grouping::Rollup(group_keys),
            Ok(GroupType::Cube) => Grouping::Cube(group_keys),
            Ok(GroupType::GroupingSets) => Grouping::GroupingSets {
                exprs: group_keys,
                sets: a
                    .grouping_sets
                    .iter()
                    .map(|s| {
                        Ok(self
                            .expressions(&s.grouping_set)?
                            .iter()
                            .map(bare)
                            .collect())
                    })
                    .collect::<Result<_>>()?,
            },
            Ok(GroupType::Pivot) => return self.pivot(a, &input, &keys, aggregates),
            Ok(GroupType::Unspecified) => {
                return Err(invalid_enum(
                    "spark.connect.Aggregate.GroupType",
                    "GROUP_TYPE_UNSPECIFIED",
                    0,
                ));
            }
            Err(_) => {
                return Err(invalid_enum(
                    "spark.connect.Aggregate.GroupType",
                    "UNRECOGNIZED",
                    a.group_type,
                ));
            }
        };
        Ok(PlanNode::Aggregate {
            input,
            grouping,
            aggregates: keys.into_iter().chain(aggregates).collect(),
        })
    }

    /// `Aggregate` with `GROUP_TYPE_PIVOT` (`groupBy(...).pivot(col, values).agg(...)`): the
    /// session's pivot ([`krishiv_plan::grouped_pivot`]). Spark computes the values of a pivot
    /// without any by running a query before planning; the translator runs none, so that case is
    /// refused.
    fn pivot(
        &mut self,
        a: &pb::Aggregate,
        input: &Arc<LogicalPlan>,
        keys: &[Arc<Expr>],
        aggregates: Vec<Arc<Expr>>,
    ) -> Result<PlanNode> {
        let Some(pivot) = &a.pivot else {
            return Err(assertion("Aggregate with GROUP_TYPE_PIVOT needs a pivot"));
        };
        let column = self.required_expression(pivot.col.as_ref())?;
        if pivot.values.is_empty() {
            return Err(unsupported(
                "a pivot without values (Spark runs a query for the distinct values)",
            ));
        }
        let values = pivot
            .values
            .iter()
            .map(|v| {
                Ok(PivotValue {
                    exprs: vec![self.literal_expression(v)?],
                    alias: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let keys: Vec<Expr> = keys.iter().map(|k| (**k).clone()).collect();
        krishiv_plan::grouped_pivot(
            input,
            &keys,
            &column,
            values,
            aggregates,
            self.conf.case_sensitive,
            || Origin::UNKNOWN,
        )
    }

    /// `Range`: `range(start, end, step, numPartitions)`, as Spark builds
    /// it (defaults: `start` 0, `numPartitions` from
    /// `spark.sql.leafNodeDefaultParallelism` else default parallelism,
    /// resolved at execution).
    fn range(&mut self, r: &pb::Range) -> Result<PlanNode> {
        use krishiv_common::error::{FailedFunctionCall, KrishivInvalidPartitionNumber};
        use krishiv_plan::{AttrId, BoundColumn};
        // The Connect path builds the resolved leaf directly, as
        // `Session::range` does; the id is fresh within this
        // translation (one per translated node).
        let id = AttrId(self.nodes as u64);
        let num_partitions = match r.num_partitions {
            Some(n) if n <= 0 => {
                return Err(KrishivInvalidPartitionNumber {
                    num_partitions: n.to_string(),
                }
                .into());
            }
            Some(n) => Some(u32::try_from(n).map_err(|_| {
                unsupported_field("Range", "num_partitions", "a partition count past u32")
            })?),
            None => None,
        };
        if r.step == 0 {
            return Err(FailedFunctionCall {
                func_name: "`range`".to_owned(),
            }
            .into());
        }
        Ok(PlanNode::Range {
            start: r.start.unwrap_or(0),
            end: r.end,
            step: r.step,
            num_slices: num_partitions,
            output: BoundColumn {
                id,
                name: "id".to_owned(),
                data_type: krishiv_types::DataType::Long,
                nullable: false,
            },
        })
    }

    /// `WithColumns`: each alias names exactly one column.
    fn with_columns(&mut self, w: &pb::WithColumns) -> Result<PlanNode> {
        let mut columns = Vec::with_capacity(w.aliases.len());
        for alias in &w.aliases {
            let [name] = alias.name.as_slice() else {
                return Err(
                    krishiv_common::error::ConnectInvalidPlanWithColumnsRequireSingleNamePart {
                        got: format!("[{}]", alias.name.join(", ")),
                    }
                    .into(),
                );
            };
            if alias.metadata.as_deref().is_some_and(|m| !m.is_empty()) {
                return Err(unsupported_field(
                    "Expression.Alias",
                    "metadata",
                    "column aliases with metadata",
                ));
            }
            columns.push((
                name.clone(),
                self.required_expression(alias.expr.as_deref())?,
            ));
        }
        let input = self.input("WithColumns", w.input.as_deref())?;
        // Generator columns are lifted as the native DataFrame API does (one column at a time,
        // each reading the ones before it).
        if columns.iter().any(|(n, e)| {
            krishiv_plan::with_column_generator(&input, n, e, || Origin::UNKNOWN).is_some()
        }) {
            // Spark binds every alias to the input; the one-at-a-time lift below would let an alias read a
            // column an earlier alias of the same call has just replaced (`x = y, y = x`).
            for (i, (_, value)) in columns.iter().enumerate() {
                if columns[..i]
                    .iter()
                    .any(|(earlier, _)| reads_column(value, earlier))
                {
                    return Err(unsupported_field(
                        "WithColumns",
                        "aliases",
                        "a generator column beside an alias that reads a column an earlier alias replaces",
                    ));
                }
            }
            let mut plan = input;
            let last = columns.len() - 1;
            for (i, (name, value)) in columns.into_iter().enumerate() {
                let node =
                    krishiv_plan::with_column_generator(&plan, &name, &value, || Origin::UNKNOWN)
                        .unwrap_or_else(|| PlanNode::WithColumns {
                            input: Arc::clone(&plan),
                            columns: vec![(name, value)],
                        });
                if i == last {
                    return Ok(node);
                }
                plan = Arc::new(LogicalPlan::new(node));
            }
            return Err(assertion("'aliases' must contain elements."));
        }
        Ok(PlanNode::WithColumns { input, columns })
    }
}

#[cfg(test)]
mod tests;

/// Whether `e` reads a column whose last name part is `name` (case-insensitively, as the resolver's default).
fn reads_column(e: &krishiv_plan::Expr, name: &str) -> bool {
    let mut stack = vec![e];
    while let Some(e) = stack.pop() {
        if let krishiv_plan::ExprNode::Column(c) = &e.node
            && c.name.last().is_some_and(|n| n.eq_ignore_ascii_case(name))
        {
            return true;
        }
        e.for_each_child(&mut |child| stack.push(child));
    }
    false
}

/// The alias of a `SubqueryAlias`: `qualifier ++ [alias]` joined by [`ALIAS_PART_SEP`], as the IR
/// holds a multi-part alias (Spark's `AliasIdentifier(alias, qualifier)`). A part holding the
/// separator could not be told from two parts, so it is refused as `KRISHIV_UNSUPPORTED_FEATURE`
/// (a deviation from note section 4: 56K00 is for `oneof` arms).
fn alias_name(qualifier: &[String], alias: &str) -> Result<String> {
    let parts = || qualifier.iter().map(String::as_str).chain([alias]);
    if parts().any(|p| p.contains(ALIAS_PART_SEP)) {
        return Err(unsupported_field(
            "SubqueryAlias",
            "qualifier",
            "relation alias parts holding the control character U+001E",
        ));
    }
    Ok(parts()
        .collect::<Vec<_>>()
        .join(&ALIAS_PART_SEP.to_string()))
}

/// The origin of a relation: its `plan_id`, when the client sent one.
fn origin_of(rel: &Relation) -> Origin {
    rel.common
        .as_ref()
        .and_then(|c| c.plan_id)
        .map_or(Origin::UNKNOWN, |id| Origin::plan(PlanId(id)))
}

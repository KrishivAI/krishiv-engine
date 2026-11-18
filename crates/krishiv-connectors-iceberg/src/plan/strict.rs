// SPDX-License-Identifier: Apache-2.0
//! The row filter of an overwrite (P3-T027, PR-36b part 2b): whether every
//! row of a file matches a predicate (`rows_must_match`) or some row may
//! (`rows_might_match`), as apache-iceberg 1.12.0's `ManifestFilterManager`
//! decides a `deleteByRowFilter` removal (`PartitionAndMetricsEvaluator`,
//! `ManifestFilterManager.java:584-631`; notes 36b2 §3-§5): per file, the
//! **residual** of the predicate after its partition (a predicate whose
//! strict projection holds for the partition is true, one whose inclusive
//! projection fails is false, the rest stay), then the strict metrics
//! evaluator (`StrictMetricsEvaluator` / `StrictEvalVisitor`) or the
//! inclusive one on the residual.
//!
//! - **Semantics.** A [`Pred`] is in Iceberg's semantics, built from a
//!   Spark-semantics [`PushdownExpr`] by [`Pred::from_spark`]: the positive
//!   predicates mean the same; Spark's `x <> v` and `NOT startswith` are FALSE
//!   on a null `x` where Iceberg's `notEq`/`notStartsWith` are true, so each
//!   is guarded by `not_null(x)`; `x = NaN` is `is_nan(x)` (Iceberg refuses a
//!   NaN literal) and a range comparison with NaN, a non-binary collation,
//!   `contains` and runtime filters are unknown (never "every row matches").
//! - **Strict metrics** (`StrictEvalVisitor.java:99-457`, notes §4): a file
//!   of no rows matches; NaN lower bounds refuse `gt`, `gtEq`, `notEq`,
//!   `notIn`; nested columns refuse everything but `is_nan`/`not_nan`. The
//!   null and NaN helpers diverge from Java's on purpose (review 36b2b B2):
//!   Java's `canContainNulls` is false when the null-count map lacks the
//!   column and `canContainNaNs` false when the NaN count is missing, so a
//!   float file written without NaN counts (older Java writers, other
//!   engines) has `x < v` proven by its upper bound (bounds exclude NaN) and
//!   is removed with its NaN rows, which Spark's `x < v` keeps (NaN is the
//!   greatest value). Krishiv reads a missing null count as "may hold
//!   nulls", and a missing NaN count of a `float`/`double` column as "may
//!   hold NaN" for `lt`, `ltEq`, `eq` and `in` (`gt`/`gtEq` hold for NaN in
//!   Spark and keep Java's rule), as `prune.rs` does: the file is kept, so
//!   the overwrite fails with the partial-match error instead.
//! - **Unknowns** (review 36b2b B1): a leaf with no Iceberg form
//!   ([`Pred::Unknown`]) or a column the schema lacks never proves a match
//!   and always may match, so such a file is never kept silently. The
//!   overwrite refuses such a filter up front ([`Pred::unsupported`]).
//! - **Strict projections** (`Projections.strict`, notes §5): the OR over the
//!   partition fields of the column; identity copies the predicate; bucket
//!   only `notEq`/`notIn` and the unary ones; truncate of ints and longs `lt
//!   t(v)`, `ltEq v` as `lt t(v+1)`, `gt t(v)`, `gtEq v` as `gt t(v-1)`,
//!   `notEq`/`notIn` (`eq`/`in` none); truncate of strings `startsWith`
//!   (shorter than W) or `eq` (exactly W), `notStartsWith`, and the range
//!   predicates as `lt`/`gt` of the truncated literal; year, month, day and
//!   hour as truncate of longs, with a partition value at or below 0 read as
//!   also holding the unit below (`fixStrictTimeProjection`: old writers
//!   rounded pre-1970 values toward zero); `void` and unknown transforms none.
//!   The inclusive side is `prune.rs`'s projection.
//!
//! Divergences, each toward keeping a file (so toward Java's "some, but not
//! all, rows match" error, never toward removing a row that does not match):
//! a string or binary upper bound at the writer's truncation length is read
//! as a prefix (`prune.rs`; Java trusts its own rounded-up bounds), uuid
//! bounds are unused, a literal of another type than the column (or a signed
//! zero) never proves a match, truncate of decimals and binaries and the
//! wrapped windows of truncated ints near their minimum project nothing, and
//! the inclusive side answers "might match" for `is_nan`, `not_nan`, `not_in`
//! and `not_starts_with`. The string truncate projections of the range
//! predicates and the exact pre-1970 fix were derived from the transforms'
//! guarantees (the notes recall them from fetch summaries only).

use std::cmp::Ordering;

use krishiv_io::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};

use super::bounds::{compare, decode, is_nan, is_unordered};
use super::entry::{ContentFile, PartitionValue};
use super::metrics::WriterMetrics;
use super::prune::{
    Op, PruneOptions, Upper, metric_facts, metric_options, partition_facts, projected, upper_bound,
    wide_truncate, wrapped_windows, wraps,
};
use super::transform::{apply, is_time, truncate_str};
use super::types::{IcebergType, PartitionFieldInfo, PartitionSpecs, SchemaField, Transform};

/// One predicate on a column, in Iceberg's semantics (module docs).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Leaf {
    /// `is_null`.
    IsNull,
    /// `not_null`.
    NotNull,
    /// `is_nan`.
    IsNaN,
    /// `not_nan`.
    NotNaN,
    /// `<`.
    Lt(ScalarValue),
    /// `<=`.
    LtEq(ScalarValue),
    /// `>`.
    Gt(ScalarValue),
    /// `>=`.
    GtEq(ScalarValue),
    /// `==`.
    Eq(ScalarValue),
    /// `!=` (true on null).
    NotEq(ScalarValue),
    /// `in`.
    In(Vec<ScalarValue>),
    /// `not in` (true on null). Spark's `NOT IN` reaches the filter as
    /// `<>` conjuncts (`PushdownExpr::to_nnf`), so only the tests build it.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Java's rule kept with its unit tests")
    )]
    NotIn(Vec<ScalarValue>),
    /// `startsWith`.
    StartsWith(String),
    /// `notStartsWith` (true on null).
    NotStartsWith(String),
}

/// A predicate in Iceberg's semantics, with no `not` (Java's `rewriteNot`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Pred {
    /// `alwaysTrue` / `alwaysFalse`.
    Const(bool),
    /// Conjunction.
    And(Vec<Pred>),
    /// Disjunction.
    Or(Vec<Pred>),
    /// A predicate on a column.
    Leaf(ColumnPath, Leaf),
    /// Something no evaluator can answer: never a match, always a maybe.
    Unknown,
}

impl Pred {
    /// The first part of the predicate the strict side cannot evaluate over
    /// `fields` (a leaf Krishiv cannot translate, or a column that is not a
    /// primitive column of the table, such as a struct under `IS NULL`), as
    /// text; `None` when every part can be. An overwrite refuses such a
    /// filter up front (Java's `SparkV2Filters.convert` / bind failure).
    pub(crate) fn unsupported(&self, fields: &[SchemaField]) -> Option<String> {
        match self {
            Self::Const(_) => None,
            Self::And(xs) | Self::Or(xs) => xs.iter().find_map(|x| x.unsupported(fields)),
            Self::Unknown => Some(
                "a predicate Iceberg's strict evaluator cannot answer (a substring match, a \
                 range comparison with NaN, a non-binary collation or a runtime filter)"
                    .to_owned(),
            ),
            Self::Leaf(c, _) => (!fields.iter().any(|f| &f.path == c)).then(|| {
                format!(
                    "the column {}, which is not a primitive column of the table",
                    c.0.join(".")
                )
            }),
        }
    }
}

impl Pred {
    /// The Iceberg-semantics form of the Spark-semantics `e` (module docs).
    pub(crate) fn from_spark(e: &PushdownExpr) -> Self {
        Self::convert(&e.to_nnf())
    }

    fn convert(e: &PushdownExpr) -> Self {
        let binary = |c: &Collation| *c == Collation::Utf8Binary;
        let leaf = |c: &ColumnPath, l: Leaf| Self::Leaf(c.clone(), l);
        let guarded = |c: &ColumnPath, l: Leaf| Self::And(vec![leaf(c, Leaf::NotNull), leaf(c, l)]);
        match e {
            PushdownExpr::Constant(b) => Self::Const(*b),
            PushdownExpr::And(xs) => Self::And(xs.iter().map(Self::convert).collect()),
            PushdownExpr::Or(xs) => Self::Or(xs.iter().map(Self::convert).collect()),
            PushdownExpr::IsNull(c) => leaf(c, Leaf::IsNull),
            PushdownExpr::IsNotNull(c) => leaf(c, Leaf::NotNull),
            PushdownExpr::Cmp {
                column,
                op,
                value,
                collation,
            } if binary(collation) && *value != ScalarValue::Null => {
                if is_nan(value) {
                    return match op {
                        CmpOp::Eq => leaf(column, Leaf::IsNaN),
                        CmpOp::NotEq => guarded(column, Leaf::NotNaN),
                        _ => Self::Unknown,
                    };
                }
                let v = value.clone();
                match op {
                    CmpOp::Eq => leaf(column, Leaf::Eq(v)),
                    CmpOp::NotEq => guarded(column, Leaf::NotEq(v)),
                    CmpOp::Lt => leaf(column, Leaf::Lt(v)),
                    CmpOp::LtEq => leaf(column, Leaf::LtEq(v)),
                    CmpOp::Gt => leaf(column, Leaf::Gt(v)),
                    CmpOp::GtEq => leaf(column, Leaf::GtEq(v)),
                }
            }
            PushdownExpr::In {
                column,
                values,
                collation,
            } if binary(collation) => {
                let (nans, rest): (Vec<_>, Vec<_>) = values
                    .iter()
                    .filter(|v| **v != ScalarValue::Null)
                    .cloned()
                    .partition(is_nan);
                let mut any = Vec::new();
                if !rest.is_empty() {
                    any.push(leaf(column, Leaf::In(rest)));
                }
                if !nans.is_empty() {
                    any.push(leaf(column, Leaf::IsNaN));
                }
                match any.len() {
                    0 => Self::Const(false),
                    1 => any.pop().unwrap_or(Self::Unknown),
                    _ => Self::Or(any),
                }
            }
            PushdownExpr::StartsWith {
                column,
                prefix,
                collation,
            } if binary(collation) => leaf(column, Leaf::StartsWith(prefix.clone())),
            PushdownExpr::Not(inner) => match inner.as_ref() {
                PushdownExpr::StartsWith {
                    column,
                    prefix,
                    collation,
                } if binary(collation) => guarded(column, Leaf::NotStartsWith(prefix.clone())),
                _ => Self::Unknown,
            },
            _ => Self::Unknown,
        }
    }
}

/// Java's `Expression.toString` of `p` (recalled, unverified: `ref(name="x")
/// == 1`, `(a and b)`, `not_null(ref(name="x"))`, ...).
impl std::fmt::Display for Pred {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let join = |f: &mut std::fmt::Formatter<'_>, xs: &[Pred], op: &str| {
            let parts: Vec<String> = xs.iter().map(ToString::to_string).collect();
            match parts.len() {
                0 => write!(f, "{}", op == "and"),
                1 => write!(f, "{}", parts[0]),
                _ => write!(f, "({})", parts.join(&format!(" {op} "))),
            }
        };
        match self {
            Self::Const(b) => write!(f, "{b}"),
            Self::And(xs) => join(f, xs, "and"),
            Self::Or(xs) => join(f, xs, "or"),
            Self::Unknown => write!(f, "unknown"),
            Self::Leaf(c, l) => {
                let r = format!("ref(name=\"{}\")", c.0.join("."));
                let lit = |v: &ScalarValue| match v {
                    ScalarValue::Utf8(s) => format!("\"{s}\""),
                    other => scalar_text(other),
                };
                let list = |vs: &[ScalarValue]| vs.iter().map(lit).collect::<Vec<_>>().join(", ");
                match l {
                    Leaf::IsNull => write!(f, "is_null({r})"),
                    Leaf::NotNull => write!(f, "not_null({r})"),
                    Leaf::IsNaN => write!(f, "is_nan({r})"),
                    Leaf::NotNaN => write!(f, "not_nan({r})"),
                    Leaf::Lt(v) => write!(f, "{r} < {}", lit(v)),
                    Leaf::LtEq(v) => write!(f, "{r} <= {}", lit(v)),
                    Leaf::Gt(v) => write!(f, "{r} > {}", lit(v)),
                    Leaf::GtEq(v) => write!(f, "{r} >= {}", lit(v)),
                    Leaf::Eq(v) => write!(f, "{r} == {}", lit(v)),
                    Leaf::NotEq(v) => write!(f, "{r} != {}", lit(v)),
                    Leaf::In(vs) => write!(f, "{r} in ({})", list(vs)),
                    Leaf::NotIn(vs) => write!(f, "{r} not in ({})", list(vs)),
                    Leaf::StartsWith(p) => write!(f, "{r} startsWith \"{p}\""),
                    Leaf::NotStartsWith(p) => write!(f, "{r} notStartsWith \"{p}\""),
                }
            }
        }
    }
}

/// A literal's text.
fn scalar_text(v: &ScalarValue) -> String {
    use ScalarValue as S;
    match v {
        S::Null => "null".to_owned(),
        S::Boolean(b) => b.to_string(),
        S::Int8(x) => x.to_string(),
        S::Int16(x) => x.to_string(),
        S::Int32(x) | S::Date32(x) => x.to_string(),
        S::Int64(x) | S::TimestampMicros(x) => x.to_string(),
        S::Float32(x) => x.to_string(),
        S::Float64(x) => x.to_string(),
        S::Decimal128 {
            unscaled, scale, ..
        } => format!("{unscaled}E-{scale}"),
        S::Utf8(s) => s.clone(),
        S::Binary(b) => format!("{b:02X?}"),
        other => format!("{other:?}"),
    }
}

/// [`Pred`] bound to a table's schema and partition specs (module docs).
#[derive(Debug, Clone)]
pub(crate) struct RowFilter<'a> {
    pred: Pred,
    fields: &'a [SchemaField],
    specs: &'a PartitionSpecs,
    metrics: Option<&'a WriterMetrics>,
}

impl<'a> RowFilter<'a> {
    /// `pred` over `fields` (the current schema's) and `specs`.
    pub(crate) fn new(pred: Pred, fields: &'a [SchemaField], specs: &'a PartitionSpecs) -> Self {
        Self {
            pred,
            fields,
            specs,
            metrics: None,
        }
    }

    /// The same filter reading metrics by the writers' modes (`prune.rs`).
    #[must_use]
    pub(crate) fn with_metrics(mut self, metrics: &'a WriterMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Whether some row of `file` may match (Java's `rowsMightMatch`: the
    /// inclusive metrics evaluator on the residual; a file of no rows
    /// cannot).
    pub(crate) fn rows_might_match(&self, file: &ContentFile) -> bool {
        file.record_count > 0
            && self.eval(&self.pred, file, true, &|field, leaf| {
                let Some(op) = inclusive_op(leaf) else {
                    return true;
                };
                let options = metric_options(self.metrics, field.id, PruneOptions::default());
                metric_facts(file, field, options).may_match(op)
            })
    }

    /// Whether every row of `file` matches (Java's `rowsMustMatch`: the
    /// strict metrics evaluator on the residual; a file of no rows does).
    pub(crate) fn rows_must_match(&self, file: &ContentFile) -> bool {
        file.record_count == 0
            || self.eval(&self.pred, file, false, &|field, leaf| {
                let options = metric_options(self.metrics, field.id, PruneOptions::default());
                strict_metrics(file, field, options, leaf)
            })
    }

    /// `p` with each leaf answered as its residual for `file`'s partition
    /// would be: true when the strict projection holds, false when the
    /// inclusive one fails, else `metrics`. `might`: the might-match side,
    /// where something unknown (an untranslated leaf, a column the schema
    /// lacks) may match; on the must-match side it never does (review 36b2b
    /// B1: either way the file is never kept silently).
    fn eval(
        &self,
        p: &Pred,
        file: &ContentFile,
        might: bool,
        metrics: &dyn Fn(&SchemaField, &Leaf) -> bool,
    ) -> bool {
        match p {
            Pred::Const(b) => *b,
            Pred::And(xs) => xs.iter().all(|x| self.eval(x, file, might, metrics)),
            Pred::Or(xs) => xs.iter().any(|x| self.eval(x, file, might, metrics)),
            Pred::Unknown => might,
            Pred::Leaf(c, leaf) => {
                let Some(field) = self.fields.iter().find(|f| &f.path == c) else {
                    return might;
                };
                let spec = self
                    .specs
                    .get(&file.spec_id)
                    .filter(|s| s.fields.len() == file.partition.len());
                let (strict, inclusive) = match spec {
                    None => (false, true),
                    Some(spec) => {
                        let strict = spec.fields.iter().zip(&file.partition).any(|(pf, pv)| {
                            pf.source_id == field.id && strict_partition(pf, &field.ty, leaf, pv)
                        });
                        let inclusive = inclusive_op(leaf).is_none_or(|op| {
                            projected(spec, field, op, |i, pf| {
                                file.partition
                                    .get(i)
                                    .map_or_else(Default::default, |v| partition_facts(v, pf))
                            })
                        });
                        (strict, inclusive)
                    }
                };
                strict || (inclusive && metrics(field, leaf))
            }
        }
    }
}

/// The leaf as `prune.rs`'s inclusive evaluator takes it (`None`: it
/// answers "might match").
fn inclusive_op(leaf: &Leaf) -> Option<Op<'_>> {
    Some(match leaf {
        Leaf::IsNull => Op::IsNull,
        Leaf::NotNull => Op::IsNotNull,
        Leaf::Lt(v) => Op::Cmp(CmpOp::Lt, v),
        Leaf::LtEq(v) => Op::Cmp(CmpOp::LtEq, v),
        Leaf::Gt(v) => Op::Cmp(CmpOp::Gt, v),
        Leaf::GtEq(v) => Op::Cmp(CmpOp::GtEq, v),
        Leaf::Eq(v) => Op::Cmp(CmpOp::Eq, v),
        Leaf::In(vs) => Op::In(vs),
        Leaf::StartsWith(p) => Op::StartsWith(p),
        // Spark's `<>` is FALSE on null where the inclusive evaluator's
        // `NotEq` is; it is guarded by `not_null`, so the same here.
        Leaf::NotEq(v) => Op::Cmp(CmpOp::NotEq, v),
        Leaf::IsNaN | Leaf::NotNaN | Leaf::NotIn(_) | Leaf::NotStartsWith(_) => return None,
    })
}

/// A file's column metrics as Java's `StrictMetricsEvaluator` reads them.
struct Stats<'f> {
    file: &'f ContentFile,
    id: i32,
    /// A `float` or `double` column (the only ones that can hold NaN).
    float: bool,
}

impl Stats<'_> {
    fn count(map: &std::collections::BTreeMap<i32, u64>, id: i32) -> Option<u64> {
        map.get(&id).copied()
    }

    /// `mayContainNull`: unknown counts may.
    fn may_contain_null(&self) -> bool {
        Self::count(&self.file.null_counts, self.id) != Some(0)
    }

    /// `containsNullsOnly`.
    fn contains_nulls_only(&self) -> bool {
        matches!(
            (
                Self::count(&self.file.value_counts, self.id),
                Self::count(&self.file.null_counts, self.id)
            ),
            (Some(v), Some(n)) if v == n
        )
    }

    /// `canContainNulls`, but an unknown count (a map without the column
    /// too) says yes, where Java's says no (module docs: a divergence).
    fn can_contain_nulls(&self) -> bool {
        self.may_contain_null()
    }

    /// `mayContainNaN`.
    fn may_contain_nan(&self) -> bool {
        Self::count(&self.file.nan_counts, self.id) != Some(0)
    }

    /// `containsNaNsOnly`.
    fn contains_nans_only(&self) -> bool {
        matches!(
            (
                Self::count(&self.file.nan_counts, self.id),
                Self::count(&self.file.value_counts, self.id)
            ),
            (Some(n), Some(v)) if n == v
        )
    }

    /// `canContainNaNs`: unknown counts say no, as Java's (used where a NaN
    /// row matches in Spark: `gt`, `gtEq`).
    fn can_contain_nans(&self) -> bool {
        Self::count(&self.file.nan_counts, self.id).is_some_and(|n| n > 0)
    }

    /// Whether NaN rows are not ruled out: a float column whose NaN count is
    /// unknown or above 0 (`lt`, `ltEq`, `eq`, `in`, where a NaN row does
    /// not match; module docs: a divergence).
    fn nan_possible(&self) -> bool {
        self.float && self.may_contain_nan()
    }
}

/// Whether every value is `< v` (`le`: `<=`) under the upper bound `u`.
fn below(u: &Upper, v: &ScalarValue, le: bool) -> bool {
    match u {
        Upper::Exact(u) => match compare(u, v) {
            Some(Ordering::Less) => true,
            Some(Ordering::Equal) => le,
            _ => false,
        },
        // "At most `u`, or a value starting with `u`".
        Upper::Prefix(u) => {
            compare(u, v) == Some(Ordering::Less) && !starts_with(v, u).unwrap_or(true)
        }
    }
}

/// Whether `v` starts with `p` (strings or binaries).
fn starts_with(v: &ScalarValue, p: &ScalarValue) -> Option<bool> {
    match (v, p) {
        (ScalarValue::Utf8(v), ScalarValue::Utf8(p)) => Some(v.starts_with(p.as_str())),
        (ScalarValue::Binary(v), ScalarValue::Binary(p)) => Some(v.starts_with(p)),
        _ => None,
    }
}

/// `StrictEvalVisitor` for one leaf on one top-level or nested column
/// (module docs; notes 36b2 §4's table).
fn strict_metrics(
    file: &ContentFile,
    field: &SchemaField,
    options: Option<PruneOptions>,
    leaf: &Leaf,
) -> bool {
    let s = Stats {
        file,
        id: field.id,
        float: matches!(field.ty, IcebergType::Float | IcebergType::Double),
    };
    let nested = field.path.0.len() > 1;
    let bounds = options.filter(|_| field.ty != IcebergType::Uuid);
    let lower = bounds
        .and(file.lower_bounds.get(&field.id))
        .and_then(|b| decode(b, &field.ty));
    let upper = bounds.and_then(|o| {
        file.upper_bounds
            .get(&field.id)
            .and_then(|b| decode(b, &field.ty))
            .map(|u| upper_bound(u, &field.ty, o))
    });
    let unordered = |v: &ScalarValue| is_unordered(v);
    let gt = |l: &ScalarValue, v: &ScalarValue, ge: bool| match compare(l, v) {
        Some(Ordering::Greater) => true,
        Some(Ordering::Equal) => ge,
        _ => false,
    };
    let no_nulls_or_nans = !(s.can_contain_nulls() || s.can_contain_nans());
    // `lt`, `ltEq`, `eq` and `in` are FALSE on a NaN row: a NaN count must
    // rule NaN out, not merely be missing.
    let no_nulls_nor_nan_rows = no_nulls_or_nans && !s.nan_possible();
    match leaf {
        Leaf::IsNull => !nested && s.contains_nulls_only(),
        Leaf::NotNull => !nested && !s.may_contain_null(),
        Leaf::IsNaN => s.contains_nans_only(),
        Leaf::NotNaN => !s.may_contain_nan() || s.contains_nulls_only(),
        _ if nested => false,
        Leaf::Lt(v) | Leaf::LtEq(v) => {
            let le = matches!(leaf, Leaf::LtEq(_));
            no_nulls_nor_nan_rows
                && !unordered(v)
                && upper.as_ref().is_some_and(|u| below(u, v, le))
        }
        Leaf::Gt(v) | Leaf::GtEq(v) => {
            let ge = matches!(leaf, Leaf::GtEq(_));
            no_nulls_or_nans
                && !unordered(v)
                && lower.as_ref().is_some_and(|l| !is_nan(l) && gt(l, v, ge))
        }
        Leaf::Eq(v) => {
            no_nulls_nor_nan_rows
                && !unordered(v)
                && exact_single(lower.as_ref(), upper.as_ref())
                    .is_some_and(|x| compare(x, v) == Some(Ordering::Equal))
        }
        Leaf::In(vs) => {
            no_nulls_nor_nan_rows
                && exact_single(lower.as_ref(), upper.as_ref()).is_some_and(|x| {
                    vs.iter()
                        .any(|v| !unordered(v) && compare(x, v) == Some(Ordering::Equal))
                })
        }
        Leaf::NotEq(v) => {
            if s.contains_nulls_only() || s.contains_nans_only() {
                return true;
            }
            if unordered(v) {
                return false;
            }
            if let Some(l) = &lower {
                if is_nan(l) {
                    return false;
                }
                if gt(l, v, false) {
                    return true;
                }
            }
            upper.as_ref().is_some_and(|u| below(u, v, false))
        }
        Leaf::NotIn(vs) => {
            if s.contains_nulls_only() || s.contains_nans_only() {
                return true;
            }
            if lower.as_ref().is_some_and(is_nan) || vs.iter().any(unordered) {
                return false;
            }
            if lower.is_none() && upper.is_none() {
                return false;
            }
            // `anyWithinBounds`: a missing bound is unbounded.
            !vs.iter().any(|v| {
                let above_lower = lower.as_ref().is_none_or(|l| !gt(l, v, false));
                let below_upper = upper.as_ref().is_none_or(|u| !beyond(u, v));
                above_lower && below_upper
            })
        }
        Leaf::StartsWith(p) => {
            let n = p.chars().count();
            let has = |b: &ScalarValue| match b {
                ScalarValue::Utf8(b) => b.chars().count() >= n && b.starts_with(p.as_str()),
                _ => false,
            };
            !s.can_contain_nulls()
                && lower.as_ref().is_some_and(has)
                && upper.as_ref().is_some_and(|u| match u {
                    Upper::Exact(u) | Upper::Prefix(u) => has(u),
                })
        }
        Leaf::NotStartsWith(p) => {
            if s.contains_nulls_only() {
                return true;
            }
            let n = u32::try_from(p.chars().count()).unwrap_or(u32::MAX);
            let cut = |b: &ScalarValue| match b {
                ScalarValue::Utf8(b) => Some(truncate_str(b, n)),
                _ => None,
            };
            let lower_above = lower
                .as_ref()
                .and_then(cut)
                .is_some_and(|l| l.as_bytes() > p.as_bytes());
            let upper_below = upper.as_ref().is_some_and(|u| match u {
                Upper::Exact(u) => cut(u).is_some_and(|u| u.as_bytes() < p.as_bytes()),
                // A prefix `u` that `p` starts with may stand for values
                // starting with `p` (module docs).
                Upper::Prefix(u) => {
                    cut(u).is_some_and(|c| c.as_bytes() < p.as_bytes())
                        && !starts_with(&ScalarValue::Utf8(p.clone()), u).unwrap_or(true)
                }
            });
            lower_above || upper_below
        }
    }
}

/// The single value both bounds pin when the upper bound is exact and equal
/// to the lower one.
fn exact_single<'b>(
    lower: Option<&'b ScalarValue>,
    upper: Option<&Upper>,
) -> Option<&'b ScalarValue> {
    let (Some(l), Some(Upper::Exact(u))) = (lower, upper) else {
        return None;
    };
    (compare(l, u) == Some(Ordering::Equal)).then_some(l)
}

/// Whether `v` is certainly above every value under the upper bound `u`.
fn beyond(u: &Upper, v: &ScalarValue) -> bool {
    match u {
        Upper::Exact(u) => compare(v, u) == Some(Ordering::Greater),
        Upper::Prefix(u) => {
            compare(v, u) == Some(Ordering::Greater) && !starts_with(v, u).unwrap_or(true)
        }
    }
}

/// A partition value as the strict projection reads it: `Some(None)` for
/// null, `None` when it cannot be read.
fn partition_value(pv: &PartitionValue, pf: &PartitionFieldInfo) -> Option<Option<ScalarValue>> {
    let ty = pf.result_type.as_ref()?;
    match pv {
        PartitionValue::Null => Some(None),
        PartitionValue::Value(b) => decode(b, ty).map(Some),
        PartitionValue::Unreadable => None,
    }
}

/// Whether the strict projection of `leaf` (on a column of type `source`)
/// through `pf` holds for the partition value `pv` (module docs): every row
/// of a file in that partition matches.
fn strict_partition(
    pf: &PartitionFieldInfo,
    source: &IcebergType,
    leaf: &Leaf,
    pv: &PartitionValue,
) -> bool {
    let t = pf.transform;
    if matches!(t, Transform::Void | Transform::Unknown) {
        return false;
    }
    let Some(p) = partition_value(pv, pf) else {
        return false;
    };
    // Every other transform maps null to null and only null to null.
    match leaf {
        Leaf::IsNull => return p.is_none(),
        Leaf::NotNull => return p.is_some(),
        _ => {}
    }
    // A null partition value proves nothing else (conservative: Iceberg's
    // `notEq` would hold on it, Spark's guard does not).
    let Some(p) = p else {
        return false;
    };
    match t {
        Transform::Identity => identity(&p, leaf),
        Transform::Bucket(_) => {
            let ne = |v: &ScalarValue| {
                apply(t, source, v).is_some_and(|b| compare(&p, &b).is_some_and(Ordering::is_ne))
            };
            match leaf {
                Leaf::NotEq(v) => ne(v),
                Leaf::NotIn(vs) => vs.iter().all(ne),
                _ => false,
            }
        }
        Transform::Truncate(_) => match source {
            IcebergType::Int | IcebergType::Long => truncate_number(t, source, &p, leaf),
            IcebergType::String => truncate_string(t, source, &p, leaf),
            _ => false,
        },
        _ if is_time(t) => time(t, source, &p, leaf),
        _ => false,
    }
}

/// Identity: the leaf on the partition value itself.
fn identity(p: &ScalarValue, leaf: &Leaf) -> bool {
    if is_nan(p) {
        return matches!(leaf, Leaf::IsNaN);
    }
    let cmp = |v: &ScalarValue| (!is_unordered(v)).then(|| compare(p, v)).flatten();
    let is = |v: &ScalarValue, f: fn(Ordering) -> bool| cmp(v).is_some_and(f);
    match leaf {
        Leaf::IsNull | Leaf::IsNaN => false,
        Leaf::NotNull | Leaf::NotNaN => true,
        Leaf::Lt(v) => is(v, Ordering::is_lt),
        Leaf::LtEq(v) => is(v, Ordering::is_le),
        Leaf::Gt(v) => is(v, Ordering::is_gt),
        Leaf::GtEq(v) => is(v, Ordering::is_ge),
        Leaf::Eq(v) => is(v, Ordering::is_eq),
        Leaf::NotEq(v) => is(v, Ordering::is_ne),
        Leaf::In(vs) => vs.iter().any(|v| is(v, Ordering::is_eq)),
        Leaf::NotIn(vs) => vs.iter().all(|v| is(v, Ordering::is_ne)),
        Leaf::StartsWith(s) => matches!(p, ScalarValue::Utf8(x) if x.starts_with(s.as_str())),
        Leaf::NotStartsWith(s) => matches!(p, ScalarValue::Utf8(x) if !x.starts_with(s.as_str())),
    }
}

/// `v + d` for an integer, date or timestamp (`None` on overflow or another
/// type).
fn step(v: &ScalarValue, d: i64) -> Option<ScalarValue> {
    use ScalarValue as S;
    Some(match v {
        S::Int32(x) => S::Int32(i32::try_from(i64::from(*x).checked_add(d)?).ok()?),
        S::Date32(x) => S::Date32(i32::try_from(i64::from(*x).checked_add(d)?).ok()?),
        S::Int64(x) => S::Int64(x.checked_add(d)?),
        S::TimestampMicros(x) => S::TimestampMicros(x.checked_add(d)?),
        _ => return None,
    })
}

/// Truncate of an int or long (`truncateIntegerStrict` / `truncateLongStrict`).
fn truncate_number(t: Transform, source: &IcebergType, p: &ScalarValue, leaf: &Leaf) -> bool {
    let tv = |v: &ScalarValue| apply(t, source, v);
    let ne = |v: &ScalarValue| tv(v).is_some_and(|x| compare(p, &x).is_some_and(Ordering::is_ne));
    match leaf {
        Leaf::NotEq(v) => return ne(v),
        Leaf::NotIn(vs) => return vs.iter().all(ne),
        _ => {}
    }
    // Java's wrapping arithmetic misplaces values near the type's minimum
    // (`prune.rs`): no range projection from a wrapped partition value or
    // literal, nor past `W = 2^30`.
    if wide_truncate(t)
        || wrapped_windows(t, source)
            .iter()
            .any(|w| compare(p, w) == Some(Ordering::Greater))
    {
        return false;
    }
    let bound = |v: Option<ScalarValue>| {
        let v = v?;
        let fv = tv(&v)?;
        (!wraps(&v, &fv)).then_some(fv)
    };
    let holds = |v: Option<ScalarValue>, f: fn(Ordering) -> bool| {
        bound(v).is_some_and(|b| compare(p, &b).is_some_and(f))
    };
    match leaf {
        Leaf::Lt(v) => holds(Some(v.clone()), Ordering::is_lt),
        Leaf::LtEq(v) => holds(step(v, 1), Ordering::is_lt),
        Leaf::Gt(v) => holds(Some(v.clone()), Ordering::is_gt),
        Leaf::GtEq(v) => holds(step(v, -1), Ordering::is_gt),
        _ => false,
    }
}

/// Truncate of a string (`Truncate.java:318-394`'s strict side).
fn truncate_string(t: Transform, source: &IcebergType, p: &ScalarValue, leaf: &Leaf) -> bool {
    let (Transform::Truncate(w), ScalarValue::Utf8(ps)) = (t, p) else {
        return false;
    };
    let width = usize::try_from(w).unwrap_or(usize::MAX);
    let tv = |v: &ScalarValue| apply(t, source, v);
    let cmp = |v: &ScalarValue, f: fn(Ordering) -> bool| {
        tv(v).is_some_and(|x| compare(p, &x).is_some_and(f))
    };
    match leaf {
        Leaf::StartsWith(s) => match s.chars().count().cmp(&width) {
            Ordering::Less => ps.starts_with(s.as_str()),
            Ordering::Equal => ps == s,
            Ordering::Greater => false,
        },
        Leaf::NotStartsWith(s) => match s.chars().count().cmp(&width) {
            Ordering::Less => !ps.starts_with(s.as_str()),
            Ordering::Equal => ps != s,
            Ordering::Greater => !ps.starts_with(truncate_str(s, w).as_str()),
        },
        Leaf::Lt(v) | Leaf::LtEq(v) => cmp(v, Ordering::is_lt),
        Leaf::Gt(v) | Leaf::GtEq(v) => cmp(v, Ordering::is_gt),
        Leaf::NotEq(v) => cmp(v, Ordering::is_ne),
        Leaf::NotIn(vs) => vs.iter().all(|v| cmp(v, Ordering::is_ne)),
        _ => false,
    }
}

/// The time unit a year, month, day or hour partition value or transformed
/// literal stands for.
fn unit(v: &ScalarValue) -> Option<i64> {
    match v {
        ScalarValue::Int32(x) | ScalarValue::Date32(x) => Some(i64::from(*x)),
        _ => None,
    }
}

/// Year, month, day and hour (`truncateLongStrict` with
/// `fixStrictTimeProjection`; module docs): a partition value `p <= 0` may
/// hold rows of the unit `p - 1`.
fn time(t: Transform, source: &IcebergType, p: &ScalarValue, leaf: &Leaf) -> bool {
    let Some(hi) = unit(p) else {
        return false;
    };
    let lo = if hi <= 0 { hi - 1 } else { hi };
    let tv = |v: Option<ScalarValue>| v.and_then(|v| apply(t, source, &v)).and_then(|x| unit(&x));
    let outside = |v: &ScalarValue| tv(Some(v.clone())).is_some_and(|u| u < lo || u > hi);
    match leaf {
        Leaf::Lt(v) => tv(Some(v.clone())).is_some_and(|u| hi < u),
        Leaf::LtEq(v) => tv(step_time(v, 1)).is_some_and(|u| hi < u),
        Leaf::Gt(v) => tv(Some(v.clone())).is_some_and(|u| lo > u),
        Leaf::GtEq(v) => tv(step_time(v, -1)).is_some_and(|u| lo > u),
        Leaf::NotEq(v) => outside(v),
        Leaf::NotIn(vs) => vs.iter().all(outside),
        _ => false,
    }
}

/// `v + d` in its own unit (a day for a date, a microsecond for a
/// timestamp).
fn step_time(v: &ScalarValue, d: i64) -> Option<ScalarValue> {
    match v {
        ScalarValue::Date32(_) | ScalarValue::TimestampMicros(_) => step(v, d),
        _ => None,
    }
}

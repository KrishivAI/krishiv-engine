// SPDX-License-Identifier: Apache-2.0
//! The might-match evaluator over Krishiv-owned structs, with no I/O (design
//! note §2.2): manifest-list partition summaries, a file's partition tuple
//! (the inclusive projection of the predicate onto each spec's partition
//! fields), and a file's column metrics. Each answers `false` only when no row
//! of the manifest or file can satisfy the predicate; anything unknown (a
//! column the schema lacks, a bound that does not decode, an unknown transform,
//! missing counts, a collation other than binary) answers `true`.
//!
//! The rules that keep it sound under Spark's row semantics:
//! - **NaN.** Bounds exclude NaN, and Spark orders NaN above every value. A
//!   float column that may hold NaN is never pruned by its bounds (review-1 M8:
//!   Iceberg's own evaluator can prune such files, so an oracle answer may lack
//!   rows Krishiv returns). A NaN or zero literal never prunes.
//! - **Truncated strings and binaries.** A lower bound is always a valid lower
//!   bound (a prefix is not larger). An upper bound is used as exact only in
//!   [`PruneMode::Scan`] and only when it is shorter than the writer's truncation
//!   length; otherwise it means "at most this value, or a value starting with
//!   it", so an equal-length-prefix tie never prunes.
//! - **Promoted columns.** Bounds decode by stored width into the read type
//!   ([`super::bounds`]).
//! - **Time transforms before 1970.** Old Java writers rounded those toward zero;
//!   the projection of `<`, `<=`, `=` and `IN` is widened by one unit for a
//!   negative value, as Iceberg's `fixInclusiveTimeProjection` does.
//!
//! [`PruneMode::Conservative`] is for T027's conflict validation, where a wrong
//! "cannot match" is a lost update (review-1 S11): no assumption about the
//! writer's truncation, and no inference that a partition summary without
//! bounds holds only nulls.

use std::cmp::Ordering;

use krishiv_io::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};

use super::bounds::{compare, decode, is_nan, is_unordered};
use super::entry::{ContentFile, ManifestFileEntry, PartitionFieldSummary, PartitionValue};
use super::transform::{apply, is_time, truncate_str};
use super::types::{
    IcebergType, PartitionFieldInfo, PartitionSpecInfo, PartitionSpecs, SchemaField, Transform,
};

/// The writer's default string and binary bound truncation
/// (`write.metadata.metrics.default` = `truncate(16)`).
pub const DEFAULT_TRUNCATE_LENGTH: usize = 16;

/// How much the evaluator may assume about writers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PruneMode {
    /// Scan planning: an upper string or binary bound shorter than
    /// [`PruneOptions::truncate_length`] is exact, and a partition summary with
    /// no bounds and no NaN holds only nulls (Java's `ManifestEvaluator`).
    #[default]
    Scan,
    /// Conflict validation (T027): every unknown means "may match"; no upper
    /// string or binary bound is taken as exact.
    Conservative,
}

/// Options of a [`Pruner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PruneOptions {
    /// The mode.
    pub mode: PruneMode,
    /// The writer's string/binary bound truncation length (code points for
    /// strings, bytes for binaries).
    pub truncate_length: usize,
}

impl Default for PruneOptions {
    fn default() -> Self {
        Self {
            mode: PruneMode::Scan,
            truncate_length: DEFAULT_TRUNCATE_LENGTH,
        }
    }
}

impl PruneOptions {
    /// Conservative options (conflict validation).
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            mode: PruneMode::Conservative,
            ..Self::default()
        }
    }
}

/// A predicate bound to a schema and the table's partition specs.
#[derive(Debug, Clone)]
pub struct Pruner<'a> {
    expr: PushdownExpr,
    fields: &'a [SchemaField],
    specs: &'a PartitionSpecs,
    options: PruneOptions,
}

/// A leaf of the predicate, bound to its column.
#[derive(Debug, Clone, Copy)]
enum Op<'e> {
    Cmp(CmpOp, &'e ScalarValue),
    In(&'e [ScalarValue]),
    IsNull,
    IsNotNull,
    StartsWith(&'e str),
}

impl<'a> Pruner<'a> {
    /// `expr` (in the read schema's names and types) against `fields` and `specs`.
    #[must_use]
    pub fn new(
        expr: &PushdownExpr,
        fields: &'a [SchemaField],
        specs: &'a PartitionSpecs,
        options: PruneOptions,
    ) -> Self {
        Self {
            expr: expr.to_nnf(),
            fields,
            specs,
            options,
        }
    }

    /// Whether the files of `manifest` may hold a matching row, by its partition
    /// summaries.
    #[must_use]
    pub fn manifest_may_match(&self, manifest: &ManifestFileEntry) -> bool {
        let Some(spec) = self.specs.get(&manifest.spec_id) else {
            return true;
        };
        let Some(summaries) = &manifest.partitions else {
            return true;
        };
        if summaries.len() != spec.fields.len() {
            return true;
        }
        self.eval(&self.expr, &|field, op| {
            projected(spec, field, op, |i, pf| {
                summary_facts(&summaries[i], pf, self.options)
            })
        })
    }

    /// Whether `file` may hold a matching row: it has rows, and neither its
    /// partition tuple nor its column metrics rule the predicate out.
    #[must_use]
    pub fn file_may_match(&self, file: &ContentFile) -> bool {
        file.record_count > 0 && self.partition_may_match(file) && self.metrics_may_match(file)
    }

    /// Whether `file`'s partition tuple allows a matching row.
    #[must_use]
    pub fn partition_may_match(&self, file: &ContentFile) -> bool {
        let Some(spec) = self.specs.get(&file.spec_id) else {
            return true;
        };
        if file.partition.len() != spec.fields.len() {
            return true;
        }
        self.eval(&self.expr, &|field, op| {
            projected(spec, field, op, |i, pf| {
                partition_facts(&file.partition[i], pf)
            })
        })
    }

    /// Whether `file`'s column metrics allow a matching row.
    #[must_use]
    pub fn metrics_may_match(&self, file: &ContentFile) -> bool {
        self.eval(&self.expr, &|field, op| {
            metric_facts(file, field, self.options).may_match(op)
        })
    }

    fn field(&self, path: &ColumnPath) -> Option<&'a SchemaField> {
        self.fields.iter().find(|f| &f.path == path)
    }

    /// The NNF tree with each bound leaf answered by `leaf` (unknown leaves: true).
    fn eval(&self, e: &PushdownExpr, leaf: &dyn Fn(&SchemaField, Op<'_>) -> bool) -> bool {
        let binary = |c: &Collation| *c == Collation::Utf8Binary;
        let at = |path: &ColumnPath, op: Op<'_>| self.field(path).is_none_or(|f| leaf(f, op));
        match e {
            PushdownExpr::Constant(b) => *b,
            PushdownExpr::And(xs) => xs.iter().all(|x| self.eval(x, leaf)),
            PushdownExpr::Or(xs) => xs.iter().any(|x| self.eval(x, leaf)),
            PushdownExpr::Cmp {
                column,
                op,
                value,
                collation,
            } if binary(collation) => at(column, Op::Cmp(*op, value)),
            PushdownExpr::In {
                column,
                values,
                collation,
            } if binary(collation) => at(column, Op::In(values)),
            PushdownExpr::IsNull(c) => at(c, Op::IsNull),
            PushdownExpr::IsNotNull(c) => at(c, Op::IsNotNull),
            PushdownExpr::StartsWith {
                column,
                prefix,
                collation,
            } if binary(collation) => at(column, Op::StartsWith(prefix)),
            // `Not` leaves after NNF, substrings, runtime filters, collations.
            _ => true,
        }
    }
}

/// An upper bound: exact, or "at most this value or any value starting with it"
/// (a string or binary bound that may have been truncated without rounding up).
#[derive(Debug, Clone)]
enum Upper {
    Exact(ScalarValue),
    Prefix(ScalarValue),
}

/// What is known about the non-null values of one column (or partition value)
/// of a file or manifest.
#[derive(Debug, Clone, Default)]
struct Facts {
    /// A value at or below every non-null, non-NaN value.
    lower: Option<ScalarValue>,
    /// A bound above every non-null, non-NaN value.
    upper: Option<Upper>,
    /// `Some(false)`: no value is null.
    has_nulls: Option<bool>,
    /// Every value is null (comparisons, `IS NOT NULL` and `startswith` fail).
    all_null: bool,
    /// Some value may be NaN: bounds are not used.
    maybe_nan: bool,
}

impl Facts {
    fn may_match(&self, op: Op<'_>) -> bool {
        match op {
            Op::IsNull => self.has_nulls != Some(false),
            Op::IsNotNull => !self.all_null,
            _ if self.all_null => false,
            Op::Cmp(op, v) => self.cmp(op, v),
            Op::In(vs) => vs.iter().any(|v| self.cmp(CmpOp::Eq, v)),
            Op::StartsWith(p) => self.starts_with(p),
        }
    }

    fn cmp(&self, op: CmpOp, v: &ScalarValue) -> bool {
        if self.maybe_nan || is_unordered(v) {
            return true;
        }
        let lo = self.lower.as_ref().and_then(|l| compare(l, v));
        match op {
            CmpOp::Eq => !(lo == Some(Ordering::Greater) || self.all_below(v)),
            CmpOp::NotEq => {
                !(lo == Some(Ordering::Equal)
                    && matches!(&self.upper, Some(Upper::Exact(u)) if compare(u, v) == Some(Ordering::Equal)))
            }
            CmpOp::Lt => !matches!(lo, Some(Ordering::Greater | Ordering::Equal)),
            CmpOp::LtEq => lo != Some(Ordering::Greater),
            CmpOp::Gt => !self.all_at_most(v),
            CmpOp::GtEq => !self.all_below(v),
        }
    }

    /// Every value is `< v`.
    fn all_below(&self, v: &ScalarValue) -> bool {
        match &self.upper {
            None => false,
            Some(Upper::Exact(u)) => compare(u, v) == Some(Ordering::Less),
            Some(Upper::Prefix(u)) => {
                compare(u, v) == Some(Ordering::Less) && !has_prefix(v, u).unwrap_or(true)
            }
        }
    }

    /// Every value is `<= v`.
    fn all_at_most(&self, v: &ScalarValue) -> bool {
        match &self.upper {
            None => false,
            Some(Upper::Exact(u)) => {
                matches!(compare(u, v), Some(Ordering::Less | Ordering::Equal))
            }
            Some(Upper::Prefix(_)) => self.all_below(v),
        }
    }

    fn starts_with(&self, p: &str) -> bool {
        let pv = ScalarValue::Utf8(p.to_owned());
        // Every value is below `p`, so none starts with it.
        if self.all_below(&pv) {
            return false;
        }
        // Every value is above every string that starts with `p`.
        if let Some(ScalarValue::Utf8(l)) = &self.lower
            && l.as_str() > p
            && !l.starts_with(p)
        {
            return false;
        }
        true
    }
}

/// Whether `v` starts with `u` (strings or binaries); `None` for other types.
fn has_prefix(v: &ScalarValue, u: &ScalarValue) -> Option<bool> {
    match (v, u) {
        (ScalarValue::Utf8(v), ScalarValue::Utf8(u)) => Some(v.starts_with(u.as_str())),
        (ScalarValue::Binary(v), ScalarValue::Binary(u)) => Some(v.starts_with(u)),
        _ => None,
    }
}

/// `v` as an upper bound: exact unless it is a string or binary that may be
/// truncated (see the module docs).
fn upper_bound(v: ScalarValue, ty: &IcebergType, options: PruneOptions) -> Upper {
    let len = match (&v, ty) {
        (ScalarValue::Utf8(s), IcebergType::String) => Some(s.chars().count()),
        (ScalarValue::Binary(b), IcebergType::Binary | IcebergType::Fixed(_)) => Some(b.len()),
        _ => None,
    };
    match len {
        Some(n) if options.mode == PruneMode::Conservative || n >= options.truncate_length => {
            Upper::Prefix(v)
        }
        _ => Upper::Exact(v),
    }
}

fn metric_facts(file: &ContentFile, field: &SchemaField, options: PruneOptions) -> Facts {
    let id = field.id;
    let values = file.value_counts.get(&id).copied();
    let nulls = file.null_counts.get(&id).copied();
    let nans = file.nan_counts.get(&id).copied();
    let all_null = matches!((values, nulls), (Some(v), Some(n)) if v == n);
    let maybe_nan = field.ty.is_float() && nans != Some(0) && !all_null;
    Facts {
        lower: file
            .lower_bounds
            .get(&id)
            .and_then(|b| decode(b, &field.ty)),
        upper: file
            .upper_bounds
            .get(&id)
            .and_then(|b| decode(b, &field.ty))
            .map(|u| upper_bound(u, &field.ty, options)),
        has_nulls: nulls.map(|n| n > 0),
        all_null,
        maybe_nan,
    }
}

fn partition_facts(value: &PartitionValue, pf: &PartitionFieldInfo) -> Facts {
    let Some(ty) = &pf.result_type else {
        return Facts::default();
    };
    match value {
        PartitionValue::Null => Facts {
            has_nulls: Some(true),
            all_null: true,
            ..Facts::default()
        },
        PartitionValue::Value(bytes) => match decode(bytes, ty) {
            Some(v) => Facts {
                maybe_nan: is_nan(&v),
                lower: Some(v.clone()),
                upper: Some(Upper::Exact(v)),
                has_nulls: Some(false),
                all_null: false,
            },
            None => Facts::default(),
        },
        PartitionValue::Unreadable => Facts::default(),
    }
}

fn summary_facts(
    s: &PartitionFieldSummary,
    pf: &PartitionFieldInfo,
    options: PruneOptions,
) -> Facts {
    let Some(ty) = &pf.result_type else {
        return Facts::default();
    };
    let lower = s.lower.as_ref().and_then(|b| decode(b, ty));
    let upper = s
        .upper
        .as_ref()
        .and_then(|b| decode(b, ty))
        .map(|u| upper_bound(u, ty, options));
    let no_nan = !ty.is_float() || s.contains_nan == Some(false);
    // Java: no lower bound means every value is null (or NaN, excluded here). Only
    // when both bounds were absent as written, never when one failed to decode.
    let all_null =
        options.mode == PruneMode::Scan && s.lower.is_none() && s.upper.is_none() && no_nan;
    Facts {
        lower,
        upper,
        has_nulls: Some(s.contains_null),
        all_null,
        maybe_nan: !no_nan,
    }
}

/// A leaf on `field` answered through each partition field of `spec` whose
/// source is `field`: false when one of them rules it out.
fn projected(
    spec: &PartitionSpecInfo,
    field: &SchemaField,
    op: Op<'_>,
    facts: impl Fn(usize, &PartitionFieldInfo) -> Facts,
) -> bool {
    for (i, pf) in spec.fields.iter().enumerate() {
        if pf.source_id != field.id
            || matches!(pf.transform, Transform::Void | Transform::Unknown)
            || pf.result_type.is_none()
        {
            continue;
        }
        if !project(pf, &field.ty, op, &facts(i, pf)) {
            return false;
        }
    }
    true
}

/// The inclusive projection of `source op` through `pf`, answered on `facts`.
fn project(pf: &PartitionFieldInfo, source: &IcebergType, op: Op<'_>, facts: &Facts) -> bool {
    let t = pf.transform;
    let f = |v: &ScalarValue| apply(t, source, v);
    match op {
        // Every non-void transform maps null to null and only null to null.
        Op::IsNull | Op::IsNotNull => facts.may_match(op),
        Op::Cmp(op, v) => {
            if is_unordered(v) {
                return true;
            }
            if t == Transform::Identity {
                return facts.may_match(Op::Cmp(op, v));
            }
            let Some(fv) = f(v) else {
                return true;
            };
            match (t, op) {
                (Transform::Bucket(_), CmpOp::Eq) => facts.may_match(Op::Cmp(CmpOp::Eq, &fv)),
                (Transform::Bucket(_), _) | (_, CmpOp::NotEq) => true,
                (_, CmpOp::Eq) => widened(t, &fv)
                    .iter()
                    .any(|x| facts.may_match(Op::Cmp(CmpOp::Eq, x))),
                (_, CmpOp::Lt | CmpOp::LtEq) => {
                    let hi = widened(t, &fv);
                    let top = hi.last().unwrap_or(&fv);
                    facts.may_match(Op::Cmp(CmpOp::LtEq, top))
                }
                (_, CmpOp::Gt | CmpOp::GtEq) => facts.may_match(Op::Cmp(CmpOp::GtEq, &fv)),
            }
        }
        Op::In(vs) => vs
            .iter()
            .any(|v| project(pf, source, Op::Cmp(CmpOp::Eq, v), facts)),
        Op::StartsWith(p) => match t {
            Transform::Identity => facts.may_match(op),
            Transform::Truncate(w) if *source == IcebergType::String => {
                let n = usize::try_from(w).unwrap_or(usize::MAX);
                if p.chars().count() <= n {
                    facts.may_match(op)
                } else {
                    let v = ScalarValue::Utf8(truncate_str(p, w));
                    facts.may_match(Op::Cmp(CmpOp::Eq, &v))
                }
            }
            _ => true,
        },
    }
}

/// `fv`, and for a negative time-transform value also `fv + 1` (see the module
/// docs), ascending.
fn widened(t: Transform, fv: &ScalarValue) -> Vec<ScalarValue> {
    let next = match fv {
        ScalarValue::Int32(x) if is_time(t) && *x < 0 => Some(ScalarValue::Int32(x + 1)),
        ScalarValue::Date32(x) if is_time(t) && *x < 0 => Some(ScalarValue::Date32(x + 1)),
        _ => None,
    };
    let mut out = vec![fv.clone()];
    out.extend(next);
    out
}

// SPDX-License-Identifier: Apache-2.0
//! Statistics and sound pruning.
//!
//! [`FileStatistics`] describe a file or one of its units (row group,
//! stripe, chunk) and feed the optimizer and pruning. [`prune`] answers, for
//! a unit, whether a predicate holds for **all** rows ([`TriState::True`]),
//! **no** row ([`TriState::False`], the unit is skipped) or neither
//! ([`TriState::Unknown`]). It is sound by construction: it uses only exact
//! bounds, and answers unknown for NaN, signed zeros, absent null counts,
//! non-binary collations and inexact (truncated or legacy-writer) bounds, so
//! a skipped unit never holds a matching row.
//!
//! **NaN.** Spark orders NaN above every other value and equal to itself,
//! but the Parquet specification keeps NaN out of float min/max statistics.
//! So unless [`ColumnStatistics::nan_count`] is exactly zero, a float unit
//! may hold NaN rows its bounds do not show: comparisons NaN satisfies
//! (`>`, `>=`, `<>`) never prune it, and comparisons NaN fails (`=`, `<`,
//! `<=`) never claim every row.

use std::cmp::Ordering;

use crate::request::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};
use krishiv_common::SketchRef;

/// A value that is exact, an estimate or bound that must not be used for
/// pruning (truncated min/max, statistics of writers known to order them
/// wrongly, INT96), or absent.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Precision<T> {
    /// The exact value.
    Exact(T),
    /// An estimate: fine for cost estimates, never for pruning.
    Inexact(T),
    /// Unknown.
    #[default]
    Absent,
}

impl<T> Precision<T> {
    /// The value if exact.
    pub fn exact(&self) -> Option<&T> {
        match self {
            Self::Exact(v) => Some(v),
            Self::Inexact(_) | Self::Absent => None,
        }
    }

    /// The value, exact or not.
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Exact(v) | Self::Inexact(v) => Some(v),
            Self::Absent => None,
        }
    }

    /// The same value, demoted to an estimate.
    #[must_use]
    pub fn to_inexact(self) -> Self {
        match self {
            Self::Exact(v) | Self::Inexact(v) => Self::Inexact(v),
            Self::Absent => Self::Absent,
        }
    }
}

/// Statistics of one column (a leaf or a top-level column).
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct ColumnStatistics {
    /// The column, in the file's names.
    pub path: ColumnPath,
    /// Number of nulls.
    pub null_count: Precision<u64>,
    /// Smallest non-null value, in the column's own type.
    pub min: Precision<ScalarValue>,
    /// Largest non-null value.
    pub max: Precision<ScalarValue>,
    /// Number of NaN values (float columns). Unless it is exactly zero, the
    /// unit may hold NaN rows that `min`/`max` do not cover (the Parquet
    /// specification keeps NaN out of them), and pruning accounts for them.
    pub nan_count: Precision<u64>,
    /// Number of distinct non-null values (an estimate or absent; never
    /// used for pruning, only for costing).
    pub distinct: Precision<u64>,
    /// A serialized distinct-count sketch (a later statistics task reads
    /// it); pruning ignores it.
    pub sketch: Option<SketchRef>,
}

/// Statistics of a file or of a unit of one. Formats without statistics
/// return [`FileStatistics::unknown`], never a guess.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct FileStatistics {
    /// Number of rows.
    pub rows: Precision<u64>,
    /// Bytes in storage.
    pub bytes: Precision<u64>,
    /// Per-column statistics; columns not listed are unknown.
    pub columns: Vec<ColumnStatistics>,
}

impl ColumnStatistics {
    /// Statistics of `path` with every value unknown; set the fields on the
    /// result.
    #[must_use]
    pub fn new(path: ColumnPath) -> Self {
        Self {
            path,
            ..Self::default()
        }
    }
}

impl FileStatistics {
    /// Statistics with a row count and nothing else known.
    #[must_use]
    pub fn with_rows(rows: Precision<u64>) -> Self {
        Self {
            rows,
            ..Self::default()
        }
    }

    /// Nothing known.
    #[must_use]
    pub fn unknown() -> Self {
        Self::default()
    }

    /// The statistics of `path`, if known.
    #[must_use]
    pub fn column(&self, path: &ColumnPath) -> Option<&ColumnStatistics> {
        self.columns.iter().find(|c| &c.path == path)
    }
}

/// A three-valued answer about the rows of a unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriState {
    /// Every row satisfies the predicate.
    True,
    /// No row satisfies it: the unit can be skipped.
    False,
    /// Some rows may satisfy it.
    Unknown,
}

/// Whether `expr` holds for all, none or some rows of the unit described by
/// `stats` (see the module docs). `expr` is in the file's names and types
/// (after [`SchemaMapper::map_predicate`](crate::SchemaMapper::map_predicate)).
#[must_use]
pub fn prune(expr: &PushdownExpr, stats: &FileStatistics) -> TriState {
    eval(&expr.to_nnf(), stats)
}

fn eval(expr: &PushdownExpr, stats: &FileStatistics) -> TriState {
    match expr {
        PushdownExpr::Constant(true) => TriState::True,
        PushdownExpr::Constant(false) => TriState::False,
        PushdownExpr::And(xs) => {
            let mut all = true;
            for x in xs {
                match eval(x, stats) {
                    TriState::False => return TriState::False,
                    TriState::Unknown => all = false,
                    TriState::True => {}
                }
            }
            if all {
                TriState::True
            } else {
                TriState::Unknown
            }
        }
        PushdownExpr::Or(xs) => {
            let mut none = true;
            for x in xs {
                match eval(x, stats) {
                    TriState::True => return TriState::True,
                    TriState::Unknown => none = false,
                    TriState::False => {}
                }
            }
            if none {
                TriState::False
            } else {
                TriState::Unknown
            }
        }
        // After `to_nnf`, only a negated `StartsWith`, `Contains` or
        // `Membership` remains under `Not`. Statistics cannot test
        // membership or substrings.
        PushdownExpr::Not(_) | PushdownExpr::Membership { .. } | PushdownExpr::Contains { .. } => {
            TriState::Unknown
        }
        PushdownExpr::IsNull(c) => is_null(stats, c),
        PushdownExpr::IsNotNull(c) => match is_null(stats, c) {
            TriState::True => TriState::False,
            TriState::False => TriState::True,
            TriState::Unknown => TriState::Unknown,
        },
        PushdownExpr::Cmp {
            column,
            op,
            value,
            collation,
        } => {
            if *collation != Collation::Utf8Binary {
                return TriState::Unknown;
            }
            Bounds::of(stats, column).map_or(TriState::Unknown, |b| b.cmp(*op, value))
        }
        PushdownExpr::In {
            column,
            values,
            collation,
        } => {
            if *collation != Collation::Utf8Binary {
                return TriState::Unknown;
            }
            let Some(b) = Bounds::of(stats, column) else {
                return TriState::Unknown;
            };
            let mut none = true;
            for v in values {
                match b.cmp(CmpOp::Eq, v) {
                    TriState::True => return TriState::True,
                    TriState::Unknown => none = false,
                    TriState::False => {}
                }
            }
            if none {
                TriState::False
            } else {
                TriState::Unknown
            }
        }
        PushdownExpr::StartsWith {
            column,
            prefix,
            collation,
        } => {
            if *collation != Collation::Utf8Binary {
                return TriState::Unknown;
            }
            Bounds::of(stats, column).map_or(TriState::Unknown, |b| b.starts_with(prefix))
        }
    }
}

fn is_null(stats: &FileStatistics, c: &ColumnPath) -> TriState {
    let Some(cs) = stats.column(c) else {
        return TriState::Unknown;
    };
    match (cs.null_count.exact(), stats.rows.exact()) {
        (Some(0), _) => TriState::False,
        (Some(n), Some(rows)) if n == rows => TriState::True,
        _ => TriState::Unknown,
    }
}

/// The usable facts about one column of a unit.
struct Bounds<'a> {
    min: Option<&'a ScalarValue>,
    max: Option<&'a ScalarValue>,
    /// No nulls (exact).
    no_nulls: bool,
    /// Every row is null (exact): no comparison can match.
    all_null: bool,
    /// The unit may hold NaN values its bounds do not cover (float columns).
    maybe_nan: bool,
}

impl<'a> Bounds<'a> {
    fn of(stats: &'a FileStatistics, c: &ColumnPath) -> Option<Self> {
        let cs = stats.column(c)?;
        let usable = |v: &'a Precision<ScalarValue>| v.exact().filter(|v| !is_unordered(v));
        let nulls = cs.null_count.exact().copied();
        Some(Self {
            min: usable(&cs.min),
            max: usable(&cs.max),
            no_nulls: nulls == Some(0),
            all_null: nulls.is_some() && nulls == stats.rows.exact().copied(),
            maybe_nan: cs.nan_count.exact() != Some(&0),
        })
    }

    fn cmp(&self, op: CmpOp, v: &ScalarValue) -> TriState {
        if self.all_null {
            return TriState::False;
        }
        if is_unordered(v) {
            return TriState::Unknown;
        }
        let lo = self.min.and_then(|m| compare(m, v)); // min vs v
        let hi = self.max.and_then(|m| compare(m, v)); // max vs v
        use Ordering::{Equal as E, Greater as G, Less as L};
        let (mut none, mut all) = match op {
            CmpOp::Eq => (
                lo == Some(G) || hi == Some(L),
                lo == Some(E) && hi == Some(E),
            ),
            CmpOp::NotEq => (
                lo == Some(E) && hi == Some(E),
                lo == Some(G) || hi == Some(L),
            ),
            CmpOp::Lt => (matches!(lo, Some(G | E)), hi == Some(L)),
            CmpOp::LtEq => (lo == Some(G), matches!(hi, Some(L | E))),
            CmpOp::Gt => (matches!(hi, Some(L | E)), lo == Some(G)),
            CmpOp::GtEq => (hi == Some(L), matches!(lo, Some(G | E))),
        };
        // NaN rows hidden from float bounds (see the module docs): NaN is
        // greater than and unequal to every non-NaN literal in Spark.
        if self.maybe_nan && matches!(v, ScalarValue::Float32(_) | ScalarValue::Float64(_)) {
            match op {
                CmpOp::Gt | CmpOp::GtEq | CmpOp::NotEq => none = false,
                CmpOp::Eq | CmpOp::Lt | CmpOp::LtEq => all = false,
            }
        }
        if none {
            TriState::False
        } else if all && self.no_nulls {
            TriState::True
        } else {
            TriState::Unknown
        }
    }

    fn starts_with(&self, prefix: &str) -> TriState {
        if self.all_null {
            return TriState::False;
        }
        let (Some(ScalarValue::Utf8(min)), Some(ScalarValue::Utf8(max))) = (self.min, self.max)
        else {
            return TriState::Unknown;
        };
        // Strings with the prefix lie in [prefix, every string above prefix
        // that does not start with it).
        if max.as_str() < prefix || (min.as_str() > prefix && !min.starts_with(prefix)) {
            TriState::False
        } else if self.no_nulls && min.starts_with(prefix) && max.starts_with(prefix) {
            TriState::True
        } else {
            TriState::Unknown
        }
    }
}

/// Values whose order in statistics does not match SQL comparison: NaN
/// (greatest in Spark, unordered in IEEE statistics) and signed zeros
/// (equal in SQL, distinct in statistics).
fn is_unordered(v: &ScalarValue) -> bool {
    match v {
        ScalarValue::Float32(f) => f.is_nan() || *f == 0.0,
        ScalarValue::Float64(f) => f.is_nan() || *f == 0.0,
        ScalarValue::Null => true,
        _ => false,
    }
}

/// SQL order of two values of the same type; `None` when they are not
/// comparable (different types or scales, NaN).
pub(crate) fn compare(a: &ScalarValue, b: &ScalarValue) -> Option<Ordering> {
    use ScalarValue as S;
    match (a, b) {
        (S::Boolean(x), S::Boolean(y)) => Some(x.cmp(y)),
        (S::Int8(x), S::Int8(y)) => Some(x.cmp(y)),
        (S::Int16(x), S::Int16(y)) => Some(x.cmp(y)),
        (S::Int32(x), S::Int32(y)) | (S::Date32(x), S::Date32(y)) => Some(x.cmp(y)),
        (S::Int64(x), S::Int64(y)) | (S::TimestampMicros(x), S::TimestampMicros(y)) => {
            Some(x.cmp(y))
        }
        (S::Float32(x), S::Float32(y)) => x.partial_cmp(y),
        (S::Float64(x), S::Float64(y)) => x.partial_cmp(y),
        (
            S::Decimal128 {
                unscaled: x,
                scale: sx,
                ..
            },
            S::Decimal128 {
                unscaled: y,
                scale: sy,
                ..
            },
        ) if sx == sy => Some(x.cmp(y)),
        (S::Utf8(x), S::Utf8(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
        (S::Binary(x), S::Binary(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

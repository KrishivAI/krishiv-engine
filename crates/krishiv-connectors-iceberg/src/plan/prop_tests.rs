// SPDX-License-Identifier: Apache-2.0
//! `prune_never_skips_a_matching_row`: random typed rows are grouped into files
//! by a random partition spec, given the metrics a writer would record (strings
//! truncated as Java does, or without rounding up, a promoted column's bounds
//! written at the old width, pre-epoch time partitions as old writers computed
//! them), and grouped into manifests with their partition summaries. For random
//! predicates, any file or manifest holding a row that matches under Spark's
//! semantics (three-valued logic, NaN above every value and equal to itself,
//! `-0.0 = 0.0`) must survive, in both modes.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use krishiv_io::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};
use proptest::prelude::*;

use super::bounds::encode;
use super::entry::{
    ContentFile, ManifestContent, ManifestFileEntry, PartitionFieldSummary, PartitionValue,
};
use super::prune::{PruneOptions, Pruner};
use super::transform::{apply, truncate_str};
use super::types::{
    IcebergType, PartitionFieldInfo, PartitionSpecInfo, PartitionSpecs, SchemaField, Transform,
};

const DAY: i64 = 86_400_000_000;
const HOUR: i64 = 3_600_000_000;

/// Columns: a long, s string, d double, p long (written as int), t timestamptz,
/// k int.
#[derive(Debug, Clone)]
struct Row {
    a: Option<i64>,
    s: Option<String>,
    d: Option<f64>,
    p: Option<i32>,
    t: Option<i64>,
    k: Option<i32>,
}

impl Row {
    fn get(&self, c: &str) -> Option<ScalarValue> {
        match c {
            "a" => self.a.map(ScalarValue::Int64),
            "s" => self.s.clone().map(ScalarValue::Utf8),
            "d" => self.d.map(ScalarValue::Float64),
            "p" => self.p.map(|v| ScalarValue::Int64(i64::from(v))),
            "t" => self.t.map(ScalarValue::TimestampMicros),
            "k" => self.k.map(ScalarValue::Int32),
            _ => None,
        }
    }
}

fn fields() -> Vec<SchemaField> {
    let f = |id, n: &str, ty| SchemaField {
        id,
        path: ColumnPath::new(n),
        ty,
    };
    vec![
        f(1, "a", IcebergType::Long),
        f(2, "s", IcebergType::String),
        f(3, "d", IcebergType::Double),
        f(4, "p", IcebergType::Long),
        f(5, "t", IcebergType::Timestamptz),
        f(6, "k", IcebergType::Int),
    ]
}

fn type_of(id: i32) -> IcebergType {
    fields()
        .into_iter()
        .find(|f| f.id == id)
        .map(|f| f.ty)
        .unwrap_or(IcebergType::Boolean)
}

fn name_of(id: i32) -> &'static str {
    ["", "a", "s", "d", "p", "t", "k"][usize::try_from(id).unwrap()]
}

fn candidate_fields() -> Vec<(i32, Transform)> {
    vec![
        (6, Transform::Identity),
        (1, Transform::Identity),
        (1, Transform::Bucket(3)),
        (1, Transform::Truncate(4)),
        (2, Transform::Truncate(2)),
        (2, Transform::Identity),
        (2, Transform::Bucket(2)),
        (3, Transform::Identity),
        (3, Transform::Void),
        (4, Transform::Identity),
        (4, Transform::Truncate(5)),
        (5, Transform::Day),
        (5, Transform::Hour),
        (5, Transform::Month),
        (5, Transform::Year),
        (5, Transform::Unknown),
    ]
}

/// How the writer recorded string bounds.
#[derive(Debug, Clone, Copy)]
enum Writer {
    /// Java: truncate to 16 code points, round the upper bound up.
    Java,
    /// Truncate to `n` code points without rounding up.
    Plain(usize),
}

/// Spark's order: NaN above every value and equal to itself, `-0.0 = 0.0`.
fn spark_cmp(a: &ScalarValue, b: &ScalarValue) -> Option<Ordering> {
    match (a, b) {
        (ScalarValue::Float64(x), ScalarValue::Float64(y)) => {
            Some(match (x.is_nan(), y.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => x.partial_cmp(y)?,
            })
        }
        (ScalarValue::Utf8(x), ScalarValue::Utf8(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
        (ScalarValue::Int64(x), ScalarValue::Int64(y))
        | (ScalarValue::TimestampMicros(x), ScalarValue::TimestampMicros(y)) => Some(x.cmp(y)),
        (ScalarValue::Int32(x), ScalarValue::Int32(y))
        | (ScalarValue::Date32(x), ScalarValue::Date32(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// The SQL value of `e` on `row` (`None`: unknown).
fn eval(e: &PushdownExpr, row: &Row) -> Option<bool> {
    let name = |c: &ColumnPath| c.root().to_owned();
    match e {
        PushdownExpr::Constant(b) => Some(*b),
        PushdownExpr::Not(x) => eval(x, row).map(|b| !b),
        PushdownExpr::And(xs) => {
            let vs: Vec<_> = xs.iter().map(|x| eval(x, row)).collect();
            if vs.contains(&Some(false)) {
                Some(false)
            } else if vs.contains(&None) {
                None
            } else {
                Some(true)
            }
        }
        PushdownExpr::Or(xs) => {
            let vs: Vec<_> = xs.iter().map(|x| eval(x, row)).collect();
            if vs.contains(&Some(true)) {
                Some(true)
            } else if vs.contains(&None) {
                None
            } else {
                Some(false)
            }
        }
        PushdownExpr::IsNull(c) => Some(row.get(&name(c)).is_none()),
        PushdownExpr::IsNotNull(c) => Some(row.get(&name(c)).is_some()),
        PushdownExpr::Cmp {
            column, op, value, ..
        } => {
            let v = row.get(&name(column))?;
            let o = spark_cmp(&v, value)?;
            Some(match op {
                CmpOp::Eq => o == Ordering::Equal,
                CmpOp::NotEq => o != Ordering::Equal,
                CmpOp::Lt => o == Ordering::Less,
                CmpOp::LtEq => o != Ordering::Greater,
                CmpOp::Gt => o == Ordering::Greater,
                CmpOp::GtEq => o != Ordering::Less,
            })
        }
        PushdownExpr::In { column, values, .. } => {
            let v = row.get(&name(column))?;
            Some(
                values
                    .iter()
                    .any(|x| spark_cmp(&v, x) == Some(Ordering::Equal)),
            )
        }
        PushdownExpr::StartsWith { column, prefix, .. } => match row.get(&name(column))? {
            ScalarValue::Utf8(s) => Some(s.starts_with(prefix.as_str())),
            _ => None,
        },
        _ => None,
    }
}

/// Java's `truncateStringMax`: the first `n` code points with the last one
/// incremented (the alphabet here never overflows).
fn round_up(s: &str, n: usize) -> String {
    let mut cs: Vec<char> = s.chars().take(n).collect();
    if let Some(last) = cs.last_mut() {
        *last = char::from_u32(u32::from(*last) + 1).unwrap();
    }
    cs.into_iter().collect()
}

fn metrics(rows: &[Row], writer: Writer, f: &mut ContentFile) {
    let count = u64::try_from(rows.len()).unwrap();
    for field in fields() {
        let name = field.path.root().to_owned();
        let values: Vec<ScalarValue> = rows.iter().filter_map(|r| r.get(&name)).collect();
        let nulls = count - u64::try_from(values.len()).unwrap();
        f.value_counts.insert(field.id, count);
        f.null_counts.insert(field.id, nulls);
        let mut ordered: Vec<ScalarValue> = values
            .iter()
            .filter(|v| !matches!(v, ScalarValue::Float64(x) if x.is_nan()))
            .cloned()
            .collect();
        if field.ty == IcebergType::Double {
            let nans = u64::try_from(values.len() - ordered.len()).unwrap();
            f.nan_counts.insert(field.id, nans);
        }
        ordered.sort_by(|x, y| match (x, y) {
            (ScalarValue::Float64(a), ScalarValue::Float64(b)) => a.total_cmp(b),
            _ => spark_cmp(x, y).unwrap(),
        });
        let (Some(lo), Some(hi)) = (ordered.first(), ordered.last()) else {
            continue;
        };
        let (lo, hi) = match (lo, hi, writer) {
            (ScalarValue::Utf8(l), ScalarValue::Utf8(h), Writer::Java) => (
                ScalarValue::Utf8(truncate_str(l, 16)),
                ScalarValue::Utf8(if h.chars().count() > 16 {
                    round_up(h, 16)
                } else {
                    h.clone()
                }),
            ),
            (ScalarValue::Utf8(l), ScalarValue::Utf8(h), Writer::Plain(n)) => {
                let n = u32::try_from(n).unwrap();
                (
                    ScalarValue::Utf8(truncate_str(l, n)),
                    ScalarValue::Utf8(truncate_str(h, n)),
                )
            }
            _ => (lo.clone(), hi.clone()),
        };
        let bytes = |v: &ScalarValue| match (field.id, v) {
            // The promoted column was `int` when the file was written.
            (4, ScalarValue::Int64(x)) => i32::try_from(*x).unwrap().to_le_bytes().to_vec(),
            _ => encode(v, &field.ty).unwrap(),
        };
        f.lower_bounds.insert(field.id, bytes(&lo));
        f.upper_bounds.insert(field.id, bytes(&hi));
    }
}

/// A partition value as the (possibly old, pre-epoch-buggy) writer computed it.
fn partition_value(pf: &PartitionFieldInfo, row: &Row, toward_zero: bool) -> Option<ScalarValue> {
    let v = row.get(name_of(pf.source_id))?;
    if toward_zero && let ScalarValue::TimestampMicros(us) = v {
        match pf.transform {
            Transform::Day => return Some(ScalarValue::Date32(i32::try_from(us / DAY).unwrap())),
            Transform::Hour => return Some(ScalarValue::Int32(i32::try_from(us / HOUR).unwrap())),
            _ => {}
        }
    }
    match pf.transform {
        // Void is always null; unknown is never serialised (no result type).
        Transform::Void | Transform::Unknown => None,
        t => apply(t, &type_of(pf.source_id), &v),
    }
}

/// Strings near the 16-code-point truncation length, so ties with a bound happen.
fn string_strategy() -> impl Strategy<Value = String> {
    let stems = vec![
        "",
        "a",
        "ab",
        "b",
        "é",
        "aaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaab",
        "aaaaaaaaaaaaaaaab",
        "bbbbbbbbbbbbbbbbbb",
        "ééééééééééééééé",
    ];
    (
        prop::sample::select(stems),
        prop::collection::vec(prop::sample::select(vec!['a', 'b', 'é']), 0..3),
    )
        .prop_map(|(stem, tail)| format!("{stem}{}", tail.into_iter().collect::<String>()))
}

fn row_strategy() -> impl Strategy<Value = Row> {
    let d = prop::sample::select(vec![-1.5, 0.0, -0.0, 2.0, f64::NAN, 3.25]);
    let t = prop_oneof![
        (-3 * DAY..3 * DAY),
        (-4i64..4).prop_map(|h| h * HOUR),
        (-40i64..40).prop_map(|m| m * 31 * DAY),
    ];
    (
        prop::option::weighted(0.8, -20i64..20),
        prop::option::weighted(0.8, string_strategy()),
        prop::option::weighted(0.8, d),
        prop::option::weighted(0.8, -30i32..30),
        prop::option::weighted(0.8, t),
        prop::option::weighted(0.8, 0i32..3),
    )
        .prop_map(|(a, s, d, p, t, k)| Row { a, s, d, p, t, k })
}

/// A literal for column `c`: random, or one of the rows' values (and near
/// values: one more or less, a string with one more character or one fewer).
fn literal(c: &str, rows: &[Row]) -> BoxedStrategy<ScalarValue> {
    let random = match c {
        "a" | "p" => (-35i64..35).prop_map(ScalarValue::Int64).boxed(),
        "s" => string_strategy().prop_map(ScalarValue::Utf8).boxed(),
        "d" => prop::sample::select(vec![-2.0, -1.5, 0.0, -0.0, 1.0, 2.0, 3.25, 9.0, f64::NAN])
            .prop_map(ScalarValue::Float64)
            .boxed(),
        "t" => prop_oneof![
            (-4 * DAY..4 * DAY),
            (-5i64..5).prop_map(|h| h * HOUR),
            (-45i64..45).prop_map(|m| m * 31 * DAY),
        ]
        .prop_map(ScalarValue::TimestampMicros)
        .boxed(),
        _ => (-1i32..4).prop_map(ScalarValue::Int32).boxed(),
    };
    let mut pool = Vec::new();
    for v in rows.iter().filter_map(|r| r.get(c)) {
        match &v {
            ScalarValue::Int64(x) => pool.extend([x - 1, x + 1].map(ScalarValue::Int64)),
            ScalarValue::Int32(x) => pool.extend([x - 1, x + 1].map(ScalarValue::Int32)),
            ScalarValue::TimestampMicros(x) => {
                pool.extend([x - 1, x + 1].map(ScalarValue::TimestampMicros));
            }
            ScalarValue::Utf8(x) => {
                pool.push(ScalarValue::Utf8(format!("{x}a")));
                let mut shorter = x.clone();
                shorter.pop();
                pool.push(ScalarValue::Utf8(shorter));
            }
            _ => {}
        }
        pool.push(v);
    }
    if pool.is_empty() {
        random
    } else {
        prop_oneof![1 => random, 2 => prop::sample::select(pool)].boxed()
    }
}

fn leaf(rows: &[Row]) -> BoxedStrategy<PushdownExpr> {
    let rows = rows.to_vec();
    let column = prop::sample::select(vec!["a", "s", "d", "p", "t", "k"]);
    let op = prop::sample::select(vec![
        CmpOp::Eq,
        CmpOp::NotEq,
        CmpOp::Lt,
        CmpOp::LtEq,
        CmpOp::Gt,
        CmpOp::GtEq,
    ]);
    column
        .prop_flat_map(move |c| {
            let path = ColumnPath::new(c);
            let collation = prop_oneof![
                9 => Just(Collation::Utf8Binary),
                1 => Just(Collation::Other("UTF8_LCASE".to_owned())),
            ];
            let prefix = prop_oneof![
                string_strategy(),
                literal("s", &rows).prop_map(|v| match v {
                    ScalarValue::Utf8(s) => s,
                    _ => String::new(),
                }),
            ];
            prop_oneof![
                4 => (op.clone(), literal(c, &rows), collation.clone()).prop_map({
                    let path = path.clone();
                    move |(op, value, collation)| PushdownExpr::Cmp {
                        column: path.clone(),
                        op,
                        value,
                        collation,
                    }
                }),
                2 => (prop::collection::vec(literal(c, &rows), 0..4), collation).prop_map({
                    let path = path.clone();
                    move |(values, collation)| PushdownExpr::In {
                        column: path.clone(),
                        values,
                        collation,
                    }
                }),
                1 => Just(PushdownExpr::IsNull(path.clone())),
                1 => Just(PushdownExpr::IsNotNull(path.clone())),
                1 => prefix.prop_map({
                    move |prefix| PushdownExpr::StartsWith {
                        column: path.clone(),
                        prefix,
                        collation: Collation::Utf8Binary,
                    }
                }),
            ]
        })
        .boxed()
}

fn predicate(rows: &[Row]) -> BoxedStrategy<PushdownExpr> {
    leaf(rows)
        .prop_recursive(3, 12, 3, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 1..4).prop_map(PushdownExpr::And),
                prop::collection::vec(inner.clone(), 1..4).prop_map(PushdownExpr::Or),
                inner.prop_map(|x| PushdownExpr::Not(Box::new(x))),
            ]
        })
        .boxed()
}

#[derive(Debug, Clone)]
struct Case {
    rows: Vec<Row>,
    spec_fields: Vec<usize>,
    writer: Writer,
    toward_zero: bool,
    manifests: usize,
    predicates: Vec<PushdownExpr>,
}

fn case() -> impl Strategy<Value = Case> {
    prop::collection::vec(row_strategy(), 1..40).prop_flat_map(|rows| {
        (
            prop::collection::vec(0..candidate_fields().len(), 0..4),
            prop_oneof![Just(Writer::Java), (1usize..17).prop_map(Writer::Plain)],
            any::<bool>(),
            1usize..4,
            prop::collection::vec(predicate(&rows), 1..8),
        )
            .prop_map(
                move |(spec_fields, writer, toward_zero, manifests, predicates)| Case {
                    rows: rows.clone(),
                    spec_fields,
                    writer,
                    toward_zero,
                    manifests,
                    predicates,
                },
            )
    })
}

/// The files of a case with their rows, and its manifests with their members.
type Built = (
    PartitionSpecs,
    Vec<(ContentFile, Vec<Row>)>,
    Vec<(ManifestFileEntry, Vec<usize>)>,
);

/// The files (partition tuple, rows, metrics) and manifests of a case.
fn build(c: &Case) -> Built {
    let candidates = candidate_fields();
    let pfs: Vec<PartitionFieldInfo> = c
        .spec_fields
        .iter()
        .enumerate()
        .map(|(i, &j)| {
            let (source, t) = candidates[j];
            PartitionFieldInfo::new(
                1000 + i32::try_from(i).unwrap(),
                source,
                format!("p{i}"),
                t,
                Some(type_of(source)),
            )
        })
        .collect();
    let specs: PartitionSpecs = [(
        1,
        PartitionSpecInfo {
            spec_id: 1,
            fields: pfs.clone(),
        },
    )]
    .into();
    // Group rows by their partition tuple (as serialised bytes).
    let mut groups: BTreeMap<Vec<Option<Vec<u8>>>, Vec<Row>> = BTreeMap::new();
    for r in &c.rows {
        let key = pfs
            .iter()
            .map(|pf| {
                let ty = pf.result_type.as_ref()?;
                match partition_value(pf, r, c.toward_zero)? {
                    // The promoted column was `int` when the file was written.
                    ScalarValue::Int64(x) if pf.source_id == 4 => {
                        Some(i32::try_from(x).unwrap().to_le_bytes().to_vec())
                    }
                    v => encode(&v, ty),
                }
            })
            .collect();
        groups.entry(key).or_default().push(r.clone());
    }
    let mut files = Vec::new();
    for (key, rows) in groups {
        let mut f = super::tests::file(u64::try_from(rows.len()).unwrap());
        f.spec_id = 1;
        f.partition = key
            .into_iter()
            .zip(&pfs)
            .map(|(v, pf)| match (v, &pf.result_type) {
                (_, None) => PartitionValue::Unreadable,
                (None, Some(_)) => PartitionValue::Null,
                (Some(b), Some(_)) => PartitionValue::Value(b),
            })
            .collect();
        metrics(&rows, c.writer, &mut f);
        files.push((f, rows));
    }
    // Deal the files into manifests and summarise each.
    let mut manifests = Vec::new();
    for m in 0..c.manifests {
        let members: Vec<usize> = (0..files.len()).filter(|i| i % c.manifests == m).collect();
        let summaries = pfs
            .iter()
            .enumerate()
            .map(|(i, pf)| {
                let Some(ty) = pf.result_type.clone() else {
                    return PartitionFieldSummary::default();
                };
                let mut s = PartitionFieldSummary {
                    contains_nan: ty.is_float().then_some(false),
                    ..PartitionFieldSummary::default()
                };
                let mut values = Vec::new();
                for &fi in &members {
                    match &files[fi].0.partition[i] {
                        PartitionValue::Null => s.contains_null = true,
                        PartitionValue::Value(b) => {
                            let v = super::bounds::decode(b, &ty).unwrap();
                            if super::bounds::is_nan(&v) {
                                s.contains_nan = Some(true);
                            } else {
                                values.push((v, b.clone()));
                            }
                        }
                        PartitionValue::Unreadable => {}
                    }
                }
                values.sort_by(|x, y| match (&x.0, &y.0) {
                    (ScalarValue::Float64(a), ScalarValue::Float64(b)) => a.total_cmp(b),
                    _ => spark_cmp(&x.0, &y.0).unwrap(),
                });
                s.lower = values.first().map(|v| v.1.clone());
                s.upper = values.last().map(|v| v.1.clone());
                s
            })
            .collect();
        let mut entry = super::tests::manifest_entry(1, None);
        entry.spec_id = 1;
        entry.content = ManifestContent::Data;
        entry.partitions = Some(summaries);
        manifests.push((entry, members));
    }
    (specs, files, manifests)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1500))]

    #[test]
    fn prune_never_skips_a_matching_row(c in case()) {
        let (specs, files, manifests) = build(&c);
        let fields = fields();
        let mut modes = vec![PruneOptions::conservative()];
        // Scan mode assumes a writer that truncates at 16 or rounds up.
        if matches!(c.writer, Writer::Java | Writer::Plain(16)) {
            modes.push(PruneOptions::default());
        }
        for e in &c.predicates {
            for &o in &modes {
                let pruner = Pruner::new(e, &fields, &specs, o);
                for (f, rows) in &files {
                    if rows.iter().any(|r| eval(e, r) == Some(true)) {
                        prop_assert!(pruner.file_may_match(f), "{:?} pruned {:?} in {:?}", e, f, o);
                    }
                }
                for (m, members) in &manifests {
                    let matching = members.iter().any(|&i| files[i].1.iter().any(|r| eval(e, r) == Some(true)));
                    if matching {
                        prop_assert!(pruner.manifest_may_match(m), "{:?} pruned {:?} in {:?}", e, m, o);
                    }
                }
            }
        }
    }
}

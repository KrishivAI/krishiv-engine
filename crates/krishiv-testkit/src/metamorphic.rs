// SPDX-License-Identifier: Apache-2.0
//! Metamorphic query oracles: ternary logic partitioning (TLP) and non-optimizing reference
//! engine construction (NoREC) (P2-T110b).
//!
//! Neither needs a second engine. For a predicate `p` over a source `S`:
//!
//! * **TLP** — the rows of `S` are exactly the rows where `p`, where `NOT p` and where
//!   `p IS NULL` (three-valued logic), and the aggregates `COUNT`, `SUM`, `MIN`, `MAX` combine
//!   over the three parts. A `HAVING` form partitions the groups the same way.
//! * **NoREC** — `COUNT(*) ... WHERE p` (which the optimizer rewrites, pushes down and prunes
//!   with) equals `SUM(CASE WHEN p THEN 1 ELSE 0 END)` (which it cannot).
//!
//! [`hunt`] draws seeded random predicates over the TPC-H tables and a NULL/NaN-heavy table and
//! reports the violations; [`reduce`] shrinks a violating predicate by expression-tree
//! reduction, so the report is a handful of tokens.

use std::fmt::{self, Write as _};
use std::io::Write;

use crate::metamorphic_reduce::{PythonSpark, Verdict, reduce_rows, render, spark_verdict};
use crate::query_oracle::{QueryRunner, normalise};

/// The NULL- and NaN-heavy table every source can read.
pub const NULLS_SETUP: &str = "CREATE TEMPORARY VIEW nulls AS SELECT * FROM VALUES \
 (1, 1, 1.5D, 'a', 0), (2, NULL, NULL, 'B', 0), (3, 3, CAST('NaN' AS DOUBLE), NULL, 1), \
 (NULL, 4, -0.0D, '', 1), (5, 5, CAST('Infinity' AS DOUBLE), 'a', NULL), (NULL, NULL, NULL, NULL, NULL), \
 (0, 0, 0.0D, 'ab', 2), (-1, 7, -2.5D, 'A', 2), (7, -1, 100.0D, ' a', 3), (7, 7, 1.5D, 'a%', 3) \
 AS nulls(a, b, d, s, g)";

/// The rows of [`NULLS_SETUP`], one `VALUES` tuple each, so a reduction can drop rows.
pub const NULLS_ROWS: [&str; 10] = [
    "(1, 1, 1.5D, 'a', 0)",
    "(2, NULL, NULL, 'B', 0)",
    "(3, 3, CAST('NaN' AS DOUBLE), NULL, 1)",
    "(NULL, 4, -0.0D, '', 1)",
    "(5, 5, CAST('Infinity' AS DOUBLE), 'a', NULL)",
    "(NULL, NULL, NULL, NULL, NULL)",
    "(0, 0, 0.0D, 'ab', 2)",
    "(-1, 7, -2.5D, 'A', 2)",
    "(7, -1, 100.0D, ' a', 3)",
    "(7, 7, 1.5D, 'a%', 3)",
];

/// The statement that (re)defines the `nulls` view over the rows `keep` of [`NULLS_ROWS`].
#[must_use]
pub fn nulls_setup(keep: &[usize]) -> String {
    let rows: Vec<&str> = keep
        .iter()
        .filter_map(|i| NULLS_ROWS.get(*i))
        .copied()
        .collect();
    format!(
        "CREATE OR REPLACE TEMPORARY VIEW nulls AS SELECT * FROM VALUES {} AS nulls(a, b, d, s, g)",
        rows.join(", ")
    )
}

/// The kind of a column, which fixes the comparisons and literals drawn for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// An integer.
    Int,
    /// A double (may be NaN or infinite).
    Dbl,
    /// A string.
    Str,
}

/// A source the oracles run over: a `FROM` clause and the columns a predicate may read.
#[derive(Debug, Clone, Copy)]
pub struct Source {
    /// The `FROM` clause, with its joins.
    pub from: &'static str,
    /// The columns of the predicate, as SQL text and kind.
    pub cols: &'static [(&'static str, Kind)],
    /// The column the aggregates read (an integer).
    pub measure: &'static str,
    /// The expressions of a `HAVING` predicate (aggregates) and the `GROUP BY` column.
    pub having: &'static [(&'static str, Kind)],
    /// The `GROUP BY` column.
    pub group: &'static str,
}

/// The sources: a table of NULLs and NaNs, one TPC-H table, and an outer join whose predicate
/// reads the null-producing side.
pub const SOURCES: [Source; 3] = [
    Source {
        from: "nulls n",
        cols: &[
            ("n.a", Kind::Int),
            ("n.b", Kind::Int),
            ("n.d", Kind::Dbl),
            ("n.s", Kind::Str),
        ],
        measure: "n.a",
        having: &[
            ("MAX(n.b)", Kind::Int),
            ("MIN(n.a)", Kind::Int),
            ("COUNT(n.d)", Kind::Int),
        ],
        group: "n.g",
    },
    Source {
        from: "orders o",
        cols: &[
            ("o.o_custkey", Kind::Int),
            ("o.o_shippriority", Kind::Int),
            ("o.o_orderstatus", Kind::Str),
            ("o.o_orderpriority", Kind::Str),
        ],
        measure: "o.o_custkey",
        having: &[
            ("MAX(o.o_custkey)", Kind::Int),
            ("COUNT(o.o_clerk)", Kind::Int),
        ],
        group: "o.o_orderstatus",
    },
    Source {
        from: "customer c LEFT JOIN orders o ON c.c_custkey = o.o_custkey AND o.o_totalprice > 300000",
        cols: &[
            ("o.o_orderkey", Kind::Int),
            ("o.o_orderstatus", Kind::Str),
            ("c.c_nationkey", Kind::Int),
        ],
        measure: "c.c_nationkey",
        having: &[
            ("COUNT(o.o_orderkey)", Kind::Int),
            ("MAX(c.c_nationkey)", Kind::Int),
        ],
        group: "c.c_mktsegment",
    },
];

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    /// `=`
    Eq,
    /// `<>`
    Ne,
    /// `<`
    Lt,
    /// `>=`
    Ge,
}

impl Cmp {
    const fn sql(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ne => "<>",
            Self::Lt => "<",
            Self::Ge => ">=",
        }
    }
}

/// A boolean predicate.
#[derive(Debug, Clone, PartialEq)]
pub enum Pred {
    /// `col op literal`
    Cmp(String, Cmp, String),
    /// `col IS [NOT] NULL`
    IsNull(String, bool),
    /// `col [NOT] IN (items)`; an item may be `NULL`.
    In(String, Vec<String>, bool),
    /// `col BETWEEN lo AND hi`
    Between(String, String, String),
    /// `col LIKE pattern`
    Like(String, String),
    /// `(a AND b)`
    And(Box<Pred>, Box<Pred>),
    /// `(a OR b)`
    Or(Box<Pred>, Box<Pred>),
    /// `NOT (a)`
    Not(Box<Pred>),
}

impl fmt::Display for Pred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cmp(c, op, l) => write!(f, "{c} {} {l}", op.sql()),
            Self::IsNull(c, neg) => write!(f, "{c} IS {}NULL", if *neg { "NOT " } else { "" }),
            Self::In(c, items, neg) => write!(
                f,
                "{c} {}IN ({})",
                if *neg { "NOT " } else { "" },
                items.join(", ")
            ),
            Self::Between(c, lo, hi) => write!(f, "{c} BETWEEN {lo} AND {hi}"),
            Self::Like(c, p) => write!(f, "{c} LIKE {p}"),
            Self::And(a, b) => write!(f, "({a} AND {b})"),
            Self::Or(a, b) => write!(f, "({a} OR {b})"),
            Self::Not(a) => write!(f, "NOT ({a})"),
        }
    }
}

/// A seeded generator (splitmix64).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// A generator for `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n.max(1) as u64).unwrap_or(0)
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

fn literal(kind: Kind, rng: &mut Rng) -> String {
    match kind {
        Kind::Int => rng
            .pick(&["0", "1", "2", "3", "5", "7", "-1", "10", "NULL"])
            .to_string(),
        Kind::Dbl => rng
            .pick(&[
                "0.0D",
                "1.5D",
                "-2.5D",
                "100.0D",
                "CAST('NaN' AS DOUBLE)",
                "CAST('Infinity' AS DOUBLE)",
                "NULL",
            ])
            .to_string(),
        Kind::Str => rng
            .pick(&[
                "'a'",
                "'B'",
                "''",
                "'ab'",
                "'F'",
                "'O'",
                "'1-URGENT'",
                "NULL",
            ])
            .to_string(),
    }
}

fn leaf(cols: &[(&str, Kind)], rng: &mut Rng) -> Pred {
    let (col, kind) = *rng.pick(cols);
    let col = col.to_owned();
    match rng.below(6) {
        0 | 1 => Pred::Cmp(
            col,
            *rng.pick(&[Cmp::Eq, Cmp::Ne, Cmp::Lt, Cmp::Ge]),
            literal(kind, rng),
        ),
        2 => Pred::IsNull(col, rng.below(2) == 0),
        3 | 4 => {
            let n = 1 + rng.below(3);
            let mut items: Vec<String> = (0..n).map(|_| literal(kind, rng)).collect();
            if rng.below(2) == 0 {
                items.push("NULL".to_owned());
            }
            Pred::In(col, items, rng.below(2) == 0)
        }
        _ => match kind {
            Kind::Str => Pred::Like(col, (*rng.pick(&["'a%'", "'%a'", "'_'", "'%'"])).to_owned()),
            _ => Pred::Between(col, literal(kind, rng), literal(kind, rng)),
        },
    }
}

/// A random predicate of depth at most `depth` over `cols`.
#[must_use]
pub fn gen_pred(cols: &[(&str, Kind)], depth: u32, rng: &mut Rng) -> Pred {
    if depth == 0 || rng.below(3) == 0 {
        return leaf(cols, rng);
    }
    match rng.below(4) {
        0 => Pred::Not(Box::new(gen_pred(cols, depth - 1, rng))),
        1 => Pred::Or(
            Box::new(gen_pred(cols, depth - 1, rng)),
            Box::new(gen_pred(cols, depth - 1, rng)),
        ),
        _ => Pred::And(
            Box::new(gen_pred(cols, depth - 1, rng)),
            Box::new(gen_pred(cols, depth - 1, rng)),
        ),
    }
}

/// Which oracle a violation came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Oracle {
    /// The three `WHERE` partitions do not make up the table.
    TlpWhere,
    /// The aggregate does not combine over the partitions.
    TlpAggregate,
    /// The three `HAVING` partitions do not make up the groups.
    TlpHaving,
    /// The filtered count differs from the unoptimizable count.
    NoRec,
}

impl Oracle {
    /// The name a regression file carries.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::TlpWhere => "tlp-where",
            Self::TlpAggregate => "tlp-aggregate",
            Self::TlpHaving => "tlp-having",
            Self::NoRec => "norec",
        }
    }

    /// The oracle named `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [
            Self::TlpWhere,
            Self::TlpAggregate,
            Self::TlpHaving,
            Self::NoRec,
        ]
        .into_iter()
        .find(|o| o.name() == name)
    }
}

/// A violated oracle.
#[derive(Debug, Clone)]
pub struct Finding {
    /// The oracle.
    pub oracle: Oracle,
    /// Index into [`SOURCES`].
    pub source: usize,
    /// The predicate.
    pub pred: Pred,
    /// What differs.
    pub detail: String,
}

pub(crate) type Rows = Vec<Vec<String>>;

pub(crate) fn canonical(mut rows: Rows) -> Rows {
    for r in &mut rows {
        for c in r.iter_mut() {
            *c = normalise(c);
        }
    }
    rows.sort();
    rows
}

pub(crate) fn number(cell: &str) -> Option<f64> {
    cell.parse().ok()
}

fn one(rows: &Rows) -> &str {
    rows.first()
        .and_then(|r| r.first())
        .map_or("NULL", String::as_str)
}

/// The three partitions of `p`.
fn parts(p: &str) -> [String; 3] {
    [
        format!("({p})"),
        format!("NOT ({p})"),
        format!("({p}) IS NULL"),
    ]
}

fn tlp_where(
    r: &mut dyn QueryRunner,
    s: &Source,
    p: &dyn fmt::Display,
) -> Result<Option<String>, String> {
    let cols = s.cols.iter().map(|c| c.0).collect::<Vec<_>>().join(", ");
    let whole = r.run(&format!("SELECT {cols} FROM {}", s.from))?;
    let mut union = Vec::new();
    for part in parts(&p.to_string()) {
        union.extend(r.run(&format!("SELECT {cols} FROM {} WHERE {part}", s.from))?);
    }
    let (want, got) = (canonical(whole), canonical(union));
    Ok((want != got).then(|| format!("{} rows, the partitions hold {}", want.len(), got.len())))
}

fn tlp_aggregate(
    r: &mut dyn QueryRunner,
    s: &Source,
    p: &dyn fmt::Display,
) -> Result<Option<String>, String> {
    let m = s.measure;
    let agg = format!("COUNT(*), SUM({m}), MIN({m}), MAX({m})");
    let whole = r.run(&format!("SELECT {agg} FROM {}", s.from))?;
    let (mut count, mut sum, mut min, mut max) = (0.0, None::<f64>, None::<f64>, None::<f64>);
    for part in parts(&p.to_string()) {
        let rows = r.run(&format!("SELECT {agg} FROM {} WHERE {part}", s.from))?;
        let row = rows.first().ok_or("an aggregate returned no row")?;
        count += number(&row[0]).unwrap_or(0.0);
        let fold = |acc: Option<f64>, cell: &str, f: fn(f64, f64) -> f64| match (acc, number(cell))
        {
            (Some(a), Some(b)) => Some(f(a, b)),
            (a, b) => a.or(b),
        };
        sum = fold(sum, &row[1], |a, b| a + b);
        min = fold(min, &row[2], f64::min);
        max = fold(max, &row[3], f64::max);
    }
    let row = whole.first().ok_or("an aggregate returned no row")?;
    let show = |v: Option<f64>| v.map_or_else(|| "NULL".to_owned(), |v| normalise(&v.to_string()));
    let want = [
        normalise(&row[0]),
        normalise(&row[1]),
        normalise(&row[2]),
        normalise(&row[3]),
    ];
    let got = [
        normalise(&count.to_string()),
        show(sum),
        show(min),
        show(max),
    ];
    Ok((want != got).then(|| format!("COUNT/SUM/MIN/MAX {want:?}, combined {got:?}")))
}

fn tlp_having(
    r: &mut dyn QueryRunner,
    s: &Source,
    p: &dyn fmt::Display,
) -> Result<Option<String>, String> {
    let q = |w: &str| {
        format!(
            "SELECT {g}, COUNT(*) FROM {} GROUP BY {g}{w}",
            s.from,
            g = s.group
        )
    };
    let whole = r.run(&q(""))?;
    let mut union = Vec::new();
    for part in parts(&p.to_string()) {
        union.extend(r.run(&q(&format!(" HAVING {part}")))?);
    }
    let (want, got) = (canonical(whole), canonical(union));
    Ok((want != got).then(|| format!("{} groups, the partitions hold {}", want.len(), got.len())))
}

fn norec(
    r: &mut dyn QueryRunner,
    s: &Source,
    p: &dyn fmt::Display,
) -> Result<Option<String>, String> {
    let filtered = r.run(&format!("SELECT COUNT(*) FROM {} WHERE {p}", s.from))?;
    let counted = r.run(&format!(
        "SELECT SUM(CASE WHEN {p} THEN 1 ELSE 0 END) FROM {}",
        s.from
    ))?;
    let (a, b) = (one(&filtered), one(&counted));
    let b = if b == "NULL" { "0" } else { b };
    Ok((number(a) != number(b)).then(|| format!("WHERE counts {a}, SUM(CASE) counts {b}")))
}

/// Checks one oracle; `Ok(None)` when it holds. An engine error is not a violation: the same
/// predicate failing alike on every form would hide nothing, and a failure of only one form
/// surfaces as an `Err` the caller counts as `resource`.
///
/// # Errors
/// The engine's failure on one of the queries.
pub fn check(
    r: &mut dyn QueryRunner,
    oracle: Oracle,
    source: usize,
    p: &dyn fmt::Display,
) -> Result<Option<String>, String> {
    let s = &SOURCES[source];
    match oracle {
        Oracle::TlpWhere => tlp_where(r, s, p),
        Oracle::TlpAggregate => tlp_aggregate(r, s, p),
        Oracle::TlpHaving => tlp_having(r, s, p),
        Oracle::NoRec => norec(r, s, p),
    }
}

/// The result of a [`hunt`].
#[derive(Debug, Default)]
pub struct Hunt {
    /// Predicates checked (each against every oracle that applies).
    pub checked: usize,
    /// Violations.
    pub findings: Vec<Finding>,
    /// Checks that failed in the engine instead of disagreeing, with the error.
    pub errors: Vec<String>,
}

/// Draws `cases` predicates from `seed` and checks every oracle on each.
pub fn hunt(r: &mut dyn QueryRunner, seed: u64, cases: usize) -> Hunt {
    let mut rng = Rng::new(seed);
    let mut out = Hunt::default();
    for _ in 0..cases {
        let source = rng.below(SOURCES.len());
        let s = &SOURCES[source];
        let p = gen_pred(s.cols, 3, &mut rng);
        let h = gen_pred(s.having, 2, &mut rng);
        for (oracle, pred) in [
            (Oracle::TlpWhere, &p),
            (Oracle::TlpAggregate, &p),
            (Oracle::NoRec, &p),
            (Oracle::TlpHaving, &h),
        ] {
            match check(r, oracle, source, pred) {
                Ok(None) => {}
                Ok(Some(detail)) => out.findings.push(Finding {
                    oracle,
                    source,
                    pred: pred.clone(),
                    detail,
                }),
                Err(e) => out.errors.push(format!("{oracle:?} {pred}: {e}")),
            }
        }
        out.checked += 1;
    }
    out
}

/// Tokens of a predicate's SQL.
#[must_use]
pub fn tokens(p: &Pred) -> usize {
    p.to_string().split_whitespace().count()
}

fn shrinks(p: &Pred) -> Vec<Pred> {
    let mut out = Vec::new();
    match p {
        Pred::And(a, b) | Pred::Or(a, b) => {
            out.push((**a).clone());
            out.push((**b).clone());
            let rebuild = |a: Pred, b: Pred| match p {
                Pred::And(..) => Pred::And(Box::new(a), Box::new(b)),
                _ => Pred::Or(Box::new(a), Box::new(b)),
            };
            out.extend(shrinks(a).into_iter().map(|x| rebuild(x, (**b).clone())));
            out.extend(shrinks(b).into_iter().map(|x| rebuild((**a).clone(), x)));
        }
        Pred::Not(a) => {
            out.push((**a).clone());
            out.extend(shrinks(a).into_iter().map(|x| Pred::Not(Box::new(x))));
        }
        Pred::In(c, items, neg) if items.len() > 1 => {
            for i in 0..items.len() {
                let mut rest = items.clone();
                rest.remove(i);
                out.push(Pred::In(c.clone(), rest, *neg));
            }
        }
        _ => {}
    }
    out
}

/// Shrinks `p` while `still_fails` holds: replaces a node by a child or drops an `IN` item,
/// repeating until no step keeps the failure.
pub fn reduce(p: &Pred, still_fails: &mut dyn FnMut(&Pred) -> bool) -> Pred {
    let mut cur = p.clone();
    'again: loop {
        for cand in shrinks(&cur) {
            if tokens(&cand) < tokens(&cur) && still_fails(&cand) {
                cur = cand;
                continue 'again;
            }
        }
        return cur;
    }
}

/// The smallest predicate that still fails the oracle of `f`.
pub fn reduce_finding(r: &mut dyn QueryRunner, f: &Finding) -> Pred {
    reduce(&f.pred, &mut |c| {
        matches!(check(r, f.oracle, f.source, c), Ok(Some(_)))
    })
}

/// A finding with its reduced predicate `small`, ready to paste into a regression test.
///
/// # Errors
/// When writing the report fails (never, in practice).
pub fn render_report(f: &Finding, small: &Pred) -> Result<String, fmt::Error> {
    let mut out = String::new();
    writeln!(
        out,
        "{:?} over `{}`: {}\n  predicate ({} tokens, from {}): {small}\n  original: {}",
        f.oracle,
        SOURCES[f.source].from,
        f.detail,
        tokens(small),
        tokens(&f.pred),
        f.pred
    )?;
    Ok(out)
}

/// A finding reduced to its smallest failing predicate, ready to paste into a regression test.
///
/// # Errors
/// When writing the report fails (never, in practice).
pub fn report(r: &mut dyn QueryRunner, f: &Finding) -> Result<String, fmt::Error> {
    let small = reduce_finding(r, f);
    render_report(f, &small)
}

/// A predicate given as SQL text.
pub(crate) struct Raw(pub(crate) String);

impl fmt::Display for Raw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The regression file of a reduced finding: its name (stable for the same oracle, source and
/// predicate) and its JSON text. The nightly hunt writes these (`--regressions DIR`); a file
/// committed under `tests/fixtures/metamorphic/regressions/` is replayed by the
/// `regressions_hold` test, so a fixed bug stays fixed.
#[must_use]
pub fn regression_file(
    oracle: Oracle,
    source: usize,
    small: &Pred,
    detail: &str,
) -> (String, String) {
    regression_file_with(oracle, source, small, detail, None, None)
}

/// [`regression_file`] with the rows of the `nulls` view the finding needs (`rows`, indices
/// into [`NULLS_ROWS`]; absent means all) and Spark's verdict text.
#[must_use]
pub fn regression_file_with(
    oracle: Oracle,
    source: usize,
    small: &Pred,
    detail: &str,
    rows: Option<&[usize]>,
    spark: Option<&str>,
) -> (String, String) {
    let predicate = small.to_string();
    // FNV-1a over the identifying parts.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in oracle
        .name()
        .bytes()
        .chain(source.to_le_bytes())
        .chain(predicate.bytes())
        .chain(
            rows.unwrap_or_default()
                .iter()
                .flat_map(|i| i.to_le_bytes()),
        )
    {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    let mut text = serde_json::json!({
        "oracle": oracle.name(),
        "source": source,
        "predicate": predicate,
        "found": detail,
    });
    if let (Some(rows), Some(obj)) = (rows, text.as_object_mut()) {
        obj.insert("rows".to_owned(), serde_json::json!(rows));
    }
    if let (Some(spark), Some(obj)) = (spark, text.as_object_mut()) {
        obj.insert("spark".to_owned(), serde_json::json!(spark));
    }
    (
        format!("{}-{h:016x}.json", oracle.name()),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&text).unwrap_or_default()
        ),
    )
}

/// Replays a regression file: `Ok(None)` when its oracle holds.
///
/// # Errors
/// A malformed file, or the engine's failure on a query.
pub fn replay_regression(r: &mut dyn QueryRunner, text: &str) -> Result<Option<String>, String> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let oracle = v
        .get("oracle")
        .and_then(serde_json::Value::as_str)
        .and_then(Oracle::from_name)
        .ok_or("missing or unknown `oracle`")?;
    let source = v
        .get("source")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n < SOURCES.len())
        .ok_or("missing or unknown `source`")?;
    let predicate = v
        .get("predicate")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `predicate`")?;
    let rows: Option<Vec<usize>> = v
        .get("rows")
        .and_then(serde_json::Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|i| i.as_u64().and_then(|i| usize::try_from(i).ok()))
                .collect()
        });
    let Some(rows) = rows.filter(|_| source == 0) else {
        return check(r, oracle, source, &Raw(predicate.to_owned()));
    };
    r.run(&nulls_setup(&rows))?;
    let out = check(r, oracle, source, &Raw(predicate.to_owned()));
    r.run(&nulls_setup(&(0..NULLS_ROWS.len()).collect::<Vec<_>>()))?;
    out
}

/// The `metamorphic-hunt` command over `args` (without the program name); `Ok(true)` when no
/// oracle was violated.
///
/// # Errors
/// A usage, I/O or engine error, as text.
pub fn hunt_cli(
    args: impl IntoIterator<Item = String>,
    out: &mut dyn Write,
) -> Result<bool, String> {
    let (mut data, mut seed, mut seconds, mut regressions) = (None, 1_u64, 60_u64, None);
    let mut spark_cmd: Option<Vec<String>> = None;
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let v = args.next().ok_or(
            "usage: metamorphic-hunt --data DIR [--seed N] [--seconds N] [--regressions DIR] [--spark CMD]",
        )?;
        match flag.as_str() {
            "--data" => data = Some(std::path::PathBuf::from(v)),
            "--seed" => seed = v.parse().map_err(|e| format!("--seed: {e}"))?,
            "--seconds" => seconds = v.parse().map_err(|e| format!("--seconds: {e}"))?,
            "--regressions" => regressions = Some(std::path::PathBuf::from(v)),
            // The command that runs `tests/scripts/spark_answers.py` (script path last).
            "--spark" => spark_cmd = Some(v.split_whitespace().map(str::to_owned).collect()),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    let data = data.ok_or("--data DIR is required")?;
    let mut runner = crate::query_oracle::SessionRunner::new(&data)?;
    crate::query_oracle::QueryRunner::run(&mut runner, NULLS_SETUP)?;
    let (start, budget) = (
        std::time::Instant::now(),
        std::time::Duration::from_secs(seconds),
    );
    let (mut checked, mut bad, mut failed) = (0_usize, 0_usize, 0_usize);
    let mut next = seed;
    while start.elapsed() < budget {
        let h = hunt(&mut runner, next, 25);
        next += 1;
        checked += h.checked * 4;
        failed += h.errors.len();
        for e in &h.errors {
            writeln!(out, "engine error (seed {next}): {e}").map_err(|e| e.to_string())?;
        }
        for f in &h.findings {
            let small = reduce_finding(&mut runner, f);
            let kept = reduce_rows(&mut runner, f, &small)?;
            let verdict = spark_cmd.as_ref().map(|command| {
                spark_verdict(
                    &mut runner,
                    &mut PythonSpark {
                        command: command.clone(),
                        data: data.clone(),
                    },
                    f,
                    &small,
                    &kept,
                )
            });
            let artefact = verdict == Some(Verdict::OracleArtefact);
            if !artefact {
                bad += 1;
            }
            let text = match &verdict {
                Some(v) => render(f, &small, &kept, v),
                None => render_report(f, &small).map_err(|e| e.to_string())?,
            };
            writeln!(out, "VIOLATION (seed batch {}):\n{text}", next - 1)
                .map_err(|e| e.to_string())?;
            if let (Some(dir), false) = (&regressions, artefact) {
                let rows = (f.source == 0).then_some(kept.as_slice());
                let spark = verdict.as_ref().map(ToString::to_string);
                let (name, body) = regression_file_with(
                    f.oracle,
                    f.source,
                    &small,
                    &f.detail,
                    rows,
                    spark.as_deref(),
                );
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                let path = dir.join(name);
                std::fs::write(&path, body).map_err(|e| format!("{}: {e}", path.display()))?;
                writeln!(out, "  regression file: {}", path.display())
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    let rate = checked as f64 / start.elapsed().as_secs_f64().max(1e-9);
    writeln!(
        out,
        "queries_checked_per_s {rate:.1} ({checked} checks, {bad} violations, {failed} engine errors)"
    )
    .map_err(|e| e.to_string())?;
    // An engine error on a generated query is a failure too: on a clean build it is a bug or a
    // resource limit to look at, not a pass.
    Ok(bad == 0 && failed == 0)
}

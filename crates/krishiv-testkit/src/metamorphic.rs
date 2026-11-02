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

use crate::query_oracle::{QueryRunner, normalise};

/// The NULL- and NaN-heavy table every source can read.
pub const NULLS_SETUP: &str = "CREATE TEMPORARY VIEW nulls AS SELECT * FROM VALUES \
 (1, 1, 1.5D, 'a', 0), (2, NULL, NULL, 'B', 0), (3, 3, CAST('NaN' AS DOUBLE), NULL, 1), \
 (NULL, 4, -0.0D, '', 1), (5, 5, CAST('Infinity' AS DOUBLE), 'a', NULL), (NULL, NULL, NULL, NULL, NULL), \
 (0, 0, 0.0D, 'ab', 2), (-1, 7, -2.5D, 'A', 2), (7, -1, 100.0D, ' a', 3), (7, 7, 1.5D, 'a%', 3) \
 AS nulls(a, b, d, s, g)";

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

type Rows = Vec<Vec<String>>;

fn canonical(mut rows: Rows) -> Rows {
    for r in &mut rows {
        for c in r.iter_mut() {
            *c = normalise(c);
        }
    }
    rows.sort();
    rows
}

fn number(cell: &str) -> Option<f64> {
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

fn tlp_where(r: &mut dyn QueryRunner, s: &Source, p: &Pred) -> Result<Option<String>, String> {
    let cols = s.cols.iter().map(|c| c.0).collect::<Vec<_>>().join(", ");
    let whole = r.run(&format!("SELECT {cols} FROM {}", s.from))?;
    let mut union = Vec::new();
    for part in parts(&p.to_string()) {
        union.extend(r.run(&format!("SELECT {cols} FROM {} WHERE {part}", s.from))?);
    }
    let (want, got) = (canonical(whole), canonical(union));
    Ok((want != got).then(|| format!("{} rows, the partitions hold {}", want.len(), got.len())))
}

fn tlp_aggregate(r: &mut dyn QueryRunner, s: &Source, p: &Pred) -> Result<Option<String>, String> {
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

fn tlp_having(r: &mut dyn QueryRunner, s: &Source, p: &Pred) -> Result<Option<String>, String> {
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

fn norec(r: &mut dyn QueryRunner, s: &Source, p: &Pred) -> Result<Option<String>, String> {
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
    p: &Pred,
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

/// A finding reduced to its smallest failing predicate, ready to paste into a regression test.
///
/// # Errors
/// When writing the report fails (never, in practice).
pub fn report(r: &mut dyn QueryRunner, f: &Finding) -> Result<String, fmt::Error> {
    let small = reduce(&f.pred, &mut |c| {
        matches!(check(r, f.oracle, f.source, c), Ok(Some(_)))
    });
    let mut out = String::new();
    writeln!(
        out,
        "{:?} over `{}`: {}\n  predicate ({} tokens, from {}): {small}\n  original: {}",
        f.oracle,
        SOURCES[f.source].from,
        f.detail,
        tokens(&small),
        tokens(&f.pred),
        f.pred
    )?;
    Ok(out)
}

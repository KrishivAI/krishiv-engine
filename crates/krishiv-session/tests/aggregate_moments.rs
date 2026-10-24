// SPDX-License-Identifier: Apache-2.0
//! The moment aggregates (`stddev`, `variance`, `skewness`, `kurtosis`,
//! `corr`, `covar_*`) end to end against Spark 4.2 `SQLQueryTestSuite`
//! goldens. Each claimed statement is found by its exact text in a vendored
//! result file and must give the golden's schema and rows; the two knobs
//! that change a result (`spark.sql.ansi.enabled` and
//! `spark.sql.legacy.statisticalAggregate`) are checked on top.
//!
//! The session has no `CREATE VIEW` or inline `VALUES` yet, so the driver
//! registers the golden's tables by hand.
//!
//! Spark computes these goldens over two partitions: each folds its rows
//! into a partial buffer and the buffers are merged. A session aggregates
//! one partition in one pass, which rounds the same values differently in
//! the last digits, so the double columns are compared to 1e-14 (relative)
//! and everything else exactly. The digits themselves are reproduced bit
//! for bit, under Spark's partitioning, by the accumulator tests of
//! `krishiv-exec` (`recorded_spark_results_are_reproduced_to_the_bit`).
//!
//! Statements of the three goldens that use these functions and are not
//! claimed (see `UNCLAIMED`) are listed with the reason, and the list is
//! checked against the files, so a new statement cannot go unnoticed.
#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// One golden statement: the query and its expected schema and output.
struct Case {
    query: String,
    schema: String,
    output: String,
}

fn load(file: &str) -> R<Vec<Case>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results")
        .join(file);
    let text = std::fs::read_to_string(path)?;
    let total = text.split("-- !query\n").skip(1).count();
    let cases: Vec<Case> = text
        .split("-- !query\n")
        .skip(1)
        .filter_map(|block| {
            let (query, rest) = block.split_once("\n-- !query schema\n")?;
            let (schema, output) = rest.split_once("\n-- !query output\n")?;
            Some(Case {
                query: query.trim().to_owned(),
                schema: schema.trim().to_owned(),
                output: output.trim_end().to_owned(),
            })
        })
        .collect();
    if cases.len() != total {
        return Err(format!("dropped {} statements of {file}", total - cases.len()).into());
    }
    Ok(cases)
}

fn find<'a>(cases: &'a [Case], query: &str) -> R<&'a Case> {
    cases
        .iter()
        .find(|c| c.query == query)
        .ok_or_else(|| format!("the golden has no statement {query:?}").into())
}

fn spark_type(t: &DataType) -> &'static str {
    match t {
        DataType::Int32 => "int",
        DataType::Int64 => "bigint",
        DataType::Float64 => "double",
        _ => "other",
    }
}

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Float64 => format!("{:?}", c.as_primitive::<Float64Type>().value(row)),
        other => format!("<{other}>"),
    }
}

/// Spark's rendering of a result: `struct<name:type,..>` and the tab
/// separated rows.
fn render(schema: &Schema, batches: &[RecordBatch]) -> (String, Vec<String>) {
    let fields = schema
        .fields()
        .iter()
        .map(|f| format!("{}:{}", f.name(), spark_type(f.data_type())))
        .collect::<Vec<_>>()
        .join(",");
    let rows = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(|r| {
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("\t")
            })
        })
        .collect();
    (format!("struct<{fields}>"), rows)
}

/// A table of nullable `int` columns.
fn table(s: &Session, name: &str, columns: &[&str], rows: &[&[Option<i32>]]) -> R {
    let fields: Vec<Field> = columns
        .iter()
        .map(|c| Field::new(*c, DataType::Int32, true))
        .collect();
    let arrays: Vec<ArrayRef> = (0..columns.len())
        .map(|c| {
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r[c]).collect::<Vec<_>>(),
            )) as ArrayRef
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    s.register_batches(name, schema, vec![batch])?;
    Ok(())
}

fn run(s: &Session, query: &str) -> Result<(Arc<Schema>, Vec<RecordBatch>), KrishivError> {
    block_on(async {
        let df = s.sql(query).await?;
        let schema = df.schema().await?;
        Ok((schema, df.collect().await?))
    })
}

/// The golden's `testData` view of `group-by.sql`.
const TEST_DATA: [[Option<i32>; 2]; 9] = [
    [Some(1), Some(1)],
    [Some(1), Some(2)],
    [Some(2), Some(1)],
    [Some(2), Some(2)],
    [Some(3), Some(1)],
    [Some(3), Some(2)],
    [None, Some(1)],
    [Some(3), None],
    [None, None],
];

/// The golden's `testRegression` view of `linear-regression.sql`.
const REGRESSION: [[Option<i32>; 3]; 6] = [
    [Some(1), Some(10), None],
    [Some(2), Some(10), Some(11)],
    [Some(2), Some(20), Some(22)],
    [Some(2), Some(25), None],
    [Some(2), Some(30), Some(35)],
    [Some(2), None, Some(40)],
];

fn session() -> R<Session> {
    let s = Session::new()?;
    let rows: Vec<&[Option<i32>]> = TEST_DATA.iter().map(|r| &r[..]).collect();
    table(&s, "testData", &["a", "b"], &rows)?;
    let rows: Vec<&[Option<i32>]> = REGRESSION.iter().map(|r| &r[..]).collect();
    table(&s, "testRegression", &["k", "y", "x"], &rows)?;
    table(
        &s,
        "t",
        &["x", "y"],
        &[
            &[Some(1), Some(1)],
            &[Some(2), Some(2)],
            &[Some(2), Some(2)],
        ],
    )?;
    Ok(s)
}

/// Whether two rows agree: cells equal as text, or doubles (written with a
/// fraction or an exponent) within 1e-14 of each other, relative.
fn agree(got: &str, want: &str) -> bool {
    let (got, want): (Vec<_>, Vec<_>) = (got.split('\t').collect(), want.split('\t').collect());
    let real = |c: &str| c.contains(['.', 'E']);
    got.len() == want.len()
        && got.iter().zip(&want).all(|(g, w)| {
            g == w
                || (real(g)
                    && real(w)
                    && g.parse::<f64>().is_ok_and(|g| {
                        w.parse::<f64>()
                            .is_ok_and(|w| (g - w).abs() <= 1e-14 * g.abs().max(w.abs()))
                    }))
        })
}

fn assert_rows(mut got: Vec<String>, mut want: Vec<String>, what: &str) {
    got.sort();
    want.sort();
    assert_eq!(got.len(), want.len(), "{what}: {got:?} vs {want:?}");
    for (g, w) in got.iter().zip(&want) {
        assert!(agree(g, w), "{what}: {g:?} vs {w:?}");
    }
}

/// Runs the golden statement `query` of `file` and compares schema and rows.
fn replay(file: &str, query: &str) -> R {
    let cases = load(file)?;
    let case = find(&cases, query)?;
    let (schema, batches) = run(&session()?, &case.query)?;
    let (schema, rows) = render(&schema, &batches);
    assert_eq!(schema, case.schema, "{query}");
    assert_rows(
        rows,
        case.output.lines().map(str::to_owned).collect(),
        query,
    );
    Ok(())
}

#[test]
fn group_by_golden_skewness_through_stddev_over_nulls() -> R {
    replay(
        "group-by.sql.out",
        "SELECT SKEWNESS(a), KURTOSIS(a), MIN(a), MAX(a), AVG(a), VARIANCE(a), STDDEV(a), SUM(a), COUNT(a)\nFROM testData",
    )
}

/// `corr(DISTINCT x, y)` of the golden's pairs; the golden statement also
/// asks for `corr(DISTINCT y, x)`, a second argument list that `DISTINCT`
/// aggregates do not support yet.
#[test]
fn distinct_pair_statistics_of_the_group_by_golden_rows() -> R {
    let cases = load("group-by.sql.out")?;
    let golden = find(
        &cases,
        "SELECT corr(DISTINCT x, y), corr(DISTINCT y, x), count(*)\n  FROM (VALUES (1, 1), (2, 2), (2, 2)) t(x, y)",
    )?;
    assert_eq!(golden.output, "1.0\t1.0\t3");
    let (schema, batches) = run(&session()?, "SELECT corr(DISTINCT x, y), count(*) FROM t")?;
    let (schema, rows) = render(&schema, &batches);
    assert_eq!(schema, "struct<corr(DISTINCT x, y):double,count(1):bigint>");
    assert_eq!(rows, ["1.0\t3"]);
    Ok(())
}

/// The `corr` columns of the `linear-regression` goldens, without their
/// `regr_r2` column (the `regr_*` family is not implemented): the golden's
/// own rows for `corr`.
#[test]
fn linear_regression_golden_corr_columns() -> R {
    let cases = load("linear-regression.sql.out")?;
    let golden = find(
        &cases,
        "SELECT k, corr(y, x), regr_r2(y, x) FROM testRegression GROUP BY k",
    )?;
    // `k <TAB> corr <TAB> regr_r2`: keep the first two cells.
    let want: Vec<String> = golden
        .output
        .lines()
        .map(|l| l.split('\t').take(2).collect::<Vec<_>>().join("\t"))
        .collect();
    assert_eq!(want, ["1\tNULL", "2\t0.9988445981121533"]);
    let (schema, batches) = run(
        &session()?,
        "SELECT k, corr(y, x) FROM testRegression GROUP BY k",
    )?;
    let (schema, rows) = render(&schema, &batches);
    assert_eq!(schema, "struct<k:int,corr(y, x):double>");
    assert_rows(rows, want, "corr by k");
    Ok(())
}

#[test]
fn moments_over_the_golden_tables_follow_the_flags() -> R {
    let s = session()?;
    let one = |query: &str| -> R<String> {
        let (schema, batches) = run(&s, query)?;
        Ok(render(&schema, &batches).1.join("|"))
    };
    // One row per group in `testData` grouped by `a`, `b`: the sample
    // statistics divide by zero.
    let q = "SELECT stddev(a), var_samp(a), covar_samp(a, b), corr(a, b), skewness(a), kurtosis(a), \
             stddev_pop(a), var_pop(a), covar_pop(a, b) FROM testData WHERE a = 1 AND b = 1";
    assert_eq!(one(q)?, "NULL\tNULL\tNULL\tNULL\tNULL\tNULL\t0.0\t0.0\t0.0");
    s.conf()
        .set("spark.sql.legacy.statisticalAggregate", "true")?;
    assert_eq!(one(q)?, "NaN\tNaN\tNaN\tNaN\tNaN\tNaN\t0.0\t0.0\t0.0");
    // The setting does not change a group with spread.
    let spread = "SELECT stddev(a), covar_pop(a, b) FROM testData";
    let with = one(spread)?;
    s.conf()
        .set("spark.sql.legacy.statisticalAggregate", "false")?;
    assert_eq!(one(spread)?, with);
    Ok(())
}

#[test]
fn corr_of_a_constant_column_follows_ansi() -> R {
    let s = session()?;
    // `a` is 3 in every row of the filtered input, `b` varies.
    let q = "SELECT corr(a, b) FROM testData WHERE a = 3 AND b IS NOT NULL";
    for legacy in ["false", "true"] {
        s.conf()
            .set("spark.sql.legacy.statisticalAggregate", legacy)?;
        s.conf().set("spark.sql.ansi.enabled", "false")?;
        let (schema, batches) = run(&s, q)?;
        assert_eq!(render(&schema, &batches).1, ["NULL"], "legacy={legacy}");
        s.conf().set("spark.sql.ansi.enabled", "true")?;
        let e = run(&s, q).expect_err("DIVIDE_BY_ZERO under ANSI");
        assert_eq!(e.error_class(), ErrorClass::DivideByZero, "legacy={legacy}");
    }
    Ok(())
}

/// The window form is not implemented: it is refused, never computed
/// as another function.
#[test]
fn moment_window_aggregates_are_refused() -> R {
    let s = session()?;
    let e = run(&s, "SELECT stddev(a) OVER (PARTITION BY b) FROM testData")
        .expect_err("not implemented");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    Ok(())
}

/// Statements of the three goldens that use these functions and are not
/// claimed, each with the reason.
const UNCLAIMED: [(&str, &str, &str); 4] = [
    (
        "group-by.sql.out",
        "SELECT corr(DISTINCT x, y), corr(DISTINCT y, x), count(*)",
        "DISTINCT aggregates over different argument lists; checked above with one",
    ),
    (
        "window.sql.out",
        "stddev(val) OVER w AS stddev,",
        "window aggregates of the moment functions, with approx_count_distinct, collect_list and collect_set in the same statement",
    ),
    (
        "linear-regression.sql.out",
        "SELECT k, corr(y, x), regr_r2(y, x) FROM testRegression GROUP BY k",
        "regr_r2 (the regr_* family is out of scope); its corr column is checked above",
    ),
    (
        "linear-regression.sql.out",
        "SELECT k, corr(y, x) FILTER (WHERE x IS NOT NULL), regr_r2(y, x) FROM testRegression GROUP BY k",
        "regr_r2 (out of scope) and aggregate FILTER clauses",
    ),
];

#[test]
fn every_statement_using_a_moment_function_is_claimed_or_listed() -> R {
    let names = [
        "stddev(",
        "stddev_samp(",
        "stddev_pop(",
        "std(",
        "variance(",
        "var_samp(",
        "var_pop(",
        "corr(",
        "covar_samp(",
        "covar_pop(",
        "skewness(",
        "kurtosis(",
    ];
    let claimed = [("group-by.sql.out", "SELECT SKEWNESS(a)")];
    for file in [
        "group-by.sql.out",
        "window.sql.out",
        "linear-regression.sql.out",
    ] {
        for case in load(file)? {
            let lower = case.query.to_ascii_lowercase();
            if !names.iter().any(|n| lower.contains(n)) {
                continue;
            }
            let known = claimed
                .iter()
                .any(|(f, q)| *f == file && case.query.starts_with(q))
                || UNCLAIMED
                    .iter()
                    .any(|(f, q, _)| *f == file && case.query.contains(q));
            assert!(known, "{file}: unlisted statement {:?}", case.query);
        }
    }
    // Every listed statement is still in its file.
    for (file, q, _) in UNCLAIMED {
        assert!(
            load(file)?.iter().any(|c| c.query.contains(q)),
            "{file}: {q}"
        );
    }
    Ok(())
}

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
use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Float32Array, Int32Array, RecordBatch};
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
        DataType::Float32 => "float",
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
        DataType::Float32 => format!("{:?}", c.as_primitive::<Float32Type>().value(row)),
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

/// The PostgreSQL `aggtest` table of the goldens (`a int`, `b float`).
fn aggtest(s: &Session) -> R {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Float32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![
                Some(56),
                Some(100),
                Some(0),
                Some(42),
            ])) as ArrayRef,
            Arc::new(Float32Array::from(vec![
                Some(7.8_f32),
                Some(99.097_f32),
                Some(0.09561_f32),
                Some(324.78_f32),
            ])) as ArrayRef,
        ],
    )?;
    s.register_batches("aggtest", schema, vec![batch])?;
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
    aggtest(&s)?;
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

/// The `aggtest` moments of `postgreSQL/aggregates_part1.sql.out`: the
/// plain, decimal-cast and pair statements, replayed exactly. The
/// decimal casts exercise the analyzer-cast-to-double path end to end.
#[test]
fn postgresql_aggregates_part1_moments() -> R {
    for query in [
        "SELECT stddev_pop(b) FROM aggtest",
        "SELECT stddev_samp(b) FROM aggtest",
        "SELECT var_pop(b) FROM aggtest",
        "SELECT var_samp(b) FROM aggtest",
        "SELECT stddev_pop(CAST(b AS Decimal(38,0))) FROM aggtest",
        "SELECT stddev_samp(CAST(b AS Decimal(38,0))) FROM aggtest",
        "SELECT var_pop(CAST(b AS Decimal(38,0))) FROM aggtest",
        "SELECT var_samp(CAST(b AS Decimal(38,0))) FROM aggtest",
        "SELECT covar_pop(b, a), covar_samp(b, a) FROM aggtest",
        "SELECT corr(b, a) FROM aggtest",
    ] {
        replay("postgreSQL/aggregates_part1.sql.out", query)?;
    }
    Ok(())
}

/// The single-row statements of `postgreSQL/aggregates_part1.sql.out`: a
/// population statistic over one row is its value, a sample statistic is
/// NULL.
#[test]
fn postgresql_aggregates_part1_single_row_moments() -> R {
    for query in [
        "SELECT var_pop(1.0), var_samp(2.0)",
        "SELECT stddev_pop(CAST(3.0 AS Decimal(38,0))), stddev_samp(CAST(4.0 AS Decimal(38,0)))",
    ] {
        replay("postgreSQL/aggregates_part1.sql.out", query)?;
    }
    Ok(())
}

/// The string-literal moments of `promoteStrings.sql.out`: `'1'` promotes
/// to double, so one row gives the population value and NULL for the
/// sample statistics, skewness and kurtosis.
#[test]
fn promote_strings_moment_literals() -> R {
    let s = Session::new()?;
    table(&s, "t", &["v"], &[&[Some(1)]])?;
    let cases = load("typeCoercion/native/promoteStrings.sql.out")?;
    for query in [
        "SELECT stddev_pop('1') FROM t",
        "SELECT stddev_samp('1') FROM t",
        "SELECT var_pop('1') FROM t",
        "SELECT var_samp('1') FROM t",
        "SELECT skewness('1') FROM t",
        "SELECT kurtosis('1') FROM t",
    ] {
        let case = find(&cases, query)?;
        let (schema, batches) = run(&s, &case.query)?;
        let (schema, rows) = render(&schema, &batches);
        assert_eq!(schema, case.schema, "{query}");
        assert_rows(
            rows,
            case.output.lines().map(str::to_owned).collect(),
            query,
        );
    }
    Ok(())
}

/// `mean` is Spark's alias of `avg`: the grouping and window planners
/// share one name table, so it plans in both.
#[test]
fn mean_plans_as_avg_in_group_by() -> R {
    let (schema, batches) = run(&session()?, "SELECT mean(a), avg(a) FROM testData")?;
    let (schema, rows) = render(&schema, &batches);
    assert_eq!(schema, "struct<mean(a):double,avg(a):double>");
    assert_eq!(rows, ["2.142857142857143\t2.142857142857143"]);
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

/// Statements of the goldens that use these functions and are not
/// claimed, each with the reason. The `udf`/`udaf` suites need SQL UDFs and
/// user-defined aggregates, which are not implemented; the window suites
/// need the window form of the moment aggregates, which is refused.
const UNCLAIMED: [(&str, &str, &str); 10] = [
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
    (
        "postgreSQL/aggregates_part1.sql.out",
        "SELECT avg(CAST(x AS DOUBLE)), var_pop(CAST(x AS DOUBLE))",
        "inline VALUES, which the session runner does not support; infinite inputs are covered by the accumulator unit tests",
    ),
    (
        "postgreSQL/window_part4.sql.out",
        "OVER (ORDER BY i ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)",
        "window form of the moment aggregates, which is refused",
    ),
    (
        "udaf/udaf-group-by.sql.out",
        "udaf(",
        "user-defined aggregate functions are not implemented; its moment columns repeat the group-by golden checked above",
    ),
    (
        "udf/postgreSQL/udf-aggregates_part1.sql.out",
        "udf(",
        "SQL UDFs are not implemented; the same statements without udf are checked above",
    ),
    (
        "udf/udf-group-by.sql.out",
        "udf(",
        "SQL UDFs are not implemented; the DISTINCT pair without udf is checked above",
    ),
    (
        "udf/udf-window.sql.out",
        "udf(",
        "window form of the moment aggregates over SQL UDFs, neither of which is implemented",
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
    let claimed = [
        ("group-by.sql.out", "SELECT SKEWNESS(a)"),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT stddev_pop(b) FROM aggtest",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT stddev_samp(b) FROM aggtest",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT var_pop(b) FROM aggtest",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT var_samp(b) FROM aggtest",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT stddev_pop(CAST(b AS Decimal(38,0))) FROM aggtest",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT stddev_samp(CAST(b AS Decimal(38,0))) FROM aggtest",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT var_pop(CAST(b AS Decimal(38,0))) FROM aggtest",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT var_samp(CAST(b AS Decimal(38,0))) FROM aggtest",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT var_pop(1.0), var_samp(2.0)",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT stddev_pop(CAST(3.0 AS Decimal(38,0)))",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT covar_pop(b, a), covar_samp(b, a) FROM aggtest",
        ),
        (
            "postgreSQL/aggregates_part1.sql.out",
            "SELECT corr(b, a) FROM aggtest",
        ),
        (
            "typeCoercion/native/promoteStrings.sql.out",
            "SELECT stddev_pop('1') FROM t",
        ),
        (
            "typeCoercion/native/promoteStrings.sql.out",
            "SELECT stddev_samp('1') FROM t",
        ),
        (
            "typeCoercion/native/promoteStrings.sql.out",
            "SELECT var_pop('1') FROM t",
        ),
        (
            "typeCoercion/native/promoteStrings.sql.out",
            "SELECT var_samp('1') FROM t",
        ),
        (
            "typeCoercion/native/promoteStrings.sql.out",
            "SELECT skewness('1') FROM t",
        ),
        (
            "typeCoercion/native/promoteStrings.sql.out",
            "SELECT kurtosis('1') FROM t",
        ),
    ];
    for file in [
        "group-by.sql.out",
        "window.sql.out",
        "linear-regression.sql.out",
        "postgreSQL/aggregates_part1.sql.out",
        "postgreSQL/window_part4.sql.out",
        "udaf/udaf-group-by.sql.out",
        "typeCoercion/native/promoteStrings.sql.out",
        "udf/postgreSQL/udf-aggregates_part1.sql.out",
        "udf/udf-group-by.sql.out",
        "udf/udf-window.sql.out",
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

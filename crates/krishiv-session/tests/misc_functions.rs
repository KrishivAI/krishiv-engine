// SPDX-License-Identifier: Apache-2.0
//! The misc-function goldens end to end: every `misc-functions.sql`,
//! `random.sql` and `current_database_catalog.sql` statement that uses only
//! the implemented functions matches Spark 4.2.0 exactly (schema, rows,
//! error class and parameters), except the multi-row `VALUES` statements:
//! the engine runs those tiny scans in one partition while Spark CI splits
//! them over two, so the row multisets differ. The single-partition draws
//! are asserted here; their decomposition into per-partition streams is
//! pinned at the binding level (`scalar/tests/random.rs`: partition 0's
//! first `uniform(10, 20, 0)` draw is 17, partition 1's next two are 16 and
//! 15 — the golden multiset — and `randstr`'s partition-1 rows match the
//! golden's rows 2-3 exactly). The nondeterministic functions are otherwise
//! deterministic per partition.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use arrow_schema::{Field, Schema};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;
use std::sync::Arc;

fn block_on<T>(f: impl Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(f)
}

/// The schema names and every row of `sql`, values as strings (`~` for
/// NULL), rows in output order.
fn rows(s: &Session, sql: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let batches = block_on(df.collect()).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let mut row = Vec::new();
            for c in b.columns() {
                if c.is_null(i) {
                    row.push("~".to_owned());
                    continue;
                }
                row.push(match c.data_type() {
                    ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                    ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                    ArrowType::Int32 => c
                        .as_primitive::<arrow_array::types::Int32Type>()
                        .value(i)
                        .to_string(),
                    ArrowType::Int64 => c
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(i)
                        .to_string(),
                    ArrowType::Float32 => c
                        .as_primitive::<arrow_array::types::Float32Type>()
                        .value(i)
                        .to_string(),
                    ArrowType::Float64 => c
                        .as_primitive::<arrow_array::types::Float64Type>()
                        .value(i)
                        .to_string(),
                    ArrowType::Boolean => c.as_boolean().value(i).to_string(),
                    ArrowType::Decimal128(_, s) => {
                        let v = c
                            .as_primitive::<arrow_array::types::Decimal128Type>()
                            .value(i);
                        render_decimal(v, *s)
                    }
                    t => panic!("{sql}: unexpected type {t}"),
                });
            }
            out.push(row);
        }
    }
    (names, out)
}

fn session() -> Session {
    Session::new().unwrap()
}

/// Renders a decimal `(unscaled, scale)` as Spark does.
fn render_decimal(unscaled: i128, scale: i8) -> String {
    let neg = unscaled < 0;
    let digits = unscaled.unsigned_abs().to_string();
    let s = scale.cast_unsigned() as usize;
    let text = if s == 0 {
        digits
    } else if digits.len() > s {
        let (i, f) = digits.split_at(digits.len() - s);
        format!("{i}.{f}")
    } else {
        format!("0.{:0>width$}", digits, width = s)
    };
    format!("{}{text}", if neg { "-" } else { "" })
}

/// A session with a three-row table (`SELECT ... FROM vals` stands in for
/// the goldens' `FROM VALUES (0), (1), (2)`, which the engine does not
/// support yet).
fn values_session() -> Session {
    use arrow_array::Int32Array;
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "col",
        ArrowType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2]))],
    )
    .unwrap();
    s.register_batches("vals", schema, vec![batch]).unwrap();
    s
}

fn error(s: &Session, sql: &str) -> KrishivError {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_err()
}

#[test]
fn typeof_names_the_analyzers_type() {
    let s = session();
    // `select typeof(null)` in `misc-functions.sql`.
    assert_eq!(
        rows(&s, "select typeof(null)"),
        (
            vec!["typeof(NULL)".to_owned()],
            vec![vec!["void".to_owned()]]
        )
    );
    assert_eq!(
        rows(&s, "select typeof(true)"),
        (
            vec!["typeof(true)".to_owned()],
            vec![vec!["boolean".to_owned()]]
        )
    );
    assert_eq!(
        rows(&s, "select typeof(1Y), typeof(1S), typeof(1), typeof(1L)"),
        (
            vec![
                "typeof(1)".to_owned(),
                "typeof(1)".to_owned(),
                "typeof(1)".to_owned(),
                "typeof(1)".to_owned(),
            ],
            vec![vec![
                "tinyint".to_owned(),
                "smallint".to_owned(),
                "int".to_owned(),
                "bigint".to_owned(),
            ]]
        )
    );
    assert_eq!(
        rows(
            &s,
            "select typeof(cast(1.0 as float)), typeof(1.0D), typeof(1.2)"
        ),
        (
            vec![
                "typeof(CAST(1.0 AS FLOAT))".to_owned(),
                "typeof(1.0)".to_owned(),
                "typeof(1.2)".to_owned(),
            ],
            vec![vec![
                "float".to_owned(),
                "double".to_owned(),
                "decimal(2,1)".to_owned(),
            ]]
        )
    );
    assert_eq!(
        rows(&s, "select typeof(x'ABCD'), typeof('SPARK')"),
        (
            vec!["typeof(X'ABCD')".to_owned(), "typeof(SPARK)".to_owned()],
            vec![vec!["binary".to_owned(), "string".to_owned()]]
        )
    );
    assert_eq!(
        rows(
            &s,
            "select typeof(date '1986-05-23'), typeof(timestamp '1986-05-23')"
        ),
        (
            vec![
                "typeof(DATE '1986-05-23')".to_owned(),
                "typeof(TIMESTAMP '1986-05-23 00:00:00')".to_owned(),
            ],
            vec![vec!["date".to_owned(), "timestamp".to_owned()]]
        )
    );
}

#[test]
fn typeof_needs_no_execution() {
    // The rewrite is a literal: the argument is never evaluated, so a
    // failing argument still names its type (with the analyzer's casts, as
    // Spark's `toPrettySQL` shows them).
    let s = session();
    assert_eq!(
        rows(&s, "select typeof(1 / 0)").0,
        vec!["typeof((CAST(1 AS DOUBLE) / CAST(0 AS DOUBLE)))".to_owned()]
    );
    assert_eq!(
        rows(&s, "select typeof(1 / 0)").1,
        vec![vec!["double".to_owned()]]
    );
}

#[test]
fn current_database_and_catalog() {
    // `current_database_catalog.sql`, exactly.
    let s = session();
    assert_eq!(
        rows(
            &s,
            "select current_database(), current_schema(), current_catalog()"
        ),
        (
            vec![
                "current_schema()".to_owned(),
                "current_schema()".to_owned(),
                "current_catalog".to_owned(),
            ],
            vec![vec![
                "default".to_owned(),
                "default".to_owned(),
                "spark_catalog".to_owned(),
            ]]
        )
    );
    assert_eq!(
        rows(&s, "select version()").1,
        vec![vec!["4.2.0".to_owned()]]
    );
}

/// Seeded draws match Spark exactly; the multi-row `VALUES` cases assert
/// the engine's single-partition stream (Spark CI splits those rows over
/// two partitions — see the module docs).
#[test]
fn rand_and_randn_match_spark_single_partition() {
    let s = session();
    // `random.sql`, exactly.
    assert_eq!(
        rows(&s, "SELECT rand(0)").1,
        vec![vec!["0.7604953758285915".to_owned()]]
    );
    assert_eq!(
        rows(&s, "SELECT rand(cast(3 / 7 AS int))").1,
        vec![vec!["0.7604953758285915".to_owned()]]
    );
    assert_eq!(
        rows(&s, "SELECT rand(NULL)").1,
        vec![vec!["0.7604953758285915".to_owned()]]
    );
    assert_eq!(
        rows(&s, "SELECT rand(cast(NULL AS int))").1,
        vec![vec!["0.7604953758285915".to_owned()]]
    );
    assert_eq!(
        rows(&s, "SELECT randn(0L)").1,
        vec![vec!["1.6034991609278433".to_owned()]]
    );
    assert_eq!(
        rows(&s, "SELECT randn(cast(3 / 7 AS long))").1,
        vec![vec!["1.6034991609278433".to_owned()]]
    );
    assert_eq!(
        rows(&s, "SELECT randn(NULL)").1,
        vec![vec!["1.6034991609278433".to_owned()]]
    );
    assert_eq!(
        rows(&s, "SELECT randn(cast(NULL AS long))").1,
        vec![vec!["1.6034991609278433".to_owned()]]
    );
    // The seed types Spark rejects.
    let e = error(&s, "SELECT rand(1.0)");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(e.to_string().contains("(\"INT\" or \"BIGINT\")"), "{e}");
    let e = error(&s, "SELECT rand('1')");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(e.to_string().contains("\"rand(1)\""), "{e}");
    // A non-foldable seed is refused.
    let v = values_session();
    let e = error(&v, "SELECT rand(col) FROM vals");
    assert_eq!(e.error_class(), ErrorClass::SeedExpressionIsUnfoldable);
    let e = error(&s, "SELECT rand(id) FROM (SELECT 1 AS id)");
    assert_eq!(e.error_class(), ErrorClass::SeedExpressionIsUnfoldable);
    // Too many seeds is an arity error.
    let e = error(&s, "SELECT rand(1, 2)");
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
}

/// Seeded values match Spark exactly; the multi-row `VALUES` case asserts
/// the engine's single-partition draws (Spark CI splits those rows over
/// two partitions — see the module docs).
#[test]
fn uniform_matches_spark_single_partition() {
    let s = session();
    let value = |sql: &str| rows(&s, sql).1.into_iter().flatten().collect::<Vec<_>>();
    assert_eq!(value("SELECT uniform(0, 1, 0) AS result"), ["0"]);
    assert_eq!(value("SELECT uniform(0, 10, 0) AS result"), ["7"]);
    assert_eq!(value("SELECT uniform(0L, 10L, 0) AS result"), ["7"]);
    assert_eq!(value("SELECT uniform(10, 20, 0) AS result"), ["17"]);
    assert_eq!(
        value("SELECT uniform(10.0F, 20.0F, 0) AS result"),
        ["17.604954"]
    );
    assert_eq!(
        value(
            "SELECT uniform(cast(10 as decimal(10, 3)), cast(20 as decimal(10, 3)), 0) AS result"
        ),
        ["17.605"]
    );
    assert_eq!(
        value(
            "SELECT uniform(cast(10 as decimal(10, 3)), cast(20 as decimal(11, 4)), 0) AS result"
        ),
        ["17.6050"]
    );
    assert_eq!(
        value("SELECT uniform(10.0D, 20.0D, CAST(3 / 7 AS LONG)) AS result"),
        ["17.604953758285916"]
    );
    assert_eq!(value("SELECT uniform(-10L, 10L, 0) AS result"), ["5"]);
    assert_eq!(value("SELECT uniform(-20L, -10L, 0) AS result"), ["-12"]);
    assert_eq!(value("SELECT uniform(-20L, -10L, -10) AS result"), ["-17"]);
    // NULL bounds are NULL results; a NULL seed reads as 0.
    assert_eq!(value("SELECT uniform(NULL, 1, 0) AS result"), ["~"]);
    assert_eq!(
        value("SELECT uniform(cast(NULL AS int), 1, 0) AS result"),
        ["~"]
    );
    assert_eq!(value("SELECT uniform(0, NULL, 0) AS result"), ["~"]);
    assert_eq!(value("SELECT uniform(0, 1, NULL) AS result"), ["0"]);
    assert_eq!(
        value("SELECT uniform(0, 1, cast(NULL as int)) AS result"),
        ["0"]
    );
    // The seed and bound checks Spark reports.
    let v = values_session();
    let e = error(&v, "SELECT uniform(10, 20, col) AS result FROM vals");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    assert!(e.to_string().contains("`seed`"), "{e}");
    let e = error(&s, "SELECT uniform(10) AS result");
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
    let e = error(&s, "SELECT uniform(10, 20, 30, 40) AS result");
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
    assert!(e.to_string().contains("[2, 3]"), "{e}");
    let e = error(&s, "SELECT uniform(10.0F, 20.0F, 0.0F) AS result");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(e.to_string().contains("third"), "{e}");
    let e = error(&s, "SELECT uniform('abc', 10, 0) AS result");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(e.to_string().contains("\"NUMERIC\""), "{e}");
    let e = error(&s, "SELECT uniform(0, 'def', 0) AS result");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(e.to_string().contains("\"NUMERIC\""), "{e}");
    assert!(e.to_string().contains("second"), "{e}");
    // Several partitions draw several streams: the engine runs the tiny
    // scan in one partition, so the rows are one stream's first draws
    // (Spark CI splits the rows over two partitions and sorts).
    let v = values_session();
    let mut got = rows(&v, "SELECT uniform(10, 20, 0) AS result FROM vals")
        .1
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    got.sort();
    assert_eq!(got, ["10", "15", "17"]);
}

/// Seeded strings match Spark exactly; the multi-row `VALUES` case asserts
/// the engine's single-partition draws (Spark CI splits those rows over
/// two partitions — see the module docs).
#[test]
fn randstr_matches_spark_single_partition() {
    let s = session();
    let value = |sql: &str| rows(&s, sql).1.into_iter().flatten().collect::<Vec<_>>();
    assert_eq!(value("SELECT randstr(1, 0) AS result"), ["c"]);
    assert_eq!(value("SELECT randstr(5, 0) AS result"), ["ceV0P"]);
    assert_eq!(value("SELECT randstr(10, 0) AS result"), ["ceV0PXaR2I"]);
    assert_eq!(value("SELECT randstr(10S, 0) AS result"), ["ceV0PXaR2I"]);
    assert_eq!(
        value("SELECT randstr(CAST(10 AS TINYINT), 0) AS result"),
        ["ceV0PXaR2I"]
    );
    assert_eq!(
        value("SELECT randstr(CAST(10 AS BIGINT), 0) AS result"),
        ["ceV0PXaR2I"]
    );
    assert_eq!(value("SELECT randstr(1.0F, 0) AS result"), ["c"]);
    assert_eq!(value("SELECT randstr(1.0D, 0) AS result"), ["c"]);
    assert_eq!(
        value("SELECT randstr(cast(1 AS DECIMAL(10, 2)), 0) AS result"),
        ["c"]
    );
    assert_eq!(value("SELECT randstr(1, -1) AS result"), ["S"]);
    assert_eq!(value("SELECT randstr(NULL, 0) AS result"), [""]);
    assert_eq!(value("SELECT randstr(0, NULL) AS result"), [""]);
    let v = values_session();
    let e = error(&v, "SELECT randstr(col, 0) AS result FROM vals");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    let e = error(&s, "SELECT randstr(10, 0, 1) AS result");
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
    let e = error(&s, "SELECT randstr(-1, 0) AS result");
    assert_eq!(e.error_class(), ErrorClass::InvalidParameterValueLength);
    // A string seed is neither an `int` nor a `bigint`.
    let e = error(&s, "SELECT randstr(10, 'a') AS result");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(e.to_string().contains("second"), "{e}");
    assert!(e.to_string().contains("(\"INT\" or \"BIGINT\")"), "{e}");
    // The engine's tiny scan is one partition: consecutive draws of one
    // stream (Spark CI splits the rows over two partitions and sorts).
    let v = values_session();
    assert_eq!(
        rows(&v, "SELECT randstr(10, 0) AS result FROM vals").1,
        [
            vec!["ceV0PXaR2I".to_owned()],
            vec!["lBhHi56d0u".to_owned()],
            vec!["CtuwSOTLiL".to_owned()]
        ]
    );
}

#[test]
fn nondeterministic_queries_are_deterministic_per_partition() {
    let s = session();
    let value = |sql: &str| rows(&s, sql).1.into_iter().flatten().collect::<Vec<_>>();
    // The same query twice draws the same stream (one partition).
    let once = value("SELECT rand(0), randn(0), uuid(42), monotonically_increasing_id()");
    let twice = value("SELECT rand(0), randn(0), uuid(42), monotonically_increasing_id()");
    assert_eq!(once, twice);
    // An unseeded call is not NULL.
    assert_eq!(value("SELECT randstr(10) IS NOT NULL AS result"), ["true"]);
    assert_eq!(
        value("SELECT uniform(10, 20.0F) IS NOT NULL AS result"),
        ["true"]
    );
    // `uuid()` draws from the query seed: two queries differ.
    let a = value("SELECT uuid()");
    let b = value("SELECT uuid()");
    assert_ne!(a, b);
    // Two unseeded calls in one query draw different streams.
    let both = rows(&s, "SELECT rand() AS a, rand() AS b").1;
    assert_ne!(both[0][0], both[0][1]);
    // Partition functions in one partition.
    assert_eq!(value("SELECT spark_partition_id()"), ["0"]);
    assert_eq!(value("SELECT monotonically_increasing_id()"), ["0"]);
    assert_eq!(value("SELECT input_file_name()"), [""]);
    assert_eq!(value("SELECT input_file_block_start()"), ["-1"]);
    assert_eq!(value("SELECT input_file_block_length()"), ["-1"]);
    // UNION ALL numbers its children's partitions one after another, as
    // Spark's `UnionRDD` does (Spark 4.2.0, local[1]: 0, 1, 2 and
    // 0, 8589934592).
    assert_eq!(
        value(
            "SELECT spark_partition_id() AS p FROM (SELECT 1 AS x UNION ALL \
             SELECT 2 AS x UNION ALL SELECT 3 AS x) ORDER BY p"
        ),
        ["0", "1", "2"]
    );
    assert_eq!(
        value(
            "SELECT monotonically_increasing_id() AS m FROM (SELECT 1 AS x \
             UNION ALL SELECT 2 AS x) ORDER BY m"
        ),
        ["0", "8589934592"]
    );
    // A stateful predicate sees the same partition stream.
    let v = values_session();
    assert_eq!(
        rows(&v, "SELECT col FROM vals WHERE rand(0) > 0.5").1,
        [vec!["0".to_owned()], vec!["1".to_owned()]]
    );
}

#[test]
fn misc_rejects_bad_calls() {
    let s = session();
    // `uuid` wants a plain literal seed.
    let e = error(&s, "SELECT uuid(1 + 1)");
    assert_eq!(e.error_class(), ErrorClass::InvalidParameterValueLong);
    // `typeof` takes exactly one argument.
    let e = error(&s, "SELECT typeof()");
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
    for sql in [
        "SELECT version(1)",
        "SELECT current_database(1)",
        "SELECT monotonically_increasing_id(1)",
        "SELECT input_file_name('x')",
        "SELECT spark_partition_id(1)",
    ] {
        let e = error(&s, sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::WrongNumArgsWithoutSuggestion,
            "{sql}"
        );
    }
}

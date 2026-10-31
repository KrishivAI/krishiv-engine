// SPDX-License-Identifier: Apache-2.0
//! SQL-bodied temporary functions through `Session::sql`: the Spark 4.2.0
//! goldens (`sql-udf.sql`, `sql-udf-name-precedence.sql` and its legacy
//! variant) replayed in file order in one session each, statement by
//! statement, and the behaviour around them.
//!
//! A golden file is replayed by its `.sql.out`, which holds every statement
//! with its schema and output (or its error class and message parameters).
//! Only the statements a test lists are replayed; each listed statement must
//! match exactly: the schema string, every row, the error class, SQLSTATE
//! and message parameters. The statements left out of each list, and why:
//!
//! - **`VALUES`** (`CREATE TEMPORARY VIEW t AS VALUES (0) t(a)` and the
//!   functions reading `t`): not claimed yet; the statements that need only
//!   temporary functions are the ones replayed.
//! - **Stored functions** (`CREATE FUNCTION` without `TEMPORARY`) live in the
//!   session's in-memory catalog and are run by the golden-report ratchet
//!   (`known-failures-cases/sql-udf*.json` lists the statements that still
//!   differ); the tests at the end of this file pin the behaviour.
//! - **Table functions** (`RETURNS TABLE`, `FROM f(...)`): not implemented
//!   yet.
//! - **`foo3_3b` under ANSI** (`SELECT foo3_3b(0)` after the function was
//!   created with ANSI on and called with it off): Spark runs the body with
//!   the SQL configuration captured when the function was created; here the
//!   body runs under the caller's.
//! - **Session variables** in `sql-udf.sql`: not claimed. The precedence
//!   file's `DECLARE x = 999` and the statements around it are replayed.
//! - **`outer_vs_param`** (`(SELECT (SELECT x) FROM v1)`): the engine does not
//!   decorrelate a scalar subquery without an aggregate, which this one
//!   becomes where the column `x` of `v1` wins over the parameter.
//! - **`typeof`, `current_time` and `grouping__id`** (the parameters named
//!   like them, and the statements around those): the engine has no
//!   `typeof` and no `TIME` type, and does not read a bare `grouping__id`
//!   as a function (in the legacy variant, where the parameter wins, the
//!   statements run).

#![cfg(test)]

use std::collections::BTreeMap;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use krishiv_types::DataType;

/// One statement of a golden `.out` file.
struct Block {
    query: String,
    schema: String,
    output: String,
}

fn blocks(file: &str) -> Vec<Block> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results")
        .join(file);
    let text = std::fs::read_to_string(&path).unwrap();
    text.split("-- !query\n")
        .skip(1)
        .map(|b| {
            let (query, rest) = b.split_once("\n-- !query schema\n").unwrap();
            let (schema, output) = rest.split_once("\n-- !query output\n").unwrap();
            Block {
                query: query.to_owned(),
                schema: schema.to_owned(),
                output: output.trim_end_matches('\n').to_owned(),
            }
        })
        .collect()
}

/// A cell as Spark's golden files print it.
fn cell(column: &ArrayRef, row: usize) -> String {
    if column.is_null(row) {
        return "NULL".to_owned();
    }
    match column.data_type() {
        ArrowType::Int32 => column.as_primitive::<Int32Type>().value(row).to_string(),
        ArrowType::Int64 => column.as_primitive::<Int64Type>().value(row).to_string(),
        ArrowType::Float64 => column.as_primitive::<Float64Type>().value(row).to_string(),
        ArrowType::Boolean => column.as_boolean().value(row).to_string(),
        ArrowType::Utf8 => column.as_string::<i32>().value(row).to_owned(),
        ArrowType::Utf8View => column.as_string_view().value(row).to_owned(),
        other => panic!("a {other} cell"),
    }
}

/// The schema as Spark prints it (`struct<a:int,b:string>`) and the rows.
fn run(s: &Session, sql: &str) -> Result<(String, Vec<String>), KrishivError> {
    let df = block_on(s.sql(sql))?;
    let schema = block_on(df.schema())?;
    let fields: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| {
            let t = DataType::from_arrow_field(f)?;
            Ok(format!("{}:{}", f.name(), t.simple_string()))
        })
        .collect::<Result<_, KrishivError>>()?;
    let mut rows = Vec::new();
    for batch in block_on(df.collect())? {
        for i in 0..batch.num_rows() {
            let cells: Vec<String> = batch.columns().iter().map(|c| cell(c, i)).collect();
            rows.push(cells.join("\t"));
        }
    }
    Ok((format!("struct<{}>", fields.join(",")), rows))
}

/// Replays the statements `claimed` of `file` (their positions in the file
/// and the text they must have, so a change of the golden shows) in order
/// in `s`, checking each against the golden.
fn replay(s: &Session, file: &str, claimed: &[(usize, &str)]) {
    let all = blocks(file);
    for &(i, query) in claimed {
        let b = &all[i];
        assert_eq!(b.query, query, "{file} #{i} is not the statement claimed");
        check(s, b, &format!("{file} #{i}: {query}"));
    }
}

fn check(s: &Session, b: &Block, what: &str) {
    let result = run(s, &b.query);
    if let Some((header, json)) = b.output.split_once('\n')
        && header.starts_with("org.apache.spark")
    {
        let expected: serde_json::Value = serde_json::from_str(json).unwrap();
        let e = match result {
            Err(e) => e,
            Ok(ok) => panic!("{what}: expected an error, got {ok:?}"),
        };
        assert_eq!(
            e.error_class().name(),
            expected["errorClass"].as_str().unwrap(),
            "{what}: {e}"
        );
        assert_eq!(e.sql_state(), expected["sqlState"].as_str(), "{what}");
        let want: BTreeMap<String, String> = expected["messageParameters"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
                    .collect()
            })
            .unwrap_or_default();
        let got: BTreeMap<String, String> = e
            .message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect();
        assert_eq!(got, want, "{what}");
        return;
    }
    let (schema, rows) = result.unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(schema, b.schema, "{what}");
    assert_eq!(rows.join("\n"), b.output, "{what}");
}

#[test]
fn sql_udf_golden_temporary_functions() {
    let s = Session::new().unwrap();
    replay(
        &s,
        "sql-udf.sql.out",
        &[
            (
                34,
                "CREATE OR REPLACE TEMPORARY FUNCTION foo1d3(a INT DEFAULT 7 COMMENT 'hello') RETURNS INT RETURN a",
            ),
            (35, "SELECT foo1d3(5), foo1d3()"),
            (331, "SET spark.sql.ansi.enabled=true"),
            (
                334,
                "CREATE TEMPORARY FUNCTION foo3_3b(x INT) RETURNS DOUBLE RETURN 1 / x",
            ),
            (335, "SET spark.sql.ansi.enabled=false"),
            (
                341,
                "CREATE OR REPLACE TEMPORARY FUNCTION foo3_3b(x INT) RETURNS DOUBLE RETURN 1 / x",
            ),
            (343, "SELECT foo3_3b(0)"),
            (
                365,
                "CREATE OR REPLACE TEMP FUNCTION foo3_4g(x INT) RETURN x + 1",
            ),
            (
                366,
                "CREATE OR REPLACE TEMP FUNCTION foo3_4h(x INT) RETURN foo3_4g(x) + 1",
            ),
            (
                374,
                "CREATE OR REPLACE TEMP FUNCTION foo3_4g(x INT) RETURN foo3_4h(x) + 1",
            ),
            (
                376,
                "CREATE TEMPORARY FUNCTION foo3_5a(x INT) RETURNS INT RETURN x",
            ),
            (565, "DROP TEMPORARY FUNCTION IF EXISTS foo1d3"),
            (566, "DROP TEMPORARY FUNCTION IF EXISTS foo3_3b"),
            (567, "DROP TEMPORARY FUNCTION IF EXISTS foo3_5a"),
            (568, "DROP TEMPORARY FUNCTION IF EXISTS foo3_5b"),
            (569, "DROP TEMPORARY FUNCTION IF EXISTS foo3_5c"),
        ],
    );
}

/// The statements of `sql-udf-name-precedence.sql` both variants replay
/// (positions in the `.out` file).
const PRECEDENCE: &[(usize, &str)] = &[
    (0, "CREATE OR REPLACE TEMPORARY VIEW v1 AS SELECT 1 AS x"),
    (
        1,
        "CREATE OR REPLACE TEMPORARY FUNCTION identity_fn(x INT) RETURNS INT RETURN x",
    ),
    (2, "SELECT identity_fn(42)"),
    (
        3,
        "CREATE OR REPLACE TEMPORARY FUNCTION col_vs_param(x INT) RETURNS INT RETURN (SELECT x FROM v1)",
    ),
    (4, "SELECT col_vs_param(42)"),
    (
        5,
        "CREATE OR REPLACE TEMPORARY FUNCTION paramless_vs_param(current_user STRING)\nRETURNS STRING RETURN current_user",
    ),
    (
        6,
        "SELECT paramless_vs_param('should_be_ignored') = current_user() AS function_won",
    ),
    (
        13,
        "CREATE OR REPLACE TEMPORARY FUNCTION lca_vs_param(x INT)\nRETURNS INT RETURN (SELECT y FROM (SELECT 999 AS x, x + 1 AS y))",
    ),
    (14, "SELECT lca_vs_param(42)"),
    (
        17,
        "CREATE OR REPLACE TEMPORARY FUNCTION outer_param_pure(x INT)\nRETURNS INT RETURN (SELECT (SELECT x))",
    ),
    (18, "SELECT outer_param_pure(42)"),
    (19, "DECLARE x = 999"),
    (20, "SELECT identity_fn(42)"),
    (
        21,
        "CREATE OR REPLACE TEMPORARY FUNCTION inner_fn(y INT) RETURNS INT RETURN x",
    ),
    (
        22,
        "CREATE OR REPLACE TEMPORARY FUNCTION outer_fn(x INT) RETURNS INT RETURN inner_fn(x)",
    ),
    (23, "SELECT outer_fn(42)"),
    (27, "DROP VIEW v1"),
    (28, "DROP TEMPORARY FUNCTION identity_fn"),
    (29, "DROP TEMPORARY FUNCTION col_vs_param"),
    (30, "DROP TEMPORARY FUNCTION paramless_vs_param"),
    (34, "DROP TEMPORARY FUNCTION lca_vs_param"),
    (36, "DROP TEMPORARY FUNCTION outer_param_pure"),
];

#[test]
fn sql_udf_name_precedence_golden() {
    let s = Session::new().unwrap();
    replay(&s, "sql-udf-name-precedence.sql.out", PRECEDENCE);
}

#[test]
fn sql_udf_name_precedence_legacy_golden() {
    let s = Session::new().unwrap();
    s.conf()
        .set(
            "spark.sql.legacy.allowUdfParameterToShadowParameterlessFunction",
            "true",
        )
        .unwrap();
    // The variant's own output: the parameter shadows `current_user`, so
    // that call differs; a bare `grouping__id` is the parameter too (the
    // call returns its argument).
    let mut claimed: Vec<(usize, &str)> = PRECEDENCE.to_vec();
    claimed.extend([
        (
            11,
            "CREATE OR REPLACE TEMPORARY FUNCTION paramless_vs_param_grouping(grouping__id INT)\nRETURNS INT RETURN grouping__id",
        ),
        (12, "SELECT paramless_vs_param_grouping(42)"),
        (33, "DROP TEMPORARY FUNCTION paramless_vs_param_grouping"),
    ]);
    claimed.sort_by_key(|(i, _)| *i);
    replay(&s, "sql-udf-name-precedence-legacy.sql.out", &claimed);
}

// ---- behaviour beyond the goldens --------------------------------------

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{Field, Schema};
use std::sync::Arc;

fn session() -> Session {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int32, false),
        Field::new("name", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![3, 1, 2])),
            Arc::new(StringArray::from(vec![Some("c"), None, Some("b")])),
        ],
    )
    .unwrap();
    s.register_batches("people", schema, vec![batch]).unwrap();
    s
}

fn ok(s: &Session, sql: &str) -> Vec<String> {
    run(s, sql).unwrap_or_else(|e| panic!("{sql}: {e}")).1
}

fn fails(s: &Session, sql: &str) -> KrishivError {
    match run(s, sql) {
        Err(e) => e,
        Ok(r) => panic!("{sql} ran: {r:?}"),
    }
}

fn class_and_params(e: &KrishivError) -> (String, BTreeMap<String, String>) {
    (
        e.error_class().name().to_owned(),
        e.message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    )
}

#[test]
fn functions_run_over_columns() {
    let s = session();
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION plus(a INT, b INT) RETURNS INT RETURN a + b",
    );
    assert_eq!(
        ok(&s, "SELECT plus(id, 10) FROM people ORDER BY id"),
        ["11", "12", "13"]
    );
    // Calls nest, and a function calls another.
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION twice(a INT) RETURNS INT RETURN plus(a, a)",
    );
    assert_eq!(
        ok(&s, "SELECT twice(plus(id, 1)) FROM people WHERE id = 1"),
        ["4"]
    );
    // The result is cast to the declared type.
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION d(a INT) RETURNS DOUBLE RETURN a",
    );
    let (schema, rows) = run(&s, "SELECT d(2)").unwrap();
    assert_eq!(schema, "struct<d(2):double>");
    assert_eq!(rows, ["2"]);
    // In WHERE and ORDER BY.
    assert_eq!(
        ok(
            &s,
            "SELECT id FROM people WHERE plus(id, 0) > 1 ORDER BY plus(0, id) DESC"
        ),
        ["3", "2"]
    );
}

#[test]
fn defaults_and_named_arguments() {
    let s = session();
    let e = fails(
        &s,
        "CREATE TEMPORARY FUNCTION f(a INT, b INT DEFAULT a) RETURNS INT RETURN a + b",
    );
    assert_eq!(
        e.error_class().name(),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION g(a INT, b INT DEFAULT 7, c STRING DEFAULT 'x') RETURNS STRING RETURN a || '-' || b || c",
    );
    assert_eq!(
        ok(&s, "SELECT g(1), g(1, 2), g(1, 2, 'y')"),
        ["1-7x\t1-2x\t1-2y"]
    );
    assert_eq!(
        ok(&s, "SELECT g(c => 'z', a => 5), g(1, c => 'q')"),
        ["5-7z\t1-7q"]
    );
    let e = fails(&s, "SELECT g(b => 2)");
    assert_eq!(class_and_params(&e).0, "REQUIRED_PARAMETER_NOT_FOUND");
    assert_eq!(class_and_params(&e).1["parameterName"], "`a`");
    let e = fails(&s, "SELECT g(1, a => 2)");
    assert_eq!(
        e.error_class().name(),
        "DUPLICATE_ROUTINE_PARAMETER_ASSIGNMENT.BOTH_POSITIONAL_AND_NAMED"
    );
    let e = fails(&s, "SELECT g(a => 1, 2)");
    assert_eq!(e.error_class().name(), "UNEXPECTED_POSITIONAL_ARGUMENT");
    let e = fails(&s, "SELECT g(1, 2, 'x', 4)");
    let (class, params) = class_and_params(&e);
    assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    assert_eq!(params["expectedNum"], "3");
    assert_eq!(params["actualNum"], "4");
    assert_eq!(params["functionName"], "`g`");
    // Named arguments are for functions that declare parameters.
    let e = fails(&s, "SELECT abs(x => 1)");
    assert_eq!(e.error_class().name(), "NAMED_PARAMETERS_NOT_SUPPORTED");
}

#[test]
fn recursion_is_rejected_when_the_function_is_created() {
    let s = session();
    let e = fails(
        &s,
        "CREATE TEMPORARY FUNCTION loopy(a INT) RETURNS INT RETURN loopy(a)",
    );
    let (class, params) = class_and_params(&e);
    assert_eq!(class, "CYCLIC_FUNCTION_REFERENCE");
    assert_eq!(params["path"], "loopy -> loopy");
    // Nothing was defined.
    let e = fails(&s, "SELECT loopy(1)");
    assert_eq!(e.error_class().name(), "UNRESOLVED_ROUTINE");
    // A replacement that closes a cycle fails and leaves the function as
    // it was.
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION a1(x INT) RETURNS INT RETURN x",
    );
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION a2(x INT) RETURNS INT RETURN a1(x) + 1",
    );
    let e = fails(
        &s,
        "CREATE OR REPLACE TEMPORARY FUNCTION a1(x INT) RETURNS INT RETURN a2(x)",
    );
    assert_eq!(class_and_params(&e).1["path"], "a1 -> a2 -> a1");
    assert_eq!(ok(&s, "SELECT a2(1)"), ["2"]);
}

#[test]
fn create_replace_and_drop_lifecycle() {
    let s = session();
    ok(&s, "CREATE TEMPORARY FUNCTION One() RETURNS INT RETURN 1");
    // Names are case-insensitive.
    assert_eq!(ok(&s, "SELECT one(), ONE()"), ["1\t1"]);
    let e = fails(&s, "CREATE TEMPORARY FUNCTION one() RETURNS INT RETURN 2");
    let (class, params) = class_and_params(&e);
    assert_eq!(class, "ROUTINE_ALREADY_EXISTS");
    assert_eq!(params["routineName"], "`one`");
    ok(
        &s,
        "CREATE OR REPLACE TEMPORARY FUNCTION one() RETURNS INT RETURN 2",
    );
    assert_eq!(ok(&s, "SELECT one()"), ["2"]);
    ok(&s, "DROP TEMPORARY FUNCTION ONE");
    assert_eq!(
        fails(&s, "SELECT one()").error_class().name(),
        "UNRESOLVED_ROUTINE"
    );
    let e = fails(&s, "DROP TEMPORARY FUNCTION one");
    let (class, params) = class_and_params(&e);
    assert_eq!(class, "ROUTINE_NOT_FOUND");
    assert_eq!(params["routineName"], "`one`");
    ok(&s, "DROP TEMPORARY FUNCTION IF EXISTS one");
    // A qualified name is never a temporary function.
    ok(&s, "DROP TEMPORARY FUNCTION IF EXISTS db.one");
    assert_eq!(
        fails(&s, "DROP TEMPORARY FUNCTION db.one")
            .error_class()
            .name(),
        "ROUTINE_NOT_FOUND"
    );
    // A function calling a dropped one fails when used.
    ok(&s, "CREATE TEMPORARY FUNCTION base1() RETURNS INT RETURN 1");
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION top1() RETURNS INT RETURN base1() + 1",
    );
    ok(&s, "DROP TEMPORARY FUNCTION base1");
    assert_eq!(
        fails(&s, "SELECT top1()").error_class().name(),
        "UNRESOLVED_ROUTINE"
    );
}

#[test]
fn functions_belong_to_their_session() {
    let a = session();
    let b = session();
    ok(&a, "CREATE TEMPORARY FUNCTION mine() RETURNS INT RETURN 1");
    assert_eq!(ok(&a, "SELECT mine()"), ["1"]);
    assert_eq!(
        fails(&b, "SELECT mine()").error_class().name(),
        "UNRESOLVED_ROUTINE"
    );
}

#[test]
fn a_function_cannot_take_the_name_of_a_built_in() {
    let s = session();
    let e = fails(
        &s,
        "CREATE TEMPORARY FUNCTION abs(x INT) RETURNS INT RETURN 100",
    );
    assert_eq!(e.error_class().name(), "ROUTINE_ALREADY_EXISTS");
    let e = fails(
        &s,
        "CREATE OR REPLACE TEMPORARY FUNCTION abs(x INT) RETURNS INT RETURN 100",
    );
    assert_eq!(
        e.error_class().name(),
        "USER_DEFINED_FUNCTIONS.CANNOT_REPLACE_NON_SQL_UDF_WITH_SQL_UDF"
    );
    assert_eq!(ok(&s, "SELECT abs(-5)"), ["5"]);
}

#[test]
fn the_return_type_is_inferred_when_omitted() {
    let s = session();
    ok(&s, "CREATE TEMPORARY FUNCTION inc(a INT) RETURN a + 1");
    let (schema, rows) = run(&s, "SELECT inc(1)").unwrap();
    assert_eq!(schema, "struct<inc(1):int>");
    assert_eq!(rows, ["2"]);
    ok(&s, "CREATE TEMPORARY FUNCTION greet() RETURN 'hi'");
    assert_eq!(
        run(&s, "SELECT greet()").unwrap().0,
        "struct<greet():string>"
    );
}

#[test]
fn a_body_that_is_a_query_is_a_scalar_subquery() {
    let s = session();
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION q() RETURNS INT RETURN SELECT 5",
    );
    assert_eq!(ok(&s, "SELECT q()"), ["5"]);
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION cnt() RETURNS BIGINT RETURN (SELECT count(*) FROM people)",
    );
    assert_eq!(ok(&s, "SELECT cnt()"), ["3"]);
    let e = fails(
        &s,
        "CREATE TEMPORARY FUNCTION two() RETURNS INT RETURN (SELECT 1, 2)",
    );
    assert_eq!(
        e.error_class().name(),
        "INVALID_SUBQUERY_EXPRESSION.SCALAR_SUBQUERY_RETURN_MORE_THAN_ONE_OUTPUT_COLUMN"
    );
}

#[test]
fn bodies_read_temporary_views_as_they_are_when_called() {
    let s = session();
    ok(&s, "CREATE TEMPORARY VIEW five AS SELECT 5 AS x");
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION v() RETURNS INT RETURN (SELECT max(x) FROM five)",
    );
    assert_eq!(ok(&s, "SELECT v()"), ["5"]);
    ok(&s, "CREATE OR REPLACE TEMPORARY VIEW five AS SELECT 6 AS x");
    assert_eq!(ok(&s, "SELECT v()"), ["6"]);
    ok(&s, "DROP VIEW five");
    let e = fails(&s, "SELECT v()");
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    // The error points into the statement that defined the function.
    let fragment = &e.query_contexts()[0].fragment;
    assert_eq!(fragment, "five");
    // A body reading a view that does not exist is refused at creation.
    let e = fails(
        &s,
        "CREATE TEMPORARY FUNCTION w() RETURNS INT RETURN (SELECT max(x) FROM five)",
    );
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn a_changed_function_is_seen_by_a_dataframe_analysed_before() {
    let s = session();
    ok(&s, "CREATE TEMPORARY FUNCTION n() RETURNS INT RETURN 1");
    let df = block_on(s.sql("SELECT n()")).unwrap();
    let first = block_on(df.collect()).unwrap();
    assert_eq!(first[0].column(0).as_primitive::<Int32Type>().value(0), 1);
    ok(
        &s,
        "CREATE OR REPLACE TEMPORARY FUNCTION n() RETURNS INT RETURN 2",
    );
    // The cached analysis is not reused: the function's version changed.
    let again = block_on(df.collect()).unwrap();
    assert_eq!(again[0].column(0).as_primitive::<Int32Type>().value(0), 2);
}

#[test]
fn declarations_are_checked_as_spark_checks_them() {
    let s = session();
    let case = |sql: &str, class: &str, want: &[(&str, &str)]| {
        let e = fails(&s, sql);
        let (got_class, got) = class_and_params(&e);
        assert_eq!(got_class, class, "{sql}");
        for (k, v) in want {
            assert_eq!(got.get(*k).map(String::as_str), Some(*v), "{sql}: {k}");
        }
    };
    // golden: foo1c1, foo1c2
    case(
        "CREATE TEMPORARY FUNCTION foo1c1(duplicate INT, DUPLICATE INT) RETURNS INT RETURN 1",
        "DUPLICATE_ROUTINE_PARAMETER_NAMES",
        &[("routineName", "foo1c1"), ("names", "`duplicate`")],
    );
    // golden: foo1d1 (the cast error shows the default)
    case(
        "CREATE OR REPLACE TEMPORARY FUNCTION foo1d1(a INT DEFAULT array(55, 17)) RETURNS INT RETURN a",
        "DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION",
        &[("srcType", "\"ARRAY<INT>\""), ("targetType", "\"INT\"")],
    );
    // golden: foo1d1 with a subquery
    case(
        "CREATE OR REPLACE TEMPORARY FUNCTION foo1d1(a INT DEFAULT (SELECT max(id) FROM people)) RETURNS INT RETURN a",
        "USER_DEFINED_FUNCTIONS.NOT_A_VALID_DEFAULT_EXPRESSION",
        &[("functionName", "foo1d1"), ("parameterName", "a")],
    );
    // golden: foo1d2
    case(
        "CREATE OR REPLACE TEMPORARY FUNCTION foo1d2(a INT DEFAULT 5, b INT, c INT DEFAULT 8) RETURNS STRING RETURN 'x'",
        "USER_DEFINED_FUNCTIONS.NOT_A_VALID_DEFAULT_PARAMETER_POSITION",
        &[
            ("functionName", "foo1d2"),
            ("parameterName", "a"),
            ("nextParameterName", "b"),
        ],
    );
    // golden: foo1e1, foo1e2
    case(
        "CREATE TEMPORARY FUNCTION foo1e1(x INT NOT NULL, y INT) RETURNS INT RETURN 1",
        "USER_DEFINED_FUNCTIONS.NOT_NULL_ON_FUNCTION_PARAMETERS",
        &[("input", "x INT NOT NULL, y INT")],
    );
    // golden: foo1f1
    case(
        "CREATE TEMPORARY FUNCTION foo1f1(x INT, y INT GENERATED ALWAYS AS (x + 10)) RETURNS INT RETURN y + 1",
        "INVALID_SQL_SYNTAX.CREATE_FUNC_WITH_GENERATED_COLUMNS_AS_PARAMETERS",
        &[],
    );
    // golden: foo31, foo9i
    case(
        "CREATE TEMPORARY FUNCTION foo9i(a VARCHAR(10)) RETURNS VARCHAR(12) RETURN a",
        "UNSUPPORTED_CHAR_OR_VARCHAR_AS_STRING",
        &[],
    );
    // golden: foo3_2b5: an aggregate in the body; an unknown name.
    case(
        "CREATE TEMPORARY FUNCTION foo3_2b5(x INT) RETURNS BOOLEAN RETURN SUM(1) + IF(x IN (SELECT 1), 1, 0)",
        "USER_DEFINED_FUNCTIONS.CANNOT_CONTAIN_COMPLEX_FUNCTIONS",
        &[("queryText", "SUM(1) + IF(x IN (SELECT 1), 1, 0)")],
    );
    case(
        "CREATE TEMPORARY FUNCTION foo3_2b5(x INT) RETURNS BOOLEAN RETURN y IN (SELECT 1)",
        "UNRESOLVED_COLUMN.WITH_SUGGESTION",
        &[("objectName", "`y`"), ("proposal", "`x`")],
    );
    // golden: foo42
    case(
        "CREATE TEMPORARY FUNCTION foo42() RETURNS TABLE(a INT) RETURN 1",
        "USER_DEFINED_FUNCTIONS.SQL_TABLE_UDF_BODY_MUST_BE_A_QUERY",
        &[("name", "foo42")],
    );
    // A body that cannot be cast to the return type.
    case(
        "CREATE TEMPORARY FUNCTION bad() RETURNS INT RETURN DATE '2020-01-01'",
        "DATATYPE_MISMATCH.CAST_WITH_FUNC_SUGGESTION",
        &[],
    );
}

#[test]
fn what_is_not_supported_is_named() {
    let s = session();
    for (sql, feature) in [
        (
            "CREATE FUNCTION other_catalog.default.p() RETURNS INT RETURN 1",
            "functions of catalogs other than spark_catalog",
        ),
        (
            "CREATE TEMPORARY FUNCTION p() RETURNS INT LANGUAGE python RETURN 1",
            "LANGUAGE python functions",
        ),
        (
            "EXPLAIN CREATE TEMPORARY FUNCTION p() RETURN 1",
            "EXPLAIN of this statement in sessions",
        ),
    ] {
        let (class, params) = class_and_params(&fails(&s, sql));
        assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE", "{sql}");
        assert_eq!(params["feature"], feature, "{sql}");
    }
    assert!(ok(&s, "SELECT 1").len() == 1);
}

#[test]
fn parse_errors_come_before_anything_runs() {
    let s = session();
    let e = fails(
        &s,
        "CREATE TEMPORARY FUNCTION f() RETURNS TABLE() RETURN SELECT 1",
    );
    assert_eq!(e.error_class().name(), "PARSE_SYNTAX_ERROR");
    let e = fails(&s, "CREATE TEMPORARY FUNCTION IF NOT EXISTS f() RETURN 1");
    assert_eq!(
        e.error_class().name(),
        "INVALID_SQL_SYNTAX.CREATE_TEMP_FUNC_WITH_IF_NOT_EXISTS"
    );
}

#[test]
fn an_argument_is_read_as_often_as_the_body_reads_it() {
    let s = session();
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION dbl(x DOUBLE) RETURNS DOUBLE RETURN x + x",
    );
    assert_eq!(
        ok(&s, "SELECT dbl(2.0), dbl(id) FROM people WHERE id = 1"),
        ["4\t2"]
    );
}

#[test]
fn a_table_functions_declaration_is_checked_before_it_is_refused() {
    let s = session();
    let case = |sql: &str, class: &str, key: &str, want: &str| {
        let (got, params) = class_and_params(&fails(&s, sql));
        assert_eq!(got, class, "{sql}");
        assert_eq!(params[key], want, "{sql}");
    };
    // golden: foo2b1, foo2b2
    case(
        "CREATE TEMPORARY FUNCTION foo2b1() RETURNS TABLE(DuPLiCatE INT, duplicate INT) RETURN SELECT 1, 2",
        "DUPLICATE_ROUTINE_RETURNS_COLUMNS",
        "columns",
        "`duplicate`",
    );
    // golden: foo1e2, foo1e3
    case(
        "CREATE TEMPORARY FUNCTION foo1e2(x INT, y INT NOT NULL) RETURNS TABLE (x INT) RETURN SELECT 1",
        "USER_DEFINED_FUNCTIONS.NOT_NULL_ON_FUNCTION_PARAMETERS",
        "input",
        "x INT, y INT NOT NULL",
    );
    case(
        "CREATE TEMPORARY FUNCTION foo1e3(x INT, y INT) RETURNS TABLE (x INT NOT NULL) RETURN SELECT 1",
        "USER_DEFINED_FUNCTIONS.NOT_NULL_ON_FUNCTION_PARAMETERS",
        "input",
        "x INT NOT NULL",
    );
}

#[test]
fn parameter_names_compare_as_the_session_does() {
    let s = session();
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    // Case sensitive: `a` and `A` are two parameters.
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION cs(a INT, A INT) RETURNS INT RETURN a + A",
    );
    assert_eq!(ok(&s, "SELECT cs(1, 2)"), ["3"]);
    s.conf().set("spark.sql.caseSensitive", "false").unwrap();
    let e = fails(
        &s,
        "CREATE TEMPORARY FUNCTION ci(a INT, A INT) RETURNS INT RETURN 1",
    );
    assert_eq!(e.error_class().name(), "DUPLICATE_ROUTINE_PARAMETER_NAMES");
}

/// Both uses of a parameter, written bare and qualified by the function's
/// name, are the argument.
#[test]
fn a_qualified_parameter_is_the_argument() {
    let s = session();
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION q(x INT) RETURNS INT RETURN q.x + x",
    );
    assert_eq!(ok(&s, "SELECT q(21)"), ["42"]);
}

/// A lambda variable of the calling query is not visible in the body, even
/// when it is named like the function's parameter.
#[test]
fn the_body_does_not_see_the_callers_lambda_variables() {
    let s = session();
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION times10(a INT) RETURNS INT RETURN a * 10",
    );
    assert_eq!(
        ok(
            &s,
            "SELECT aggregate(transform(array(1, 2, 3), a -> times10(a + 1)), 0, (acc, v) -> acc + v)"
        ),
        ["90"]
    );
}

#[test]
fn a_stored_function_lives_in_the_catalog_of_the_session() {
    let s = session();
    ok(
        &s,
        "CREATE FUNCTION stored_add(a INT, b INT) RETURNS INT RETURN a + b",
    );
    // Bare, namespace-qualified and fully qualified calls reach it, in any
    // letter case.
    assert_eq!(ok(&s, "SELECT stored_add(1, 2)"), ["3"]);
    assert_eq!(ok(&s, "SELECT default.STORED_ADD(1, 2)"), ["3"]);
    assert_eq!(
        ok(&s, "SELECT spark_catalog.default.stored_add(1, 2)"),
        ["3"]
    );
    // Spark names the column after the catalog's identifier.
    let (schema, _) = run(&s, "SELECT stored_add(1, 2)").unwrap();
    assert_eq!(schema, "struct<spark_catalog.default.stored_add(1, 2):int>");
    // Another session has not got it.
    let other = Session::new().unwrap();
    let (class, _) = class_and_params(&fails(&other, "SELECT stored_add(1, 2)"));
    assert_eq!(class, "UNRESOLVED_ROUTINE");
    // It follows `USE`: bare names resolve in the current namespace only.
    ok(&s, "CREATE SCHEMA other_ns");
    ok(&s, "USE other_ns");
    let (class, _) = class_and_params(&fails(&s, "SELECT stored_add(1, 2)"));
    assert_eq!(class, "UNRESOLVED_ROUTINE");
    assert_eq!(ok(&s, "SELECT default.stored_add(1, 2)"), ["3"]);
    ok(
        &s,
        "CREATE FUNCTION stored_add(a INT) RETURNS INT RETURN a * 10",
    );
    assert_eq!(ok(&s, "SELECT stored_add(4)"), ["40"]);
    assert_eq!(ok(&s, "SELECT default.stored_add(4, 5)"), ["9"]);
}

#[test]
fn a_stored_function_is_created_replaced_and_dropped_as_spark_does() {
    let s = session();
    ok(&s, "CREATE FUNCTION f1() RETURNS INT RETURN 1");
    let (class, params) = class_and_params(&fails(&s, "CREATE FUNCTION f1() RETURNS INT RETURN 2"));
    assert_eq!(class, "ROUTINE_ALREADY_EXISTS");
    assert_eq!(params["routineName"], "`spark_catalog`.`default`.`f1`");
    ok(&s, "CREATE OR REPLACE FUNCTION f1() RETURNS INT RETURN 2");
    assert_eq!(ok(&s, "SELECT f1()"), ["2"]);
    // A stored and a temporary function of one name coexist; the temporary
    // one is found first, and the stored one by its qualified name.
    ok(&s, "CREATE TEMPORARY FUNCTION f1() RETURNS INT RETURN 3");
    assert_eq!(ok(&s, "SELECT f1()"), ["3"]);
    assert_eq!(ok(&s, "SELECT default.f1()"), ["2"]);
    ok(&s, "DROP TEMPORARY FUNCTION f1");
    assert_eq!(ok(&s, "SELECT f1()"), ["2"]);
    ok(&s, "DROP FUNCTION f1");
    let (class, _) = class_and_params(&fails(&s, "SELECT f1()"));
    assert_eq!(class, "UNRESOLVED_ROUTINE");
    let (class, _) = class_and_params(&fails(&s, "DROP FUNCTION f1"));
    assert_eq!(class, "ROUTINE_NOT_FOUND");
    ok(&s, "DROP FUNCTION IF EXISTS f1");
    // A namespace the catalog lacks.
    let (class, _) = class_and_params(&fails(&s, "CREATE FUNCTION nope.f() RETURNS INT RETURN 1"));
    assert_eq!(class, "SCHEMA_NOT_FOUND");
    let (class, _) = class_and_params(&fails(&s, "DROP FUNCTION nope.f"));
    assert_eq!(class, "SCHEMA_NOT_FOUND");
}

#[test]
fn a_namespace_with_functions_is_not_empty() {
    let s = session();
    ok(&s, "CREATE SCHEMA ns1");
    ok(&s, "CREATE FUNCTION ns1.f() RETURNS INT RETURN 1");
    assert_eq!(ok(&s, "SELECT ns1.f()"), ["1"]);
    let (class, _) = class_and_params(&fails(&s, "DROP SCHEMA ns1 RESTRICT"));
    assert_eq!(class, "SCHEMA_NOT_EMPTY");
    ok(&s, "DROP SCHEMA ns1 CASCADE");
    let (class, _) = class_and_params(&fails(&s, "SELECT ns1.f()"));
    assert_eq!(class, "UNRESOLVED_ROUTINE");
}

#[test]
fn stored_functions_call_each_other_but_not_themselves() {
    let s = session();
    ok(
        &s,
        "CREATE FUNCTION inner_f(x INT) RETURNS INT RETURN x + 1",
    );
    ok(
        &s,
        "CREATE FUNCTION outer_f(x INT) RETURNS INT RETURN inner_f(x) * 2",
    );
    assert_eq!(ok(&s, "SELECT outer_f(4)"), ["10"]);
    // The caller reads the callee as it is when called.
    ok(
        &s,
        "CREATE OR REPLACE FUNCTION inner_f(x INT) RETURNS INT RETURN x + 100",
    );
    assert_eq!(ok(&s, "SELECT outer_f(4)"), ["208"]);
    let (class, _) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION rec(x INT) RETURNS INT RETURN rec(x)",
    ));
    assert_eq!(class, "CYCLIC_FUNCTION_REFERENCE");
    // Arity errors name the stored function by the catalog's identifier.
    let (class, params) = class_and_params(&fails(&s, "SELECT inner_f()"));
    assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    assert_eq!(
        params["functionName"],
        "`spark_catalog`.`default`.`inner_f`"
    );
}

#[test]
fn a_stored_function_cannot_read_a_temporary_view() {
    let s = session();
    ok(&s, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS x");
    let (class, params) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION reads_tv() RETURNS INT RETURN (SELECT max(x) FROM tv)",
    ));
    assert_eq!(class, "INVALID_TEMP_OBJ_REFERENCE");
    // Spark's parameters (sql-udf.sql.out, foo3_5d).
    assert_eq!(params["obj"], "FUNCTION");
    assert_eq!(params["objName"], "`default`.`reads_tv`");
    assert_eq!(params["tempObj"], "VIEW");
    assert_eq!(params["tempObjName"], "`tv`");
    // Nor a temporary function.
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION tmp_f(x INT) RETURNS INT RETURN x",
    );
    let (class, params) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION calls_tmp(x INT) RETURNS INT RETURN tmp_f(x)",
    ));
    assert_eq!(class, "INVALID_TEMP_OBJ_REFERENCE");
    assert_eq!(params["tempObj"], "FUNCTION");
    assert_eq!(params["tempObjName"], "`tmp_f`");
    // A temporary function may read it.
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION reads_tv() RETURNS INT RETURN (SELECT max(x) FROM tv)",
    );
    assert_eq!(ok(&s, "SELECT reads_tv()"), ["1"]);
}

#[test]
fn a_temporary_function_in_a_subquery_filter_is_found() {
    let s = session();
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION tmp_g(x INT) RETURNS INT RETURN x",
    );
    let (class, params) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION g() RETURNS INT RETURN (SELECT max(c) FROM (SELECT 1 AS c) WHERE c > tmp_g(0))",
    ));
    assert_eq!(class, "INVALID_TEMP_OBJ_REFERENCE");
    assert_eq!(params["tempObj"], "FUNCTION");
    assert_eq!(params["tempObjName"], "`tmp_g`");
}

#[test]
fn a_table_function_returns_its_declared_columns() {
    let s = session();
    ok(
        &s,
        "CREATE FUNCTION tf(a INT) RETURNS TABLE (c1 BIGINT, c2 STRING) RETURN SELECT a + 1, 'x'",
    );
    // Columns are renamed and cast to the declared ones.
    let (schema, rows) = run(&s, "SELECT * FROM tf(1)").unwrap();
    assert_eq!(schema, "struct<c1:bigint,c2:string>");
    assert_eq!(rows, ["2\tx"]);
    assert_eq!(ok(&s, "SELECT * FROM default.tf(5)"), ["6\tx"]);
    assert_eq!(
        ok(&s, "SELECT * FROM spark_catalog.default.tf(7)"),
        ["8\tx"]
    );
    assert_eq!(
        ok(&s, "SELECT t.c1 FROM tf(2) AS t WHERE t.c2 = 'x'"),
        ["3"]
    );
    // A table function reads the table of the session by its own name and
    // joins like any relation.
    assert_eq!(
        ok(
            &s,
            "SELECT p.id, t.c1 FROM people p JOIN tf(0) t ON p.id = t.c1"
        ),
        ["1\t1"]
    );
    // Defaults and the arity check are the scalar ones.
    ok(
        &s,
        "CREATE FUNCTION tf_d(a INT, b INT DEFAULT 10) RETURNS TABLE (s INT) RETURN SELECT a + b",
    );
    assert_eq!(ok(&s, "SELECT * FROM tf_d(1)"), ["11"]);
    assert_eq!(ok(&s, "SELECT * FROM tf_d(1, 2)"), ["3"]);
    let (class, _) = class_and_params(&fails(&s, "SELECT * FROM tf_d()"));
    assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    // Without a column list the columns are the query's own.
    ok(
        &s,
        "CREATE FUNCTION tf_i() RETURNS TABLE RETURN SELECT 1 AS id, 'hello' AS name",
    );
    let (schema, rows) = run(&s, "SELECT * FROM tf_i()").unwrap();
    assert_eq!(schema, "struct<id:int,name:string>");
    assert_eq!(rows, ["1\thello"]);
    // A temporary table function.
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION tt() RETURNS TABLE (c BIGINT) RETURN SELECT 7",
    );
    assert_eq!(ok(&s, "SELECT c + 1 FROM tt()"), ["8"]);
    ok(&s, "DROP TEMPORARY FUNCTION tt");
    let (class, _) = class_and_params(&fails(&s, "SELECT * FROM tt()"));
    assert_eq!(class, "UNRESOLVABLE_TABLE_VALUED_FUNCTION");
}

#[test]
fn a_table_function_is_checked_when_it_is_created() {
    let s = session();
    // Spark's parameters (sql-udf.sql.out, foo32 and foo33).
    let (class, params) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION foo32() RETURNS TABLE(a INT) RETURN SELECT 1, 2",
    ));
    assert_eq!(class, "USER_DEFINED_FUNCTIONS.RETURN_COLUMN_COUNT_MISMATCH");
    assert_eq!(params["name"], "spark_catalog.default.foo32");
    assert_eq!(params["outputSize"], "2");
    assert_eq!(params["returnParamSize"], "1");
    let (_, params) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION foo33() RETURNS TABLE(a INT, b INT) RETURN SELECT 1",
    ));
    assert_eq!(params["outputSize"], "1");
    assert_eq!(params["returnParamSize"], "2");
    // A body that is not a query.
    let (class, _) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION foo42() RETURNS TABLE(a INT) RETURN 1",
    ));
    assert_eq!(
        class,
        "USER_DEFINED_FUNCTIONS.SQL_TABLE_UDF_BODY_MUST_BE_A_QUERY"
    );
    // Recursion, directly and through another function.
    let (class, _) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION rec(x INT) RETURNS TABLE (a INT) RETURN SELECT * FROM rec(x)",
    ));
    assert_eq!(class, "CYCLIC_FUNCTION_REFERENCE");
    // A table function is not a scalar one, nor the other way round.
    ok(
        &s,
        "CREATE FUNCTION tf2() RETURNS TABLE (a INT) RETURN SELECT 1",
    );
    ok(&s, "CREATE FUNCTION sf2() RETURNS INT RETURN 1");
    // Spark's classes and parameters (sql-udf.sql.out, foo3_14a and foo3_14b).
    let (class, params) = class_and_params(&fails(&s, "SELECT tf2()"));
    assert_eq!(class, "NOT_A_SCALAR_FUNCTION");
    assert_eq!(params["functionName"], "`spark_catalog`.`default`.`tf2`");
    let (class, params) = class_and_params(&fails(&s, "SELECT * FROM sf2()"));
    assert_eq!(class, "NOT_A_TABLE_FUNCTION");
    assert_eq!(params["functionName"], "`spark_catalog`.`default`.`sf2`");
    // A table function over another.
    ok(
        &s,
        "CREATE FUNCTION tf3(x INT) RETURNS TABLE (a INT) RETURN SELECT a + x FROM tf2()",
    );
    assert_eq!(ok(&s, "SELECT * FROM tf3(10)"), ["11"]);
}

#[test]
fn a_table_function_takes_deterministic_arguments_and_no_temporary_ones() {
    let s = session();
    ok(
        &s,
        "CREATE FUNCTION tf_r(x DOUBLE) RETURNS TABLE (a DOUBLE, b DOUBLE) RETURN SELECT x, x",
    );
    let e = fails(&s, "SELECT * FROM tf_r(rand())");
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    // A stored function cannot call a temporary table function.
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION tmp_t() RETURNS TABLE (a INT) RETURN SELECT 1",
    );
    let (class, params) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION calls_tmp_t() RETURNS TABLE (a INT) RETURN SELECT * FROM tmp_t()",
    ));
    assert_eq!(class, "INVALID_TEMP_OBJ_REFERENCE");
    assert_eq!(params["tempObj"], "FUNCTION");
    assert_eq!(params["tempObjName"], "`tmp_t`");
    let (class, _) = class_and_params(&fails(
        &s,
        "CREATE FUNCTION calls_tmp_q() RETURNS TABLE (a INT) RETURN SELECT * FROM session.tmp_t()",
    ));
    assert_eq!(class, "INVALID_TEMP_OBJ_REFERENCE");
    // A quoted name with a dot is one name.
    ok(
        &s,
        "CREATE TEMPORARY FUNCTION `d.t`() RETURNS TABLE (a INT) RETURN SELECT 5",
    );
    assert_eq!(ok(&s, "SELECT * FROM `d.t`()"), ["5"]);
}

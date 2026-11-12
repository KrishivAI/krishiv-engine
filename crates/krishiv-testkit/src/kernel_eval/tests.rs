// SPDX-License-Identifier: Apache-2.0
//! Tests of the kernel evaluator.

use std::collections::BTreeMap;

use serde_json::json;

use super::*;
use crate::fn_matrix::{Field, SparkError};

fn rows(ty: &str, cell: Value) -> Expected {
    Expected::Rows {
        schema: vec![Field {
            name: "c".to_owned(),
            data_type: ty.to_owned(),
            nullable: true,
        }],
        rows: vec![],
        render: vec![],
        ordered: false,
        truncated: false,
        typed_rows: Some(vec![vec![cell]]),
        typed_unavailable: None,
    }
}

fn error(class: &str, message: &str, phase: &str) -> Expected {
    Expected::Error {
        schema: None,
        error: SparkError {
            error_class: class.to_owned(),
            sqlstate: None,
            message: message.to_owned(),
            message_parameters: BTreeMap::new(),
            phase: phase.to_owned(),
            stable: true,
        },
    }
}

fn cell(function: &str, sql: &str, mode: Mode, expected: Expected) -> CellOutcome {
    run_cell(&MatrixCase {
        id: "x#1/ansi_on".to_owned(),
        base_id: "x#1".to_owned(),
        function: function.to_owned(),
        family: "predicate".to_owned(),
        kind: "seed0".to_owned(),
        mode,
        sql: sql.to_owned(),
        conf: BTreeMap::new(),
        expected,
    })
}

#[test]
fn shapes_cover_every_operand_kind() {
    let on = Mode::AnsiOn;
    let pass = |f, sql, e| assert_eq!(cell(f, sql, on, e), CellOutcome::Pass, "{sql}");
    pass("<", "SELECT 'a' < 'b'", rows("boolean", json!(true)));
    pass("=", "SELECT X'01' = X'02'", rows("boolean", json!(false)));
    pass("<=", "SELECT true <= false", rows("boolean", json!(false)));
    pass(
        ">",
        "SELECT TIMESTAMP'2024-01-02 00:00:00.5' > TIMESTAMP'2024-01-01 00:00:00'",
        rows("boolean", json!(true)),
    );
    pass(
        "=",
        "SELECT TIMESTAMP_NTZ'2024-01-01 00:00:00' = TIMESTAMP_NTZ'2024-01-01 00:00:00'",
        rows("boolean", json!(true)),
    );
    pass("<=>", "SELECT 2 <=> NULL", rows("boolean", json!(false)));
    pass(
        "string",
        "SELECT string(TIMESTAMP_NTZ'2024-01-01 00:00:00')",
        rows("string", json!("2024-01-01 00:00:00")),
    );
    pass(
        "timestamp",
        "SELECT timestamp(DATE'2024-01-01')",
        rows(
            "timestamp",
            json!({"ts_us": 1_704_067_200_000_000_i64, "tz": "UTC"}),
        ),
    );
    pass("+", "SELECT 1Y + 2S", rows("smallint", json!(3)));
    pass(
        "+",
        "SELECT 1Y + 1.5BD",
        rows("decimal(5,1)", json!({"unscaled": "25", "scale": 1})),
    );
    pass(
        "-",
        "SELECT -(1.5F)",
        rows("float", json!({"f32": "bfc00000"})),
    );
    pass(
        "*",
        "SELECT 99999999999999999999 * 1",
        rows(
            "decimal(22,0)",
            json!({"unscaled": "99999999999999999999", "scale": 0}),
        ),
    );
    pass(
        "cast",
        "SELECT CAST('1' AS BIGINT) + CAST(1 AS DECIMAL)",
        rows("decimal(21,0)", json!({"unscaled": "2", "scale": 0})),
    );
    pass(
        "cast",
        "SELECT CAST(1 AS NUMERIC(5, 2))",
        rows("decimal(5,2)", json!({"unscaled": "100", "scale": 2})),
    );
    pass(
        "cast",
        "SELECT CAST(1 AS REAL) = 1L",
        rows("boolean", json!(true)),
    );
    pass("=", "SELECT 1.5 = 1", rows("boolean", json!(false)));
    pass(
        "abs",
        "SELECT abs(1L) + 1e0",
        rows("double", json!({"f64": "4000000000000000"})),
    );
}

#[test]
fn failures_and_reasons() {
    let (on, off) = (Mode::AnsiOn, Mode::AnsiOff);
    let fail = |o: CellOutcome| assert!(matches!(o, CellOutcome::Fail(_)), "{o:?}");
    let not_run = |o: CellOutcome| assert!(matches!(o, CellOutcome::NotRun(_)), "{o:?}");
    fail(cell("+", "SELECT 1 + 1", on, rows("bigint", json!(2))));
    fail(cell("+", "SELECT 1 + 1", on, rows("int", json!(3))));
    fail(cell(
        "+",
        "SELECT 1 + 1",
        on,
        error("DIVIDE_BY_ZERO", "x", "execution"),
    ));
    fail(cell(
        "div",
        "SELECT 1 div 0",
        on,
        rows("bigint", Value::Null),
    ));
    fail(cell(
        "div",
        "SELECT 1 div 0",
        on,
        error("DIVIDE_BY_ZERO", "other text", "execution"),
    ));
    not_run(cell(
        "factorial",
        "SELECT factorial(1)",
        on,
        rows("double", Value::Null),
    ));
    not_run(cell("+", "SELECT 1 + true", on, error("X", "", "analysis")));
    not_run(cell(
        "+",
        "SELECT 1 + rand()",
        on,
        Expected::Nondeterministic { schema: vec![] },
    ));
    not_run(cell("+", "SELECT 1 +", on, rows("int", json!(1))));
    not_run(cell("+", "SELECT a + 1 FROM t", on, rows("int", json!(1))));
    not_run(cell(
        "+",
        "SELECT 1 + factorial(1)",
        on,
        rows("double", json!(1)),
    ));
    not_run(cell("+", "SELECT 1 + rand()", on, rows("double", json!(1))));
    not_run(cell("+", "SELECT '1' div 2", off, rows("bigint", json!(0))));
    not_run(cell("+", "SELECT 1.5D div 2", on, rows("bigint", json!(0))));
    not_run(cell("+", "SELECT ~1", on, rows("int", json!(-2))));
    not_run(cell("+", "SELECT 1 | 2", on, rows("int", json!(3))));
    not_run(cell("+", "SELECT X'01' + 1", on, rows("int", json!(3))));
    not_run(cell("-", "SELECT -X'01'", on, rows("int", json!(3))));
    not_run(cell(
        "cast",
        "SELECT CAST(1 AS ARRAY<INT>)",
        on,
        rows("int", json!(3)),
    ));
    not_run(cell(
        "cast",
        "SELECT CAST(1 AS INTERVAL DAY)",
        on,
        rows("int", json!(3)),
    ));
    not_run(cell(
        "cast",
        "SELECT CAST(1 AS DECIMAL(x))",
        on,
        rows("int", json!(3)),
    ));
    not_run(cell(
        "cast",
        "SELECT CAST(X'01' AS INT)",
        on,
        rows("int", json!(3)),
    ));
    not_run(cell(
        "try_add",
        "SELECT try_add(1)",
        on,
        rows("int", json!(3)),
    ));
    not_run(cell(
        "=",
        "SELECT DATE'2024-01-01' = X'01'",
        on,
        rows("boolean", json!(3)),
    ));
    not_run(cell(
        "=",
        "SELECT TIME'01:00:00' = 1",
        on,
        rows("boolean", json!(3)),
    ));
    not_run(cell(
        "=",
        "SELECT TIMESTAMP'2024-01-01' = 1",
        on,
        rows("boolean", json!(3)),
    ));
    not_run(cell(
        "=",
        "SELECT 1e400BD = 1",
        on,
        rows("boolean", json!(3)),
    ));
    not_run(cell(
        "=",
        "SELECT 1S = 99999S",
        on,
        rows("boolean", json!(3)),
    ));
    for sql in [
        "SELECT 1e2000000000BD = 1",
        "SELECT 1e-2147483648BD = 1",
        "SELECT TIMESTAMP'2024-01-01 99999999999999:00:00' = 1",
        "SELECT TIMESTAMP'2024-01-01 25:00:00' = 1",
        "SELECT TIMESTAMP'1000000-01-01 00:00:00' = 1",
    ] {
        not_run(cell("=", sql, on, rows("boolean", json!(3))));
    }
    let pass = |f, sql, m, e| assert_eq!(cell(f, sql, m, e), CellOutcome::Pass, "{sql}");
    pass("+", "SELECT NULL + NULL", on, rows("double", Value::Null));
    pass(
        "cast",
        "SELECT CAST('a' AS BINARY)",
        on,
        rows("binary", json!({"hex": "61"})),
    );
    pass(
        "div",
        "SELECT NULL div NULL",
        on,
        rows("bigint", Value::Null),
    );
    pass(
        "=",
        "SELECT '1.21' = 1.2",
        off,
        rows("boolean", json!(false)),
    );
    pass(
        "=",
        "SELECT X'0A' = X'0A'",
        on,
        rows("boolean", json!(true)),
    );
    pass(
        "+",
        "SELECT 1Y + 1.5BD",
        on,
        rows("decimal(5,1)", json!({"unscaled": "25", "scale": 1})),
    );
    // An unstable Spark error only needs to fail on the same row.
    let mut unstable = error("UNCLASSIFIED.ArithmeticException", "x", "execution");
    if let Expected::Error { error, .. } = &mut unstable {
        error.stable = false;
    }
    pass("div", "SELECT 1 div 0", on, unstable);
    let mut tz = MatrixCase {
        id: "x#2/ansi_on".to_owned(),
        base_id: "x#2".to_owned(),
        function: "+".to_owned(),
        family: "math".to_owned(),
        kind: "seed0".to_owned(),
        mode: on,
        sql: "SELECT 1 + 1".to_owned(),
        conf: BTreeMap::new(),
        expected: rows("int", json!(2)),
    };
    tz.conf.insert(
        "spark.sql.session.timeZone".to_owned(),
        "Mars/Olympus".to_owned(),
    );
    not_run(run_cell(&tz));
}

#[test]
fn string_functions_run_in_every_shape() {
    let on = Mode::AnsiOn;
    let hex = |h: &str| json!({ "hex": h });
    for (f, sql, ty, want) in [
        ("upper", "SELECT upper('abc')", "string", json!("ABC")),
        ("length", "SELECT length(X'0102')", "int", json!(2)),
        (
            "trim",
            "SELECT trim(LEADING 'x' FROM 'xxa')",
            "string",
            json!("a"),
        ),
        ("||", "SELECT 'a' || 1 || 'b'", "string", json!("a1b")),
        ("concat", "SELECT concat()", "string", json!("")),
        (
            "concat",
            "SELECT concat(X'01', X'02')",
            "binary",
            hex("0102"),
        ),
        (
            "lpad",
            "SELECT lpad(X'01', 3, X'02')",
            "binary",
            hex("020201"),
        ),
        (
            "substr",
            "SELECT substr(X'AABBCC', 2)",
            "binary",
            hex("BBCC"),
        ),
        (
            "substring",
            "SELECT substring('Spark SQL' FROM 5 FOR 1)",
            "string",
            json!("k"),
        ),
        (
            "replace",
            "SELECT replace('abc', 'b')",
            "string",
            json!("ac"),
        ),
        (
            "startswith",
            "SELECT startswith(X'AB01', X'AB')",
            "boolean",
            json!(true),
        ),
        (
            "like",
            "SELECT 'abc' NOT LIKE 'a%'",
            "boolean",
            json!(false),
        ),
        (
            "like",
            "SELECT like('a%c', 'a$%c', '$')",
            "boolean",
            json!(true),
        ),
        (
            "split",
            "SELECT split('a,b', ',', 1)",
            "array<string>",
            json!(["a,b"]),
        ),
        (
            "split",
            "SELECT split('a.b', '\\\\.')",
            "array<string>",
            json!(["a", "b"]),
        ),
        (
            "split",
            "SELECT split('a', '[ab]')",
            "array<string>",
            json!(["", ""]),
        ),
        (
            "split",
            "SELECT split('a', NULL)",
            "array<string>",
            Value::Null,
        ),
    ] {
        let outcome = cell(f, sql, on, rows(ty, want));
        assert_eq!(outcome, CellOutcome::Pass, "{sql}");
    }
    let escape = error("INVALID_FORMAT.ESC_AT_THE_END", "other text", "execution");
    let fails = [
        ("upper", "SELECT upper('a')", rows("string", json!("a"))),
        ("like", "SELECT 'a' LIKE 'a'", rows("boolean", json!(false))),
        (
            "split",
            "SELECT split('a,b', ',')",
            rows("array<string>", json!(["a"])),
        ),
        ("like", "SELECT 'a' LIKE 'a\\\\'", escape),
    ];
    for (f, sql, want) in fails {
        let outcome = cell(f, sql, on, want);
        assert!(
            matches!(outcome, CellOutcome::Fail(_)),
            "{sql}: {outcome:?}"
        );
    }
    for (f, sql) in [
        ("like", "SELECT 'a' ILIKE 'A'"),
        ("like", "SELECT 'a' LIKE 'a' ESCAPE 'ab'"),
        ("upper", "SELECT upper()"),
        ("upper", "SELECT upper(INTERVAL '1' DAY)"),
        ("upper", "SELECT upper(array(1))"),
        ("upper", "SELECT upper(split('a', ','))"),
        ("substr", "SELECT substr('a', split('a', ','))"),
        ("split", "SELECT split('a', CAST(1 / 0 AS STRING))"),
        ("replace", "SELECT replace('a')"),
        ("like", "SELECT like('a', 'a', NULL)"),
        ("split", "SELECT split('a')"),
        ("substr", "SELECT substr('a', ARRAY(1))"),
    ] {
        let outcome = cell(f, sql, on, rows("string", json!("a")));
        assert!(
            matches!(outcome, CellOutcome::NotRun(_)),
            "{sql}: {outcome:?}"
        );
    }
}

#[test]
fn math_functions_run_in_every_shape() {
    let on = Mode::AnsiOn;
    let off = Mode::AnsiOff;
    let dbl = |v: f64| json!({ "f64": format!("{:016x}", v.to_bits()), "repr": v.to_string() });
    let dec = |scale: i64, unscaled: &str| json!({ "scale": scale, "unscaled": unscaled });
    for (f, sql, ty, want) in [
        ("sqrt", "SELECT sqrt(4)", "double", dbl(2.0)),
        ("sqrt", "SELECT sqrt('9')", "double", dbl(3.0)),
        ("pow", "SELECT pow(2, 10)", "double", dbl(1024.0)),
        ("log", "SELECT log(2, 8)", "double", dbl(3.0)),
        ("ln", "SELECT ln(0)", "double", Value::Null),
        ("pi", "SELECT pi()", "double", dbl(std::f64::consts::PI)),
        ("round", "SELECT round(2.5, 0)", "decimal(2,0)", dec(0, "3")),
        (
            "bround",
            "SELECT bround(2.5, 0)",
            "decimal(2,0)",
            dec(0, "2"),
        ),
        ("round", "SELECT round(15, -1)", "int", json!(20)),
        (
            "round",
            "SELECT round(CAST(2.675 AS DOUBLE), 2)",
            "double",
            dbl(2.68),
        ),
        (
            "round",
            "SELECT round(CAST(2.675 AS FLOAT), 2)",
            "float",
            json!({ "f32": format!("{:08x}", 2.67_f32.to_bits()) }),
        ),
        (
            "ceil",
            "SELECT ceil(3.1411, 3)",
            "decimal(5,3)",
            dec(3, "3142"),
        ),
        (
            "floor",
            "SELECT floor(CAST(12345 AS INT), -2)",
            "decimal(11,0)",
            dec(0, "12300"),
        ),
        ("floor", "SELECT floor(-1.5D)", "bigint", json!(-2)),
        ("ceil", "SELECT ceil(7)", "bigint", json!(7)),
        (
            "ceil",
            "SELECT ceil(CAST(1.5 AS DECIMAL(10,2)))",
            "decimal(9,0)",
            dec(0, "2"),
        ),
        (
            "ceil",
            "SELECT ceil(1.1D, 1)",
            "decimal(17,1)",
            dec(1, "11"),
        ),
        (
            "floor",
            "SELECT floor('1', 3)",
            "decimal(24,3)",
            dec(3, "1000"),
        ),
        ("pmod", "SELECT pmod(-7, 3)", "int", json!(2)),
        ("pmod", "SELECT pmod(-7.5, 2)", "decimal(2,1)", dec(1, "5")),
        ("pmod", "SELECT pmod(-7.5D, 2)", "double", dbl(0.5)),
        ("pmod", "SELECT pmod('7', 4)", "bigint", json!(3)),
        (
            "greatest",
            "SELECT greatest(1, 2.5)",
            "decimal(11,1)",
            dec(1, "25"),
        ),
        ("greatest", "SELECT greatest(1, NULL, 3)", "int", json!(3)),
        (
            "greatest",
            "SELECT greatest(TRUE, FALSE)",
            "boolean",
            json!(true),
        ),
        (
            "greatest",
            "SELECT greatest(X'01', X'02')",
            "binary",
            json!({ "hex": "02" }),
        ),
        (
            "greatest",
            "SELECT greatest(CAST('NaN' AS DOUBLE), 1D)",
            "double",
            dbl(f64::NAN),
        ),
        ("least", "SELECT least('b', 'a', 'c')", "string", json!("a")),
        (
            "least",
            "SELECT least(DATE'2020-01-02', DATE'2020-01-01')",
            "date",
            json!({ "days": 18_262 }),
        ),
        (
            "least",
            "SELECT least(CAST(1.5 AS FLOAT), 2, 3)",
            "double",
            dbl(1.5),
        ),
        (
            "least",
            "SELECT least(1, 2L, CAST(3 AS SMALLINT))",
            "bigint",
            json!(1),
        ),
        (
            "least",
            "SELECT least(CAST(1 AS TINYINT), CAST(2 AS TINYINT))",
            "tinyint",
            json!(1),
        ),
        ("least", "SELECT least(1S, 2S)", "smallint", json!(1)),
        ("greatest", "SELECT greatest(1.5D, 1)", "double", dbl(1.5)),
    ] {
        let outcome = cell(f, sql, on, rows(ty, want));
        assert_eq!(outcome, CellOutcome::Pass, "{sql}");
    }
    // Legacy mode: tinyint overflow wraps; a float and an int stay float.
    assert_eq!(
        cell(
            "round",
            "SELECT round(CAST(127 AS TINYINT), -1)",
            off,
            rows("tinyint", json!(-126))
        ),
        CellOutcome::Pass
    );
    assert_eq!(
        cell(
            "greatest",
            "SELECT greatest(CAST(1.5 AS FLOAT), 2)",
            off,
            rows(
                "float",
                json!({ "f32": format!("{:08x}", 2.0_f32.to_bits()) })
            )
        ),
        CellOutcome::Pass
    );
    for (f, sql, want) in [
        (
            "sqrt",
            "SELECT sqrt(4)",
            rows(
                "double",
                json!({ "f64": format!("{:016x}", 3.0_f64.to_bits()) }),
            ),
        ),
        ("round", "SELECT round(15, -1)", rows("int", json!(10))),
        ("pmod", "SELECT pmod(-7, 3)", rows("int", json!(1))),
    ] {
        let outcome = cell(f, sql, on, want);
        assert!(
            matches!(outcome, CellOutcome::Fail(_)),
            "{sql}: {outcome:?}"
        );
    }
    for (f, sql) in [
        ("round", "SELECT round(2.5, NULL)"),
        ("round", "SELECT round(true, 0)"),
        ("round", "SELECT round(2.5, 'x')"),
        ("sqrt", "SELECT sqrt(true)"),
        ("sqrt", "SELECT sqrt()"),
        ("pi", "SELECT pi(1)"),
        ("pow", "SELECT pow(2)"),
        ("ceil", "SELECT ceil(DATE'2020-01-01')"),
        ("ceil", "SELECT ceil(2.5, true)"),
        ("ceil", "SELECT ceil(true, 1)"),
        ("ceil", "SELECT ceil(1, 2, 3)"),
        ("pmod", "SELECT pmod(1)"),
        ("pmod", "SELECT pmod(true, 1)"),
        ("greatest", "SELECT greatest(1)"),
        ("greatest", "SELECT greatest(1, 'a')"),
        ("greatest", "SELECT greatest(NULL, NULL)"),
        ("greatest", "SELECT greatest(array(1), array(2))"),
    ] {
        let outcome = cell(f, sql, on, rows("double", json!(1)));
        assert!(
            matches!(outcome, CellOutcome::NotRun(_)),
            "{sql}: {outcome:?}"
        );
    }
}

#[test]
fn datetime_functions_run_in_every_shape() {
    let on = Mode::AnsiOn;
    let day = |d: i64| json!({ "days": d });
    let micros = |m: i64| json!({ "ts_us": m, "tz": "UTC" });
    for (f, sql, ty, want) in [
        (
            "date_add",
            "SELECT date_add('2016-07-30', 1)",
            "date",
            day(17_013),
        ),
        (
            "date_sub",
            "SELECT date_sub(DATE'2016-07-30', 1S)",
            "date",
            day(17_011),
        ),
        (
            "datediff",
            "SELECT datediff(DATE'2024-02-29', '2009-07-30')",
            "int",
            json!(5_327),
        ),
        (
            "year",
            "SELECT year(TIMESTAMP'2024-02-29 12:34:56')",
            "int",
            json!(2024),
        ),
        (
            "extract",
            "SELECT extract(DOY FROM DATE'2024-03-10')",
            "int",
            json!(70),
        ),
        (
            "date_part",
            "SELECT date_part('week', DATE'2024-03-10')",
            "int",
            json!(10),
        ),
        (
            "extract",
            "SELECT extract(SECOND FROM TIMESTAMP'2024-03-10 13:45:56.789012')",
            "decimal(8,6)",
            json!({"unscaled": "56789012", "scale": 6}),
        ),
        (
            "+",
            "SELECT DATE'2020-01-31' + INTERVAL '1' MONTH",
            "date",
            day(18_321),
        ),
        (
            "-",
            "SELECT DATE'2020-01-01' - interval 90 days",
            "date",
            day(18_262 - 90),
        ),
        (
            "+",
            "SELECT INTERVAL 14 MONTHS + DATE'2020-01-01'",
            "date",
            day(18_687),
        ),
        ("+", "SELECT DATE'2020-01-01' + 2", "date", day(18_264)),
        (
            "+",
            "SELECT DATE'2020-01-01' + INTERVAL 1 DAY 2 HOURS 3 MINUTES 4.5 SECONDS",
            "timestamp",
            micros(1_577_930_584_500_000),
        ),
        (
            "to_date",
            "SELECT to_date(' 2020-01-01 10:00:00')",
            "date",
            day(18_262),
        ),
        (
            "to_timestamp",
            "SELECT to_timestamp('2020-01-01 10:00:00+02:00')",
            "timestamp",
            micros(1_577_865_600_000_000),
        ),
        (
            "try_add",
            "SELECT try_add(DATE'9999-12-31', INTERVAL 2147483647 MONTH)",
            "date",
            Value::Null,
        ),
    ] {
        let outcome = cell(f, sql, on, rows(ty, want));
        assert_eq!(outcome, CellOutcome::Pass, "{sql}");
    }
    // Day arithmetic past the `int` range wraps like Java `int` under ANSI too.
    let sql = "SELECT date_add('2016-07-30', 2147483647)";
    let wrapped = rows("date", day(-2_147_466_637));
    assert_eq!(
        cell("date_add", sql, Mode::AnsiOff, wrapped.clone()),
        CellOutcome::Pass
    );
    assert_eq!(cell("date_add", sql, on, wrapped), CellOutcome::Pass);
    // The clock functions only check the result type.
    let nd = |ty: &str| Expected::Nondeterministic {
        schema: vec![Field {
            name: "c".to_owned(),
            data_type: ty.to_owned(),
            nullable: false,
        }],
    };
    for (f, sql, ty, pass) in [
        ("current_date", "SELECT current_date()", "date", true),
        ("now", "SELECT now()", "timestamp", true),
        ("current_date", "SELECT current_date()", "timestamp", false),
    ] {
        let outcome = cell(f, sql, on, nd(ty));
        assert_eq!(outcome == CellOutcome::Pass, pass, "{sql}: {outcome:?}");
    }
    let fail = |o: CellOutcome| assert!(matches!(o, CellOutcome::Fail(_)), "{o:?}");
    fail(cell(
        "date_add",
        "SELECT date_add(DATE'2020-01-01', 1)",
        on,
        rows("date", day(0)),
    ));
    let v = rows("date", day(0));
    for (f, sql) in [
        ("to_date", "SELECT to_date('2020', 'yyyy')"),
        ("extract", "SELECT extract(century FROM DATE'2020-01-01')"),
        ("extract", "SELECT extract(YEAR FROM INTERVAL '1' MONTH)"),
        ("+", "SELECT DATE'2020-01-01' * INTERVAL '1' DAY"),
        ("+", "SELECT INTERVAL '1' DAY - DATE'2020-01-01'"),
        ("+", "SELECT INTERVAL '1' DAY + INTERVAL '1' DAY"),
        ("+", "SELECT DATE'2020-01-01' + INTERVAL '1' YEAR '1' DAY"),
        ("+", "SELECT DATE'2020-01-01' + INTERVAL '1.5' DAY"),
        (
            "+",
            "SELECT DATE'2020-01-01' + INTERVAL '99999999999999999999999999999999' DAY",
        ),
        ("+", "SELECT DATE'2020-01-01' + INTERVAL '2147483648' MONTH"),
        (
            "+",
            "SELECT DATE'2020-01-01' + INTERVAL '1 2' DAY TO MINUTE",
        ),
        ("date_add", "SELECT date_add(1.5D, 1)"),
        ("date_add", "SELECT date_add(DATE'2020-01-01', 1L)"),
        ("current_date", "SELECT current_date(1)"),
    ] {
        let outcome = cell(f, sql, on, v.clone());
        assert!(
            matches!(outcome, CellOutcome::NotRun(_)),
            "{sql}: {outcome:?}"
        );
    }
}

#[test]
fn golden_statements() {
    let text = "-- Automatically generated by SQLQueryTestSuite\n\
                -- !query\nset spark.sql.ansi.enabled=false\n-- !query schema\nstruct<key:string,value:string>\n-- !query output\nspark.sql.ansi.enabled\tfalse\n\n\n\
                -- !query\nselect 1 div 0\n-- !query schema\nstruct<(1 div 0):bigint>\n-- !query output\nNULL\n\n\n\
                -- !query\nset spark.sql.ansi.enabled=true\n-- !query schema\nstruct<>\n-- !query output\n\n\n\n\
                -- !query\nselect 1 div 0\n-- !query schema\nstruct<>\n-- !query output\norg.apache.spark.SparkArithmeticException\n{\"errorClass\" : \"DIVIDE_BY_ZERO\", \"sqlState\" : \"22012\"}\n\n\n\
                -- !query\nselect 1, 2.5D\n-- !query schema\nstruct<1:int,2.5:double>\n-- !query output\n1\t2.5\n\n\n\
                -- !query\nselect 1 + 1\n-- !query schema\nstruct<(1 + 1):int>\n-- !query output\n3\n\n\n\
                -- !query\nselect 1 div 0\n-- !query schema\nstruct<>\n-- !query output\norg.apache.spark.SparkArithmeticException\n{\"errorClass\" : \"ARITHMETIC_OVERFLOW\"}\n\n\n\
                -- !query\nselect 1 + 1\n-- !query schema\nstruct<>\n-- !query output\norg.apache.spark.SparkArithmeticException\n{\"errorClass\" : \"ARITHMETIC_OVERFLOW\"}\n\n\n\
                -- !query\nselect 1 + 2147483647\n-- !query schema\nstruct<>\n-- !query output\n3\n\n\n\
                -- !query\nselect factorial(1)\n-- !query schema\nstruct<>\n-- !query output\n1\n\n\n\
                -- !query\nselect CAST(X'01' AS INT)\n-- !query schema\nstruct<>\n-- !query output\n1\n\n\n\
                -- !query\nselect 1 + rand()\n-- !query schema\nstruct<>\n-- !query output\n1\n\n\n\
                -- !query\nselect 1 limit 0\n-- !query schema\nstruct<1:int>\n-- !query output\n\n\n\n\
                -- !query\nselect * from t\n-- !query schema\nstruct<>\n-- !query output\n1\n\n\n\
                -- !query\nshow tables\n-- !query schema\nstruct<>\n-- !query output\n1\n";
    let t = run_golden(text, true).unwrap();
    assert_eq!((t.statements, t.evaluated, t.passed), (15, 7, 3), "{t:?}");
    assert_eq!(t.failed.len(), 4, "{t:?}");
    assert_eq!(t.needs_engine.len(), 6, "{t:?}");
    assert!(schema_types_match(
        "struct<a,b:int,c:double>",
        &["int".to_owned(), "double".to_owned()]
    ));
    assert!(!schema_types_match("struct<a:int>", &[]));
    assert!(!schema_types_match("nope", &["int".to_owned()]));
    assert!(!schema_types_match(
        "struct<a:int,b:int>",
        &["bigint".to_owned(), "int".to_owned()]
    ));
}

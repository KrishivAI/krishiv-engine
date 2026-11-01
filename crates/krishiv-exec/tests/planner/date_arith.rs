// SPDX-License-Identifier: Apache-2.0
//! `date + int`, `int + date` and `date - int` are `date_add` and
//! `date_sub` (Spark 4.2.0, ANSI on and off; values and names from a local
//! Spark run). Every other mix of a date and a number stays refused.

use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::block_on;
use krishiv_plan::analyzer::{Analyzer as _, DefaultAnalyzer};

use super::{env, names};

#[test]
fn a_date_and_a_day_count_are_date_add_and_date_sub() {
    let env = env();
    for (expr, name, want) in [
        (
            "DATE'2020-01-30' + 5",
            "date_add(DATE '2020-01-30', 5)",
            "2020-02-04",
        ),
        (
            "5 + DATE'2020-01-30'",
            "date_add(DATE '2020-01-30', 5)",
            "2020-02-04",
        ),
        (
            "DATE'2020-01-30' - 5",
            "date_sub(DATE '2020-01-30', 5)",
            "2020-01-25",
        ),
        (
            "DATE'2020-03-01' - CAST(1 AS TINYINT)",
            "date_sub(DATE '2020-03-01', CAST(1 AS TINYINT))",
            "2020-02-29",
        ),
    ] {
        let sql = format!("SELECT {expr}");
        let plan = env.analyzed(&sql).plan;
        let schema = plan.root().resolved_output().unwrap();
        assert_eq!(schema[0].name, name, "{sql}");
        let (_, rows) = env.rows(&format!("SELECT CAST({expr} AS STRING)"));
        assert_eq!(rows, [want], "{sql}");
    }
    // A column, per row: `a` is 0, 1, 2, NULL, ... and `NULL + 5` is NULL.
    let (_, rows) =
        env.rows("SELECT CAST(DATE'2020-01-30' + a AS STRING) FROM t WHERE b <= 30 ORDER BY b");
    assert_eq!(rows, ["2020-01-30", "2020-01-31", "2020-02-01", "NULL"]);
    assert_eq!(names(&env.rows("SELECT 1 AS x").0), ["x"]);
}

#[test]
fn other_mixes_of_dates_and_numbers_stay_refused() {
    let env = env();
    for sql in [
        // Spark: DATATYPE_MISMATCH (date_add takes an int); here refused.
        "SELECT DATE'2020-01-30' + CAST(5 AS BIGINT)",
        // `int - date` is not a date_sub.
        "SELECT 5 - DATE'2020-01-30'",
        // Two dates subtract to an interval.
        "SELECT DATE'2020-01-30' - DATE'2020-01-01'",
        "SELECT TIMESTAMP'2020-01-30 10:00:00' + 5",
    ] {
        let plan =
            krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .unwrap();
        let e = block_on(DefaultAnalyzer.analyze(&plan, &env.ctx)).unwrap_err();
        assert_eq!(
            e.error.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{sql}: {e}"
        );
    }
}

#[test]
fn a_date_plus_a_decimal_is_spark_s_type_mismatch() {
    let env = env();
    let plan = krishiv_sql::parse_and_lower(
        "SELECT DATE'2020-01-30' + 5.5",
        &krishiv_sql::ParserOptions::default(),
        SourceId(0),
    )
    .unwrap();
    let e = block_on(DefaultAnalyzer.analyze(&plan, &env.ctx)).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
        "{e}"
    );
}

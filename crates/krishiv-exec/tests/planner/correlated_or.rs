// SPDX-License-Identifier: Apache-2.0
//! A correlated subquery whose filter is `(k = o AND p) OR (k = o AND q)`
//! (TPC-DS q41): the shared correlation is factored out, as Spark's
//! optimizer factors a disjunction's common conjuncts. Rows are from a
//! local Spark 4.2.0 run over `t` of the planner tests.

use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::block_on;
use krishiv_plan::analyzer::{Analyzer as _, DefaultAnalyzer};

use super::env;

#[test]
fn a_shared_correlation_under_or_is_a_correlation() {
    let env = env();
    for (sql, want) in [
        (
            "SELECT CAST(a AS STRING) FROM t x WHERE (SELECT count(*) FROM t y \
             WHERE (y.a = x.a AND y.b > 50) OR (y.a = x.a AND y.b < 20)) > 0 ORDER BY a",
            vec!["0", "1", "6", "8", "9"],
        ),
        (
            "SELECT CONCAT(CAST(a AS STRING), '|', CAST((SELECT count(*) FROM t y \
             WHERE (y.a = x.a AND y.b > 50) OR (y.a = x.a AND y.b < 20)) AS STRING)) \
             FROM t x WHERE a IS NOT NULL ORDER BY a",
            vec!["0|1", "1|1", "2|0", "4|0", "5|0", "6|1", "8|1", "9|1"],
        ),
        (
            "SELECT CAST(a AS STRING) FROM t x WHERE EXISTS (SELECT 1 FROM t y \
             WHERE (y.a = x.a AND y.b > 50) OR (y.a = x.a AND y.s = 's1')) ORDER BY a",
            vec!["1", "6", "8", "9"],
        ),
        // `k = o OR (k = o AND p)` is `k = o`.
        (
            "SELECT CAST(a AS STRING) FROM t x WHERE (SELECT count(*) FROM t y \
             WHERE (y.a = x.a AND y.b = 10) OR (y.a = x.a)) > 0 ORDER BY a",
            vec!["0", "1", "2", "4", "5", "6", "8", "9"],
        ),
    ] {
        let (_, rows) = env.rows(sql);
        assert_eq!(rows, want, "{sql}");
    }
}

#[test]
fn a_correlation_only_one_disjunct_holds_stays_refused() {
    let env = env();
    let sql = "SELECT a FROM t x WHERE (SELECT count(*) FROM t y \
               WHERE (y.a = x.a AND y.b > 50) OR y.b < 20) > 0";
    let plan =
        krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .unwrap();
    let e = block_on(DefaultAnalyzer.analyze(&plan, &env.ctx)).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "KRISHIV_UNSUPPORTED_FEATURE",
        "{e}"
    );
}

#[test]
fn a_nondeterministic_disjunction_is_not_factored() {
    let env = env();
    let sql = "SELECT a FROM t x WHERE (SELECT count(*) FROM t y \
               WHERE (y.a = x.a AND rand() < 0.5) OR (y.a = x.a AND y.b > 50)) > 0";
    let plan =
        krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .unwrap();
    let e = block_on(DefaultAnalyzer.analyze(&plan, &env.ctx)).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "KRISHIV_UNSUPPORTED_FEATURE",
        "{e}"
    );
}

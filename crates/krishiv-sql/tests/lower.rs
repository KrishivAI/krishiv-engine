// SPDX-License-Identifier: Apache-2.0
//! SQL lowering: the 22 TPC-H queries lower to the hand-built plans of
//! `krishiv-plan` (their pretty-printed snapshots are the oracle);
//! unsupported constructs are named; literals are typed as Spark types
//! them; deep statements lower on a small stack and long join lists hit the
//! depth cap.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::fmt::Write as _;
use std::path::Path;

use krishiv_plan::{Plan, SourceId};
use krishiv_sql::{ParserOptions, parse_and_lower};

fn lower(sql: &str) -> Result<Plan, krishiv_sql::ParseError> {
    parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
}

/// One line per statement: the lowered plan's root line, or the error.
fn report(stmts: &[&str]) -> String {
    let mut out = String::new();
    for sql in stmts {
        let line = match lower(sql) {
            Ok(p) => p.root().to_string().replace('\n', " | "),
            Err(e) => format!("{} @ {}..{}", e.error, e.span.start, e.span.end),
        };
        let _ = writeln!(out, "{sql}\n  => {line}");
    }
    out
}

#[test]
fn tpch_lowers_to_the_hand_built_plans() {
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    for i in 1..=22 {
        let snap = root.join(format!(
            "crates/krishiv-plan/src/tests/snapshots/krishiv_plan__tests__tpch__q{i:02}.snap"
        ));
        let snap = std::fs::read_to_string(snap).unwrap();
        let oracle = snap.splitn(3, "---\n").nth(2).unwrap();
        let sql = std::fs::read_to_string(root.join(format!("bench/queries/tpch/q{i:02}.sql")));
        let sql = sql.unwrap();
        let plan = lower(sql.trim().trim_end_matches(';')).unwrap();
        assert_eq!(plan.root().to_string(), oracle, "q{i:02}");
    }
}

#[test]
fn constructs_lower_or_are_named_as_unsupported() {
    insta::assert_snapshot!(report(&[
        "SELECT 1",
        "SELECT DISTINCT a FROM t",
        "SELECT /*+ BROADCAST(t) */ a FROM t",
        "SELECT a FROM t UNION SELECT a FROM u",
        "VALUES (1)",
        "SELECT * FROM VALUES (1) AS v(a)",
        "SELECT * FROM range(3)",
        "SELECT a FROM t LATERAL VIEW explode(b) x AS y",
        "SELECT row_number() OVER (ORDER BY a) FROM t",
        "SELECT a FROM t GROUP BY ROLLUP (a)",
        "SELECT a FROM t GROUP BY a WITH CUBE",
        "SELECT * FROM t NATURAL JOIN u",
        "SELECT * FROM t, LATERAL (SELECT a)",
        "SELECT * FROM t VERSION AS OF 1",
        "SELECT a FROM t CLUSTER BY a",
        "SELECT a FROM t DISTRIBUTE BY a",
        "SELECT a FROM t LIMIT 1 OFFSET 2",
        "FROM t SELECT a LIMIT 1 OFFSET 2",
        "FROM t SELECT a CLUSTER BY a",
        "FROM t SELECT a DISTRIBUTE BY a",
        "FROM t SELECT a WINDOW w AS (ORDER BY a)",
        "SELECT a FROM t WINDOW w AS (ORDER BY a)",
        "SELECT a FROM t QUALIFY a > 1",
        "WITH RECURSIVE r AS (SELECT a FROM t) SELECT * FROM r",
        "SELECT * EXCEPT (a) FROM t",
        "SELECT x[0] FROM t",
        "SELECT (a).b.c FROM t",
        "SELECT f(a).b FROM t",
        "SELECT transform(a, x -> x) FROM t",
        "SELECT (a, b) FROM t",
        "SELECT a LIKE ANY ('x', 'y') FROM t",
        "SELECT ? FROM t",
        "SELECT INTERVAL 1 DAY FROM t",
        "SELECT TIMESTAMP '2020-01-01 00:00:00' FROM t",
        "SELECT db.f(a) FROM t",
        "SELECT f(x => 1) FROM t",
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY a) FROM t",
        "SELECT first(a) IGNORE NULLS FROM t",
        "SELECT a FROM t PIVOT (sum(a) FOR b IN (1))",
        "SELECT * FROM t WITH (k = x)",
        "FROM t SELECT a SELECT b",
        "SELECT * FROM parquet.`s3a://k:Canary-7f3a9c@b/p` WITH ('password' = 'Canary-7f3a9c')",
        "WITH c (x) AS (SELECT a FROM t) SELECT x FROM c ORDER BY x LIMIT ALL",
        "FROM t SELECT a SORT BY a",
        "SELECT a FROM t WHERE (a, b) IN (SELECT a, d FROM u) AND a NOT BETWEEN 1 AND 2",
        "SELECT a <=> 1, a IS DISTINCT FROM 1, a IS NOT DISTINCT FROM 1, a IS TRUE, a IS NOT FALSE, a IS UNKNOWN, a IS NOT UNKNOWN FROM t",
        "SELECT +a, ~a, a DIV 2, a & 1, a | 1, a ^ 1, a << 1, a >> 1, a >>> 1, a || b FROM t",
        "SELECT CASE a WHEN 1 THEN 'x' END, b ILIKE 'x' ESCAPE '/', b NOT RLIKE 'x', TRY_CAST(a AS INT) FROM t",
        "SELECT * FROM (t JOIN u ON true) x CROSS JOIN v LEFT SEMI JOIN w ON true",
        "TABLE t",
        "(SELECT a FROM t) ORDER BY a",
        "FROM t SELECT a UNION SELECT b",
        "WITH RECURSIVE c MAX RECURSION LEVEL 3 AS (SELECT a FROM t) SELECT a FROM c",
        "SELECT explode(a) AS (x, y) FROM t",
        "SELECT a FROM t, u PIVOT (sum(a) FOR b IN (1))",
        "SELECT * FROM t JOIN LATERAL (SELECT a) s",
        "SELECT * FROM t WITH (k = true, n = 2) RIGHT JOIN (SELECT a FROM u) USING (a)",
        "SELECT -a, a % 2, a IS NULL, NULL FROM t ORDER BY a DESC NULLS FIRST, b NULLS LAST",
        "SELECT (t.a).b, ((a)).c.d FROM t",
        "SELECT 1BD, 1e3BD FROM t",
    ]));
}

#[test]
fn literals_are_typed_as_spark_types_them() {
    insta::assert_snapshot!(report(&[
        "SELECT 2147483647, 2147483648, 9223372036854775808, 1Y, 1S, 1L, -0.050, .5, 5., 1e3, 1.5D, 1.5F, 1.5BD FROM t",
        "SELECT X'1F', X'abc', DATE '2020-2-3', DATE '+2020', DATE '2020-02-29T10:00', DATE '-0044-03-15' FROM t",
        "SELECT 300Y FROM t",
        "SELECT 40000S FROM t",
        "SELECT 9223372036854775808L FROM t",
        "SELECT 1e400 FROM t",
        "SELECT 1e40F FROM t",
        "SELECT 123456789012345678901234567890123456789 FROM t",
        "SELECT DATE '2021-02-29' FROM t",
        "SELECT DATE '2021-13-01' FROM t",
        "SELECT X'zz' FROM t",
        "SELECT CAST(a AS unknown_type) FROM t",
    ]));
}

/// A thread with a 2 MiB stack, the size of a tokio worker's.
fn on_small_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn deep_queries_lower_on_a_small_stack_and_long_joins_hit_the_cap() {
    on_small_stack(|| {
        let nested = format!("SELECT {}a{} FROM t", "(".repeat(950), ")".repeat(950));
        assert!(lower(&nested).is_ok());
        let chain = format!("SELECT a{} FROM t", " + a".repeat(950));
        assert!(lower(&chain).is_ok());
        let subqueries = (0..120).fold("SELECT a FROM t".to_owned(), |q, _| {
            format!("SELECT (SELECT max(a) FROM ({q}) s) FROM t")
        });
        let r = lower(&subqueries);
        assert!(
            r.is_ok(),
            "nested subqueries: {:?}",
            r.err().map(|e| e.error)
        );
        // Join lists do not nest in the syntax tree but do in the plan.
        let joins = format!("SELECT a FROM t{}", ", t".repeat(1200));
        let e = lower(&joins).unwrap_err();
        assert_eq!(e.error.error_class().name(), "FAILED_TO_PARSE_TOO_COMPLEX");
        let joins = format!("SELECT a FROM t{}", " JOIN t ON true".repeat(1200));
        assert!(lower(&joins).is_err());
        let dotted = format!("SELECT {} FROM t", vec!["a"; 3000].join("."));
        assert!(lower(&dotted).is_ok());
    });
}

// SPDX-License-Identifier: Apache-2.0
//! SQL lowering: the 22 TPC-H queries lower to the hand-built plans of
//! `krishiv-plan` (their pretty-printed snapshots are the oracle);
//! unsupported constructs are named; literals are typed as Spark types
//! them; deep statements lower on a small stack and long join lists hit the
//! depth cap.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use krishiv_plan::{Plan, PlanNode, SourceId};
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
        "SELECT a FROM t INTERSECT SELECT a FROM u",
        "SELECT a FROM t EXCEPT ALL SELECT a FROM u",
        "SELECT /*+ BROADCAST(t), MERGE(u) */ /*+ COALESCE(2) */ DISTINCT a FROM t, u",
        "SELECT a FROM t UNION ALL SELECT b FROM u UNION SELECT c FROM v",
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
        "SELECT sum(a) IGNORE NULLS FROM t",
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

/// `first(x) IGNORE NULLS` is `first(x, true)`; `RESPECT NULLS` is `false`,
/// whatever flag was written; a flag that is not a literal stays for the
/// analyzer to refuse.
#[test]
fn null_treatment_sets_the_ignore_nulls_argument() {
    let shown = |sql: &str| lower(sql).unwrap().root().to_string();
    let cases = [
        ("first(a) IGNORE NULLS", "first(a, true)"),
        ("LAST_VALUE(a) RESPECT NULLS", "LAST_VALUE(a, false)"),
        ("first(a, false) IGNORE NULLS", "first(a, true)"),
        ("last(a, true) RESPECT NULLS", "last(a, false)"),
        ("first(a, b) IGNORE NULLS", "first(a, b)"),
    ];
    for (call, shown_call) in cases {
        let plan = shown(&format!("SELECT {call} FROM t"));
        assert!(
            plan.contains(&format!("Project [{shown_call}]")),
            "{call}: {plan}"
        );
    }
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

#[test]
fn nested_grouping_sets_lower_on_a_small_stack() {
    on_small_stack(|| {
        // 900 nested `GROUPING SETS (...)`: the parser accepts it (below
        // `MAX_DEPTH`), and lowering must hop stacks through the recursion.
        let sql = format!(
            "SELECT a FROM t GROUP BY {}a{}",
            "GROUPING SETS (".repeat(900),
            ")".repeat(900)
        );
        // One set `((a))`: every level unwraps to the single inner set.
        let expected = "Aggregate [a] GROUPING SETS ((a)), [a]\n+- Read t\n";
        assert_eq!(
            lower(&sql)
                .expect("nested GROUPING SETS lowers")
                .root()
                .to_string(),
            expected
        );
        // A single level nests the same way and keeps its sets.
        assert_eq!(
            lower("SELECT a FROM t GROUP BY GROUPING SETS (GROUPING SETS ((a)))")
                .expect("nested sets lower")
                .root()
                .to_string(),
            expected
        );
    });
}

#[test]
fn nested_grouping_sets_cap_counts_inside_the_loop() {
    // Two 16-element cubes hold 65,536 sets each: the union passes the
    // 65,536 cap on the second item, at the outer `GROUPING SETS`, before
    // the cross-product check runs.
    let a: Vec<String> = (0..16).map(|i| format!("a{i}")).collect();
    let b: Vec<String> = (0..16).map(|i| format!("b{i}")).collect();
    let sql = format!(
        "SELECT a FROM t GROUP BY GROUPING SETS (CUBE ({}), CUBE ({}))",
        a.join(", "),
        b.join(", ")
    );
    let e = lower(&sql).expect_err("two 16-cubes exceed the grouping-sets cap");
    assert!(
        e.error
            .to_string()
            .contains("more than 65536 grouping sets"),
        "cap message, got {}",
        e.error
    );
    assert_eq!(
        e.span.start,
        sql.find("GROUPING SETS").expect("the item"),
        "the cap cites the GROUPING SETS item, not GROUP BY"
    );
}

/// The inline table of `sql`: the first `PlanNode::InlineTable` under its
/// root, with the alias above it unwrapped.
fn inline_table(sql: &str) -> Option<(Vec<String>, Vec<Vec<String>>)> {
    let plan = lower(sql).unwrap();
    let mut stack = vec![Arc::clone(plan.root())];
    while let Some(p) = stack.pop() {
        if let PlanNode::InlineTable { rows, columns } = &p.node {
            let rows = rows
                .iter()
                .map(|r| r.iter().map(ToString::to_string).collect())
                .collect();
            return Some((columns.clone(), rows));
        }
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    None
}

#[test]
fn values_lower_to_inline_tables() {
    // A query body without an alias: columns are `col1`, ....
    assert_eq!(
        inline_table("VALUES (1, 'a'), (2, 'b')").expect("an inline table"),
        (
            vec!["col1".to_owned(), "col2".to_owned()],
            vec![
                vec!["1".to_owned(), "'a'".to_owned()],
                vec!["2".to_owned(), "'b'".to_owned()],
            ]
        )
    );
    // A query body with an alias: the alias column list renames them, and
    // the alias itself stays a `SubqueryAlias` above.
    let plan = lower("VALUES (1, 'a'), (2, 'b') AS t(x, y)").unwrap();
    let PlanNode::SubqueryAlias {
        input,
        alias,
        columns,
    } = &plan.root().node
    else {
        panic!("alias above:\n{}", plan.root());
    };
    assert_eq!(alias, "t");
    assert_eq!(columns, &["x", "y"]);
    assert!(
        matches!(&input.node, PlanNode::InlineTable { .. }),
        "inline table below:\n{}",
        plan.root()
    );
    assert_eq!(
        inline_table("VALUES (1, 'a'), (2, 'b') AS t(x, y)")
            .expect("an inline table")
            .0,
        ["x", "y"]
    );
    // A table factor with an alias column list.
    assert_eq!(
        inline_table("SELECT * FROM VALUES ('one', 1) AS data(a, b)")
            .expect("an inline table")
            .0,
        ["a", "b"]
    );
    // A table factor without an alias keeps `col1`, ....
    assert_eq!(
        inline_table("SELECT * FROM VALUES ('one', 1)")
            .expect("an inline table")
            .0,
        ["col1", "col2"]
    );
    // A single-column row written without parentheses is one column, and
    // one row per expression.
    assert_eq!(
        inline_table("SELECT * FROM VALUES 1, 2, 3 AS data(a)").expect("an inline table"),
        (
            vec!["a".to_owned()],
            vec![
                vec!["1".to_owned()],
                vec!["2".to_owned()],
                vec!["3".to_owned()],
            ]
        )
    );
    // Row aliases are stripped: the schema never comes from them (the
    // outer alias renames `col1`, ... here).
    assert_eq!(
        inline_table(
            "SELECT * FROM (VALUES (1 AS id, 'a' AS name), (2 AS id, 'b' AS name)) AS t(a, b)"
        )
        .expect("an inline table"),
        (
            vec!["col1".to_owned(), "col2".to_owned()],
            vec![
                vec!["1".to_owned(), "'a'".to_owned()],
                vec!["2".to_owned(), "'b'".to_owned()],
            ]
        )
    );
    assert_eq!(
        inline_table("SELECT * FROM VALUES (1 AS a, 2 AS b)")
            .expect("an inline table")
            .0,
        ["col1", "col2"]
    );
}

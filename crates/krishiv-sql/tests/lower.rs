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

use krishiv_plan::{ExprNode, Grouping, JoinCondition, JoinKind, Plan, PlanNode, SourceId};
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
        "SELECT * FROM t WITH (k = x)",
        "FROM t SELECT a SELECT b",
        "FROM t",
        "FROM t SELECT a ORDER BY a LIMIT 1 SELECT b WHERE b > 1",
        "FROM t SELECT DISTINCT a SELECT b GROUP BY b",
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
        "SELECT * FROM t JOIN LATERAL (SELECT a) s",
        "SELECT * FROM t WITH (k = true, n = 2) RIGHT JOIN (SELECT a FROM u) USING (a)",
        "SELECT -a, a % 2, a IS NULL, NULL FROM t ORDER BY a DESC NULLS FIRST, b NULLS LAST",
        "SELECT (t.a).b, ((a)).c.d FROM t",
        "SELECT 1BD, 1e3BD FROM t",
    ]));
}

/// `PIVOT` and `UNPIVOT` lower over the relation they follow (per
/// relation and at the `FROM`-clause level); the alias becomes a
/// `SubqueryAlias`. A `FROM` clause with both is rejected, as Spark's
/// parser rejects it, as is one relation carrying both (Spark's
/// `NOT_ALLOWED_IN_FROM.UNPIVOT_WITH_PIVOT`).
#[test]
fn pivot_and_unpivot_lower_over_the_relation_they_follow() {
    insta::assert_snapshot!(report(&[
        "SELECT a FROM t PIVOT (sum(a) FOR b IN (1))",
        "SELECT a FROM t, u PIVOT (sum(a) FOR b IN (1))",
        "SELECT * FROM t PIVOT (sum(a) AS s, avg(b) FOR (c, d) IN ((1, 2) AS one, (3, 4) two)) p",
        "SELECT * FROM t UNPIVOT INCLUDE NULLS (v FOR n IN (a AS x, b)) AS u",
        "SELECT * FROM t UNPIVOT EXCLUDE NULLS ((v1, v2) FOR n IN ((a, b) AS ab, (c, d)))",
        "SELECT * FROM t JOIN u ON t.a = u.a PIVOT (count(*) FOR a IN ('x')) p JOIN v ON true",
        "SELECT * FROM t PIVOT (sum(a) FOR b IN (1)) UNPIVOT (x FOR y IN (z))",
        "SELECT * FROM t, u PIVOT (sum(a) FOR b IN (1)) UNPIVOT (x FOR y IN (z))",
    ]));
}

/// The parser attaches `PIVOT` to the relation, but the `FROM` clause can
/// carry one too (after `LATERAL VIEW`, which lowering rejects): moving a
/// parsed pivot there must lower the same plan, and a clause with both a
/// pivot and an unpivot is rejected, as Spark's parser rejects it.
#[test]
fn from_clause_level_pivot_lowers_over_the_comma_join() {
    use krishiv_plan::SourceId;
    use krishiv_sql::ast::{Pivot, RelationExt, SetExprKind, Unpivot};
    use krishiv_sql::{ParserOptions, lower, parse_query};
    let mut q = parse_query(
        "SELECT a FROM t, u PIVOT (sum(a) FOR b IN (1))",
        &ParserOptions::default(),
    )
    .unwrap();
    let krishiv_sql::ast::SetExprKind::Select(select) = &mut q.body.kind else {
        panic!("a SELECT query");
    };
    let from = select.from.as_mut().unwrap();
    assert!(from.pivot.is_none());
    let relation = &mut from.relations[1];
    let Some(RelationExt::Pivot(pivot)) = relation.extensions.pop() else {
        panic!("a pivot extension");
    };
    assert!(relation.extensions.is_empty());
    from.pivot = Some(pivot);
    let plan = lower(&q, SourceId(0)).unwrap();
    assert_eq!(
        plan.root().to_string().replace('\n', " | "),
        "Project [a] | +- Pivot [b], [[1]], [sum(a)] |    +- Join Inner |       :- Read t |       +- Read u | "
    );
    // Both at once is `NOT_ALLOWED_IN_FROM.UNPIVOT_WITH_PIVOT`.
    let mut q = parse_query("SELECT a FROM t", &ParserOptions::default()).unwrap();
    let SetExprKind::Select(select) = &mut q.body.kind else {
        panic!("a SELECT query");
    };
    let from = select.from.as_mut().unwrap();
    let pivot: Pivot = parse_query(
        "SELECT a FROM t PIVOT (sum(a) FOR b IN (1))",
        &ParserOptions::default(),
    )
    .map(|mut p| {
        let SetExprKind::Select(s) = &mut p.body.kind else {
            panic!("a SELECT query");
        };
        let Some(RelationExt::Pivot(pivot)) =
            s.from.as_mut().unwrap().relations[0].extensions.pop()
        else {
            panic!("a pivot extension");
        };
        pivot
    })
    .unwrap();
    let unpivot: Unpivot = parse_query(
        "SELECT a FROM t UNPIVOT (x FOR y IN (z))",
        &ParserOptions::default(),
    )
    .map(|mut p| {
        let SetExprKind::Select(s) = &mut p.body.kind else {
            panic!("a SELECT query");
        };
        let Some(RelationExt::Unpivot(unpivot)) =
            s.from.as_mut().unwrap().relations[0].extensions.pop()
        else {
            panic!("an unpivot extension");
        };
        unpivot
    })
    .unwrap();
    from.pivot = Some(pivot);
    from.unpivot = Some(unpivot);
    let e = lower(&q, SourceId(0)).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "NOT_ALLOWED_IN_FROM.UNPIVOT_WITH_PIVOT"
    );
    // A `FROM`-clause unpivot lowers over the comma join too.
    let mut q = parse_query(
        "SELECT a FROM t, u UNPIVOT (x FOR y IN (z))",
        &ParserOptions::default(),
    )
    .unwrap();
    let SetExprKind::Select(select) = &mut q.body.kind else {
        panic!("a SELECT query");
    };
    let from = select.from.as_mut().unwrap();
    let relation = &mut from.relations[1];
    let Some(RelationExt::Unpivot(unpivot)) = relation.extensions.pop() else {
        panic!("an unpivot extension");
    };
    assert!(relation.extensions.is_empty());
    from.unpivot = Some(unpivot);
    let plan = lower(&q, SourceId(0)).unwrap();
    assert_eq!(
        plan.root().to_string().replace('\n', " | "),
        "Project [a] | +- Unpivot [x FOR y IN ([z])] |    +- Join Inner |       :- Read t |       +- Read u | "
    );
    // A pivot value with a column-list alias is rejected, like any other
    // multi-name alias.
    let mut q = parse_query(
        "SELECT a FROM t PIVOT (sum(a) FOR b IN (1))",
        &ParserOptions::default(),
    )
    .unwrap();
    let SetExprKind::Select(select) = &mut q.body.kind else {
        panic!("a SELECT query");
    };
    let from = select.from.as_mut().unwrap();
    let relation = &mut from.relations[0];
    let Some(RelationExt::Pivot(mut pivot)) = relation.extensions.pop() else {
        panic!("a pivot extension");
    };
    pivot.values[0].alias = Some(krishiv_sql::ast::Alias::List(vec![]));
    from.pivot = Some(pivot);
    let e = lower(&q, SourceId(0)).unwrap_err();
    assert_eq!(e.error.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
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
        ("any_value(a) IGNORE NULLS", "any_value(a, true)"),
        ("any_value(a, true) RESPECT NULLS", "any_value(a, false)"),
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

/// `GROUP BY ALL` lowers to a lone `all` key carrying the whole clause's
/// span (the analyzer expands it, or resolves the column); any case goes.
#[test]
fn group_by_all_lowers_to_a_clause_spanned_all_key() {
    for (sql, clause) in [
        ("SELECT a, COUNT(*) FROM t GROUP BY ALL", "GROUP BY ALL"),
        ("SELECT a, COUNT(*) FROM t GROUP BY aLl", "GROUP BY aLl"),
    ] {
        let plan = lower(sql).unwrap();
        let PlanNode::Aggregate { grouping, .. } = &plan.root().node else {
            panic!("{sql}: expected an Aggregate, got {}", plan.root());
        };
        let Grouping::GroupBy(keys) = grouping else {
            panic!("{sql}: expected GROUP BY, got {grouping:?}");
        };
        assert_eq!(keys.len(), 1, "{sql}");
        let ExprNode::Column(c) = &keys[0].node else {
            panic!("{sql}: expected a column key, got {}", keys[0]);
        };
        assert!(
            c.name.len() == 1 && c.name[0].eq_ignore_ascii_case("all"),
            "{sql}: {c:?}"
        );
        let (source, span) = keys[0].origin.sql.expect("a SQL span");
        assert_eq!(source, SourceId(0), "{sql}");
        let at = sql.find(clause).expect("the clause");
        assert_eq!(
            (span.start, span.end),
            (
                u32::try_from(at).unwrap(),
                u32::try_from(at + clause.len()).unwrap()
            ),
            "{sql}: the key cites the clause, not the column"
        );
    }
}

/// `NATURAL` joins lower with a marker condition (the analyzer computes
/// the common columns); `ORDER BY ALL` lowers as a plain `all` key (the
/// analyzer expands it).
#[test]
fn natural_join_lowers_to_a_marker_condition() {
    for (sql, kind) in [
        ("SELECT * FROM t NATURAL JOIN u", JoinKind::Inner),
        ("SELECT * FROM t NATURAL LEFT JOIN u", JoinKind::LeftOuter),
    ] {
        let plan = lower(sql).unwrap();
        let PlanNode::Project { input, .. } = &plan.root().node else {
            panic!("{sql}: expected a Project, got {}", plan.root());
        };
        let PlanNode::Join {
            kind: got,
            condition,
            ..
        } = &input.node
        else {
            panic!("{sql}: expected a Join, got {input}");
        };
        assert_eq!(*got, kind, "{sql}");
        assert!(
            matches!(condition, JoinCondition::Natural),
            "{sql}: expected a NATURAL marker, got {condition:?}"
        );
    }
    let plan = lower("SELECT * FROM t ORDER BY ALL").unwrap();
    let PlanNode::Sort { order, .. } = &plan.root().node else {
        panic!("expected a Sort, got {}", plan.root());
    };
    assert_eq!(order.len(), 1);
    assert!(
        matches!(&order[0].expr.node, ExprNode::Column(_)),
        "ORDER BY ALL stays a column for the analyzer, got {}",
        order[0].expr
    );
}

/// `QUALIFY` lowers to a filter with a wrapped predicate (the analyzer
/// tells it apart from `WHERE` and `HAVING` by the wrapper).
#[test]
fn qualify_lowers_to_a_wrapped_filter() {
    let plan = lower("SELECT a, RANK() OVER (ORDER BY b) AS r FROM t QUALIFY r > 1").unwrap();
    let PlanNode::Filter { predicate, .. } = &plan.root().node else {
        panic!("expected a Filter, got {}", plan.root());
    };
    let ExprNode::Function(f) = &predicate.node else {
        panic!("expected the qualify wrapper, got {predicate}");
    };
    assert_eq!((f.name.as_str(), f.args.len()), ("qualify", 1));
}

/// `* EXCEPT` lowers onto the star: bare and qualified entries, including
/// nested paths (the analyzer drops the former and refuses the latter).
#[test]
fn star_except_lowers_onto_the_star() {
    for (sql, qualifier, except) in [
        (
            "SELECT * EXCEPT (a, b) FROM t",
            Vec::new(),
            vec![vec!["a".to_owned()], vec!["b".to_owned()]],
        ),
        (
            "SELECT t.* EXCEPT (t.a, b.c) FROM t",
            vec!["t".to_owned()],
            vec![
                vec!["t".to_owned(), "a".to_owned()],
                vec!["b".to_owned(), "c".to_owned()],
            ],
        ),
    ] {
        let plan = lower(sql).unwrap();
        let PlanNode::Project { exprs, .. } = &plan.root().node else {
            panic!("{sql}: expected a Project, got {}", plan.root());
        };
        assert_eq!(exprs.len(), 1, "{sql}");
        let ExprNode::Star(s) = &exprs[0].node else {
            panic!("{sql}: expected a star, got {}", exprs[0]);
        };
        assert_eq!(s.qualifier, qualifier, "{sql}");
        assert_eq!(s.except, except, "{sql}");
    }
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

#[test]
fn pipe_operators_lower_left_to_right() {
    insta::assert_snapshot!(report(&[
        "FROM t |> WHERE a > 1 |> SELECT a, b |> ORDER BY a LIMIT 3",
        "SELECT * FROM t |> EXTEND a + 1 AS c |> DROP b |> AS u",
        "FROM t |> AGGREGATE sum(a) AS s GROUP BY b",
        "FROM t |> AGGREGATE GROUP BY b AS k",
        "FROM t |> JOIN u ON t.a = u.a |> UNION ALL (SELECT 1, 2)",
        "FROM t |> SELECT DISTINCT a",
        "FROM t |> AGGREGATE sum(a) GROUP BY 1",
        "FROM t |> SET a = 1",
        "FROM t |> TABLESAMPLE (3 ROWS)",
    ]));
}

/// `lower_identified` marks exactly the columns at the given spans (the names a session
/// substituted for `IDENTIFIER(..)`), compared by position: the same name elsewhere is unmarked.
#[test]
fn columns_at_identifier_spans_are_marked() {
    let sql = "SELECT `c1`.`c2`, c1.c2, `c1`.`c2` + 1 FROM t";
    let query = krishiv_sql::parse_query(sql, &ParserOptions::default()).unwrap();
    let span = krishiv_sql::ast::Span::new(7, 16);
    let plan = krishiv_sql::lower_identified(&query, SourceId(0), false, &[span]).unwrap();
    let PlanNode::Project { exprs, .. } = &plan.root().node else {
        panic!("expected a Project, got {}", plan.root());
    };
    let marked = |e: &krishiv_plan::Expr| match &e.node {
        ExprNode::Column(c) => Some(c.by_identifier),
        _ => None,
    };
    assert_eq!(marked(&exprs[0]), Some(true));
    assert_eq!(marked(&exprs[1]), Some(false));
    assert_eq!(marked(&exprs[2]), None);
    let plain = krishiv_sql::lower_with(&query, SourceId(0), false).unwrap();
    let PlanNode::Project { exprs, .. } = &plain.root().node else {
        panic!("expected a Project, got {}", plain.root());
    };
    assert_eq!(marked(&exprs[0]), Some(false));
}

/// A substituted name followed by field parts (`IDENTIFIER('c1').c2` is `` `c1`.c2 ``) is marked:
/// Spark 4.2.0 names it `c1.c2`. A column that only contains the span, or starts inside it, is not.
#[test]
fn a_substituted_name_with_fields_after_it_is_marked() {
    let sql = "SELECT `c1`.c2, `c1`.c2.c3, x.`c1` FROM t";
    let query = krishiv_sql::parse_query(sql, &ParserOptions::default()).unwrap();
    let spans = [
        krishiv_sql::ast::Span::new(7, 11),
        krishiv_sql::ast::Span::new(16, 20),
        krishiv_sql::ast::Span::new(30, 34),
    ];
    let plan = krishiv_sql::lower_identified(&query, SourceId(0), false, &spans).unwrap();
    let PlanNode::Project { exprs, .. } = &plan.root().node else {
        panic!("expected a Project, got {}", plan.root());
    };
    let marked: Vec<_> = exprs
        .iter()
        .map(|e| match &e.node {
            ExprNode::Column(c) => Some(c.by_identifier),
            _ => None,
        })
        .collect();
    assert_eq!(marked, [Some(true), Some(true), Some(false)]);
}

/// The plan line of `sql`, or its error, for exact comparisons.
fn line(sql: &str) -> String {
    match lower(sql) {
        Ok(p) => p.root().to_string().replace('\n', " | "),
        Err(e) => format!("ERR {}", e.error),
    }
}

#[test]
fn cube_and_rollup_list_their_grouping_sets_in_spark_order() {
    assert_eq!(
        line("SELECT a, b FROM t GROUP BY CUBE (a, b)"),
        "Aggregate [a, b] GROUPING SETS ((a, b), (a), (b), ()), [a, b] | +- Read t | "
    );
    assert_eq!(
        line("SELECT a, b, c FROM t GROUP BY CUBE (a, b, c)"),
        "Aggregate [a, b, c] GROUPING SETS ((a, b, c), (a, b), (a, c), (a), (b, c), (b), (c), ()), [a, b, c] \
         | +- Read t | "
    );
    assert_eq!(
        line("SELECT a, b FROM t GROUP BY ROLLUP (a, b)"),
        "Aggregate [a, b] GROUPING SETS ((a, b), (a), ()), [a, b] | +- Read t | "
    );
}

#[test]
fn only_inner_cross_and_left_joins_may_be_lateral() {
    let lateral = |kind: &str| line(&format!("SELECT * FROM t {kind} JOIN LATERAL (SELECT a) s"));
    for (kind, name) in [
        ("RIGHT", "RIGHT OUTER"),
        ("FULL", "FULL OUTER"),
        ("LEFT SEMI", "LEFT SEMI"),
        ("LEFT ANTI", "LEFT ANTI"),
    ] {
        let got = lateral(kind);
        assert!(
            got.starts_with(&format!(
                "ERR [INVALID_LATERAL_JOIN_TYPE] The {name} JOIN with LATERAL correlation is not allowed"
            )),
            "{kind}: {got}"
        );
    }
    assert!(lateral("LEFT").contains("Join LateralLeft"));
    assert!(lateral("").contains("Join LateralInner"));
    assert!(lateral("CROSS").contains("Join LateralInner"));
    // without LATERAL the same kinds are ordinary joins
    assert!(!line("SELECT * FROM t RIGHT JOIN u ON t.a = u.a").starts_with("ERR"));
}

#[test]
fn generators_in_from_are_generate_nodes_and_other_table_functions_are_not() {
    for (sql, node) in [
        (
            "SELECT * FROM explode(array(1, 2))",
            "Generate explode(array(1, 2)), false, []",
        ),
        (
            "SELECT * FROM posexplode(array(1, 2))",
            "Generate posexplode(array(1, 2)), false, []",
        ),
        (
            "SELECT * FROM inline(array(struct(1, 2)))",
            "Generate inline(array(struct(1, 2))), false, []",
        ),
        (
            "SELECT * FROM stack(2, 1, 2)",
            "Generate inline(stack(2, 1, 2)), false, []",
        ),
        (
            "SELECT * FROM explode_outer(array(1, 2))",
            "Generate explode_outer(array(1, 2)), true, []",
        ),
    ] {
        assert_eq!(
            line(sql),
            format!("Project [*] | +- {node} |    +- OneRowRelation | "),
            "{sql}"
        );
    }
    // `stack_outer` is not a generator; it stays a table function
    assert_eq!(
        line("SELECT * FROM stack_outer(2, 1, 2)"),
        "Project [*] | +- TableFunction stack_outer [2, 1, 2] | "
    );
    // a table argument and a four-part name are refused
    for sql in ["SELECT * FROM f(TABLE(t))", "SELECT * FROM a.b.c.d(1)"] {
        assert!(
            line(sql).contains("table-valued functions"),
            "{sql}: {}",
            line(sql)
        );
    }
}

#[test]
fn inline_tables_name_their_columns_and_paths_are_told_from_names() {
    assert_eq!(
        line("SELECT * FROM VALUES (1, 2) AS v"),
        "Project [*] | +- SubqueryAlias v |    +- InlineTable [col1, col2] (1 rows) | "
    );
    assert_eq!(
        line("SELECT * FROM VALUES (1, 2) AS v (x, y)"),
        "Project [*] | +- SubqueryAlias v (x, y) |    +- InlineTable [x, y] (1 rows) | "
    );
    // a quoted path is read as files; a quoted name that is not a path stays a table name
    assert_eq!(
        line("SELECT * FROM parquet.`s3://b/p`"),
        "Project [*] | +- Read parquet [s3://b/p] | "
    );
    assert_eq!(
        line("SELECT * FROM parquet.`not a path`"),
        "Project [*] | +- Read parquet.`not a path` | "
    );
}

#[test]
fn a_statement_nested_past_the_inline_limit_still_parses() {
    // 16 levels is the limit for the calling thread's stack; 900 is below the engine's 1,000 and needs the big one
    let deep = format!("SELECT {}1{} FROM t", "(".repeat(900), ")".repeat(900));
    assert!(
        lower(&deep).is_ok(),
        "{}",
        line(&deep).chars().take(200).collect::<String>()
    );
}

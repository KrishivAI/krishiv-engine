// SPDX-License-Identifier: Apache-2.0
//! Spark 4.3's `ASOF JOIN ... MATCH_CONDITION (...)`: the grammar, the parse
//! errors and the `AstBuilder` checks of Spark's `join-asof-grammar.sql`,
//! `join-asof-errors.sql` and `join-asof-gating.sql` goldens.

use super::*;
use crate::ast::{AsOfFilter, JoinConstraint, JoinKind, RelationExt};

fn on() -> ParserOptions {
    ParserOptions {
        asof_join: true,
        ..ParserOptions::default()
    }
}

/// The first join of `sql`, parsed with the flag on.
fn join(sql: &str) -> ast::Join {
    let q = match parse_query(sql, &on()) {
        Ok(q) => q,
        Err(e) => panic!("{sql}: {}", e.error),
    };
    let SetExprKind::Select(s) = q.body.kind else {
        panic!("{sql}")
    };
    let rel = s.from.unwrap().relations.into_iter().next().unwrap();
    match rel.extensions.into_iter().next() {
        Some(RelationExt::Join(j)) => j,
        other => panic!("{sql}: {other:?}"),
    }
}

/// Class and message parameters of `sql`'s error under `opts`.
fn error(sql: &str, opts: &ParserOptions) -> (String, Vec<(String, String)>, Span) {
    let e = err(sql, opts);
    let params = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    (e.error.error_class().name().to_owned(), params, e.span)
}

fn param(params: &[(String, String)], key: &str) -> String {
    params
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
        .unwrap_or_default()
}

#[test]
fn asof_join_kinds_and_criteria() {
    let sql = "SELECT * FROM trades t ASOF JOIN quotes q\n  MATCH_CONDITION (t.ts >= q.ts)\n  ON t.s = q.s";
    let j = join(sql);
    assert_eq!(j.kind, JoinKind::Inner);
    assert!(!j.natural && !j.lateral);
    let JoinConstraint::AsOf(c) = &j.constraint else {
        panic!("{j:?}")
    };
    assert_eq!(c.match_condition.span.text(sql), "t.ts >= q.ts");
    let Some(AsOfFilter::On(e)) = &c.filter else {
        panic!("{c:?}")
    };
    assert_eq!(e.span.text(sql), "t.s = q.s");
    assert!(j.span.text(sql).starts_with("ASOF JOIN quotes q"));
    assert!(c.span.text(sql).starts_with("MATCH_CONDITION"));

    for (kind, expected) in [
        ("INNER", JoinKind::Inner),
        ("LEFT", JoinKind::Left),
        ("LEFT OUTER", JoinKind::Left),
    ] {
        let sql =
            format!("SELECT * FROM t {kind} ASOF JOIN q MATCH_CONDITION (t.a < q.a) USING (k)");
        let j = join(&sql);
        assert_eq!(j.kind, expected, "{sql}");
        assert!(j.span.text(&sql).starts_with(kind), "{sql}");
        let JoinConstraint::AsOf(c) = &j.constraint else {
            panic!("{sql}")
        };
        let Some(AsOfFilter::Using(cols)) = &c.filter else {
            panic!("{sql}")
        };
        assert_eq!(cols.len(), 1);
    }
    // No alias on either side: `ASOF` and `MATCH_CONDITION` are not aliases.
    let j = join("SELECT * FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.a > quotes.a)");
    assert!(j.right.alias.is_none());
    let JoinConstraint::AsOf(c) = &j.constraint else {
        panic!()
    };
    assert!(c.filter.is_none());
    // A pipe join.
    let q = parse_query(
        "FROM t |> ASOF JOIN q MATCH_CONDITION (t.a >= q.a) ON t.k = q.k",
        &on(),
    )
    .unwrap();
    let SetExprKind::Pipe(pipe) = &q.body.kind else {
        panic!("{q:?}")
    };
    let [op] = pipe.ops.as_slice() else {
        panic!("{:?}", pipe.ops)
    };
    let ast::PipeOpKind::Join(j) = &op.kind else {
        panic!("{op:?}")
    };
    let JoinConstraint::AsOf(c) = &j.constraint else {
        panic!("{j:?}")
    };
    assert!(matches!(c.filter, Some(AsOfFilter::On(_))), "{c:?}");
}

#[test]
fn asof_join_parse_errors() {
    for kind in ["RIGHT", "FULL", "CROSS", "LEFT SEMI", "ANTI", "RIGHT OUTER"] {
        let sql = format!("SELECT * FROM t x {kind} ASOF JOIN q MATCH_CONDITION (x.a >= q.a)");
        let (class, params, _) = error(&sql, &on());
        assert_eq!(class, "PARSE_SYNTAX_ERROR", "{sql}");
        assert_eq!(param(&params, "error"), "'ASOF'", "{sql}");
        assert_eq!(param(&params, "hint"), ": extra input 'ASOF'", "{sql}");
    }
    let sql = "SELECT * FROM trades NATURAL ASOF JOIN quotes MATCH_CONDITION (a >= b)";
    let (class, params, _) = error(sql, &on());
    assert_eq!(class, "PARSE_SYNTAX_ERROR");
    assert_eq!(param(&params, "hint"), ": extra input 'ASOF'");
    for (sql, at) in [
        (
            "SELECT * FROM trades t ASOF JOIN quotes q ON t.s = q.s",
            "'ON'",
        ),
        (
            "SELECT * FROM t ASOF JOIN q MATCH_CONDITION (t.a >= q.a) ON t.k = q.k USING (k)",
            "'USING'",
        ),
    ] {
        let (class, params, _) = error(sql, &on());
        assert_eq!(class, "PARSE_SYNTAX_ERROR", "{sql}");
        assert_eq!(param(&params, "error"), at, "{sql}");
        assert_eq!(param(&params, "hint"), "", "{sql}");
    }
}

#[test]
fn asof_join_match_operator() {
    for (cond, operator) in [
        ("t.a = q.a", "="),
        ("t.a == q.a", "="),
        ("t.a <> q.a", "<>"),
        ("t.a != q.a", "!="),
        ("t.a <=> q.a", "<=>"),
        ("t.a IS DISTINCT FROM q.a", "IS DISTINCT FROM"),
        ("t.a IS NOT DISTINCT FROM q.a", "IS NOT DISTINCT FROM"),
        ("t.a >= q.a AND t.k = q.k", "AND"),
        ("t.a >= q.a OR t.k = q.k", "OR"),
        ("NOT (t.a >= q.a)", ">="),
        ("t.a BETWEEN q.a AND q.b", "t.a BETWEEN q.a AND q.b"),
        // Spark echoes the text as written; a string literal's text is never echoed (`?`).
        ("'canary-secret'", "?"),
        ("t.s LIKE 'canary%'", "t.s LIKE ?"),
        ("t.a IN (1, 2)", "t.a IN (1, 2)"),
        // An operator name inside a string is no operator, and stays hidden.
        ("t.s LIKE '>='", "t.s LIKE ?"),
    ] {
        let sql =
            format!("SELECT * FROM t ASOF JOIN q\n  MATCH_CONDITION ({cond})\n  ON t.k = q.k");
        let (class, params, span) = error(&sql, &on());
        assert_eq!(class, "ASOF_JOIN_MATCH_CONDITION_INVALID_OPERATOR", "{sql}");
        assert_eq!(param(&params, "operator"), operator, "{sql}");
        assert_eq!(
            span.text(&sql),
            format!("ASOF JOIN q\n  MATCH_CONDITION ({cond})\n  ON t.k = q.k"),
        );
    }
    for cond in [
        "t.a >= q.a",
        "t.a > q.a",
        "t.a <= q.a",
        "t.a < q.a",
        "((t.a < q.a))",
    ] {
        join(&format!(
            "SELECT * FROM t ASOF JOIN q MATCH_CONDITION ({cond})"
        ));
    }
}

#[test]
fn asof_join_gate() {
    let sql = "SELECT * FROM VALUES (1, 'a') AS t(ts, s) ASOF JOIN\n     VALUES (0, 'a') AS q(ts, s)\n  MATCH_CONDITION (t.ts >= q.ts)\n  ON t.s = q.s";
    let (class, params, span) = error(sql, &ParserOptions::default());
    assert_eq!(class, "UNSUPPORTED_FEATURE.ASOF_JOIN");
    assert_eq!(
        param(&params, "config"),
        "\"spark.sql.join.asofJoin.enabled\""
    );
    assert_eq!(span.range(), sql.find("ASOF").unwrap()..sql.len());
    assert!(span.text(sql).starts_with("ASOF JOIN\n"));
    // The flag is read from the session's configs.
    let opts =
        ParserOptions::from_conf(|k| (k == "spark.sql.join.asofJoin.enabled").then_some("true"));
    assert!(opts.asof_join && !ParserOptions::default().asof_join);
    // Off, `ASOF` is still Spark 4.2's alias: `t AS ASOF`, then an ordinary join.
    let q = parse_query(
        "SELECT * FROM t ASOF JOIN q ON true",
        &ParserOptions::default(),
    )
    .unwrap();
    let SetExprKind::Select(s) = q.body.kind else {
        panic!()
    };
    let rel = s.from.unwrap().relations.into_iter().next().unwrap();
    let alias = rel.primary.alias.as_ref().and_then(|a| a.name.as_ref());
    assert_eq!(alias.map(|n| n.value.as_str()), Some("ASOF"), "{rel:?}");
    let Some(RelationExt::Join(j)) = rel.extensions.first() else {
        panic!("{rel:?}")
    };
    assert!(matches!(j.constraint, JoinConstraint::On(_)), "{j:?}");
    // On, it lowers to the IR's as-of join, the operands as written.
    let q = parse_query(
        "SELECT * FROM t LEFT ASOF JOIN q MATCH_CONDITION (q.a < t.a) USING (k)",
        &on(),
    )
    .unwrap();
    let plan = lower(&q, krishiv_plan::SourceId(0)).unwrap();
    let text = format!("{:?}", plan);
    assert!(text.contains("AsOfJoin"), "{text}");
    assert!(
        text.contains("LeftOuter") && text.contains("Forward"),
        "{text}"
    );
}

/// Inside a query on the right side, `match_condition (a)` is a table alias with a column list
/// again: the as-of side's reading of the word ends at the subquery.
#[test]
fn asof_join_right_subquery_reads_match_condition_as_an_alias() {
    let sql = "SELECT * FROM t ASOF JOIN (SELECT * FROM r match_condition (a)) q\n  \
               MATCH_CONDITION (t.ts >= q.a) ON t.s = q.s";
    let j = join(sql);
    assert!(matches!(j.constraint, JoinConstraint::AsOf(_)), "{j:?}");
}

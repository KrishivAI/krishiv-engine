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
    // The flag is read from the session's configs (the grammar from its Spark profile).
    let opts =
        ParserOptions::from_conf(|k| (k == "spark.sql.join.asofJoin.enabled").then_some("true"));
    assert!(opts.asof_join && !ParserOptions::default().asof_join);
    // `ASOF JOIN q ON ..` names `t` `ASOF` (Spark 4.2 and 4.3 alike), then an ordinary join.
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
    let JoinConstraint::AsOf(c) = &j.constraint else {
        panic!("{j:?}")
    };
    assert_eq!(c.match_condition.span.text(sql), "t.ts >= q.a");
    let Some(AsOfFilter::On(e)) = &c.filter else {
        panic!("{c:?}")
    };
    assert_eq!(e.span.text(sql), "t.s = q.s");
    assert_eq!(
        j.right
            .alias
            .as_ref()
            .and_then(|a| a.name.as_ref())
            .map(|n| n.value.as_str()),
        Some("q")
    );
    // Inside the subquery, `r` is aliased `match_condition` with the column list `(a)`.
    let ast::TableFactorKind::Subquery(inner) = &j.right.kind else {
        panic!("the right side is a subquery: {j:?}")
    };
    let ast::SetExprKind::Select(s) = &inner.body.kind else {
        panic!("{inner:?}")
    };
    let r = &s.from.as_ref().unwrap().relations[0].primary;
    let alias = r.alias.as_ref().unwrap();
    assert_eq!(
        alias.name.as_ref().map(|n| n.value.as_str()),
        Some("match_condition")
    );
    let columns: Vec<&str> = alias.columns.iter().map(|c| c.value.as_str()).collect();
    assert_eq!(columns, ["a"]);
}

/// Inside parentheses the as-of right side is not that side: `MATCH_CONDITION` there is an
/// ordinary alias again.
#[test]
fn asof_right_side_parenthesized_relation_clears_the_flag() {
    let sql = "SELECT * FROM l AS l ASOF JOIN (r MATCH_CONDITION (x)) AS rr \
               MATCH_CONDITION (l.x >= rr.x)";
    let j = join(sql);
    let JoinConstraint::AsOf(c) = &j.constraint else {
        panic!("{j:?}")
    };
    assert_eq!(c.match_condition.span.text(sql), "l.x >= rr.x");
    assert!(c.filter.is_none(), "{c:?}");
    // The nested relation is `r` aliased `MATCH_CONDITION` with the column list `(x)`, as Spark
    // 4.3.0 reads it (review of #990).
    let ast::TableFactorKind::Nested(inner) = &j.right.kind else {
        panic!("the right side is a parenthesized relation: {j:?}")
    };
    let alias = inner.primary.alias.as_ref().unwrap();
    assert_eq!(
        alias.name.as_ref().map(|n| n.value.as_str()),
        Some("MATCH_CONDITION")
    );
    let columns: Vec<&str> = alias.columns.iter().map(|c| c.value.as_str()).collect();
    assert_eq!(columns, ["x"]);
}

/// Spark 4.3's grammar (the session's 4.3 profile), with the flag `on` or off.
fn grammar(on: bool) -> ParserOptions {
    ParserOptions {
        asof_grammar: true,
        asof_join: on,
        ..ParserOptions::default()
    }
}

/// The left relation's alias and the first extension of `sql`'s first relation.
fn left(sql: &str, opts: &ParserOptions) -> (Option<String>, Option<RelationExt>) {
    let q = parse_query(sql, opts).unwrap_or_else(|e| panic!("{sql}: {}", e.error));
    let SetExprKind::Select(s) = q.body.kind else {
        panic!("{sql}")
    };
    let rel = s.from.unwrap().relations.into_iter().next().unwrap();
    let alias = rel
        .primary
        .alias
        .as_ref()
        .and_then(|a| a.name.as_ref())
        .map(|n| n.value.clone());
    (alias, rel.extensions.into_iter().next())
}

/// SMOKE-asof-error (Spark 4.3.0 runs, flag off and on, 2026-11-24): with Spark 4.3's grammar an
/// as-of join with no alias on its left side is still an as-of join, so with the flag off the
/// error is `UNSUPPORTED_FEATURE.ASOF_JOIN`, not a syntax error at `MATCH_CONDITION (trades.`.
#[test]
fn asof_join_without_a_left_alias_is_gated_not_misread() {
    for l in [
        "trades",
        "trades AS t",
        "trades t",
        "(SELECT * FROM trades) t",
    ] {
        for r in ["quotes", "quotes q", "quotes AS q"] {
            let sql = format!(
                "SELECT * FROM {l} ASOF JOIN {r} MATCH_CONDITION (trades.t >= quotes.t) \
                 ON trades.s = quotes.s"
            );
            let (class, params, span) = error(&sql, &grammar(false));
            assert_eq!(class, "UNSUPPORTED_FEATURE.ASOF_JOIN", "{sql}");
            assert_eq!(
                param(&params, "config"),
                "\"spark.sql.join.asofJoin.enabled\""
            );
            assert!(span.text(&sql).starts_with("ASOF JOIN"), "{sql}");
            let (_, ext) = left(&sql, &grammar(true));
            assert!(
                matches!(ext, Some(RelationExt::Join(ref j)) if matches!(j.constraint, JoinConstraint::AsOf(_))),
                "{sql}: {ext:?}"
            );
        }
    }
    // `LEFT ASOF JOIN ... USING`, a chained as-of join, an invalid operator (the gate comes first,
    // as in `withAsOfJoin`) and a dotted one-sided condition: gated.
    for sql in [
        "SELECT * FROM t LEFT ASOF JOIN q MATCH_CONDITION (t.a >= q.a) USING (s)",
        "SELECT * FROM t ASOF JOIN q ASOF JOIN r MATCH_CONDITION (q.a >= r.a)",
        "SELECT * FROM t ASOF JOIN q MATCH_CONDITION (t.a = q.a)",
        "SELECT * FROM t ASOF JOIN q MATCH_CONDITION (q.a)",
    ] {
        assert_eq!(
            error(sql, &grammar(false)).0,
            "UNSUPPORTED_FEATURE.ASOF_JOIN",
            "{sql}"
        );
    }
    // Spark 4.2's grammar (the parser's defaults) keeps 4.2's reading: `ASOF` is `trades`' alias.
    let sql = "SELECT * FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.t >= quotes.t)";
    assert_eq!(
        error(sql, &ParserOptions::default()).0,
        "PARSE_SYNTAX_ERROR"
    );
}

/// Where both readings parse, ANTLR takes the alias: `ASOF` names the left side and
/// `MATCH_CONDITION (a)` is the right side's alias with a column list (Spark 4.3.0, flag off and
/// on). With no `MATCH_CONDITION` at all `ASOF` is the alias too.
#[test]
fn asof_join_ambiguous_with_an_alias_column_list_reads_the_alias() {
    for on in [false, true] {
        for sql in [
            "SELECT * FROM t ASOF JOIN one MATCH_CONDITION (a)",
            "SELECT * FROM t ASOF JOIN one MATCH_CONDITION (a) ON true",
            "SELECT * FROM t ASOF JOIN one MATCH_CONDITION (a) USING (a)",
            "SELECT * FROM t ASOF JOIN q MATCH_CONDITION (a, s) ON ASOF.a = MATCH_CONDITION.a",
            "SELECT * FROM t ASOF JOIN q ON true",
        ] {
            let (alias, ext) = left(sql, &grammar(on));
            assert_eq!(alias.as_deref(), Some("ASOF"), "{sql} on={on}");
            let Some(RelationExt::Join(j)) = ext else {
                panic!("{sql}: {ext:?}")
            };
            assert!(
                !matches!(j.constraint, JoinConstraint::AsOf(_)),
                "{sql} on={on}"
            );
        }
    }
}

/// A long chain of `ASOF JOIN`s is read once: linear time, and no nesting (a 2 MiB stack).
#[test]
fn asof_join_chains_are_read_in_linear_time() {
    let mut sql = String::from("SELECT * FROM t");
    for _ in 0..4000 {
        sql.push_str(" ASOF JOIN t");
    }
    let handle = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || {
            let started = std::time::Instant::now();
            let _ = parse_query(&sql, &grammar(true));
            started.elapsed()
        })
        .unwrap();
    let took = handle.join().unwrap();
    assert!(took < std::time::Duration::from_secs(2), "{took:?}");
}

/// `| ASOF` starts a pipe operator in Spark 4.3 (`isOperatorPipeStart`), flag or not; in 4.2 it is
/// a bitwise or.
#[test]
fn pipe_asof_follows_the_grammar() {
    let sql = "SELECT x | asof FROM (SELECT 1 x, 2 asof)";
    assert_eq!(error(sql, &grammar(false)).0, "PARSE_SYNTAX_ERROR");
    assert!(parse_query(sql, &ParserOptions::default()).is_ok());
}

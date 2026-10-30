// SPDX-License-Identifier: Apache-2.0
//! Common-subexpression elimination: what hoists and what does not, rule
//! idempotence, the per-rule switch, and results with the rule on and off.
//!
//! Each negative case (never hoisted) shares its test with a positive
//! control in the same query, so the test fails both without the rule
//! (nothing hoists) and with a naive rule (everything hoists).

use krishiv_optimizer::{Options, optimize, optimize_with_options};
use krishiv_plan::{LogicalPlan, Plan};

use crate::support::{self, data, eval};

/// The occurrences of `needle` in the plan text.
fn occurrences(plan: &LogicalPlan, needle: &str) -> usize {
    plan.to_string().matches(needle).count()
}

/// The number of extracted expressions: one `AS cse_*` alias each.
fn extractions(plan: &LogicalPlan) -> usize {
    occurrences(plan, "AS cse_")
}

/// Optimises `sql` with CSE on and off.
fn both(sql: &str) -> (Plan, Plan) {
    let ctx = support::ctx();
    let plan = support::resolve(sql, &ctx);
    let on = optimize(&plan).unwrap();
    let off = optimize_with_options(
        &plan,
        &support::AllDistinct,
        &Options {
            cse: false,
            ..Options::default()
        },
    )
    .unwrap();
    (on, off)
}

/// `on` and `off` return the same rows on `db`: multisets, compared sorted
/// (the interpreter is exact fixed point, so any float tolerance holds).
fn assert_same_rows(db: &eval::Db, on: &LogicalPlan, off: &LogicalPlan, what: &str) {
    assert_eq!(
        eval::rows(db, off),
        eval::rows(db, on),
        "{what}: results differ with CSE on"
    );
}

/// A `CASE WHEN b <> 0 THEN a / b END` appearing twice is not hoisted: the
/// division is evaluated only for rows that reach the branch. The query
/// also repeats `a + b` unconditionally, which is hoisted, so the test
/// fails without the rule too.
#[test]
fn case_guarded_division_twice_is_not_hoisted() {
    let (on, off) = both(
        "SELECT CASE WHEN b <> 0 THEN a / b ELSE NULL END AS x, CASE WHEN b <> 0 THEN a / b ELSE NULL END AS y, (a + b) + (a + b) AS z FROM t",
    );
    let text = on.root().to_string();
    // The guarded CASE and `a + b`, then (second round) the `CAST(b)` both
    // now read in the projection below.
    assert_eq!(extractions(on.root()), 3, "{text}");
    assert_eq!(
        occurrences(on.root(), "CASE WHEN"),
        1,
        "the whole guarded CASE hoists as one unit, its division staying guarded:\n{text}"
    );
    assert_eq!(extractions(off.root()), 0, "the switch disables the rule");
    assert_same_rows(&div_db(), on.root(), off.root(), "guarded division");
}

/// Rows for division tests: `b` holds only strings the reference
/// interpreter can cast (`"2"`, `"0"`) or NULL, including a zero that the
/// `CASE` guard skips.
fn div_db() -> eval::Db {
    let mut db = eval::Db::default();
    let rows = vec![
        vec![eval::V::int(4), eval::V::Str("2".into()), eval::V::int(1)],
        vec![eval::V::int(4), eval::V::Str("0".into()), eval::V::int(1)],
        vec![eval::V::int(-4), eval::V::Str("0".into()), eval::V::int(2)],
        vec![eval::V::Null, eval::V::Str("2".into()), eval::V::int(2)],
        vec![eval::V::int(4), eval::V::Null, eval::V::int(3)],
    ];
    db.tables.insert(
        "t".to_owned(),
        (vec!["a".to_owned(), "b".to_owned(), "c".to_owned()], rows),
    );
    db
}

/// `a / b` evaluated surely once and once inside a `CASE` branch is
/// hoisted: every row evaluates it anyway.
#[test]
fn unconditional_once_plus_conditional_is_hoisted() {
    let (on, off) = both("SELECT a / b AS x, CASE WHEN a > 0 THEN a / b ELSE NULL END AS y FROM t");
    let text = on.root().to_string();
    assert_eq!(extractions(on.root()), 1, "{text}");
    assert!(text.contains("THEN cse_"), "{text}");
    assert_eq!(occurrences(on.root(), " / "), 1, "{text}");
    assert_same_rows(&div_db(), on.root(), off.root(), "sure plus conditional");
}

/// `rand()` twice stays two calls: nondeterministic calls are never
/// extracted. The query also repeats `a + 1` unconditionally, which is.
#[test]
fn rand_twice_stays_two_calls() {
    let (on, off) = both("SELECT rand() AS x, rand() AS y, (a + 1) + (a + 1) AS z FROM t");
    assert_eq!(occurrences(on.root(), "rand()"), 2, "{}", on.root());
    assert_eq!(occurrences(off.root(), "rand()"), 2, "{}", off.root());
    assert_eq!(extractions(on.root()), 1, "{}", on.root());
}

/// A `try_*` subtree is never extracted, even when repeated: the query
/// also repeats `a + 1`, which is.
#[test]
fn try_subtree_is_never_extracted() {
    let (on, _) =
        both("SELECT try_add(a, a) AS x, try_add(a, a) AS y, (c + 1) + (c + 1) AS z FROM t");
    let text = on.root().to_string();
    assert_eq!(occurrences(on.root(), "try_add"), 2, "{text}");
    assert_eq!(extractions(on.root()), 1, "{text}");
}

/// Nested and shared (`Arc`) subexpressions are handled: `((a + b) * c)`
/// twice becomes one shared computation.
#[test]
fn nested_and_shared_subexpressions() {
    let (on, off) = both("SELECT ((a + b) * c) + ((a + b) * c) AS x FROM t");
    let text = on.root().to_string();
    assert_eq!(extractions(on.root()), 1, "{text}");
    assert!(!text.contains("((a + b) * c) + ((a + b) * c)"), "{text}");
    for seed in 1..=8 {
        assert_same_rows(&data::small(seed), on.root(), off.root(), "nested");
    }
}

/// Aggregate arguments are deduplicated: `sum(a * b)` and `count(a * b)`
/// compute the product once.
#[test]
fn aggregate_arguments_are_deduplicated() {
    let (on, off) = both("SELECT sum(a * b) AS s, count(a * b) AS n FROM t GROUP BY c");
    let text = on.root().to_string();
    assert_eq!(extractions(on.root()), 1, "{text}");
    assert!(text.contains("sum(cse_"), "{text}");
    assert!(text.contains("count(cse_"), "{text}");
    for seed in 1..=8 {
        assert_same_rows(&data::small(seed), on.root(), off.root(), "aggregate args");
    }
}

/// Aggregate arguments under `FILTER (WHERE ...)` are not hoisted: they
/// are evaluated only for rows the filter keeps. A post-aggregation scalar
/// over a grouping key is left alone too (it is evaluated per group, not
/// per row). The filter predicate repeats unconditionally and is hoisted,
/// so the test fails without the rule too.
#[test]
fn filter_where_arguments_are_not_hoisted() {
    let (on, off) = both(
        "SELECT sum(a + c) FILTER (WHERE c > 0) AS s, sum(a + c) FILTER (WHERE c > 0) AS t, (c + 1) + (c + 1) AS u FROM t GROUP BY c",
    );
    let text = on.root().to_string();
    assert_eq!(extractions(on.root()), 1, "{text}");
    assert_eq!(occurrences(on.root(), "FILTER (WHERE cse_"), 2, "{text}");
    assert_eq!(occurrences(on.root(), "c#2 + 1L"), 2, "{text}");
    for seed in 1..=8 {
        assert_same_rows(&data::small(seed), on.root(), off.root(), "filter args");
    }
}

/// `nullif`/`nvl`/`ifnull`/`nvl2` meet their `CASE`/`coalesce` forms:
/// `nvl(a, b)` twice and `coalesce(a, b)` once compute once.
#[test]
fn null_forms_meet_coalesce() {
    let (on, off) = both("SELECT nvl(a, b) AS x, nvl(a, b) AS y, coalesce(a, b) AS z FROM t");
    let text = on.root().to_string();
    assert_eq!(extractions(on.root()), 1, "{text}");
    assert!(!text.contains("nvl("), "null forms resolve first:\n{text}");
    assert_eq!(
        occurrences(on.root(), "coalesce("),
        1,
        "all three spellings compute one coalesce:\n{text}"
    );
    for seed in 1..=8 {
        assert_same_rows(&data::small(seed), on.root(), off.root(), "null forms");
    }
}

/// A scalar subquery repeated twice is never extracted: subquery
/// expressions stay where the analyzer put them. A plain column both
/// subqueries read is hoisted, so the test fails without the rule too.
#[test]
fn subquery_expressions_are_never_extracted() {
    let (on, off) = both(
        "SELECT (SELECT max(a) FROM u) AS x, (SELECT max(a) FROM u) AS y, (a + b) + (a + b) AS z FROM t",
    );
    let text = on.root().to_string();
    assert_eq!(extractions(on.root()), 1, "{text}");
    assert_eq!(occurrences(on.root(), "scalar-subquery"), 2, "{text}");
    for seed in 1..=8 {
        assert_same_rows(&data::small(seed), on.root(), off.root(), "subqueries");
    }
}

/// Q1's projection computes the repeated product once.
#[test]
fn tpch_q1_computes_the_product_once() {
    let ctx = support::ctx();
    let plan = support::resolve(&support::tpch_sql(1), &ctx);
    let optimized = optimize(&plan).unwrap();
    let text = optimized.root().to_string();
    assert_eq!(
        occurrences(optimized.root(), "(l_extendedprice#5 * (1 - l_discount#6))"),
        1,
        "{text}"
    );
    assert!(text.contains("AS cse_"), "{text}");
}

/// The rule is idempotent: optimizing twice changes nothing, on plans
/// with and without extractions.
#[test]
fn rule_is_idempotent() {
    let ctx = support::ctx();
    for sql in [
        "SELECT ((a + b) * c) + ((a + b) * c) AS x FROM t",
        "SELECT a / b AS x, CASE WHEN a > 0 THEN a / b ELSE NULL END AS y FROM t",
        "SELECT sum(a * b) AS s, count(a * b) AS n FROM t GROUP BY c",
        "SELECT a FROM t WHERE (a + b) > 1 AND (a + b) < 3",
        "SELECT (a + b) * 2 AS p, CASE WHEN c > 0 THEN (a + b) * 2 END AS q, a + b AS r, CASE WHEN c > 0 THEN a + b END AS s FROM t",
    ] {
        let plan = support::resolve(sql, &ctx);
        let once = optimize(&plan).unwrap();
        let twice = optimize(&once).unwrap();
        assert!(twice.structurally_eq(&once), "{sql}:\n{}", twice.root());
    }
}

/// A filter whose predicate is rewritten keeps its input's columns, so a
/// `UNION ALL` branch (which reads its inputs by position) still resolves.
#[test]
fn a_filtered_union_branch_keeps_its_width() {
    let (on, off) = both(
        "SELECT a, b, c FROM t WHERE (a + b) > 1 AND (a + b) < 3 UNION ALL SELECT a, b, c FROM t",
    );
    let text = on.root().to_string();
    assert_eq!(extractions(on.root()), 1, "{text}");
    assert!(on.root().resolved_output().is_some(), "{text}");
    assert_eq!(on.root().resolved_output(), off.root().resolved_output());
    for seed in 1..=8 {
        assert_same_rows(&data::small(seed), on.root(), off.root(), "filtered union");
    }
}

/// A post-aggregation scalar over a grouping key is evaluated above the
/// aggregate, where a `cse_*` column of the input is not available; only
/// the aggregate calls' own arguments are rewritten.
#[test]
fn post_aggregation_scalars_stay_above_the_aggregate() {
    let (on, off) = both("SELECT c + 1 AS k, sum(c + 1) AS s, count(c + 1) AS n FROM t GROUP BY c");
    let text = on.root().to_string();
    assert_eq!(extractions(on.root()), 1, "{text}");
    assert_eq!(
        occurrences(on.root(), "c#2 + 1L"),
        2,
        "the output `c + 1 AS k` and the extraction's definition:\n{text}"
    );
    for seed in 1..=8 {
        assert_same_rows(
            &data::small(seed),
            on.root(),
            off.root(),
            "post-aggregation",
        );
    }
}

/// `IN` list items and the upper bound of `BETWEEN` are evaluated only when
/// the value is not NULL and no earlier item matched (or the lower bound
/// held), so repeating one there never hoists it. The unconditional
/// repeat of `a + b` is the positive control.
#[test]
fn in_items_and_between_upper_bounds_are_not_hoisted() {
    let (on, _) = both(
        "SELECT a IN (0, c / b) AS p, c IN (0, c / b) AS q, a BETWEEN 0 AND c / b AS r, c BETWEEN 0 AND c / b AS u, (a + b) + (a + b) AS z FROM t",
    );
    let text = on.root().to_string();
    // Casts of the columns hoist; no division does.
    let hoisted_divisions = text
        .lines()
        .filter(|l| l.contains("AS cse_") && l.contains(" / "))
        .count();
    assert_eq!(hoisted_divisions, 0, "{text}");
    assert!(extractions(on.root()) >= 1, "{text}");
}

/// The per-rule switch really disables the rule: with it off no `cse_*`
/// column appears, while outputs and (unoptimised) results are unchanged.
#[test]
fn switch_disables_the_rule() {
    let ctx = support::ctx();
    let plan = support::resolve("SELECT ((a + b) * c) + ((a + b) * c) AS x FROM t", &ctx);
    let off = optimize_with_options(
        &plan,
        &support::AllDistinct,
        &Options {
            cse: false,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(extractions(off.root()), 0);
    assert_eq!(off.root().resolved_output(), plan.root().resolved_output());
    for seed in 1..=8 {
        let db = data::small(seed);
        assert_eq!(eval::rows(&db, off.root()), eval::rows(&db, plan.root()));
    }
}

/// Results with the rule on equal results with it off, as multisets, over
/// the in-process TPC-H generator: a tiny scale by default, and the full
/// scale when `KRISHIV_CSE_FULL` is set (not part of default CI). TPC-DS
/// has no in-process generator, so it is not covered here.
#[test]
fn differential_results_match() {
    let orders: i64 = if std::env::var("KRISHIV_CSE_FULL").is_ok() {
        150
    } else {
        24
    };
    let ctx = support::ctx();
    let db = data::tpch(7, orders);
    let mut nonempty = 0;
    let mut extracted = 0;
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let on = optimize(&plan).unwrap();
        let off = optimize_with_options(
            &plan,
            &support::AllDistinct,
            &Options {
                cse: false,
                ..Options::default()
            },
        )
        .unwrap();
        extracted += on.root().to_string().matches("AS cse_").count();
        let (mut want, mut got) = (eval::rows(&db, off.root()), eval::rows(&db, on.root()));
        assert_eq!(got.len(), want.len(), "q{q:02} row counts");
        want.sort();
        got.sort();
        assert_eq!(got, want, "q{q:02} as multisets");
        nonempty += usize::from(!want.is_empty());
    }
    assert!(nonempty >= 15, "only {nonempty} queries return rows");
    assert!(
        extracted > 0,
        "CSE fired nowhere: the differential test is vacuous"
    );
}

/// A filter that roots a subquery plan keeps the subquery's width when its
/// repeated expression is extracted: the consumer reads the output by
/// position.
#[test]
fn a_filter_rooting_a_subquery_keeps_its_width() {
    let (on, off) = both(
        "SELECT a FROM t WHERE a IN (SELECT a FROM t u WHERE (u.b + u.c) > 1 AND (u.b + u.c) < 3)",
    );
    let text = on.root().to_string();
    assert_eq!(extractions(on.root()), 1, "{text}");
    for seed in 1..=8 {
        assert_same_rows(&data::small(seed), on.root(), off.root(), "subquery root");
    }
}

/// A user alias named like a CSE column (`cse_<n>`) is a user column: it
/// must survive the pass that narrows union inputs, and results must not
/// change.
#[test]
fn a_user_alias_named_like_a_cse_column_is_left_alone() {
    let (on, off) = both(
        "SELECT a AS cse_1, (c + 1) * (c + 1) AS x FROM t \
         UNION ALL SELECT a, c FROM t",
    );
    let text = on.root().to_string();
    // The user's `AS cse_1` and one extraction (`AS cse_0` of the pass).
    assert_eq!(occurrences(on.root(), "AS cse_1#"), 1, "{text}");
    assert_eq!(occurrences(on.root(), "AS cse_0#"), 1, "{text}");
    assert_eq!(on.root().resolved_output(), off.root().resolved_output());
    for seed in 1..=8 {
        assert_same_rows(&data::small(seed), on.root(), off.root(), "cse_1 alias");
    }

/// A property-test find: a self-join over a filter with repeated
/// `coalesce` calls is a fixed point of `optimize`.
#[test]
fn a_self_join_with_repeated_coalesce_is_a_fixed_point() {
    let ctx = support::ctx();
    let sql = "SELECT * FROM (SELECT x1.c0 AS c0 FROM (SELECT x3.d AS c0, x3.d AS c1, x3.d AS c2 \
        FROM u AS x2 JOIN u AS x3 ON x2.a = x3.a AND (x2.d IS NULL OR x3.d IS NOT NULL) \
        WHERE x2.d <> 'x') AS x1 LEFT ANTI JOIN u AS x4 ON coalesce(x1.c0, 'x') = 'x' \
        WHERE ((coalesce(x1.c2, 'x') = 'z' OR coalesce(x1.c0, 'z') = 'z') \
        AND coalesce(x1.c0, 'x') = 'y')) ORDER BY c0 LIMIT 1";
    let plan = support::resolve(sql, &ctx);
    let optimized = optimize(&plan).unwrap();
    let again = optimize(&optimized).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "{}\n---\n{}",
        optimized.root(),
        again.root()
    );
}

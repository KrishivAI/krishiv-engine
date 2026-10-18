// SPDX-License-Identifier: Apache-2.0
//! Join strategy hints: the build side and join kind they pick, and the
//! warnings for the hints Spark ignores (`semantics` checks their rows
//! against Spark).

use krishiv_exec::planner::ExplainNode;

use super::env;

/// The first node named `name`, depth first.
fn find<'a>(n: &'a ExplainNode, name: &str) -> Option<&'a ExplainNode> {
    if n.name == name {
        return Some(n);
    }
    n.children.iter().find_map(|c| find(c, name))
}

/// The join of `sql` (a self-join of `t` as `x` and `y`, `y` filtered on
/// `b > 50`) and whether it builds the filtered side, with the warnings.
fn join(sql: &str) -> (ExplainNode, bool, Vec<String>) {
    let env = env();
    let warnings = env.analyzed(sql).warnings;
    let plan = env.plan(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut all = warnings;
    all.extend(plan.warnings().iter().cloned());
    let root = plan.explain().clone();
    let j = find(&root, "HashJoin")
        .or_else(|| find(&root, "NestedLoopJoin"))
        .unwrap_or_else(|| panic!("{sql}: no join in\n{root}"))
        .clone();
    let builds_y = j.children[1].to_string().contains("50");
    (j, builds_y, all)
}

const EQUI: &str = "FROM t x JOIN (SELECT * FROM t WHERE b > 50) y ON x.a = y.a";

#[test]
fn broadcast_picks_the_build_side() {
    // The optimizer builds the smaller, filtered side.
    let (_, y, w) = join(&format!("SELECT x.s, y.s {EQUI}"));
    assert!(y && w.is_empty(), "{w:?}");
    let (j, y, w) = join(&format!("SELECT /*+ BROADCAST(x) */ x.s, y.s {EQUI}"));
    assert!(!y && w.is_empty(), "{j}\n{w:?}");
    for hint in ["BROADCASTJOIN(x)", "MAPJOIN(x)", "SHUFFLE_HASH(x)"] {
        let (j, y, _) = join(&format!("SELECT /*+ {hint} */ x.s, y.s {EQUI}"));
        assert!(!y, "{hint}: {j}");
    }
    // Both sides hinted: the default side.
    let (_, y, w) = join(&format!("SELECT /*+ BROADCAST(x, y) */ x.s, y.s {EQUI}"));
    assert!(y && w.is_empty(), "{w:?}");
}

#[test]
fn hint_precedence_and_overrides() {
    // BROADCAST over MERGE over SHUFFLE_HASH, across the sides.
    let (_, y, w) = join(&format!(
        "SELECT /*+ SHUFFLE_HASH(y), BROADCAST(x) */ x.s, y.s {EQUI}"
    ));
    assert!(!y && w.is_empty(), "{w:?}");
    let (_, y, w) = join(&format!(
        "SELECT /*+ SHUFFLE_HASH(x), MERGE(y) */ x.s, y.s {EQUI}"
    ));
    assert!(y, "MERGE wins and keeps the default side");
    assert_eq!(
        w,
        [
            "Hint (strategy=merge) is not supported in the query: no sort-merge join in a single-node plan; a hash join runs instead."
        ]
    );
    // On one relation the first hint written wins.
    let (_, y, w) = join(&format!(
        "SELECT /*+ BROADCAST(x), MERGE(x) */ x.s, y.s {EQUI}"
    ));
    assert!(!y);
    assert_eq!(
        w,
        ["Hint (strategy=merge) is overridden by another hint and will not take effect."]
    );
    let (_, _, w) = join(&format!(
        "SELECT /*+ MERGE(x) */ /*+ BROADCAST(x) */ x.s, y.s {EQUI}"
    ));
    assert_eq!(
        w[0],
        "Hint (strategy=broadcast) is overridden by another hint and will not take effect."
    );
    // A hint in a derived table reaches the join through its projection.
    let (_, y, w) = join(
        "SELECT x.s FROM (SELECT /*+ BROADCAST */ * FROM t) x \
         JOIN (SELECT * FROM t WHERE b > 50) y ON x.a = y.a",
    );
    assert!(!y && w.is_empty(), "{w:?}");
}

#[test]
fn unsupported_build_sides_warn() {
    // Spark cannot build the preserved side of an outer join either.
    let (j, y, w) = join(
        "SELECT /*+ BROADCAST(x) */ x.s FROM t x LEFT JOIN (SELECT * FROM t WHERE b > 50) y \
         ON x.a = y.a",
    );
    assert!(y, "{j}");
    assert_eq!(
        w,
        [
            "Hint (strategy=broadcast) is not supported in the query: build left for LEFT OUTER join."
        ]
    );
    // Spark's shuffled hash join builds it; a single-node hash join does
    // not.
    let (_, y, w) = join(
        "SELECT /*+ SHUFFLE_HASH(x) */ x.s FROM t x LEFT JOIN (SELECT * FROM t WHERE b > 50) y \
         ON x.a = y.a",
    );
    assert!(y);
    assert_eq!(
        w,
        [
            "Hint (strategy=shuffle_hash) is not supported in the query: build left for LEFT OUTER join (a single-node hash join cannot)."
        ]
    );
    let (_, _, w) = join(
        "SELECT /*+ SHUFFLE_HASH(x) */ x.s FROM t x LEFT SEMI JOIN (SELECT * FROM t WHERE b > 50) y \
         ON x.a = y.a",
    );
    assert_eq!(
        w,
        [
            "Hint (strategy=shuffle_hash) is not supported in the query: build left for LEFT SEMI join."
        ]
    );
    // A nested loop builds either side in Spark.
    let (j, y, w) = join(
        "SELECT /*+ BROADCAST(x) */ x.s FROM t x LEFT JOIN (SELECT * FROM t WHERE b > 50) y \
         ON x.a < y.a",
    );
    assert!(y, "{j}");
    assert_eq!(
        w,
        [
            "Hint (strategy=broadcast) is not supported in the query: build left for LEFT OUTER join (a single-node nested-loop join cannot)."
        ]
    );
    let (j, y, w) = join(
        "SELECT /*+ BROADCAST(x) */ x.s FROM t x JOIN (SELECT * FROM t WHERE b > 50) y \
         ON x.a < y.a",
    );
    assert!(!y && w.is_empty(), "{j}\n{w:?}");
}

#[test]
fn replicate_nl_makes_inner_joins_nested_loops() {
    let (j, _, w) = join(&format!(
        "SELECT /*+ SHUFFLE_REPLICATE_NL(x) */ x.s, y.s {EQUI}"
    ));
    assert_eq!(j.name, "NestedLoopJoin");
    assert!(j.detail.contains("= a#"), "the equality is the filter: {j}");
    assert!(w.is_empty());
    // Not for an outer join (Spark ignores it there).
    let (j, _, w) = join(
        "SELECT /*+ SHUFFLE_REPLICATE_NL(x) */ x.s FROM t x LEFT JOIN \
         (SELECT * FROM t WHERE b > 50) y ON x.a = y.a",
    );
    assert_eq!(j.name, "HashJoin");
    assert!(w.is_empty());
}

#[test]
fn ignored_hints_warn() {
    let (_, _, w) = join(&format!(
        "SELECT /*+ FOO(x, 1), BROADCAST(nope, x.a), MAPJOIN('db.y') */ x.s {EQUI}"
    ));
    assert_eq!(
        w,
        [
            // Inner hints first, as Spark resolves them.
            "Count not find relation 'db.y' specified in hint 'MAPJOIN('db.y')'.",
            "Count not find relation 'nope' specified in hint 'BROADCAST(nope, x.a)'.",
            "Count not find relation 'x.a' specified in hint 'BROADCAST(nope, x.a)'.",
            "Unrecognized hint: FOO(x, 1)",
        ]
    );
    let env = env();
    let sql = "SELECT /*+ BROADCAST(t) */ a FROM t";
    let plan = env.plan(sql).unwrap();
    assert_eq!(
        plan.warnings(),
        ["A join hint (strategy=broadcast) is specified but it is not part of a join relation."]
    );
    let sql = "SELECT /*+ REPARTITION(3) */ a FROM t";
    let plan = krishiv_sql::parse_and_lower(
        sql,
        &krishiv_sql::ParserOptions::default(),
        krishiv_plan::SourceId(0),
    )
    .unwrap();
    let err = krishiv_plan::analyzer::testing::block_on(krishiv_plan::analyzer::Analyzer::analyze(
        &krishiv_plan::analyzer::DefaultAnalyzer,
        &plan,
        &env.ctx,
    ))
    .unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

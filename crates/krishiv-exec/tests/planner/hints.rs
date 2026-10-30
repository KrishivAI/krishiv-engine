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
    // A `WITH` reference and a table by a suffix of its name.
    let (_, y, w) = join(
        "WITH x AS (SELECT * FROM t) SELECT /*+ BROADCAST(x) */ x.s FROM x \
         JOIN (SELECT * FROM t WHERE b > 50) y ON x.a = y.a",
    );
    assert!(!y && w.is_empty(), "{w:?}");
    let (_, y, w) = join(
        "SELECT /*+ BROADCAST('default.t') */ t.s FROM t \
         JOIN (SELECT * FROM t WHERE b > 50) y ON t.a = y.a",
    );
    assert!(!y && w.is_empty(), "{w:?}");
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
    // A FULL OUTER join is never broadcast: it builds the right input,
    // partitioned, and the hint is reported ignored.
    let (j, _, w) = join(
        "SELECT /*+ BROADCAST(y) */ x.s FROM t x FULL JOIN (SELECT * FROM t WHERE b > 50) y \
         ON x.a = y.a",
    );
    assert!(!j.to_string().contains("Broadcast"), "{j}");
    assert_eq!(
        w,
        [
            "Hint (strategy=broadcast) is not supported in the query: build right for FULL OUTER join."
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
    // An existence join (EXISTS under OR) never builds its left side.
    let (_, _, w) = join(
        "SELECT /*+ SHUFFLE_HASH(x) */ x.s FROM t x \
         WHERE EXISTS (SELECT 1 FROM t y WHERE y.a = x.a AND y.b > 50) OR x.s = 's0'",
    );
    assert_eq!(
        w,
        [
            "Hint (strategy=shuffle_hash) is not supported in the query: build left for EXISTENCE join."
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
    // Partitioning hints are checked, then have no effect on one node.
    let sql = "SELECT /*+ REPARTITION(3, a), COALESCE(1), REBALANCE */ \
               /*+ REPARTITION_BY_RANGE('b') */ s FROM t";
    env.plan(sql).unwrap();
    assert_eq!(
        env.analyzed(sql).warnings,
        [
            "Hint REPARTITION_BY_RANGE('b') has no effect in a single-node plan.",
            "Hint REBALANCE() has no effect in a single-node plan.",
            "Hint COALESCE(1) has no effect in a single-node plan.",
            "Hint REPARTITION(3, a) has no effect in a single-node plan.",
        ]
    );
}

/// The analysis errors of set operations and hint parameters (their rows
/// against Spark are in the `semantics` goldens).
#[test]
fn set_operation_and_hint_errors() {
    use krishiv_common::error::ErrorClass;
    let env = env();
    for (sql, class) in [
        (
            "SELECT a, b FROM t UNION ALL SELECT a FROM t",
            ErrorClass::NumColumnsMismatch,
        ),
        (
            "SELECT a FROM t UNION ALL SELECT DATE '2020-01-01'",
            ErrorClass::IncompatibleColumnType,
        ),
        (
            "SELECT /*+ BROADCAST(1) */ a FROM t",
            ErrorClass::LegacyErrorTemp1046,
        ),
        (
            "SELECT a FROM t INTERSECT ALL SELECT a, b FROM t",
            ErrorClass::NumColumnsMismatch,
        ),
        (
            "SELECT /*+ COALESCE(0) */ a FROM t",
            ErrorClass::KrishivInvalidPartitionNumber,
        ),
        (
            "SELECT /*+ REPARTITION(-1, a) */ a FROM t",
            ErrorClass::KrishivInvalidPartitionNumber,
        ),
        (
            "SELECT /*+ REBALANCE(0) */ a FROM t",
            ErrorClass::KrishivInvalidPartitionNumber,
        ),
        (
            "SELECT /*+ COALESCE(2, 3) */ a FROM t",
            ErrorClass::LegacyErrorTemp1048,
        ),
        (
            "SELECT /*+ REPARTITION(0, 1.5, NULL, true, 3L, a) */ a FROM t",
            ErrorClass::LegacyErrorTemp1047,
        ),
        (
            "SELECT /*+ REPARTITION(a) */ b FROM t GROUP BY b",
            ErrorClass::UnresolvedColumnWithSuggestion,
        ),
    ] {
        let plan = krishiv_sql::parse_and_lower(
            sql,
            &krishiv_sql::ParserOptions::default(),
            krishiv_plan::SourceId(0),
        )
        .unwrap();
        let err =
            krishiv_plan::analyzer::testing::block_on(krishiv_plan::analyzer::Analyzer::analyze(
                &krishiv_plan::analyzer::DefaultAnalyzer,
                &plan,
                &env.ctx,
            ))
            .unwrap_err();
        assert_eq!(err.error.error_class(), class, "{sql}: {}", err.error);
    }
}

/// Spark's messages for partitioning hint parameters: invalid ones by
/// value, a partition number below 1.
#[test]
fn partitioning_hint_messages() {
    let env = env();
    for (sql, message) in [
        (
            "SELECT /*+ REPARTITION(0, 1.5, NULL, true, 3L, 4, 1Y, 2S, a) */ a FROM t",
            "REPARTITION Hint parameters should include an optional integral partitionNum \
             and/or columns, but \"1.5\", \"NULL\", \"true\", \"3\", \"4\", \"1\", \"2\" can \
             not be recognized as either partitionNum or columns.",
        ),
        (
            "SELECT /*+ REPARTITION_BY_RANGE(-2, a) */ a FROM t",
            "Number of partitions (-2) must be positive.",
        ),
        (
            "SELECT /*+ coalesce(a) */ a FROM t",
            "COALESCE Hint expects a partition number as a parameter.",
        ),
    ] {
        let plan = krishiv_sql::parse_and_lower(
            sql,
            &krishiv_sql::ParserOptions::default(),
            krishiv_plan::SourceId(0),
        )
        .unwrap();
        let err =
            krishiv_plan::analyzer::testing::block_on(krishiv_plan::analyzer::Analyzer::analyze(
                &krishiv_plan::analyzer::DefaultAnalyzer,
                &plan,
                &env.ctx,
            ))
            .unwrap_err();
        assert_eq!(err.error.message(), message, "{sql}");
    }
}

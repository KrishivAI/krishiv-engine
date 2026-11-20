// SPDX-License-Identifier: Apache-2.0
//! The prefix cache (`CachingAnalyzer`, design P3-T013a) against the cold analyzer over TPC-H and the
//! Spark-golden corpus: each plan, then the plan grown by a `withColumn` and a `limit`, analysed through
//! one cache, equals the cold analysis exactly (ids, declared types and comments), or fails alike.

use super::*;
use crate::analyzer::CachingAnalyzer;

/// `plan` with a `withColumn` above it, then a `limit`.
fn grown(plan: LogicalPlan) -> LogicalPlan {
    LogicalPlan::new(PlanNode::WithColumns {
        input: Arc::new(plan),
        columns: vec![("__prefix_step".to_owned(), Arc::new(int(1)))],
    })
    .limit(Arc::new(int(10)))
}

/// Analyses `plan`, then it grown once and twice, through one cache and cold; the number of warm
/// starts.
fn warm_like_cold(name: &str, plan: LogicalPlan, ctx: &AnalysisContext) -> usize {
    let cache = CachingAnalyzer::new();
    let once = grown(plan.clone());
    for p in [plan, once.clone(), grown(once)] {
        let p = Plan::try_new(p).unwrap();
        let warm = block_on(cache.analyze(&p, ctx));
        let cold = block_on(DefaultAnalyzer.analyze(&p, ctx));
        match (warm, cold) {
            (Ok(w), Ok(k)) => {
                assert!(
                    w == k,
                    "{name}: warm != cold\n{}\n{}",
                    w.plan.root(),
                    k.plan.root()
                );
                assert_eq!(w.output_declared, k.output_declared, "{name}");
                assert_eq!(w.output_comments, k.output_comments, "{name}");
            }
            (Err(w), Err(k)) => assert_eq!(w, k, "{name}"),
            (w, k) => panic!("{name}: warm {:?} but cold {:?}", w.err(), k.err()),
        }
    }
    usize::try_from(cache.warm_starts()).unwrap()
}

#[test]
fn tpch_grown_plans_analyse_warm_like_cold() {
    let ctx = ctx();
    let mut hits = 0;
    for (i, q) in QUERIES.iter().enumerate() {
        hits += warm_like_cold(&format!("q{:02}", i + 1), q(), &ctx);
    }
    // Two grown plans per query; a query that reads a table twice outside its first-child spine is a
    // miss by design (an equal relation outside the prefix).
    assert!(hits >= 42, "{hits} warm starts of 44 (42 when written)");
}

#[test]
fn golden_corpus_grown_plans_analyse_warm_like_cold() {
    let mut hits = 0;
    for (sql, conf, plan) in corpus() {
        hits += warm_like_cold(sql, plan, &context(catalog(), conf));
    }
    // Most corpus cases raise, and a failed analysis seeds nothing.
    assert!(
        hits >= 14,
        "{hits} warm starts over the corpus (14 when written)"
    );
}

fn zip_of(left: LogicalPlan, right: LogicalPlan) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Zip {
        left: Arc::new(left),
        right: Arc::new(right),
    })
}

fn names(r: &Result<AnalyzedPlan, AnalysisError>) -> Vec<String> {
    r.as_ref()
        .expect("analyzes")
        .output
        .iter()
        .map(|c| c.name.clone())
        .collect()
}

/// `Zip` of two projections of one table: the left columns, then the right, and a fixed point.
#[test]
fn zip_of_projections_of_one_table_lists_left_then_right() {
    let ctx = ctx();
    let t = || LogicalPlan::table(["t"]);
    let r = run(
        zip_of(
            t().project(vec![c("a")]),
            t().project(vec![c("b").alias("bb")]),
        ),
        &ctx,
    );
    assert_eq!(names(&r), ["a", "bb"]);
    let a = r.unwrap();
    assert!(a.plan.root().is_resolved());
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
    assert!(again.plan.structurally_eq(&a.plan));
}

/// A bare table zipped with a projection of it, and with itself.
#[test]
fn zip_of_a_table_with_a_projection_or_itself_resolves() {
    let ctx = ctx();
    let t = || LogicalPlan::table(["t"]);
    let one = run(zip_of(t(), t().project(vec![c("a").alias("a2")])), &ctx);
    let n = names(&one).len();
    assert!(n >= 3 && names(&one).last().unwrap() == "a2", "{one:?}");
    let twice = run(zip_of(t(), t()), &ctx);
    assert_eq!(names(&twice).len() % 2, 0);
}

/// Two inputs over different bases are `ZIP_PLANS_NOT_MERGEABLE`.
#[test]
fn zip_of_different_bases_is_not_mergeable() {
    let ctx = ctx();
    let r = run(
        zip_of(
            LogicalPlan::table(["t"]).project(vec![c("a")]),
            LogicalPlan::table(["u"]).project(vec![c("a")]),
        ),
        &ctx,
    );
    assert_eq!(class(&r), "ZIP_PLANS_NOT_MERGEABLE");
}

/// A shared producer under two names keeps both names (Spark's `ResolveZip` re-aliases the dropped one).
#[test]
fn zip_of_one_producer_under_two_names_keeps_both_names() {
    let ctx = ctx();
    let t = || LogicalPlan::table(["t"]);
    let r = run(
        zip_of(
            t().project(vec![c("a").alias("x")]),
            t().project(vec![c("a").alias("y")]),
        ),
        &ctx,
    );
    assert_eq!(names(&r), ["x", "y"]);
}

/// A filter between the base and the projection makes the bases differ unless both have it.
#[test]
fn zip_requires_the_same_filtered_base() {
    let ctx = ctx();
    let filtered = || LogicalPlan::table(["t"]).filter(eq(c("a"), c("a")));
    let same = run(
        zip_of(
            filtered().project(vec![c("a")]),
            filtered().project(vec![c("b")]),
        ),
        &ctx,
    );
    assert_eq!(names(&same), ["a", "b"]);
    let different = run(
        zip_of(
            filtered().project(vec![c("a")]),
            LogicalPlan::table(["t"]).project(vec![c("b")]),
        ),
        &ctx,
    );
    assert_eq!(class(&different), "ZIP_PLANS_NOT_MERGEABLE");
}

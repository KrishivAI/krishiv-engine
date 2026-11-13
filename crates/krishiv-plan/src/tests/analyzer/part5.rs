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

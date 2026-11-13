// SPDX-License-Identifier: Apache-2.0
//! The analysis prefix cache (`CachingAnalyzer`, design P3-T013a) over TPC-H and TPC-DS: each query is
//! analysed through the cache, then the query plus two steps above it (`withColumn`, then `limit`), as a
//! client that grows a plan resends it. The warm analysis must equal a cold one exactly, ids included.

use std::path::Path;
use std::sync::Arc;

use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, CachingAnalyzer, DefaultAnalyzer, ResolutionConf,
};
use krishiv_plan::{Literal, LogicalPlan, Plan, PlanNode, SourceId};

/// `plan` with a `withColumn` and a `limit` above it.
fn grown(plan: &Plan) -> Plan {
    let step = LogicalPlan::new(PlanNode::WithColumns {
        input: Arc::clone(plan.root()),
        columns: vec![(
            "__prefix_step".to_owned(),
            Arc::new(Literal::Integer(1).into()),
        )],
    })
    .limit(Arc::new(Literal::Integer(10).into()));
    Plan::try_new(step).unwrap()
}

/// Analyses `text` and then the query grown by two steps through one cache; each warm result equals
/// the cold one (or both fail alike). `false` when the query does not parse.
fn warm_equals_cold(name: &str, text: &str, ctx: &AnalysisContext) -> bool {
    let Ok(parsed) =
        krishiv_sql::parse_and_lower(text, &krishiv_sql::ParserOptions::default(), SourceId(0))
    else {
        return false;
    };
    let cache = CachingAnalyzer::new();
    for plan in [parsed.clone(), grown(&parsed), grown(&grown(&parsed))] {
        let warm = block_on(cache.analyze(&plan, ctx));
        let cold = block_on(DefaultAnalyzer.analyze(&plan, ctx));
        match (warm, cold) {
            (Ok(w), Ok(c)) => {
                assert!(
                    w == c,
                    "{name}: warm != cold\nwarm:\n{}\ncold:\n{}",
                    w.plan.root(),
                    c.plan.root()
                );
                assert_eq!(w.output_declared, c.output_declared, "{name}");
                assert_eq!(w.output_comments, c.output_comments, "{name}");
            }
            (Err(w), Err(c)) => assert_eq!(w.to_string(), c.to_string(), "{name}"),
            (w, c) => panic!("{name}: warm {:?} but cold {:?}", w.err(), c.err()),
        }
    }
    true
}

#[test]
fn tpch_warm_analysis_equals_cold() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let tpch = super::tpch::parquet_tables(&dir, &rt);
    for q in 1..=22 {
        assert!(
            warm_equals_cold(&format!("tpch q{q:02}"), &super::tpch::sql(q), &tpch.ctx),
            "tpch q{q:02} parses"
        );
    }
}

#[test]
fn tpcds_warm_analysis_equals_cold() {
    let (catalog, _) = super::tpcds_merge::tables();
    let ctx = context(catalog, ResolutionConf::default());
    let mut checked = 0;
    for q in 1..=99 {
        let path = super::tpch::root().join(format!("bench/queries/tpcds/q{q:02}.sql"));
        let text = std::fs::read_to_string(path).unwrap();
        checked += usize::from(warm_equals_cold(&format!("tpcds q{q:02}"), &text, &ctx));
    }
    assert!(checked >= 90, "{checked} TPC-DS queries parsed");
}

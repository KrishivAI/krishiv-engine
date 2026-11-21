// SPDX-License-Identifier: Apache-2.0
//! An as-of join's left scan offers runtime-filter sites to a join above it only where the merge
//! evaluates nothing that may raise: a filter dropping left rows first would hide the row whose
//! residual or distance raises under ANSI, which Spark raises on.

use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_exec::planner::ProviderStatistics;
use krishiv_plan::analyzer::testing::block_on;
use krishiv_plan::analyzer::{Analyzer as _, DefaultAnalyzer};
use krishiv_plan::{Plan, SourceId};

use super::Runnable;
use super::runtime_filters::{configured, table, write};
use super::tpch::{Tpch, parquet_files};

fn planned(t: &Tpch, text: &str) -> Result<Runnable, KrishivError> {
    let options = krishiv_sql::ParserOptions {
        asof_join: true,
        ..krishiv_sql::ParserOptions::default()
    };
    let plan =
        krishiv_sql::parse_and_lower(text, &options, SourceId(0)).map_err(KrishivError::from)?;
    let plan: Plan = block_on(DefaultAnalyzer.analyze(&plan, &t.ctx))?.plan;
    let stats = ProviderStatistics(Arc::clone(&t.tables) as _);
    let plan = krishiv_optimizer::optimize_with(&plan, &stats)?;
    t.runnable(&plan)
}

#[test]
fn a_raising_residual_or_distance_keeps_runtime_filters_off_the_left_scan() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 41;
    write(
        dir.path(),
        "big.parquet",
        &table(&mut rng, 20_000, 4_000, 0),
    );
    write(dir.path(), "mid.parquet", &table(&mut rng, 4_000, 4_000, 0));
    write(
        dir.path(),
        "small.parquet",
        &table(&mut rng, 200, 40_000, 0),
    );
    let sources = [
        ("big", vec!["big.parquet".to_owned()]),
        ("mid", vec!["mid.parquet".to_owned()]),
        ("small", vec!["small.parquet".to_owned()]),
    ];
    let mut t = parquet_files(dir.path(), &sources, true, &rt);
    configured(
        &mut t,
        &[
            ("krishiv.runtimeFilter.bloom.minApplicationRows", "0"),
            ("krishiv.runtimeFilter.adaptive.maxPassPercent", "100"),
        ],
    );
    let query = |op: &str, extra: &str| {
        format!(
            "SELECT x.v, s.v FROM (SELECT b.k32, b.v FROM big b ASOF JOIN mid m \
             MATCH_CONDITION (b.v {op} m.v) ON b.k32 = m.k32{extra}) x \
             JOIN small s ON x.k32 = s.k32"
        )
    };
    let to_big = |text: &str| {
        let plan = planned(&t, text).unwrap_or_else(|e| panic!("{text}: {e}"));
        plan.to_string().matches(".default.big.k32").count()
    };
    // Backward without a residual the SQL scan evaluates nothing: the join above filters `big`.
    let quiet = to_big(&query(">=", ""));
    assert!(quiet > 0, "no runtime filter reached the left scan");
    // Forward, the distance `m.v - b.v` may overflow under ANSI.
    assert_eq!(to_big(&query("<=", "")), 0, "forward distance");
    // A residual that may overflow.
    assert_eq!(
        to_big(&query(">=", " AND b.v + m.v > 0")),
        0,
        "raising residual"
    );
    // A residual that cannot raise keeps the sites.
    assert_eq!(
        to_big(&query(">=", " AND b.v <> m.v")),
        quiet,
        "quiet residual"
    );
}

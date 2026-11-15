// SPDX-License-Identifier: Apache-2.0
//! `EXPLAIN` never prints the planner's internal ids, which start at 2^62
//! (PY-008: `sum(amount#1)#4611686018427387904`).
#![forbid(unsafe_code)]

use futures::executor::block_on;
use krishiv_session::{ExplainMode, Session};

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[test]
fn an_aggregate_plan_has_small_ids() -> R {
    let s = Session::new()?;
    let df = block_on(
        s.sql("SELECT k, sum(v) AS revenue FROM VALUES ('a', 1), ('b', 2) AS t(k, v) GROUP BY k"),
    )?;
    for mode in [
        ExplainMode::Simple,
        ExplainMode::Extended,
        ExplainMode::Formatted,
    ] {
        let text = block_on(df.explain(mode))?;
        assert!(!text.contains("4611686018427387"), "{text}");
    }
    Ok(())
}

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

#[test]
fn explain_analyze_prints_the_same_ids_as_explain() -> R {
    let s = Session::new()?;
    let q = "SELECT k, sum(v) AS revenue FROM VALUES ('a', 1), ('b', 2) AS t(k, v) GROUP BY k";
    let df = block_on(s.sql(q))?;
    let profile = block_on(df.explain_analyze())?;
    let text = profile.to_json();
    assert!(!text.contains("4611686018427387"), "{text}");
    let explain = block_on(df.explain(ExplainMode::Simple))?;
    let ids = |t: &str| -> std::collections::BTreeSet<String> {
        t.split('#')
            .skip(1)
            .map(|r| r.chars().take_while(char::is_ascii_digit).collect())
            .collect()
    };
    assert!(ids(&explain).is_subset(&ids(&text)), "{explain}\n{text}");
    Ok(())
}

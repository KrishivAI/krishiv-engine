// SPDX-License-Identifier: Apache-2.0
//! Floating-point edge sweep against Spark 4.2 (Comet 1.2.0 correctness check 1): `-0.0`, `NaN`,
//! `+/-Infinity`, subnormals, the extremes and NULL crossed with the arithmetic, comparison,
//! aggregate, ordering, cast, math, hashing and collection functions, DOUBLE and FLOAT, in both
//! ANSI modes (`tests/fixtures/spark-4.2.0/float-edge-sweep.json`, written by
//! `tests/scripts/gen_float_edge_sweep.py` from a real Spark run), replayed on `krishiv-session`
//! through the expression differential's machinery. Mismatches are listed in
//! `float-edge-sweep-known-failures.json`; a new one fails the test, and so does a listed one
//! that now agrees (`KRISHIV_UPDATE_MANIFEST=1` rewrites the list).
#![forbid(unsafe_code)]

use krishiv_testkit::expr_diff::{
    Evaluator, Expect, Fixture, SessionEvaluator, parse_known, ratchet, render_known, run,
};

type R = Result<(), Box<dyn std::error::Error>>;

fn fixture_dir() -> std::path::PathBuf {
    krishiv_testkit::tests_dir().join("fixtures/spark-4.2.0")
}

fn fixture() -> Result<Fixture, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(fixture_dir().join("float-edge-sweep.json"))?;
    Ok(Fixture::parse(&text)?)
}

#[test]
fn matches_spark_up_to_known_failures() -> R {
    let fx = fixture()?;
    let mut ev = SessionEvaluator::new_materialized(&fx.setup)?;
    let found = run(&fx, &mut ev);
    let known_path = fixture_dir().join("float-edge-sweep-known-failures.json");
    if std::env::var_os("KRISHIV_UPDATE_MANIFEST").is_some() {
        std::fs::write(&known_path, render_known(&found))?;
        return Ok(());
    }
    let known = parse_known(&std::fs::read_to_string(&known_path)?)?;
    let r = ratchet(&found, &known);
    let detail: Vec<String> = found
        .iter()
        .filter(|m| r.new.contains(&m.key()))
        .map(|m| {
            let (spark, got) = (format!("{:?}", m.spark), format!("{:?}", m.got));
            format!(
                "{}\n  spark: {}\n  got:   {}",
                m.key(),
                &spark[..spark.len().min(300)],
                &got[..got.len().min(300)]
            )
        })
        .collect();
    assert!(r.new.is_empty(), "new mismatches:\n{}", detail.join("\n"));
    assert!(
        r.fixed.is_empty(),
        "listed mismatches that now agree, remove them:\n{}",
        r.fixed.join("\n")
    );
    Ok(())
}

/// Spark's answers for every case, replayed from the fixture itself.
struct Replay<'a>(&'a Fixture);

impl Evaluator for Replay<'_> {
    fn eval(&mut self, sql: &str, ansi: bool) -> Expect {
        self.0
            .cases
            .iter()
            .find(|c| c.sql == sql && c.ansi == ansi)
            .map_or_else(|| Expect::Error("MISSING".into()), |c| c.expect.clone())
    }
}

/// Spark's answers except for one planted kernel bug: negating a double loses the sign of zero.
struct Planted<'a>(Replay<'a>);

impl Evaluator for Planted<'_> {
    fn eval(&mut self, sql: &str, ansi: bool) -> Expect {
        match self.0.eval(sql, ansi) {
            Expect::Rows { schema, rows } if sql == "(- x)" => Expect::Rows {
                schema,
                rows: rows
                    .into_iter()
                    .map(|r| if r == "-0.0" { "0.0".to_owned() } else { r })
                    .collect(),
            },
            other => other,
        }
    }
}

#[test]
fn replaying_spark_s_own_answers_finds_nothing() -> R {
    let fx = fixture()?;
    assert!(run(&fx, &mut Replay(&fx)).is_empty());
    Ok(())
}

#[test]
fn a_planted_negative_zero_bug_is_found() -> R {
    let fx = fixture()?;
    let found = run(&fx, &mut Planted(Replay(&fx)));
    assert!(
        found.iter().any(|m| m.sql == "(- x)"),
        "the sweep missed a lost sign of zero"
    );
    Ok(())
}

#[test]
fn the_sweep_covers_the_edge_values_and_both_float_widths() -> R {
    let fx = fixture()?;
    for needle in ["'NaN'", "'-0.0'", "'Infinity'", "'4.9E-324'", "'1.4E-45'"] {
        assert!(fx.setup.contains(needle), "{needle} is not in the table");
    }
    assert!(
        fx.cases
            .iter()
            .any(|c| c.sql == "sum(d) OVER (PARTITION BY p)")
    );
    assert!(fx.cases.iter().any(|c| c.sql == "(fx + fy)"));
    assert!(fx.cases.iter().any(|c| !c.ansi) && fx.cases.iter().any(|c| c.ansi));
    Ok(())
}

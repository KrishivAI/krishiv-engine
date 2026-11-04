// SPDX-License-Identifier: Apache-2.0
//! Timestamp type and time zone labels (Comet 1.2.0 correctness check 3): the same wall clocks as
//! TIMESTAMP and TIMESTAMP_NTZ crossed with the conversions where the label matters, under two
//! session zones (`timestamp-labels-Kolkata.json`, `timestamp-labels-Los_Angeles.json`, written
//! by `tests/scripts/gen_timestamp_labels.py` from a real Spark run). A case's answer is its
//! Spark type and its rows, so a value with the right instant and the wrong type, or rendered in
//! the wrong zone, is a mismatch. The mismatches that remain are listed in
//! `timestamp-labels-known-failures.json` (keys carry the zone); a new one fails the test, and so
//! does a listed one that now agrees (`KRISHIV_UPDATE_MANIFEST=1` rewrites the list).
#![forbid(unsafe_code)]

use krishiv_testkit::expr_diff::{
    Evaluator, Expect, Fixture, SessionEvaluator, parse_known, ratchet_keys, render_known_keys, run,
};

type R = Result<(), Box<dyn std::error::Error>>;

const ZONES: [&str; 2] = ["Kolkata", "Los_Angeles"];

fn fixture_dir() -> std::path::PathBuf {
    krishiv_testkit::tests_dir().join("fixtures/spark-4.2.0")
}

fn fixture(zone: &str) -> Result<Fixture, Box<dyn std::error::Error>> {
    let text =
        std::fs::read_to_string(fixture_dir().join(format!("timestamp-labels-{zone}.json")))?;
    Ok(Fixture::parse(&text)?)
}

fn evaluator(fx: &Fixture) -> Result<SessionEvaluator, Box<dyn std::error::Error>> {
    Ok(SessionEvaluator::for_fixture(fx)?)
}

#[test]
fn matches_spark_up_to_known_failures() -> R {
    let mut keys = Vec::new();
    let mut detail = std::collections::BTreeMap::new();
    for zone in ZONES {
        let fx = fixture(zone)?;
        let mut ev = evaluator(&fx)?;
        for m in run(&fx, &mut ev) {
            let key = format!("{zone}|{}", m.key());
            let cut = |e: &Expect| format!("{e:?}").chars().take(300).collect::<String>();
            detail.insert(
                key.clone(),
                format!(
                    "{key}\n  spark: {}\n  got:   {}",
                    cut(&m.spark),
                    cut(&m.got)
                ),
            );
            keys.push(key);
        }
    }
    let known_path = fixture_dir().join("timestamp-labels-known-failures.json");
    if std::env::var_os("KRISHIV_UPDATE_MANIFEST").is_some() {
        std::fs::write(&known_path, render_known_keys(&keys))?;
        return Ok(());
    }
    let known = parse_known(&std::fs::read_to_string(&known_path)?)?;
    let r = ratchet_keys(&keys, &known);
    let shown: Vec<&str> = r
        .new
        .iter()
        .filter_map(|k| detail.get(k))
        .map(String::as_str)
        .collect();
    assert!(r.new.is_empty(), "new mismatches:\n{}", shown.join("\n"));
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

/// Spark's answers except that `ts` comes back labeled TIMESTAMP_NTZ (right instant, wrong type).
struct Mislabeled<'a>(Replay<'a>);

impl Evaluator for Mislabeled<'_> {
    fn eval(&mut self, sql: &str, ansi: bool) -> Expect {
        match self.0.eval(sql, ansi) {
            Expect::Rows { schema, rows } if sql == "ts" => Expect::Rows {
                schema: schema.replace("timestamp", "timestamp_ntz"),
                rows,
            },
            other => other,
        }
    }
}

/// Spark's answers except that `ts` is rendered five and a half hours earlier, as UTC would show
/// a Kolkata wall clock (right type, wrong zone).
struct OtherZone<'a>(Replay<'a>);

impl Evaluator for OtherZone<'_> {
    fn eval(&mut self, sql: &str, ansi: bool) -> Expect {
        match self.0.eval(sql, ansi) {
            Expect::Rows { schema, rows } if sql == "ts" => Expect::Rows {
                schema,
                rows: rows
                    .into_iter()
                    .map(|r| r.replacen("1970-01-01 00:00:00", "1969-12-31 18:30:00", 1))
                    .collect(),
            },
            other => other,
        }
    }
}

#[test]
fn replaying_spark_s_own_answers_finds_nothing() -> R {
    for zone in ZONES {
        let fx = fixture(zone)?;
        assert!(run(&fx, &mut Replay(&fx)).is_empty(), "{zone}");
    }
    Ok(())
}

#[test]
fn a_mislabeled_type_is_a_mismatch() -> R {
    let fx = fixture("Kolkata")?;
    let found = run(&fx, &mut Mislabeled(Replay(&fx)));
    assert!(
        found.iter().any(|m| m.sql == "ts"),
        "a TIMESTAMP labeled NTZ went unnoticed"
    );
    Ok(())
}

#[test]
fn a_value_in_the_wrong_zone_is_a_mismatch() -> R {
    let fx = fixture("Kolkata")?;
    // Spark shows the wall clock the setup gave, in the session zone.
    let epoch = fx
        .cases
        .iter()
        .find(|c| c.sql == "ts")
        .map(|c| format!("{:?}", c.expect))
        .ok_or("no `ts` case")?;
    assert!(epoch.contains("1970-01-01 00:00:00"), "{epoch}");
    let found = run(&fx, &mut OtherZone(Replay(&fx)));
    assert!(
        found.iter().any(|m| m.sql == "ts"),
        "a value in another zone went unnoticed"
    );
    Ok(())
}

#[test]
fn the_fixtures_carry_their_zone_and_both_timestamp_types() -> R {
    for (zone, id) in [
        ("Kolkata", "Asia/Kolkata"),
        ("Los_Angeles", "America/Los_Angeles"),
    ] {
        let fx = fixture(zone)?;
        assert_eq!(fx.time_zone.as_deref(), Some(id));
        assert!(fx.setup.contains("TIMESTAMP_NTZ") && fx.setup.contains("AS TIMESTAMP)"));
        assert!(
            fx.cases
                .iter()
                .any(|c| c.sql == "typeof(coalesce(ts, ntz))")
        );
    }
    Ok(())
}

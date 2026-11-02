// SPDX-License-Identifier: Apache-2.0
//! Expression-level differential test against Spark 4.2 (P2-T101): the seeded fixture
//! `tests/fixtures/spark-4.2.0/expr-differential.json` (`tests/scripts/gen_expr_differential.py`)
//! replayed on `krishiv-session`. The minimal mismatches are listed in
//! `expr-differential-known-failures.json`; a new one fails the test, and so does a listed one that
//! now agrees (`KRISHIV_UPDATE_MANIFEST=1` rewrites the list). The nightly runs a 10,000-expression
//! fixture through the `expr-diff` binary.
#![forbid(unsafe_code)]

use krishiv_testkit::expr_diff::{
    Evaluator, Expect, Fixture, SessionEvaluator, cli, parse_known, ratchet, render_known, run,
};

type R = Result<(), Box<dyn std::error::Error>>;

fn fixture_dir() -> std::path::PathBuf {
    krishiv_testkit::tests_dir().join("fixtures/spark-4.2.0")
}

fn fixture() -> Result<Fixture, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(fixture_dir().join("expr-differential.json"))?;
    Ok(Fixture::parse(&text)?)
}

#[test]
fn matches_spark_up_to_known_failures() -> R {
    let fx = fixture()?;
    let mut ev = SessionEvaluator::new(&fx.setup)?;
    let found = run(&fx, &mut ev);
    let known_path = fixture_dir().join("expr-differential-known-failures.json");
    if std::env::var_os("KRISHIV_UPDATE_MANIFEST").is_some() {
        std::fs::write(&known_path, render_known(&found))?;
        return Ok(());
    }
    let known = parse_known(&std::fs::read_to_string(&known_path)?)?;
    let r = ratchet(&found, &known);
    let detail: Vec<String> = found
        .iter()
        .filter(|m| r.new.contains(&m.key()))
        .map(|m| format!("{}\n  spark: {}\n  got:   {}", m.key(), m.spark, m.got))
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

/// Answers as Spark does except for one planted kernel bug: `ceil` of a value gives the wrong rows.
struct Planted<'a>(Replay<'a>);

impl Evaluator for Planted<'_> {
    fn eval(&mut self, sql: &str, ansi: bool) -> Expect {
        match self.0.eval(sql, ansi) {
            Expect::Rows { schema, mut rows } if sql.starts_with("ceil(") => {
                rows.reverse();
                Expect::Rows {
                    schema,
                    rows: rows.into_iter().chain(["PLANTED".to_owned()]).collect(),
                }
            }
            other => other,
        }
    }
}

#[test]
fn replaying_spark_has_no_mismatch() -> R {
    let fx = fixture()?;
    assert!(run(&fx, &mut Replay(&fx)).is_empty());
    Ok(())
}

/// Acceptance: a planted kernel bug is found within the budget, narrowed to its smallest case.
#[test]
fn planted_bug_is_found_and_minimised() -> R {
    let fx = fixture()?;
    let found = run(&fx, &mut Planted(Replay(&fx)));
    assert!(
        !found.is_empty(),
        "the planted ceil bug was not found in {} cases",
        fx.cases.len()
    );
    // Minimal: only the planted `ceil(` cases are reported, not the roots above them, because a
    // parent of a mismatching case is never minimal.
    for m in &found {
        assert!(
            m.sql.starts_with("ceil("),
            "{} reported but not planted",
            m.sql
        );
    }
    Ok(())
}

#[test]
fn fixture_parse_and_ratchet() -> R {
    let text = r#"{"setup":"s","cases":[
        {"sql":"a","ansi":true,"kids":[],"root":false,"schema":"int","rows":["1"]},
        {"sql":"f(a)","ansi":true,"kids":["a"],"root":true,"error":"DIVIDE_BY_ZERO"}]}"#;
    let fx = Fixture::parse(text)?;
    assert_eq!(fx.cases.len(), 2);
    struct Wrong;
    impl Evaluator for Wrong {
        fn eval(&mut self, _: &str, _: bool) -> Expect {
            Expect::Rows {
                schema: "int".into(),
                rows: vec!["1".into()],
            }
        }
    }
    // `a` agrees; `f(a)` does not: it is the one minimal mismatch.
    let found = run(&fx, &mut Wrong);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].key(), "1|f(a)");
    let r = ratchet(&found, &["1|gone".to_owned()]);
    assert_eq!(r.new, ["1|f(a)"]);
    assert_eq!(r.fixed, ["1|gone"]);
    assert_eq!(parse_known(&render_known(&found))?, ["1|f(a)"]);
    Ok(())
}

/// The `expr-diff` command: a wrong expectation is a new mismatch (exit false), a listed one is
/// not, a bad flag or file is an error.
#[test]
fn cli_reports_new_and_known_mismatches() -> R {
    let setup = fixture()?.setup;
    let dir = std::env::temp_dir().join(format!("expr-diff-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let fx = serde_json::json!({"setup": setup, "cases": [
        {"sql":"1 + 1","ansi":true,"kids":[],"root":true,"schema":"int","rows":["999"]}]});
    let (fxp, known, out) = (
        dir.join("fx.json"),
        dir.join("known.json"),
        dir.join("out.json"),
    );
    std::fs::write(&fxp, fx.to_string())?;
    let args = |extra: &[&std::path::Path]| {
        let mut a = vec!["--fixture".to_owned(), fxp.display().to_string()];
        for (flag, p) in ["--known", "--out"].iter().zip(extra) {
            a.extend([(*flag).to_owned(), p.display().to_string()]);
        }
        a
    };
    let (mut o, mut e) = (Vec::new(), Vec::new());
    std::fs::write(&known, "[]")?;
    assert!(!cli(args(&[&known, &out]), &mut o, &mut e)?);
    let printed = String::from_utf8(o)?;
    assert!(printed.contains("1|1 + 1"), "{printed}");
    assert!(std::fs::read_to_string(&out)?.contains("\"new\": true"));
    std::fs::write(&known, "[\"1|1 + 1\"]")?;
    assert!(cli(args(&[&known]), &mut Vec::new(), &mut Vec::new())?);
    assert!(cli(args(&[]), &mut Vec::new(), &mut Vec::new())?.eq(&false));
    assert!(
        cli(
            ["--bogus".to_owned(), "x".to_owned()],
            &mut Vec::new(),
            &mut Vec::new()
        )
        .is_err()
    );
    assert!(cli(Vec::new(), &mut Vec::new(), &mut Vec::new()).is_err());
    assert!(
        cli(
            ["--fixture".to_owned(), "/nonexistent".to_owned()],
            &mut Vec::new(),
            &mut Vec::new()
        )
        .is_err()
    );
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

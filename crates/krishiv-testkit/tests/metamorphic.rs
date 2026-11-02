// SPDX-License-Identifier: Apache-2.0
//! TLP and NoREC oracles (P2-T110b) on the session engine: a clean run, three planted bugs found
//! with fixed seeds, and the reducer.
#![forbid(unsafe_code)]

use krishiv_testkit::metamorphic::{
    Cmp, NULLS_SETUP, Oracle, Pred, Rng, SOURCES, check, gen_pred, hunt, hunt_cli, reduce,
    reduce_finding, regression_file, replay_regression, report, tokens,
};
use krishiv_testkit::query_oracle::{QueryRunner, SessionRunner};

type R = Result<(), Box<dyn std::error::Error>>;

fn runner() -> Result<SessionRunner, String> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../krishiv-exec/tests/fixtures/tpch-sf0.01-parquet");
    let mut r = SessionRunner::new(&dir)?;
    r.run(NULLS_SETUP)?;
    Ok(r)
}

/// Planted bug (a): a `NOT` filter over `IN` with NULLs drops the NULLs from the list (a rewrite
/// that treats NULL as false), so `WHERE NOT (x IN (1, NULL))` keeps every `x` outside `1`
/// instead of none. Only that form is wrong, so the three partitions overlap.
struct NullInIgnored(SessionRunner);

impl QueryRunner for NullInIgnored {
    fn run(&mut self, sql: &str) -> Result<Vec<Vec<String>>, String> {
        let sql = if sql.contains("WHERE NOT (") || sql.contains("HAVING NOT (") {
            sql.replace(", NULL)", ")")
        } else {
            sql.to_owned()
        };
        self.0.run(&sql)
    }
}

/// Planted bug (b): a `WHERE` over an outer join is moved into the `ON` clause, as a pushdown
/// below the null-producing side would.
struct PushedIntoOn(SessionRunner);

impl QueryRunner for PushedIntoOn {
    fn run(&mut self, sql: &str) -> Result<Vec<Vec<String>>, String> {
        let sql = if sql.contains("LEFT JOIN") && sql.starts_with("SELECT COUNT(*)") {
            sql.replacen(" WHERE ", " AND ", 1)
        } else {
            sql.to_owned()
        };
        self.0.run(&sql)
    }
}

/// Planted bug (c): a filtered `COUNT(*)` over `orders` loses a row (a pruned row group).
struct PrunedCount(SessionRunner);

impl QueryRunner for PrunedCount {
    fn run(&mut self, sql: &str) -> Result<Vec<Vec<String>>, String> {
        let mut rows = self.0.run(sql)?;
        if sql.starts_with("SELECT COUNT(*) FROM orders o WHERE ") {
            for c in rows.iter_mut().flatten() {
                if let Ok(n @ 1..) = c.parse::<i64>() {
                    *c = (n - 1).to_string();
                }
            }
        }
        Ok(rows)
    }
}

const SEEDS: [u64; 3] = [1, 2, 3];

#[test]
fn clean_engine_has_no_violation() -> R {
    let mut r = runner()?;
    for seed in SEEDS {
        let h = hunt(&mut r, seed, 30);
        assert_eq!(h.checked, 30);
        let mut msg = Vec::new();
        for f in &h.findings {
            msg.push(report(&mut r, f)?);
        }
        assert!(h.findings.is_empty(), "seed {seed}:\n{}", msg.join("\n"));
        assert!(h.errors.is_empty(), "seed {seed}: {:?}", h.errors);
    }
    Ok(())
}

fn detections(mut r: impl QueryRunner, oracle: Oracle) -> usize {
    SEEDS
        .iter()
        .map(|&s| {
            hunt(&mut r, s, 30)
                .findings
                .iter()
                .filter(|f| f.oracle == oracle)
                .count()
        })
        .inspect(|&n| assert!(n >= 1, "a seeded run missed the planted bug"))
        .sum()
}

#[test]
fn planted_null_in_bug_is_found_by_tlp() -> R {
    assert!(detections(NullInIgnored(runner()?), Oracle::TlpWhere) >= 3);
    Ok(())
}

#[test]
fn planted_outer_join_pushdown_is_found_by_norec() -> R {
    assert!(detections(PushedIntoOn(runner()?), Oracle::NoRec) >= 3);
    Ok(())
}

#[test]
fn planted_wrong_count_is_found_by_norec() -> R {
    assert!(detections(PrunedCount(runner()?), Oracle::NoRec) >= 3);
    Ok(())
}

#[test]
fn a_finding_reduces_to_a_few_tokens() -> R {
    let mut bugged = NullInIgnored(runner()?);
    let h = hunt(&mut bugged, 1, 30);
    let f = h
        .findings
        .iter()
        .max_by_key(|f| tokens(&f.pred))
        .ok_or("no finding")?;
    let text = report(&mut bugged, f)?;
    let small = reduce(&f.pred, &mut |c| {
        matches!(check(&mut bugged, f.oracle, f.source, c), Ok(Some(_)))
    });
    assert!(
        tokens(&small) <= 30 || tokens(&small) * 5 <= tokens(&f.pred),
        "{text}"
    );
    assert!(
        matches!(check(&mut bugged, f.oracle, f.source, &small), Ok(Some(_))),
        "the reduced predicate must still fail"
    );
    assert!(text.contains("tokens"));
    Ok(())
}

#[test]
fn predicates_render_and_reduce() {
    let p = Pred::And(
        Box::new(Pred::Cmp("x".into(), Cmp::Lt, "1".into())),
        Box::new(Pred::Not(Box::new(Pred::In(
            "y".into(),
            vec!["1".into(), "NULL".into()],
            false,
        )))),
    );
    assert_eq!(p.to_string(), "(x < 1 AND NOT (y IN (1, NULL)))");
    // Fails while a NULL-bearing IN is present: the comparison and the NOT drop away.
    let small = reduce(&p, &mut |c| c.to_string().contains("NULL"));
    assert_eq!(small.to_string(), "y IN (NULL)");
}

#[test]
fn generator_is_seeded() {
    let draw = |seed| gen_pred(SOURCES[0].cols, 3, &mut Rng::new(seed)).to_string();
    assert_eq!(draw(9), draw(9));
    assert!((0..20).any(|s| draw(s) != draw(s + 100)));
}

#[test]
fn hunt_command_runs_a_budget_and_reports_usage_errors() -> R {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../krishiv-exec/tests/fixtures/tpch-sf0.01-parquet");
    let mut out = Vec::new();
    let args = [
        "--data",
        &dir.display().to_string(),
        "--seed",
        "4",
        "--seconds",
        "0",
    ];
    assert!(hunt_cli(args.map(str::to_owned), &mut out)?);
    assert!(String::from_utf8(out)?.contains("queries_checked_per_s"));
    assert!(hunt_cli(Vec::new(), &mut Vec::new()).is_err());
    assert!(hunt_cli(["--nope".to_owned(), "1".to_owned()], &mut Vec::new()).is_err());
    assert!(hunt_cli(["--seed".to_owned(), "x".to_owned()], &mut Vec::new()).is_err());
    Ok(())
}

#[test]
fn a_reduced_finding_is_written_and_replays_as_failing_then_passing() -> R {
    // Found on the planted-bug engine: the file replays as a failure there and passes on the clean one.
    let mut bugged = NullInIgnored(runner()?);
    let h = hunt(&mut bugged, 1, 30);
    let f = h.findings.first().ok_or("no finding")?;
    let small = reduce_finding(&mut bugged, f);
    let (name, body) = regression_file(f.oracle, f.source, &small, &f.detail);
    assert!(
        name.starts_with(f.oracle.name()) && name.ends_with(".json"),
        "{name}"
    );
    assert_eq!(name, regression_file(f.oracle, f.source, &small, "other").0);
    assert!(replay_regression(&mut bugged, &body)?.is_some());
    assert!(replay_regression(&mut runner()?, &body)?.is_none());
    assert!(replay_regression(&mut bugged, "{}").is_err());
    assert!(
        replay_regression(
            &mut bugged,
            "{\"oracle\":\"norec\",\"source\":99,\"predicate\":\"1=1\"}"
        )
        .is_err()
    );
    Ok(())
}

/// The committed regression files hold on the engine.
#[test]
fn regressions_hold() -> R {
    let dir = krishiv_testkit::tests_dir().join("fixtures/metamorphic/regressions");
    let mut r = runner()?;
    let mut failed = Vec::new();
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json") {
            seen += 1;
            if let Some(why) = replay_regression(&mut r, &std::fs::read_to_string(&path)?)? {
                failed.push(format!("{}: {why}", path.display()));
            }
        }
    }
    assert!(seen > 0, "no regression files in {}", dir.display());
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    Ok(())
}

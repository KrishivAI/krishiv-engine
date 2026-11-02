// SPDX-License-Identifier: Apache-2.0
//! Statement-level reduction and Spark's verdict on a reduced finding (P2-T110b).
#![forbid(unsafe_code)]

use std::collections::HashMap;

use krishiv_testkit::metamorphic::{
    NULLS_ROWS, NULLS_SETUP, Oracle, hunt, nulls_setup, reduce_finding, regression_file_with,
    replay_regression,
};
use krishiv_testkit::metamorphic_reduce::{
    Canned, PythonSpark, Verdict, ddmin, reduce_rows, render, spark_verdict, statements,
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

/// Planted data-dependent bug: while the `nulls` view holds the NaN row, a filtered `COUNT(*)`
/// over it loses a row. Only that row is needed to see it.
struct NanLosesARow {
    inner: SessionRunner,
    has_nan: bool,
}

impl QueryRunner for NanLosesARow {
    fn run(&mut self, sql: &str) -> Result<Vec<Vec<String>>, String> {
        if sql.contains("VIEW nulls AS") {
            self.has_nan = sql.contains("'NaN'");
        }
        let mut rows = self.inner.run(sql)?;
        if self.has_nan && sql.starts_with("SELECT COUNT(*) FROM nulls n WHERE ") {
            for c in rows.iter_mut().flatten() {
                if let Ok(n) = c.parse::<i64>() {
                    *c = (n - 1).to_string();
                }
            }
        }
        Ok(rows)
    }
}

fn bugged() -> Result<NanLosesARow, String> {
    Ok(NanLosesARow {
        inner: runner()?,
        has_nan: true,
    })
}

#[test]
fn ddmin_finds_the_one_needed_subset() {
    let all: Vec<usize> = (0..10).collect();
    let mut calls = 0;
    let kept = ddmin(&all, &mut |s| {
        calls += 1;
        s.contains(&3) && s.contains(&7)
    });
    assert_eq!(kept, vec![3, 7]);
    assert!(calls < 60, "{calls} calls");
    // One needed item, and an input that is already minimal.
    assert_eq!(ddmin(&all, &mut |s| s.contains(&9)), vec![9]);
    assert_eq!(ddmin(&[4], &mut |_| true), vec![4]);
    assert_eq!(ddmin(&[1, 2], &mut |s| s.len() == 2), vec![1, 2]);
}

#[test]
fn nulls_setup_over_every_row_is_the_setup() {
    let all: Vec<usize> = (0..NULLS_ROWS.len()).collect();
    assert_eq!(
        nulls_setup(&all).replace("CREATE OR REPLACE", "CREATE"),
        NULLS_SETUP.split_whitespace().collect::<Vec<_>>().join(" ")
    );
}

fn nan_finding(r: &mut NanLosesARow) -> Result<krishiv_testkit::metamorphic::Finding, String> {
    for seed in 1..20 {
        let h = hunt(r, seed, 30);
        if let Some(f) = h
            .findings
            .into_iter()
            .find(|f| f.oracle == Oracle::NoRec && f.source == 0)
        {
            return Ok(f);
        }
    }
    Err("no NoREC finding over `nulls`".to_owned())
}

#[test]
fn a_finding_reduces_to_the_rows_it_needs() -> R {
    let mut r = bugged()?;
    let f = nan_finding(&mut r)?;
    let small = reduce_finding(&mut r, &f);
    let kept = reduce_rows(&mut r, &f, &small)?;
    assert_eq!(kept, vec![2], "only the NaN row is needed");
    // The runner has the full view again.
    assert_eq!(r.run("SELECT COUNT(*) FROM nulls")?[0][0], "10");
    Ok(())
}

#[test]
fn spark_agreeing_with_the_engine_makes_it_an_oracle_artefact() -> R {
    let mut r = bugged()?;
    let f = nan_finding(&mut r)?;
    let small = reduce_finding(&mut r, &f);
    let kept = reduce_rows(&mut r, &f, &small)?;
    r.run(&nulls_setup(&kept))?;
    let same: HashMap<_, _> = statements(&mut r, f.oracle, f.source, &small)
        .into_iter()
        .collect();
    let v = spark_verdict(&mut r, &mut Canned(same), &f, &small, &kept);
    assert_eq!(v, Verdict::OracleArtefact);
    Ok(())
}

#[test]
fn spark_disagreeing_with_the_engine_keeps_its_rows_as_evidence() -> R {
    let mut r = bugged()?;
    let f = nan_finding(&mut r)?;
    let small = reduce_finding(&mut r, &f);
    let kept = reduce_rows(&mut r, &f, &small)?;
    // What a correct engine (and so Spark) answers over the reduced view.
    let mut clean = runner()?;
    clean.run(&nulls_setup(&kept))?;
    let theirs: HashMap<_, _> = statements(&mut clean, f.oracle, f.source, &small)
        .into_iter()
        .collect();
    r.run(&nulls_setup(&kept))?;
    let v = spark_verdict(&mut r, &mut Canned(theirs), &f, &small, &kept);
    let Verdict::EngineDiffers {
        query,
        engine,
        spark,
    } = &v
    else {
        return Err(format!("{v:?}").into());
    };
    assert!(query.starts_with("SELECT COUNT(*) FROM nulls n WHERE "));
    assert_ne!(engine, spark);
    let text = render(&f, &small, &kept, &v);
    assert!(text.contains("rows [2] of 10") && text.contains("engine differs from Spark 4.2"));
    Ok(())
}

#[test]
fn an_unreachable_spark_is_reported_not_guessed() -> R {
    let mut r = bugged()?;
    let f = nan_finding(&mut r)?;
    let small = reduce_finding(&mut r, &f);
    let mut spark = PythonSpark {
        command: vec!["/nonexistent/spark-command".to_owned()],
        data: std::path::PathBuf::new(),
    };
    let v = spark_verdict(&mut r, &mut spark, &f, &small, &[2]);
    assert!(matches!(v, Verdict::Unavailable(_)), "{v:?}");
    Ok(())
}

#[test]
fn spark_output_is_parsed_strictly() {
    let ok = PythonSpark::parse(
        r#"[{"rows": [["1", "NULL"]]}, {"error": "DIVIDE_BY_ZERO"}]"#,
        2,
    );
    assert_eq!(
        ok,
        Ok(vec![
            Ok(vec![vec!["1".to_owned(), "NULL".to_owned()]]),
            Err("DIVIDE_BY_ZERO".to_owned())
        ])
    );
    assert!(PythonSpark::parse("[]", 1).is_err());
    assert!(PythonSpark::parse("{}", 0).is_err());
    assert!(PythonSpark::parse("nope", 0).is_err());
    assert!(PythonSpark::parse(r#"[{"rows": [[1]]}]"#, 1).is_err());
    assert!(PythonSpark::parse(r"[{}]", 1).is_err());
}

#[test]
fn a_regression_file_replays_on_its_reduced_rows() -> R {
    let mut r = bugged()?;
    let f = nan_finding(&mut r)?;
    let small = reduce_finding(&mut r, &f);
    let (name, body) = regression_file_with(
        f.oracle,
        f.source,
        &small,
        &f.detail,
        Some(&[2]),
        Some("engine differs from Spark 4.2"),
    );
    assert!(name.starts_with("norec-"));
    assert!(body.contains("\"rows\"") && body.contains("\"spark\""));
    // The bug is still there, so the file fails; the clean engine passes it.
    assert!(replay_regression(&mut r, &body)?.is_some());
    assert_eq!(replay_regression(&mut runner()?, &body)?, None);
    // And the full view is back afterwards.
    assert_eq!(r.run("SELECT COUNT(*) FROM nulls")?[0][0], "10");
    Ok(())
}

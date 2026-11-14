// SPDX-License-Identifier: Apache-2.0
//! `TABLESAMPLE` (Spark's `AstBuilder.withSample`): `n ROWS` is a limit, a
//! Bernoulli fraction of 1 or more keeps every row and one of 0 or less
//! none. Every expected answer is what a Spark 4.2.0 run returned.
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn session() -> R<Session> {
    let s = Session::new()?;
    block_on(async {
        s.sql("CREATE TEMPORARY VIEW t AS SELECT * FROM VALUES (0, 'abc'), (1, 'def') t(x, y)")
            .await?
            .collect()
            .await
    })?;
    Ok(s)
}

/// The first column (`INT`) of every row, sorted.
fn xs(s: &Session, sql: &str) -> R<Vec<i32>> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })?;
    let mut out: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_primitive::<Int32Type>();
            (0..c.len()).map(|i| c.value(i)).collect::<Vec<_>>()
        })
        .collect();
    out.sort_unstable();
    Ok(out)
}

/// The error class `sql` fails with; `None` when it runs.
fn class(s: &Session, sql: &str) -> Option<ErrorClass> {
    block_on(async { s.sql(sql).await?.collect().await })
        .err()
        .map(|e| e.error_class())
}

#[test]
fn whole_and_empty_fractions_and_rows() -> R {
    let s = session()?;
    for (sql, want) in [
        ("SELECT * FROM t TABLESAMPLE (100 PERCENT)", vec![0, 1]),
        (
            "SELECT * FROM t TABLESAMPLE (100 PERCENT) REPEATABLE (0)",
            vec![0, 1],
        ),
        (
            "SELECT * FROM t TABLESAMPLE (100.00001 PERCENT)",
            vec![0, 1],
        ),
        (
            "SELECT * FROM t TABLESAMPLE (BUCKET 1 OUT OF 1)",
            vec![0, 1],
        ),
        (
            "SELECT * FROM t TABLESAMPLE (BUCKET 2 OUT OF 2)",
            vec![0, 1],
        ),
        ("SELECT * FROM t TABLESAMPLE (0 PERCENT)", vec![]),
        ("SELECT * FROM t TABLESAMPLE (-0.00001 PERCENT)", vec![]),
        ("SELECT * FROM t TABLESAMPLE (0 ROWS)", vec![]),
        (
            "SELECT * FROM t TABLESAMPLE (2 ROWS) REPEATABLE (0)",
            vec![0, 1],
        ),
        ("SELECT s.x FROM t TABLESAMPLE (100 PERCENT) s", vec![0, 1]),
        (
            "SELECT * FROM (SELECT x FROM t) TABLESAMPLE (100 PERCENT)",
            vec![0, 1],
        ),
        (
            "TABLE t |> TABLESAMPLE (100 PERCENT) |> TABLESAMPLE (5 ROWS)",
            vec![0, 1],
        ),
    ] {
        assert_eq!(xs(&s, sql)?, want, "{sql}");
    }
    // One of the two rows (which one is the plan's first).
    assert_eq!(xs(&s, "SELECT * FROM t TABLESAMPLE (1 ROWS) s")?.len(), 1);
    Ok(())
}

#[test]
fn rows_are_checked_as_a_limit() -> R {
    let s = session()?;
    assert_eq!(
        class(&s, "TABLE t |> TABLESAMPLE (-5 ROWS)"),
        Some(ErrorClass::InvalidLimitLikeExpressionIsNegative)
    );
    assert_eq!(
        class(&s, "TABLE t |> TABLESAMPLE (x ROWS)"),
        Some(ErrorClass::InvalidLimitLikeExpressionIsUnfoldable)
    );
    Ok(())
}

#[test]
fn system_sampling_is_refused() -> R {
    // Partial fractions and buckets are sampled (`tablesample_goldens.rs`).
    let s = session()?;
    let sql = "SELECT * FROM t TABLESAMPLE SYSTEM (100 PERCENT)";
    assert_eq!(
        class(&s, sql),
        Some(ErrorClass::KrishivUnsupportedFeature),
        "{sql}"
    );
    Ok(())
}

// SPDX-License-Identifier: Apache-2.0
//! `window()`, `session_window()` and `window_time()` through SQL. The expected values come from a
//! Spark 4.3.0 run (`spark-sql --master local[1]`, UTC) of the same statements.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// The rows of `sql`, cells joined by tabs (every column selected as a string).
fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })?;
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(r) {
                        "NULL".to_owned()
                    } else {
                        match c.as_string_opt::<i32>() {
                            Some(a) => a.value(r).to_owned(),
                            None => c.as_string_view().value(r).to_owned(),
                        }
                    }
                })
                .collect();
            out.push(cells.join("\t"));
        }
    }
    Ok(out)
}

fn utc() -> R<Session> {
    let s = Session::new()?;
    block_on(async {
        s.sql("SET spark.sql.session.timeZone=UTC")
            .await?
            .collect()
            .await
    })?;
    Ok(s)
}

#[test]
#[ignore = "known gap (docs/concepts/known-differences.md): Krishiv leaves a string time column a string"]
fn a_session_window_over_a_string_column_has_timestamp_fields() -> R {
    // Spark: timestamp, timestamp, 2024-01-01 00:00:00, 2024-01-01 00:05:00 and the same at 00:02.
    let s = utc()?;
    let got = rows(
        &s,
        "SELECT typeof(w.start), typeof(w.end), CAST(w.start AS STRING), CAST(w.end AS STRING) \
         FROM (SELECT session_window(s, '5 minutes') AS w \
               FROM VALUES ('2024-01-01 00:00:00'), ('2024-01-01 00:02:00') AS t(s))",
    )?;
    assert_eq!(
        got,
        [
            "timestamp\ttimestamp\t2024-01-01 00:00:00\t2024-01-01 00:05:00",
            "timestamp\ttimestamp\t2024-01-01 00:02:00\t2024-01-01 00:07:00",
        ]
    );
    Ok(())
}

#[test]
fn rows_with_one_timestamp_are_one_session() -> R {
    // Spark: [(00:00, 00:05, 3), (00:10, 00:15, 1)].
    let s = utc()?;
    for _ in 0..20 {
        let got = rows(
            &s,
            "SELECT CAST(session_window.start AS STRING), CAST(session_window.end AS STRING), \
                    CAST(count(*) AS STRING) \
             FROM VALUES (timestamp'2024-01-01 00:00:00'), (timestamp'2024-01-01 00:00:00'), \
                         (timestamp'2024-01-01 00:00:00'), (timestamp'2024-01-01 00:10:00') AS t(ts) \
             GROUP BY session_window(ts, '5 minutes') ORDER BY 1",
        )?;
        assert_eq!(
            got,
            [
                "2024-01-01 00:00:00\t2024-01-01 00:05:00\t3",
                "2024-01-01 00:10:00\t2024-01-01 00:15:00\t1",
            ]
        );
    }
    Ok(())
}

#[test]
fn window_time_inside_an_aggregate_of_a_grouped_window() -> R {
    // Spark: 2024-01-01 00:04:59.999999.
    let s = utc()?;
    let got = rows(
        &s,
        "SELECT CAST(max(window_time(window)) AS STRING) \
         FROM VALUES (timestamp'2024-01-01 00:01:00'), (timestamp'2024-01-01 00:02:00') AS t(ts) \
         GROUP BY window(ts, '5 minutes')",
    )?;
    assert_eq!(got, ["2024-01-01 00:04:59.999999"]);
    Ok(())
}

#[test]
fn an_enormous_overlap_is_an_error_not_an_abort() -> R {
    // One day of window over a one microsecond slide is 86.4 billion windows for one row.
    let s = utc()?;
    let r = block_on(async {
        s.sql(
            "SELECT window(ts, '1 day', '1 microsecond') FROM VALUES (timestamp'2024-01-01 00:00:00') AS t(ts)",
        )
        .await?
        .collect()
        .await
    });
    assert!(r.is_err(), "expected an error for 86.4 billion windows");
    Ok(())
}

#[test]
fn a_window_inside_an_aggregate_of_a_nested_grouped_window_is_the_inner_window() -> R {
    // Spark 4.3.0 (UTC): `min(window)` = {00:01:00, 00:02:00}, `max(window.start)` = 00:03:00, and
    // `window.start` outside an aggregate is MISSING_AGGREGATION, as the bare `window` is.
    let s = utc()?;
    let from = "FROM VALUES (timestamp'2026-01-01 00:01:30'), (timestamp'2026-01-01 00:03:30') AS t(ts) \
                GROUP BY window(window(ts, '1 minute'), '5 minutes')";
    let got = rows(
        &s,
        &format!(
            "SELECT CAST(min(window) AS STRING), CAST(max(window.start) AS STRING), \
                    CAST(max(window.end) AS STRING), CAST(count(*) AS STRING) {from}"
        ),
    )?;
    assert_eq!(
        got,
        ["{2026-01-01 00:01:00, 2026-01-01 00:02:00}\t2026-01-01 00:03:00\t2026-01-01 00:04:00\t2"]
    );
    let err = block_on(async {
        s.sql(&format!("SELECT window.start, count(*) {from}"))
            .await?
            .collect()
            .await
    });
    let text = format!("{:?}", err.err().ok_or("expected MISSING_AGGREGATION")?);
    assert!(text.contains("MissingAggregation"), "{text}");
    // Spark 4.3.0: inside an aggregate of HAVING or ORDER BY the inner window does not resolve.
    for tail in [
        "HAVING max(window.start) = timestamp'2026-01-01 00:03:00'",
        "ORDER BY max(window.start)",
    ] {
        let err = block_on(async {
            s.sql(&format!("SELECT count(*) {from} {tail}"))
                .await?
                .collect()
                .await
        });
        let text = format!("{:?}", err.err().ok_or("expected UNRESOLVED_COLUMN")?);
        assert!(
            text.contains("UnresolvedColumnWithSuggestion"),
            "{tail}: {text}"
        );
        assert!(
            text.contains("(\"proposal\", \"`count(1)`\")"),
            "{tail}: {text}"
        );
    }
    // The proposal lists the aggregate's output names (Spark 4.3.0), and a select list that reads the window
    // outside an aggregate is MISSING_AGGREGATION first.
    let having = "HAVING max(window.start) = timestamp'2026-01-01 00:03:00'";
    for (select, tail, want) in [
        ("count(*) AS c", having, "(\"proposal\", \"`c`\")"),
        (
            "max(window.start) AS m, count(*)",
            having,
            "(\"proposal\", \"`count(1)`, `m`\")",
        ),
        ("window, count(*)", having, "MissingAggregation"),
        (
            "window, count(*)",
            "ORDER BY max(window.start)",
            "MissingAggregation",
        ),
    ] {
        let err = block_on(async {
            s.sql(&format!("SELECT {select} {from} {tail}"))
                .await?
                .collect()
                .await
        });
        let text = format!("{:?}", err.err().ok_or("expected an error")?);
        assert!(text.contains(want), "{select} {tail}: {text}");
    }
    Ok(())
}

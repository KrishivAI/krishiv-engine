// SPDX-License-Identifier: Apache-2.0
//! `sequence` over dates and timestamps against the Spark 4.2.0 runs in `tests/spark-golden/functions/sequence_xml/results/`
//! (SQL-6): every statement of the case files runs in order in one session; a statement whose Spark output is a value is compared
//! as text, one whose output is an exception must fail here too (the messages are not compared). The statements that still differ
//! are listed in `KNOWN_GAPS` and must keep differing (a ratchet, as the golden runner's).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

const FILES: &[&str] = &[
    "sequence_dates",
    "sequence_timestamps",
    "sequence_tz",
    "sequence_calendar",
];

/// `file#n` (1-based statement number) the engine does not match yet; each is a Spark quirk or a missing feature named in the
/// golden README.
const KNOWN_GAPS: &[&str] = &[
    // TIMESTAMP beside TIMESTAMP_NTZ: Spark casts the NTZ bound; the registry's input rule does not yet.
    "sequence_timestamps#49",
    // `spark.sql.legacy.interval.enabled` interval literals (`interval 1 month 2 days`, `interval 6 hours` as CALENDAR): a parser feature.
    "sequence_calendar#68",
    "sequence_calendar#69",
    "sequence_calendar#70",
    "sequence_calendar#71",
    "sequence_calendar#72",
    "sequence_calendar#73",
    "sequence_calendar#74",
];

struct Case {
    sql: String,
    output: String,
}

fn cases(file: &str) -> Vec<Case> {
    let path = format!(
        "{}/../../tests/spark-golden/functions/sequence_xml/results/spark-4.2.0/{file}.sql.out",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("golden file");
    let mut out = Vec::new();
    for block in text.split("-- !query\n").skip(1) {
        let (sql, rest) = block
            .split_once("-- !query schema\n")
            .expect("schema marker");
        let (_, output) = rest
            .split_once("-- !query output\n")
            .expect("output marker");
        out.push(Case {
            sql: sql.trim().to_owned(),
            output: output.trim().to_owned(),
        });
    }
    out
}

fn is_error(output: &str) -> bool {
    output.starts_with("org.apache.") || output.starts_with("java.") || output.starts_with("scala.")
}

fn render(s: &Session, sql: &str) -> Result<Vec<String>, String> {
    let wrapped = format!("SELECT CAST(x AS STRING) FROM ({sql}) AS t(x)");
    let df = block_on(s.sql(&wrapped)).map_err(|e| e.to_string())?;
    let batches = block_on(df.collect()).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for b in &batches {
        let col = b.column(0).as_string_view();
        for i in 0..col.len() {
            rows.push(if col.is_null(i) {
                "NULL".to_owned()
            } else {
                col.value(i).to_owned()
            });
        }
    }
    Ok(rows)
}

#[test]
fn sequence_over_dates_and_timestamps_follows_the_spark_runs() {
    let mut gaps: Vec<String> = Vec::new();
    for file in FILES {
        let s = Session::new().unwrap();
        for (n, case) in cases(file).iter().enumerate() {
            let id = format!("{file}#{}", n + 1);
            let upper = case.sql.to_ascii_uppercase();
            if !upper.starts_with("SELECT") {
                // SET, CREATE VIEW ...: run for their effect.
                let _ = block_on(s.sql(&case.sql)).and_then(|df| block_on(df.collect()));
                continue;
            }
            let got = render(&s, &case.sql);
            let ok = if is_error(&case.output) {
                got.is_err()
            } else {
                match &got {
                    Ok(rows) => rows.join("\n") == case.output,
                    Err(_) => false,
                }
            };
            if !ok {
                gaps.push(format!("{id}: {}", case.sql));
                eprintln!(
                    "MISMATCH {id}: {}\n  want {}\n  got  {got:?}",
                    case.sql, case.output
                );
            }
        }
    }
    let ids: Vec<&str> = gaps
        .iter()
        .map(|g| g.split(':').next().unwrap_or(""))
        .collect();
    assert_eq!(ids, KNOWN_GAPS, "mismatches: {gaps:#?}");
}

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
    "sequence_errors",
];

/// `file#n` (1-based statement number) the engine does not match yet; each is a Spark quirk or a missing feature named in the
/// golden README.
const KNOWN_GAPS: &[&str] = &[
    // TIMESTAMP beside TIMESTAMP_NTZ: Spark casts the NTZ bound; the registry's input rule does not yet.
    "sequence_timestamps#49",
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

fn render(s: &Session, sql: &str) -> Result<Vec<String>, krishiv_common::error::KrishivError> {
    // Every column is cast to a string (a tab joins them), so a statement of any width is compared as Spark prints it.
    let width = block_on(block_on(s.sql(sql))?.schema())?.fields().len();
    let names: Vec<String> = (0..width).map(|i| format!("c{i}")).collect();
    let casts: Vec<String> = names
        .iter()
        .map(|n| format!("CAST({n} AS STRING)"))
        .collect();
    let wrapped = format!(
        "SELECT {} FROM ({sql}) AS t({})",
        casts.join(", "),
        names.join(", ")
    );
    let df = block_on(s.sql(&wrapped))?;
    let batches = block_on(df.collect())?;
    let mut rows = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = (0..width)
                .map(|c| {
                    let col = b.column(c).as_string_view();
                    if col.is_null(i) {
                        "NULL".to_owned()
                    } else {
                        col.value(i).to_owned()
                    }
                })
                .collect();
            rows.push(cells.join("\t"));
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
                // A setup statement that Spark ran must run here too, or the later results mean nothing.
                let done = block_on(s.sql(&case.sql)).and_then(|df| block_on(df.collect()));
                if let Err(e) = done
                    && !is_error(&case.output)
                {
                    panic!("setup statement {id} failed: {}: {e:?}", case.sql);
                }
                continue;
            }
            let got = render(&s, &case.sql);
            let ok = if is_error(&case.output) {
                // A failure, and when Spark names an error class (a `_LEGACY_ERROR_TEMP_*` too) the same class; only an exception
                // without a class is checked as a failure alone.
                let want = case
                    .output
                    .lines()
                    .find_map(|l| l.strip_prefix("errorClass: "))
                    .map(|c| c.trim().to_owned());
                match (&got, want) {
                    (Err(e), Some(c)) => e.error_class().name() == c,
                    (Err(_), _) => true,
                    _ => false,
                }
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

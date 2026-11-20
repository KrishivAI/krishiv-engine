// SPDX-License-Identifier: Apache-2.0
//! `from_protobuf` and `to_protobuf` against the Spark runs in `tests/spark-golden/functions/from_to_protobuf/results/` (P3-T087): every
//! statement of a case file runs in order in one session; a statement whose Spark output is a value is compared as text, one whose
//! output is an exception must fail here with the same error class. The statements that still differ are listed in `KNOWN_GAPS` (and
//! the per-version extras) and must keep differing (a ratchet, as the Avro runner's). The goldens name the descriptor files under
//! `/golden/`, the read-only mount of the Spark containers; here that is the golden tree itself.
//!
//! 3.5.7 has no SQL function (its goldens come from the DataFrame API with other column names): that version only checks that the
//! SQL names do not resolve.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

/// The case files, in `inputs/`.
const FILES: &[&str] = &[
    "basic",
    "conditional",
    "errors_args",
    "modes_options",
    "ansi_tz",
    "proto2",
    "to_types",
    "types_any",
    "types_empty",
    "types_kinds",
    "types_recursion",
    "types_tz",
    "types_wkt",
];

/// (golden directory, session profile) of each version that has the SQL functions. 4.3.0 differs from 4.2.0 in one message text.
const VERSIONS: &[(&str, SparkProfile)] = &[
    ("spark-4.0.0", SparkProfile::V4_0),
    ("spark-4.1.0", SparkProfile::V4_1),
    ("spark-4.2.0", SparkProfile::V4_2),
    ("spark-4.3.0", SparkProfile::V4_2),
];

/// `file#n` (1-based statement number) the engine does not match yet, on every version.
const KNOWN_GAPS: &[&str] = &[
    // A SQL string or binary holding bytes that are not UTF-8 (`cast(X'ff' as string)`, a `bytes` field printed inside a struct):
    // the engine's cast to string refuses them where Spark prints U+FFFD. The protobuf functions themselves agree with Spark here.
    "basic#12",
    "basic#53",
    "types_kinds#4",
    "types_kinds#5",
    "types_kinds#6",
    // `spark.sql.caseSensitive=true` is not read when the call is bound (the binder sees no session configuration): a struct that
    // names two proto fields that differ only by case is refused as in the case-insensitive default.
    "proto2#43",
    "proto2#44",
    // Spark raises `INTERNAL_ERROR` for a first argument of `from_protobuf` that is not BINARY (a Spark bug, "the replacement is
    // unresolved"); the engine raises the proper `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`, as for `from_avro`.
    "ansi_tz#8",
    "ansi_tz#9",
    "ansi_tz#13",
    "ansi_tz#14",
    "basic#102",
    "errors_args#16",
    "errors_args#17",
    "errors_args#18",
];

/// Extra `file#n` that differ on one version only: (golden directory, id).
const VERSION_GAPS: &[(&str, &str)] = &[
    // `spark.sql.function.protobufExtensions.enabled` (Spark 4.2+): proto2 extension fields are not read or written.
    ("spark-4.2.0", "proto2#23"),
    ("spark-4.2.0", "proto2#24"),
    ("spark-4.2.0", "proto2#25"),
    ("spark-4.3.0", "proto2#23"),
    ("spark-4.3.0", "proto2#24"),
    ("spark-4.3.0", "proto2#25"),
];

struct Case {
    sql: String,
    output: String,
}

fn golden_root() -> String {
    format!(
        "{}/../../tests/spark-golden/functions",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn cases(version: &str, file: &str) -> Vec<Case> {
    let path = format!(
        "{}/from_to_protobuf/results/{version}/{file}.sql.out",
        golden_root()
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
    output.starts_with("org.apache.")
        || output.starts_with("org.sparkproject.")
        || output.starts_with("java.")
        || output.starts_with("scala.")
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

fn run_version(version: &str, profile: SparkProfile) -> Vec<String> {
    let mut gaps: Vec<String> = Vec::new();
    let root = golden_root();
    for file in FILES {
        let s = Session::builder().profile(profile).build().unwrap();
        // The goldens were run in UTC (their README); the host zone must not leak into the results.
        block_on(
            block_on(s.sql("SET spark.sql.session.timeZone=UTC"))
                .unwrap()
                .collect(),
        )
        .unwrap();
        for (n, case) in cases(version, file).iter().enumerate() {
            let id = format!("{file}#{}", n + 1);
            let sql = case.sql.replace("/golden/", &format!("{root}/"));
            let upper = sql.to_ascii_uppercase();
            if upper.starts_with("DESCRIBE") || upper.starts_with("EXPLAIN") {
                continue;
            }
            if !upper.starts_with("SELECT") {
                // SET, CREATE VIEW ...: run for their effect.
                let done = block_on(s.sql(&sql)).and_then(|df| block_on(df.collect()));
                if let Err(e) = done
                    && !is_error(&case.output)
                {
                    panic!("setup statement {id} failed: {}: {e:?}", case.sql);
                }
                continue;
            }
            let got = render(&s, &sql);
            let ok = if is_error(&case.output) {
                // A failure, and when Spark names an error class the same class; an exception without a class is checked as a failure alone.
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
                    "MISMATCH {version} {id}: {}\n  want {}\n  got  {got:?}",
                    case.sql, case.output
                );
            }
        }
    }
    gaps
}

#[test]
fn protobuf_functions_follow_the_spark_runs() {
    let mut failures = Vec::new();
    for (version, profile) in VERSIONS {
        let gaps = run_version(version, *profile);
        let mut have: Vec<&str> = gaps
            .iter()
            .map(|g| g.split(':').next().unwrap_or(""))
            .collect();
        have.sort_unstable();
        let mut want: Vec<&str> = KNOWN_GAPS.to_vec();
        want.extend(
            VERSION_GAPS
                .iter()
                .filter(|(v, _)| v == version)
                .map(|(_, id)| *id),
        );
        want.sort_unstable();
        if have != want {
            let new: Vec<&&str> = have.iter().filter(|h| !want.contains(h)).collect();
            let fixed: Vec<&&str> = want.iter().filter(|w| !have.contains(w)).collect();
            failures.push(format!(
                "{version}: new mismatches {new:?}, fixed (remove from the gap list) {fixed:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn spark_3_5_has_no_sql_protobuf_functions() {
    let s = Session::builder()
        .profile(SparkProfile::V3_5)
        .build()
        .unwrap();
    let err = block_on(s.sql("SELECT from_protobuf(X'0801', 'a.B', X'')"))
        .and_then(|df| block_on(df.collect()))
        .unwrap_err();
    assert_eq!(err.error_class().name(), "UNRESOLVED_ROUTINE");
}

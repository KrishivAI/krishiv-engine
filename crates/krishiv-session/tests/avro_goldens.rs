// SPDX-License-Identifier: Apache-2.0
//! `from_avro`, `to_avro` and `is_valid_variant` against the Spark 4.2.0 runs in `tests/spark-golden/functions/{from_to_avro,variant_misc}/
//! results/` (T087, T088): every statement of a case file runs in order in one session; a statement whose Spark output is a value is
//! compared as text, one whose output is an exception must fail here with the same error class. The statements that still differ are
//! listed in `KNOWN_GAPS` and must keep differing (a ratchet, as the golden runner's).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

/// (directory, file) pairs of the golden cases that run here.
const FILES: &[(&str, &str)] = &[
    ("from_to_avro", "basic"),
    ("from_to_avro", "conditional"),
    ("from_to_avro", "errors_args"),
    ("from_to_avro", "modes_options"),
    ("from_to_avro", "ansi_tz"),
    ("variant_misc", "is_valid_variant"),
    ("variant_misc", "variant_edge"),
];

/// `file#n` (1-based statement number) the engine does not match yet: a Spark quirk, a missing feature, or the nullability
/// difference of `to_avro` (see each group's comment). Also a difference no golden shows: `to_avro` rescales a decimal to the
/// schema's precision and scale where Spark's `AvroSerializer` requires them to be equal (the Avro file writer relies on the
/// rescale).
const KNOWN_GAPS: &[&str] = &[
    // Spark types a cast of a literal as nullable (`to_avro` infers `[T, "null"]` and writes the branch index); a folded literal reaches the binding without that nullability.
    "basic#29",
    "basic#30",
    "basic#31",
    "basic#34",
    // `to_avro` of an untyped NULL (`NullType`) is not bound.
    "basic#23",
    // A `timestamp-millis` target schema for a timestamp value is not written yet.
    "basic#52",
    // A SQL string holding bytes that are not UTF-8 (`cast(X'ff' as string)`): strings here are always valid UTF-8.
    "ansi_tz#40",
    "ansi_tz#41",
    "basic#57",
    "basic#149",
    // `to_avro` infers its schema from the physical nullability of the argument, where Spark uses the analyzer's (a `cast` of a
    // literal is nullable there): the bytes differ by the union branch index. Needs the analyzer's nullability, or owner sign-off.
    "basic#133",
    "basic#140",
    // Spark quirks the owner put under opt-in incompatibility: a `["null","int"]` reader returns 0 for the null branch.
    "basic#78",
    "basic#91",
    "basic#92",
    // Avro unions of several non-null types read or written as `member0`..`memberN` structs, and `["null"]` as an empty struct.
    "basic#86",
    "basic#89",
    "basic#106",
    "basic#107",
    "basic#110",
    // Spark raises INTERNAL_ERROR for a first argument that is not BINARY (a Spark bug); the engine raises the proper DATATYPE_MISMATCH.
    "ansi_tz#8",
    "ansi_tz#9",
    "ansi_tz#18",
    "ansi_tz#19",
    "basic#113",
    "errors_args#13",
    "errors_args#14",
    "errors_args#16",
    // `schema_of_avro` is not implemented.
    "basic#150",
    "basic#151",
    // A NULL map key in the options: Spark raises NULL_MAP_KEY, here the options are refused as not a literal.
    "basic#116",
    // A schema or options map that is foldable but not a literal (`concat(..)`, `map_concat(..)`): refused here
    // (UNSUPPORTED_FEATURE) where Spark folds it.
    "errors_args#21",
    "errors_args#22",
    "errors_args#23",
    // The `avroSchema` option (a reader schema).
    "modes_options#14",
    "modes_options#16",
    // `enableStableIdentifiersForUnionType` and its prefix, and their validation.
    "modes_options#18",
    "modes_options#19",
    "modes_options#20",
    "modes_options#21",
    "modes_options#22",
    "modes_options#23",
    // `recursiveFieldMaxDepth` (recursive Avro schemas).
    "modes_options#25",
    "modes_options#26",
    "modes_options#28",
    // `datetimeRebaseMode` on a date read.
    "modes_options#33",
    "modes_options#34",
    "modes_options#35",
];

struct Case {
    sql: String,
    output: String,
}

fn cases(dir: &str, file: &str) -> Vec<Case> {
    let path = format!(
        "{}/../../tests/spark-golden/functions/{dir}/results/spark-4.2.0/{file}.sql.out",
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
fn avro_and_variant_functions_follow_the_spark_runs() {
    let mut gaps: Vec<String> = Vec::new();
    for (dir, file) in FILES {
        let s = Session::new().unwrap();
        // The goldens were run in UTC (their README); the host zone must not leak into the results.
        block_on(
            block_on(s.sql("SET spark.sql.session.timeZone=UTC"))
                .unwrap()
                .collect(),
        )
        .unwrap();
        for (n, case) in cases(dir, file).iter().enumerate() {
            let id = format!("{file}#{}", n + 1);
            let upper = case.sql.to_ascii_uppercase();
            if upper.starts_with("DESCRIBE") || upper.starts_with("EXPLAIN") {
                continue;
            }
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
    let mut want: Vec<&str> = KNOWN_GAPS.to_vec();
    want.sort_unstable();
    let mut have = ids;
    have.sort_unstable();
    assert_eq!(have, want, "mismatches: {gaps:#?}");
}

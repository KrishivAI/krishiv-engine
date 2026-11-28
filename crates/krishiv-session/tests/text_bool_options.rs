// SPDX-License-Identifier: Apache-2.0
//! The boolean options of the JSON, CSV and XML expressions and file reads
//! against the Spark 4.2.1 and 4.3.0 runs of
//! `tests/scripts/probe_text_bool_options.py`
//! (`tests/fixtures/spark-<v>/text-bool-options.json`). Spark reads them in
//! two ways, never trimmed: Scala's `toBoolean` (every JSON flag, `multiLine`,
//! `enableDateTimeParsingFallback`, XML `prefersDecimal` and
//! `useLegacyXMLParser`: Java's `equalsIgnoreCase`, so `fal\u{17f}e` is
//! false; anything else a classless `IllegalArgumentException`, here
//! `KRISHIV_INVALID_FORMAT_OPTION_VALUE`), and `CSVOptions` / `XmlOptions`
//! `getBool` (the rest: `toLowerCase`, else `_LEGACY_ERROR_TEMP_2147`).
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn lit(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// Options Krishiv refuses for `entry` with `KRISHIV_UNSUPPORTED_FEATURE`
/// when set to `true` (not implemented there; Spark formats the output with
/// them). `false` and a value that is not a boolean still go as in Spark.
fn unsupported(format: &str, option: &str, entry: &str, value: &str) -> bool {
    let on = value.eq_ignore_ascii_case("true");
    match format {
        "json" => {
            on && entry == "to_json"
                && matches!(option, "pretty" | "writeNonAsciiCharacterAsCodePoint")
        }
        "xml" => {
            entry != "schema_of_xml" && matches!(option, "excludeAttribute" | "ignoreNamespace")
        }
        _ => false,
    }
}

fn run(
    s: &Session,
    dir: &std::path::Path,
    format: &str,
    entry: &str,
    o: &str,
    v: &str,
) -> Result<(), KrishivError> {
    let m = format!("map({}, {})", lit(o), lit(v));
    let sql = match entry {
        "from_json" => format!("SELECT from_json('{{\"a\":1}}', 'a int', {m})"),
        "to_json" => format!("SELECT to_json(named_struct('a', 1, 'b', null), {m})"),
        "schema_of_json" => format!("SELECT schema_of_json('{{\"a\":1}}', {m})"),
        "from_csv" | "from_csv_legacy" => format!("SELECT from_csv('1', 'a int', {m})"),
        "to_csv" | "to_csv_legacy" => format!("SELECT to_csv(named_struct('a', 1), {m})"),
        "schema_of_csv" => format!("SELECT schema_of_csv('1', {m})"),
        "from_xml" => format!("SELECT from_xml('<r><a>1</a></r>', 'a int', {m})"),
        "to_xml" => format!("SELECT to_xml(named_struct('a', 1), {m})"),
        "schema_of_xml" => format!("SELECT schema_of_xml('<r><a>1</a></r>', {m})"),
        "read" | "read_legacy" => {
            // The test writes both files before the first read.
            let file = dir.join(if format == "json" { "a.json" } else { "a.csv" });
            let df = s
                .read()
                .format(format)
                .option(o, v)
                .load([file.to_str().unwrap()]);
            return block_on(df.collect()).map(|_| ());
        }
        other => panic!("unknown entry {other}"),
    };
    block_on(async { s.sql(&sql).await?.collect().await }).map(|_| ())
}

#[test]
fn boolean_options_are_read_as_spark_reads_them() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.json"), "{\"a\": 1}\n").unwrap();
    std::fs::write(dir.path().join("a.csv"), "a\n1\n").unwrap();
    let mut checked = 0;
    let mut wrong = Vec::new();
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/text-bool-options.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let probe: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(probe["spark"], version);
        let s = Session::builder().profile(profile).build().unwrap();
        let legacy = Session::builder().profile(profile).build().unwrap();
        block_on(legacy.sql("SET spark.sql.legacy.timeParserPolicy = LEGACY")).unwrap();
        for format in ["json", "csv", "xml"] {
            for (option, values) in probe[format].as_object().unwrap() {
                for (value, entries) in values.as_object().unwrap() {
                    for (entry, want) in entries.as_object().unwrap() {
                        let want = want.as_str().unwrap();
                        let session = if entry.ends_with("_legacy") {
                            &legacy
                        } else {
                            &s
                        };
                        let at = format!("{version} {format} {entry} {option}={value:?}");
                        let got = run(session, dir.path(), format, entry, option, value);
                        let ok = match (want, &got) {
                            ("ok", Ok(())) => true,
                            ("ok", Err(e)) => {
                                unsupported(format, option, entry, value)
                                    && e.error_class() == ErrorClass::KrishivUnsupportedFeature
                            }
                            (_, Ok(())) => false,
                            ("java.lang.IllegalArgumentException", Err(e)) => {
                                e.error_class() == ErrorClass::KrishivInvalidFormatOptionValue
                            }
                            (class, Err(e)) => e.error_class().name() == class,
                        };
                        if !ok {
                            let got = got.map_err(|e| e.to_string());
                            wrong.push(format!("{at}: Spark {want}, Krishiv {got:?}"));
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} cases differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    assert_eq!(checked, 2 * 11 * (18 * 4 + 11 * 7 + 9 * 3));
}

/// The first column of the first row, as text.
fn first(s: &Session, sql: &str) -> String {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let b = batches.iter().find(|b| b.num_rows() > 0).unwrap();
    let c = arrow_cast::cast(b.column(0), &arrow_schema::DataType::Utf8).unwrap();
    arrow_array::cast::AsArray::as_string::<i32>(&c)
        .value(0)
        .to_owned()
}

/// The `single_quotes` section: `JSONOptions` allows single-quoted names and
/// strings unless `allowSingleQuotes=false`, in file reads and expressions.
/// The `csv_values` section: `inferSchema`, `enforceSchema`, `columnPruning`
/// and `multiLine` change none of the CSV expressions' results.
#[test]
fn csv_expressions_ignore_the_file_only_flags_as_spark_does() {
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/text-bool-options.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let probe: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let s = Session::builder().profile(profile).build().unwrap();
        let cases = probe["csv_values"].as_object().unwrap();
        assert_eq!(cases.len(), 12, "{version}");
        for (case, want) in cases {
            let (o, v) = case.split_once('=').unwrap();
            let m = if v == "None" {
                String::new()
            } else {
                format!(", map({}, {})", lit(o), lit(v))
            };
            for (entry, sql) in [
                (
                    "from_csv",
                    format!("SELECT to_json(from_csv('1,x', 'a int, b string'{m}))"),
                ),
                (
                    "to_csv",
                    format!("SELECT to_csv(named_struct('a', 1, 'b', 'x'){m})"),
                ),
                ("schema_of_csv", format!("SELECT schema_of_csv('1,x'{m})")),
            ] {
                assert_eq!(
                    first(&s, &sql),
                    want[entry].as_str().unwrap(),
                    "{version} {case} {entry}"
                );
            }
        }
    }
}

#[test]
fn single_quotes_are_allowed_by_default_as_spark_does() {
    let dir = tempfile::tempdir().unwrap();
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/text-bool-options.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let probe: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let want = &probe["single_quotes"];
        let s = Session::builder().profile(profile).build().unwrap();
        let file = dir.path().join(format!("{version}.json"));
        std::fs::write(&file, "{'a': 'x', 'b': 1}\n").unwrap();
        let df = s.read().format("json").load([file.to_str().unwrap()]);
        block_on(df.create_or_replace_temp_view("v")).unwrap();
        let schema = first(
            &s,
            "SELECT concat('struct<a:', typeof(a), ',b:', typeof(b), '>') FROM v",
        );
        assert_eq!(schema, want["read_schema"].as_str().unwrap(), "{version}");
        let rows = first(&s, "SELECT to_json(array(a, cast(b AS STRING))) FROM v");
        let want_rows = want["read_rows"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.to_string().trim_matches('"').to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            format!("[\"{}\",\"{}\"]", want_rows[0], want_rows[1]),
            "{version}"
        );
        for (key, sql) in [
            (
                "from_json",
                r#"SELECT to_json(from_json("{'a': 'x'}", 'a string'))"#,
            ),
            ("schema_of_json", r#"SELECT schema_of_json("{'a': 'x'}")"#),
            (
                "from_json_off",
                r#"SELECT to_json(from_json("{'a': 'x'}", 'a string', map('allowSingleQuotes', 'false')))"#,
            ),
        ] {
            assert_eq!(
                first(&s, sql),
                want[key].as_str().unwrap(),
                "{version} {key}"
            );
        }
    }
}

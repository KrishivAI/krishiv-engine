// SPDX-License-Identifier: Apache-2.0
//! The YAML read against the metric-view goldens of Spark 4.2.1 and 4.3.0
//! (`tests/fixtures/spark-{4.2.1,4.3.0}/metric-views/cases.json`).

use super::*;
use krishiv_sql::ast::Statement;

/// Classes Spark raises before it reads the YAML (the parser's, and the clause checks).
const BEFORE_YAML: &[&str] = &[
    "PARSE_SYNTAX_ERROR",
    "DUPLICATE_CLAUSES",
    "CREATE_VIEW_WITH_IF_NOT_EXISTS_AND_REPLACE",
    "_LEGACY_ERROR_TEMP_0035",
    "MISSING_CLAUSES_FOR_OPERATION",
    "INTERNAL_ERROR",
];

fn read(text: &str) -> Result<MetricViewSpec, String> {
    yaml::read(text, &ParserOptions::default())
}

#[test]
fn create_bodies_agree_with_spark() {
    let mut checked = 0;
    let mut failures = vec![];
    for version in ["4.2.1", "4.3.0"] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/metric-views/cases.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for case in doc["cases"].as_array().unwrap() {
            for step in case["steps"].as_array().unwrap() {
                let sql = step["sql"].as_str().unwrap_or_default();
                let Ok(Statement::CreateMetricView(view)) =
                    krishiv_sql::parse_statement(sql, &ParserOptions::default())
                else {
                    continue;
                };
                let error = &step["outcome"]["error"];
                let class = error["condition"].as_str().unwrap_or_default();
                let ours = read(&view.body);
                let agrees = if class == "INVALID_METRIC_VIEW_YAML" {
                    let theirs = error["parameters"]["message"].as_str().unwrap();
                    match (&ours, step["match"].as_str()) {
                        (Err(m), Some("exact")) => m == theirs,
                        // A SnakeYAML message or a Java runtime message: the prefix up to the
                        // first `:`'s text, then ours.
                        (Err(m), _) => {
                            let prefix = theirs.split_inclusive(": ").next().unwrap_or(theirs);
                            m.starts_with(prefix)
                        }
                        (Ok(_), _) => false,
                    }
                } else if BEFORE_YAML.contains(&class)
                    && error["message"].as_str().is_some_and(|m| {
                        // An expression's syntax error comes after the YAML read.
                        !m.contains("== SQL (line 1, position 1) ==")
                    })
                {
                    continue;
                } else {
                    ours.is_ok()
                };
                checked += 1;
                if !agrees {
                    failures.push(format!(
                        "{version} {}: {:?}\n  spark: {class} {}\n  ours: {ours:?}",
                        case["name"], view.body, error["parameters"]["message"]
                    ));
                }
            }
        }
    }
    assert!(checked > 200, "only {checked} bodies");
    assert!(
        failures.is_empty(),
        "{}\n{} failures",
        failures.join("\n"),
        failures.len()
    );
}

#[test]
fn the_spec_keeps_the_yaml_order() {
    let spec = read(
        "\nversion: 0.1\nsource: SELECT * FROM t\nfilter: a > 1\nmeasures:\n  - name: m\n    expr: sum(a)\n\
         dimensions:\n  - name: d\n    expr: upper(b)\n  - name: e\n    expr: b\n",
    )
    .unwrap();
    assert_eq!(spec.source, MetricViewSource::Sql("SELECT * FROM t".into()));
    assert_eq!(spec.filter.as_deref(), Some("a > 1"));
    let names: Vec<&str> = spec.dimensions.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["d", "e"]);
    assert_eq!(
        spec.measures,
        [MetricViewColumn {
            name: "m".into(),
            expr: "sum(a)".into()
        }]
    );
    let spec = read("version: 0.1\nsource: db.t").unwrap();
    assert_eq!(spec.source, MetricViewSource::Asset("db.t".into()));
}

#[test]
fn a_view_text_that_is_no_yaml_mapping() {
    // `ALTER VIEW mv AS SELECT ..` leaves a query where the YAML was (golden
    // `parse_alter_view_as_query`).
    assert_eq!(
        read("SELECT 1 AS x").unwrap_err(),
        "Failed to parse YAML: Cannot construct instance of \
         `org.apache.spark.sql.metricview.serde.YAMLVersion` (although at least one Creator exists): \
         no String-argument constructor/factory method to deserialize from String value \
         ('SELECT 1 AS x')\n at [Source: (StringReader); line: 1, column: 1]"
    );
}

#[test]
fn scalars_of_other_shapes_where_a_column_or_list_is_expected() {
    let e = read("version: 0.1\nsource: t\ndimensions: 5\n").unwrap_err();
    assert!(
        e.contains("from Integer value (token `JsonToken.VALUE_NUMBER_INT`)"),
        "{e}"
    );
    let e = read("version: 0.1\nsource: t\ndimensions:\n  - 1.5\n").unwrap_err();
    assert!(e.contains("no double/Double-argument constructor/factory method to deserialize from Number value (1.5)"), "{e}");
    let e = read("version: 0.1\nsource: t\ndimensions:\n  - true\n").unwrap_err();
    assert!(e.contains("Boolean value (true)"), "{e}");
    let e = read("version: 0.1\nsource: t\ndimensions:\n  - [a]\n").unwrap_err();
    assert!(e.contains("ColumnV01` from Array value"), "{e}");
    let e = read("version: 0.1\nsource: t\ndimensions:\n  - ~\n").unwrap_err();
    assert!(e.contains("because \"column\" is null"), "{e}");
    let e = read("version: [1]\n").unwrap_err();
    assert!(e.contains("YAMLVersion[\"version\"]"), "{e}");
    assert!(read("~").unwrap_err().contains("\"yamlVersion\" is null"));
    assert!(read("\n").unwrap_err().contains("\"yamlVersion\" is null"));
    assert!(
        read("[a]")
            .unwrap_err()
            .contains("YAMLVersion` from Array value")
    );
}

#[test]
fn unknown_keys_after_every_parameter_are_reported_where_they_stand() {
    let e = read(
        "version: 0.1\nsource: t\nfilter: x\ndimensions: []\nmeasures: []\nowner: me\nmore: 1\n",
    )
    .unwrap_err();
    assert!(e.contains("Unrecognized field \"owner\""), "{e}");
    assert!(e.contains("line: 6, column: 10]"), "{e}");
    // A missing required parameter wins over a held-back key.
    let e = read("version: 0.1\nowner: me\n").unwrap_err();
    assert!(
        e.contains("Missing required creator property 'source' (index 1)"),
        "{e}"
    );
}

#[test]
fn nulls_and_tags() {
    let e = read("version: 0.1\nsource: t\ndimensions:\n  - name: ~\n    expr: a\n").unwrap_err();
    assert!(e.contains("'name'"), "{e}");
    let e = read("version: 0.1\nsource: t\ndimensions:\n  - name: a\n    expr: !!null x\n")
        .unwrap_err();
    assert!(e.contains("'expr'"), "{e}");
    // A quoted `~` is text, not null; a quoted empty filter is an empty string.
    assert_eq!(
        read("version: 0.1\nsource: \"~\"\n").unwrap_err(),
        "Invalid source: ~"
    );
    let spec = read("version: !!str 0.1\nsource: 't'\nfilter: ''\n").unwrap();
    assert_eq!(spec.source, MetricViewSource::Asset("t".into()));
    assert_eq!(spec.filter.as_deref(), Some(""));
    let spec =
        read("version: 0.1\nsource: t\ndimensions:\n  - {name: !!int 1, expr: .inf}\n").unwrap();
    assert_eq!(spec.dimensions[0].name, "1");
    assert!(
        read("version: 0.1\nsource: t\ndimensions:\n  - {name: [a], expr: b}\n")
            .unwrap_err()
            .contains("java.lang.String")
    );
    assert!(read("version: 0.1\nsource: t\n? [a]\n: b\n").is_err());
    assert!(
        parse_metric_view_yaml("version: 2", &ParserOptions::default())
            .is_err_and(|e| e.error_class().name() == "INVALID_METRIC_VIEW_YAML")
    );
}

#[test]
fn scalar_shapes() {
    use saphyr_parser::ScalarStyle::{Plain, SingleQuoted};
    for (text, want) in [
        ("1_000", "Int"),
        ("-3", "Int"),
        ("1.5e3", "Float"),
        ("+.5", "Float"),
        (".NaN", "Float"),
        ("1e", "Str"),
        ("1.2.3", "Str"),
        ("Null", "Null"),
        ("FALSE", "False"),
    ] {
        assert_eq!(format!("{:?}", yaml::shape_of(text, Plain)), want, "{text}");
    }
    assert_eq!(format!("{:?}", yaml::shape_of("1", SingleQuoted)), "Str");
}

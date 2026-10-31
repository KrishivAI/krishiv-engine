// SPDX-License-Identifier: Apache-2.0
//! `SET`/`RESET` and temporary views through `Session::sql`: result
//! schemas and rows, configuration updates, view lifecycles and Spark's
//! error classes.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn people() -> (Arc<Schema>, RecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int32, false),
        Field::new("name", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![3, 1, 2])),
            Arc::new(StringArray::from(vec![Some("c"), None, Some("b")])),
        ],
    )
    .unwrap();
    (schema, batch)
}

fn session() -> Session {
    let s = Session::new().unwrap();
    let (schema, batch) = people();
    s.register_batches("people", schema, vec![batch]).unwrap();
    s
}

fn class<T: std::fmt::Debug>(r: Result<T, KrishivError>) -> ErrorClass {
    r.unwrap_err().error_class()
}

/// The rows of `sql` as strings (`~` for NULL), with the schema names.
fn rows(s: &Session, sql: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let batches = block_on(df.collect()).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let mut row = Vec::new();
            for c in b.columns() {
                if c.is_null(i) {
                    row.push("~".to_owned());
                } else {
                    match c.data_type() {
                        ArrowType::Utf8View => row.push(c.as_string_view().value(i).to_owned()),
                        ArrowType::Utf8 => {
                            row.push(c.as_string::<i32>().value(i).to_owned());
                        }
                        ArrowType::Int32 => {
                            row.push(c.as_primitive::<Int32Type>().value(i).to_string());
                        }
                        other => panic!("{sql}: unexpected type {other}"),
                    }
                }
            }
            out.push(row);
        }
    }
    (names, out)
}

#[test]
fn set_returns_spark_rows_and_sets() {
    let s = Session::new().unwrap();
    // Nothing set: no rows, but the `key`/`value` string columns.
    let df = block_on(s.sql("SET")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["key", "value"]
    );
    assert_eq!(schema.field(0).data_type(), &ArrowType::Utf8View);
    assert!(!schema.field(0).is_nullable() && !schema.field(1).is_nullable());
    assert_eq!(block_on(df.count()).unwrap(), 0);
    // `SET key = value` sets through the session configuration and shows
    // the effective value; the key reads back the same.
    let (names, found) = rows(&s, "SET my.app.theme = dark");
    assert_eq!(names, ["key", "value"]);
    assert_eq!(found, [vec!["my.app.theme", "dark"]]);
    assert_eq!(s.conf().get("my.app.theme").as_deref(), Some("dark"));
    let (_, found) = rows(&s, "SET my.app.theme");
    assert_eq!(found, [vec!["my.app.theme", "dark"]]);
    // The value is the raw rest of the statement, trimmed.
    let (_, found) = rows(&s, "SET my.app.greeting =  hello = world  ;");
    assert_eq!(found, [vec!["my.app.greeting", "hello = world"]]);
    // Only backquotes quote keys and values: a single-quoted value stays
    // quoted, as in Spark.
    let (_, found) = rows(&s, "SET `my.app.quoted` = 'a b'");
    assert_eq!(found, [vec!["my.app.quoted", "'a b'"]]);
    let (_, found) = rows(&s, "SET `sp ace` = `q`");
    assert_eq!(found, [vec!["sp ace", "q"]]);
    // An assignment shows the value as written, not canonicalised.
    let (_, found) = rows(&s, "SET spark.sql.ansi.enabled=TRUE");
    assert_eq!(found, [vec!["spark.sql.ansi.enabled", "TRUE"]]);
    assert_eq!(
        s.conf().get("spark.sql.ansi.enabled").as_deref(),
        Some("true")
    );
    // A registered key shows its effective value, whatever the spelling.
    let (_, found) = rows(&s, "SET spark.sql.ansi.enabled");
    assert_eq!(found, [vec!["spark.sql.ansi.enabled", "true"]]);
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    assert_eq!(
        s.conf().get("spark.sql.ansi.enabled").as_deref(),
        Some("false")
    );
    let (_, found) = rows(&s, "SET krishiv.sql.ansi.enabled");
    assert_eq!(found, [vec!["krishiv.sql.ansi.enabled", "false"]]);
    // `SET` lists every explicitly set key, in the engine's order:
    // sorted by key, as in Spark.
    let (names, listed) = rows(&s, "SET");
    assert_eq!(names, ["key", "value"]);
    let keys: Vec<&str> = listed.iter().map(|r| r[0].as_str()).collect();
    assert_eq!(
        keys,
        [
            "krishiv.sql.ansi.enabled",
            "my.app.greeting",
            "my.app.quoted",
            "my.app.theme",
            "sp ace",
        ]
    );
    assert!(keys.windows(2).all(|w| w[0] <= w[1]));
    // An unset key shows `<undefined>`, as in Spark.
    let (_, found) = rows(&s, "SET my.app.missing");
    assert_eq!(found, [vec!["my.app.missing", "<undefined>"]]);
}

#[test]
fn set_verbose_shows_meaning_and_version() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET my.app.x = on")).unwrap();
    let df = block_on(s.sql("SET -v")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["key", "value", "meaning", "Since version"]
    );
    // All four columns are non-nullable strings.
    for f in schema.fields() {
        assert_eq!(f.data_type(), &ArrowType::Utf8View, "{}", f.name());
        assert!(!f.is_nullable(), "{}", f.name());
    }
    let (names, listed) = rows(&s, "SET -v");
    assert_eq!(names, ["key", "value", "meaning", "Since version"]);
    // Registered entries carry the registry's meaning and version.
    let ansi = listed
        .iter()
        .find(|r| r[0] == "krishiv.sql.ansi.enabled")
        .unwrap();
    assert_eq!(ansi[1], "true");
    assert!(ansi[2].contains("ANSI"), "{ansi:?}");
    assert_eq!(ansi[3], "0.1.0");
    // Explicitly set unregistered keys are not listed, as in Spark.
    assert!(listed.iter().all(|r| r[0] != "my.app.x"));
    for r in &listed {
        assert!(!r.iter().any(|c| c == "~"), "{r:?}");
    }
    // Sorted by key, in the engine's order.
    let keys: Vec<&str> = listed.iter().map(|r| r[0].as_str()).collect();
    assert!(!keys.is_empty());
    assert!(keys.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn set_failures_come_from_the_config() {
    let s = Session::new().unwrap();
    // An unknown `krishiv.*` key fails as setting it does.
    assert_eq!(
        class(block_on(s.sql("SET krishiv.nope = 1"))),
        ErrorClass::KrishivUnknownConfig
    );
    // An invalid value fails as setting it does.
    let e = block_on(s.sql("SET krishiv.exec.batchSize = many")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidConfigValue);
    assert_eq!(
        s.conf().get("krishiv.exec.batchSize").as_deref(),
        Some("4096")
    );
    // A static entry cannot change at runtime.
    assert_eq!(
        class(block_on(s.sql("SET krishiv.memory.limit = 1g"))),
        ErrorClass::CannotModifyStaticConfig
    );
    assert_eq!(
        class(block_on(s.sql("RESET krishiv.memory.limit"))),
        ErrorClass::CannotModifyStaticConfig
    );
}

#[test]
fn bucketing_max_buckets_limit_is_configured_and_validated() {
    let s = Session::new().unwrap();
    // Invalid values are rejected by the session configuration.
    for bad in ["0", "-2", "many"] {
        let e = block_on(s.sql(&format!(
            "SET spark.sql.sources.bucketing.maxBuckets = {bad}"
        )))
        .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivInvalidConfigValue,
            "{bad}"
        );
    }
    // The default limit applies: above 100,000 buckets fails.
    let over = "CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 100001 BUCKETS";
    assert_eq!(class(block_on(s.sql(over))), ErrorClass::InvalidBucketCount);
    // The configured limit applies to later statements.
    block_on(s.sql("SET spark.sql.sources.bucketing.maxBuckets = 2")).unwrap();
    let e = block_on(s.sql("CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 3 BUCKETS")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidBucketCount);
    assert!(e.to_string().contains("(`2`). Got `3`"), "{e}");
    // A count within it passes validation, and the table is created.
    block_on(async {
        s.sql("CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 2 BUCKETS")
            .await?
            .collect()
            .await
    })
    .unwrap();
    // Resetting restores the default limit.
    block_on(s.sql("RESET spark.sql.sources.bucketing.maxBuckets")).unwrap();
    assert_eq!(class(block_on(s.sql(over))), ErrorClass::InvalidBucketCount);
    // The engine key aliases the Spark one: setting through it applies the
    // same limit, and resetting through it restores the default.
    block_on(s.sql("SET krishiv.sql.sources.bucketing.maxBuckets = 2")).unwrap();
    let e = block_on(s.sql("CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 3 BUCKETS")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidBucketCount);
    assert!(e.to_string().contains("(`2`). Got `3`"), "{e}");
    block_on(async {
        s.sql("CREATE TABLE t2 (a INT) CLUSTERED BY (a) INTO 2 BUCKETS")
            .await?
            .collect()
            .await
    })
    .unwrap();
    block_on(s.sql("RESET krishiv.sql.sources.bucketing.maxBuckets")).unwrap();
    assert_eq!(class(block_on(s.sql(over))), ErrorClass::InvalidBucketCount);
}

#[test]
fn reset_restores_the_session() {
    let s = Session::builder()
        .config("spark.sql.ansi.enabled", "false")
        .config("my.app.keep", "yes")
        .build()
        .unwrap();
    block_on(s.sql("SET spark.sql.ansi.enabled = true")).unwrap();
    block_on(s.sql("SET krishiv.exec.batchSize = 1024")).unwrap();
    block_on(s.sql("SET my.app.keep = no")).unwrap();
    block_on(s.sql("SET my.app.temp = x")).unwrap();
    // `RESET key` restores the value the session was built with: the
    // builder's value, or the default when neither sets it.
    let df = block_on(s.sql("RESET spark.sql.ansi.enabled")).unwrap();
    assert_eq!(block_on(df.schema()).unwrap().fields().len(), 0);
    assert_eq!(block_on(df.count()).unwrap(), 0);
    assert_eq!(
        s.conf().get("spark.sql.ansi.enabled").as_deref(),
        Some("false")
    );
    block_on(s.sql("RESET my.app.keep")).unwrap();
    assert_eq!(s.conf().get("my.app.keep").as_deref(), Some("yes"));
    block_on(s.sql("RESET krishiv.exec.batchSize")).unwrap();
    assert_eq!(
        s.conf().get("krishiv.exec.batchSize").as_deref(),
        Some("4096")
    );
    // Resetting an unknown key is a no-op, as unsetting it is.
    block_on(s.sql("RESET my.app.never")).unwrap();
    // Resetting a registered key nobody set keeps its default.
    let fresh = Session::new().unwrap();
    block_on(fresh.sql("RESET krishiv.exec.batchSize")).unwrap();
    assert_eq!(
        fresh.conf().get("krishiv.exec.batchSize").as_deref(),
        Some("4096")
    );
    // `RESET` restores everything the session set, keeping the builder's
    // values, and returns an empty frame as in Spark.
    block_on(s.sql("RESET")).unwrap();
    assert_eq!(
        s.conf().get("spark.sql.ansi.enabled").as_deref(),
        Some("false")
    );
    assert_eq!(s.conf().get("my.app.keep").as_deref(), Some("yes"));
    assert_eq!(s.conf().get("my.app.temp"), None);
    let (names, listed) = rows(&s, "SET");
    assert_eq!(names, ["key", "value"]);
    // Builder values survive under their canonical keys.
    let mut keys: Vec<&str> = listed.iter().map(|r| r[0].as_str()).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["krishiv.sql.ansi.enabled", "my.app.keep"]);
}

#[test]
fn temp_views_created_by_sql_are_visible_to_later_queries() {
    let s = session();
    // Creation returns an empty frame as in Spark, and the view reads in
    // a later query.
    let df = block_on(s.sql("CREATE TEMPORARY VIEW v AS SELECT id, name FROM people")).unwrap();
    assert_eq!(block_on(df.schema()).unwrap().fields().len(), 0);
    assert_eq!(block_on(df.count()).unwrap(), 0);
    assert_eq!(s.temp_table_names(), ["people", "v"]);
    let (names, found) = rows(&s, "SELECT * FROM v ORDER BY id");
    assert_eq!(names, ["id", "name"]);
    assert_eq!(
        found,
        [
            vec!["1".to_owned(), "~".to_owned()],
            vec!["2".to_owned(), "b".to_owned()],
            vec!["3".to_owned(), "c".to_owned()],
        ]
    );
    // The view is a subquery: qualified by its own name.
    assert_eq!(rows(&s, "SELECT v.id FROM v ORDER BY id").1.len(), 3);
    // A view over a view reads through it.
    block_on(s.sql("CREATE TEMP VIEW w AS SELECT id FROM v WHERE id > 1")).unwrap();
    assert_eq!(rows(&s, "SELECT * FROM w ORDER BY id").1.len(), 2);
    // A view follows the table it reads: replacing the table changes the
    // next query.
    let (schema, _) = people();
    s.register_batches("people", schema, vec![]).unwrap();
    assert_eq!(rows(&s, "SELECT * FROM v").1.len(), 0);
    // Dropping the view hides it again, as in Spark.
    let df = block_on(s.sql("DROP VIEW v")).unwrap();
    assert_eq!(block_on(df.schema()).unwrap().fields().len(), 0);
    let e = block_on(s.sql("SELECT * FROM v")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    // `w` read through `v`: it fails too now that `v` is gone.
    assert_eq!(
        class(block_on(s.sql("SELECT * FROM w")).and_then(|df| block_on(df.schema()).map(|_| ()))),
        ErrorClass::TableOrViewNotFound
    );
    block_on(s.sql("DROP VIEW w")).unwrap();
    assert_eq!(s.temp_table_names(), ["people"]);
}

#[test]
fn create_view_options_replace_errors_and_arity() {
    let s = session();
    // `OR REPLACE` swaps the definition; without it the view already
    // exists, exactly as in Spark.
    block_on(s.sql("CREATE TEMP VIEW v AS SELECT id FROM people")).unwrap();
    let e = block_on(s.sql("CREATE TEMPORARY VIEW v AS SELECT id FROM people")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TempTableOrViewAlreadyExists);
    assert_eq!(
        e.to_string(),
        "[TEMP_TABLE_OR_VIEW_ALREADY_EXISTS] Cannot create the temporary view `v` because it \
         already exists.\nChoose a different name, drop or replace the existing view. \
         SQLSTATE: 42P07"
    );
    // A clash with a table is the same error.
    let e = block_on(s.sql("CREATE TEMP VIEW people AS SELECT 1")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TempTableOrViewAlreadyExists);
    block_on(s.sql("CREATE OR REPLACE TEMP VIEW v AS SELECT id AS x FROM people")).unwrap();
    assert_eq!(rows(&s, "SELECT * FROM v").0, ["x"]);
    // `IF NOT EXISTS` on a temporary view fails at parse time, as in
    // Spark, with or without `OR REPLACE`.
    let e = block_on(s.sql("CREATE TEMP VIEW IF NOT EXISTS v AS SELECT 99 AS y")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp0053);
    assert!(
        e.message()
            .contains("It is not allowed to define a TEMPORARY view with IF NOT EXISTS.")
    );
    let e = block_on(s.sql("CREATE OR REPLACE TEMP VIEW IF NOT EXISTS v AS SELECT 1")).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::CreateViewWithIfNotExistsAndReplace
    );
    // Declared columns rename the output, with comments allowed.
    block_on(s.sql("CREATE OR REPLACE TEMP VIEW v (a COMMENT 'the id') AS SELECT id FROM people"))
        .unwrap();
    assert_eq!(rows(&s, "SELECT * FROM v").0, ["a"]);
    // Arity mismatches fail as in Spark, naming both column lists.
    let e = block_on(s.sql("CREATE OR REPLACE TEMP VIEW v (a, b) AS SELECT id FROM people"))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::CreateViewColumnArityMismatchNotEnoughDataColumns
    );
    assert_eq!(e.sql_state(), Some("21S01"));
    assert!(
        e.message()
            .starts_with("Cannot create view `v`, the reason is not enough data columns"),
        "{e}"
    );
    assert_eq!(
        e.message_parameters(),
        [
            ("viewName", "`v`".to_owned()),
            ("viewColumns", "`a`, `b`".to_owned()),
            ("dataColumns", "`id`".to_owned()),
        ]
    );
    block_on(s.sql("CREATE OR REPLACE TEMP VIEW v AS SELECT id, name FROM people")).unwrap();
    let e = block_on(s.sql("CREATE OR REPLACE TEMP VIEW v (a) AS SELECT id, name FROM people"))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::CreateViewColumnArityMismatchTooManyDataColumns
    );
    // The view's query is analyzed at creation: its errors come first.
    let e = block_on(s.sql("CREATE TEMP VIEW broken AS SELECT nope FROM people")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnresolvedColumnWithSuggestion);
    let e = block_on(s.sql("CREATE TEMP VIEW missing AS SELECT * FROM nosuch")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert!(!s.temp_table_names().contains(&"broken".to_owned()));
    // So is an inline table: it analyzes, then reads back.
    block_on(s.sql("CREATE TEMP VIEW vals AS VALUES (1, 2)")).unwrap();
    let (names, found) = rows(&s, "SELECT * FROM vals");
    assert_eq!(names, ["col1", "col2"]);
    assert_eq!(found, [vec!["1", "2"]]);
}

#[test]
fn create_view_rejects_duplicate_column_names() {
    // Spark reports a repeated view column with `COLUMN_ALREADY_EXISTS`,
    // after aliases are applied: both a duplicated alias list and
    // duplicate output names fail, ignoring case by default.
    let s = session();
    let e =
        block_on(s.sql("CREATE TEMP VIEW v (a, a) AS SELECT id, name FROM people")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::ColumnAlreadyExists);
    assert_eq!(e.sql_state(), Some("42711"));
    assert_eq!(e.message_parameters(), [("columnName", "`a`".to_owned())]);
    assert!(!s.temp_table_names().contains(&"v".to_owned()));
    let e =
        block_on(s.sql("CREATE TEMP VIEW w AS SELECT id AS a, name AS a FROM people")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::ColumnAlreadyExists);
    assert_eq!(e.message_parameters(), [("columnName", "`a`".to_owned())]);
    assert!(!s.temp_table_names().contains(&"w".to_owned()));
    // Distinct names create the view.
    block_on(s.sql("CREATE TEMP VIEW ok (a, b) AS SELECT id, name FROM people")).unwrap();
    assert_eq!(rows(&s, "SELECT * FROM ok").0, ["a", "b"]);
    // Case-sensitive sessions distinguish letter case.
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    block_on(s.sql("CREATE TEMP VIEW cs (a, A) AS SELECT id, name FROM people")).unwrap();
    let e =
        block_on(s.sql("CREATE TEMP VIEW cs2 (a, a) AS SELECT id, name FROM people")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::ColumnAlreadyExists);
    s.conf().unset("spark.sql.caseSensitive").unwrap();
}

#[test]
fn create_view_with_both_flags_fails_before_the_query_is_checked() {
    // `OR REPLACE` with `IF NOT EXISTS` fails in the parser, as in Spark,
    // so the flag error comes before any error in the view's query, and an
    // `EXPLAIN` of such a statement fails instead of printing plan text.
    let s = session();
    let e = block_on(s.sql("CREATE OR REPLACE TEMP VIEW IF NOT EXISTS v AS SELECT 1")).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::CreateViewWithIfNotExistsAndReplace
    );
    assert_eq!(e.sql_state(), Some("42601"));
    let e =
        block_on(s.sql("CREATE OR REPLACE TEMP VIEW IF NOT EXISTS v AS SELECT DATE '2020-02-30'"))
            .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::CreateViewWithIfNotExistsAndReplace,
        "{e}"
    );
    let e = block_on(s.sql("EXPLAIN CREATE OR REPLACE TEMP VIEW IF NOT EXISTS v AS SELECT 1"))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::CreateViewWithIfNotExistsAndReplace,
        "{e}"
    );
    assert!(!s.temp_table_names().contains(&"v".to_owned()));
}

#[test]
fn drop_view_needs_the_view() {
    let s = session();
    block_on(s.sql("CREATE TEMP VIEW v AS SELECT id FROM people")).unwrap();
    // A missing view fails as in Spark, naming the view and search path.
    let e = block_on(s.sql("DROP VIEW nosuch")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert_eq!(e.sql_state(), Some("42P01"));
    assert!(e.message().contains("`nosuch`"), "{e}");
    assert_eq!(
        e.message_parameters(),
        [
            ("relationName", "`nosuch`".to_owned()),
            (
                "searchPath",
                "[`system`.`builtin`, `system`.`session`, `spark_catalog`.`default`]".to_owned()
            ),
        ]
    );
    // `IF EXISTS` tolerates it, returning an empty frame as in Spark.
    let df = block_on(s.sql("DROP VIEW IF EXISTS nosuch")).unwrap();
    assert_eq!(block_on(df.schema()).unwrap().fields().len(), 0);
    assert_eq!(block_on(df.count()).unwrap(), 0);
    // Qualified names never reach a temporary view.
    assert_eq!(
        class(
            block_on(s.sql("SELECT * FROM default.v"))
                .and_then(|df| block_on(df.schema()).map(|_| ()))
        ),
        ErrorClass::TableOrViewNotFound
    );
    block_on(s.sql("DROP VIEW v")).unwrap();
    assert_eq!(
        class(block_on(s.sql("SELECT * FROM v")).and_then(|df| block_on(df.schema()).map(|_| ()))),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn views_resolve_like_subqueries() {
    let s = session();
    block_on(s.sql("CREATE TEMP VIEW v AS SELECT id FROM people")).unwrap();
    // A self-join reads the view twice.
    let (_, found) = rows(
        &s,
        "SELECT a.id FROM v a JOIN v b ON a.id = b.id ORDER BY a.id",
    );
    assert_eq!(found.len(), 3);
    // Views read inside subquery expressions too.
    let (_, found) = rows(
        &s,
        "SELECT id FROM people WHERE EXISTS (SELECT 1 FROM v WHERE v.id = people.id) ORDER BY id",
    );
    assert_eq!(found.len(), 3);
    // A `WITH` definition shadows a view of the same name, as in Spark.
    assert_eq!(
        rows(&s, "WITH v AS (SELECT 99 AS id) SELECT * FROM v").1,
        [vec!["99".to_owned()]]
    );
    // So does one inside the view: it keeps its own definitions.
    block_on(s.sql("CREATE OR REPLACE TEMP VIEW w AS WITH v AS (SELECT 7 AS id) SELECT * FROM v"))
        .unwrap();
    assert_eq!(rows(&s, "SELECT * FROM w").1, [vec!["7".to_owned()]]);
    assert_eq!(rows(&s, "SELECT * FROM v ORDER BY id").1.len(), 3);
}

#[test]
fn views_with_read_options_are_rejected() {
    // Options on a view read reach the analyzer's view handling, which
    // the engine does not implement: the read fails naming views, as a
    // scan of an unimplemented source does.
    let s = session();
    block_on(s.sql("CREATE TEMP VIEW v AS SELECT id FROM people")).unwrap();
    let e = block_on(s.sql("SELECT * FROM v WITH ('a' = 'b')")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("views"), "{e}");
}

#[test]
fn recursive_views_fail_when_created() {
    let s = session();
    block_on(s.sql("CREATE TEMP VIEW a AS SELECT id FROM people")).unwrap();
    block_on(s.sql("CREATE TEMP VIEW b AS SELECT id FROM a")).unwrap();
    // `OR REPLACE` pointing `a` back at `b` closes a cycle: it fails when
    // created with `RECURSIVE_VIEW`, as in Spark, and `a` keeps its old
    // definition.
    let e = block_on(s.sql("CREATE OR REPLACE TEMP VIEW a AS SELECT id FROM b")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::RecursiveView);
    assert_eq!(e.sql_state(), Some("42K0H"));
    assert_eq!(
        e.message(),
        "Recursive view `a` detected (cycle: `a` -> `b` -> `a`)."
    );
    assert_eq!(rows(&s, "SELECT * FROM a ORDER BY id").1.len(), 3);
    // A direct self-reference fails the same way.
    let e = block_on(s.sql("CREATE OR REPLACE TEMP VIEW b AS SELECT id FROM b")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::RecursiveView);
    assert!(e.message().contains("cycle: `b` -> `b`"), "{e}");
    // A fresh view over its own missing name resolves nowhere: analysis
    // reports it as missing, as in Spark.
    let e = block_on(s.sql("CREATE TEMP VIEW fresh AS SELECT * FROM fresh")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
}

#[test]
fn dataframe_temp_views_snapshot_their_input() {
    use krishiv_session::functions::{col, lit};
    let s = Session::new().unwrap();
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = arrow_array::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Int32Array::from(vec![1]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    // A DataFrame view built from a table of the same name reads the old
    // definition, as in Spark: the input is snapshotted.
    block_on(
        s.table("t")
            .select([(col("id") + lit(1)).alias("id")])
            .create_or_replace_temp_view("t"),
    )
    .unwrap();
    assert_eq!(rows(&s, "SELECT * FROM t").1, [vec!["2".to_owned()]]);
}

#[test]
fn view_names_follow_case_sensitivity() {
    // Temporary views resolve case-insensitively by default, exactly when
    // registered with `spark.sql.caseSensitive` on.
    let s = session();
    block_on(s.sql("CREATE TEMP VIEW V AS SELECT id FROM people")).unwrap();
    assert_eq!(rows(&s, "SELECT * FROM v").1.len(), 3);
    block_on(s.sql("DROP VIEW v")).unwrap();
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    block_on(s.sql("CREATE TEMP VIEW V AS SELECT id FROM people")).unwrap();
    let e = block_on(s.sql("SELECT * FROM v")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert_eq!(rows(&s, "SELECT * FROM V").1.len(), 3);
    let e = block_on(s.sql("DROP VIEW v")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    block_on(s.sql("DROP VIEW V")).unwrap();
}

#[test]
fn create_or_replace_temp_view_takes_any_dataframe() {
    // The SQL views reuse this machinery, so a DataFrame over anything
    // but a read by path becomes a view over its plan too.
    let s = session();
    block_on(
        s.table("people")
            .select([krishiv_session::functions::col("id")])
            .create_or_replace_temp_view("v"),
    )
    .unwrap();
    assert_eq!(rows(&s, "SELECT * FROM v ORDER BY id").1.len(), 3);
    // Analysis errors still come first.
    let e = block_on(s.table("missing").create_or_replace_temp_view("w"));
    assert_eq!(class(e), ErrorClass::TableOrViewNotFound);
}

#[test]
fn set_rejects_bad_syntax() {
    let s = Session::new().unwrap();
    // Anything but `SET`, `SET -v`, `SET key` or `SET key = value` with a
    // plain key fails with `INVALID_SET_SYNTAX`, as in Spark.
    for sql in ["SET a b", "SET -x", "SET = 1", "SET -v x", "SET 'k'=1"] {
        let e = block_on(s.sql(sql)).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InvalidSetSyntax, "{sql}");
        assert_eq!(e.sql_state(), Some("42000"), "{sql}");
        assert!(
            e.message().contains("Expected format is 'SET'"),
            "{sql}: {e}"
        );
    }
    // Nothing was set by the failures.
    assert_eq!(
        block_on(block_on(s.sql("SET")).unwrap().count()).unwrap(),
        0
    );
    // A double-quoted key is a syntax error, as in Spark.
    let e = block_on(s.sql("SET \"k\"=\"v\"")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.sql_state(), Some("42601"));
}

#[test]
fn reset_rejects_bad_format() {
    let s = Session::new().unwrap();
    // Anything but `RESET` or `RESET key` with a plain key fails with
    // `INVALID_RESET_COMMAND_FORMAT`, as in Spark.
    let e = block_on(s.sql("RESET a b")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidResetCommandFormat);
    assert_eq!(e.sql_state(), Some("42000"));
    assert!(e.message().contains("Expected format is 'RESET'"), "{e}");
}

#[test]
fn set_time_zone_sets_the_session_zone() {
    let s = Session::new().unwrap();
    // `SET TIME ZONE '<zone>'` sets the session zone and shows it, as in
    // Spark.
    let (names, found) = rows(&s, "SET TIME ZONE 'UTC'");
    assert_eq!(names, ["key", "value"]);
    assert_eq!(found, [vec!["spark.sql.session.timeZone", "UTC"]]);
    assert_eq!(
        s.conf().get("spark.sql.session.timeZone").as_deref(),
        Some("UTC")
    );
    let (_, found) = rows(&s, "SET spark.sql.session.timeZone");
    assert_eq!(found, [vec!["spark.sql.session.timeZone", "UTC"]]);
    // `LOCAL` resolves to the system's zone.
    let (_, found) = rows(&s, "SET TIME ZONE LOCAL");
    assert_eq!(found.len(), 1);
    assert_eq!(found[0][0], "spark.sql.session.timeZone");
    assert!(!found[0][1].is_empty());
    assert_eq!(
        s.conf().get("spark.sql.session.timeZone").as_deref(),
        Some(found[0][1].as_str())
    );
    // An invalid zone fails through the session configuration.
    let e = block_on(s.sql("SET TIME ZONE 'not a zone!'")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidConfigValue);
    // `INTERVAL` zones name their unsupported form.
    let e = block_on(s.sql("SET TIME ZONE INTERVAL '1' DAY")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("INTERVAL"), "{e}");
}

#[test]
fn create_temp_view_rejects_qualified_names() {
    let s = Session::new().unwrap();
    // A qualified temporary view name parses and fails with
    // `INVALID_TEMP_OBJ_QUALIFIER`, naming the last part and its qualifier,
    // as in Spark 4.2.0.
    let e = block_on(s.sql("CREATE TEMP VIEW a.b AS SELECT 1")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidTempObjQualifier);
    assert_eq!(e.sql_state(), Some("42602"));
    assert_eq!(
        e.message_parameters(),
        [
            ("objectType", "VIEW".to_owned()),
            ("objectName", "`b`".to_owned()),
            ("qualifier", "`a`".to_owned()),
        ]
    );
    assert!(!s.temp_table_names().contains(&"a".to_owned()));
}

#[test]
fn cte_bodies_see_preceding_ctes_only() {
    let s = session();
    block_on(s.sql("CREATE TEMP VIEW c AS SELECT 1 AS id")).unwrap();
    // The CTE body reads the view of its own name, while the outer query
    // reads the CTE, as in Spark.
    assert_eq!(
        rows(&s, "WITH c AS (SELECT * FROM c) SELECT * FROM c").1,
        [vec!["1".to_owned()]]
    );
    // A later CTE still reads an earlier one.
    assert_eq!(
        rows(
            &s,
            "WITH a AS (SELECT 1 AS id), b AS (SELECT * FROM a) SELECT * FROM b"
        )
        .1,
        [vec!["1".to_owned()]]
    );
}

#[test]
fn explain_runs_commands() {
    let s = session();
    // `EXPLAIN` of a command prints Krishiv's command plan text: the
    // physical-plan header Spark prints, then the command itself (see
    // `command_plan`). The text is exact, not a fragment.
    for (sql, expected) in [
        ("EXPLAIN SET x=1", "== Physical Plan ==\nSET x=1\n"),
        ("EXPLAIN SET", "== Physical Plan ==\nSET\n"),
        ("EXPLAIN SET -v", "== Physical Plan ==\nSET -v\n"),
        ("EXPLAIN RESET", "== Physical Plan ==\nRESET\n"),
        (
            "EXPLAIN RESET spark.sql.ansi.enabled",
            "== Physical Plan ==\nRESET spark.sql.ansi.enabled\n",
        ),
        (
            "EXPLAIN CREATE TEMP VIEW v AS SELECT 1",
            "== Physical Plan ==\nCREATE TEMPORARY VIEW `v` AS ...\n",
        ),
        (
            "EXPLAIN DROP VIEW v",
            "== Physical Plan ==\nDROP VIEW `v`\n",
        ),
        (
            "EXPLAIN SET TIME ZONE 'UTC'",
            "== Physical Plan ==\nSET TIME ZONE 'UTC'\n",
        ),
    ] {
        let (_, found) = rows(&s, sql);
        assert_eq!(found.len(), 1, "{sql}");
        assert_eq!(found[0], [expected.to_owned()], "{sql}");
    }
    // `EXPLAIN` of an `EXPLAIN` stays unsupported.
    let e = block_on(s.sql("EXPLAIN EXPLAIN SET x=1")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

/// DDL and `USE` statements parse but have no execution yet: `Session::sql`
/// rejects them with `KRISHIV_UNSUPPORTED_FEATURE`, never a parse error.
#[test]
fn unexecuted_commands_parse_and_report_unsupported() {
    let s = session();
    let sql = "EXPLAIN CREATE TABLE t (a INT)";
    let e = block_on(s.sql(sql)).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{sql}"
    );
    assert!(e.to_string().contains("in sessions"), "{sql}: {e}");
    // Executed since the permanent catalog: table DDL on the session catalog.
    block_on(async { s.sql("CREATE TABLE t LIKE people").await?.collect().await }).unwrap();
    block_on(async { s.sql("DROP TABLE IF EXISTS t PURGE").await?.collect().await }).unwrap();
    // ... and the namespace commands.
    for sql in [
        "CREATE DATABASE IF NOT EXISTS db COMMENT 'c'",
        "CREATE SCHEMA s",
        "USE s",
        "USE NAMESPACE db",
        "DROP SCHEMA IF EXISTS s CASCADE",
    ] {
        block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
}

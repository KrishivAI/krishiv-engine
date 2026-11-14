// SPDX-License-Identifier: Apache-2.0
//! `Session::catalog()` (Spark's `spark.catalog`, T082): the PySpark `Catalog` doc examples, as Rust.
#![cfg(test)]

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{BoxFuture, PolicyVersion};
use krishiv_session::{
    AccessDecision, AccessPolicy, EngineRuntime, Principal, PrincipalKind, Privilege, Securable,
    Session,
};

fn session(dir: &std::path::Path) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .unwrap();
    s
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

#[test]
fn current_database_and_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let c = s.catalog();
    assert_eq!(c.current_database(), "default");
    assert_eq!(c.current_catalog(), "spark_catalog");
    run(&s, "CREATE DATABASE db1");
    block_on(c.set_current_database("db1")).unwrap();
    assert_eq!(c.current_database(), "db1");
    let e = block_on(c.set_current_database("nope")).unwrap_err();
    assert_eq!(e.error_class().name(), "SCHEMA_NOT_FOUND");
    assert_eq!(c.list_catalogs(), ["spark_catalog", "system"]);
}

#[test]
fn databases_list_get_exists() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE db1 COMMENT 'first'");
    let c = s.catalog();
    let names: Vec<_> = block_on(c.list_databases(None))
        .unwrap()
        .into_iter()
        .map(|d| d.name)
        .collect();
    assert_eq!(names, ["db1", "default"]);
    let d = block_on(c.get_database("DB1")).unwrap();
    assert_eq!(d.name, "db1");
    assert_eq!(d.catalog, "spark_catalog");
    assert_eq!(d.description.as_deref(), Some("first"));
    assert!(d.location_uri.ends_with("db1.db"));
    assert!(block_on(c.database_exists("default")).unwrap());
    assert!(!block_on(c.database_exists("zzz")).unwrap());
    assert_eq!(block_on(c.list_databases(Some("d*"))).unwrap().len(), 2);
    assert_eq!(block_on(c.list_databases(Some("db*"))).unwrap().len(), 1);
}

#[test]
fn tables_list_get_exists() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE tbl1 (name STRING, age INT) USING parquet");
    run(&s, "CREATE VIEW pv AS SELECT 1 AS x");
    run(&s, "CREATE TEMPORARY VIEW tab1 AS SELECT 1 AS x");
    run(&s, "CREATE GLOBAL TEMPORARY VIEW gv AS SELECT 1 AS x");
    let c = s.catalog();
    let t = block_on(c.list_tables(None, None)).unwrap();
    let got: Vec<_> = t
        .iter()
        .map(|t| (t.name.as_str(), t.table_type.as_str(), t.is_temporary))
        .collect();
    assert_eq!(
        got,
        [
            ("pv", "VIEW", false),
            ("tbl1", "MANAGED", false),
            ("tab1", "TEMPORARY", true)
        ]
    );
    let tbl1 = block_on(c.get_table("tbl1")).unwrap();
    assert_eq!(tbl1.catalog.as_deref(), Some("spark_catalog"));
    assert_eq!(tbl1.namespace, Some(vec!["default".to_owned()]));
    assert_eq!(tbl1.description, None);
    let tab1 = block_on(c.get_table("tab1")).unwrap();
    assert_eq!((tab1.catalog, tab1.namespace), (None, Some(vec![])));
    let gv = block_on(c.get_table("global_temp.gv")).unwrap();
    assert_eq!(gv.namespace, Some(vec!["global_temp".to_owned()]));
    // the global views and the local ones, as `SHOW TABLES IN global_temp` lists them
    let global: Vec<_> = block_on(c.list_tables(Some("global_temp"), None))
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(global, ["gv", "tab1"]);
    assert_eq!(block_on(c.list_tables(None, Some("tb*"))).unwrap().len(), 1);
    assert!(block_on(c.table_exists("default.tbl1")).unwrap());
    assert!(block_on(c.table_exists("spark_catalog.default.tbl1")).unwrap());
    assert!(!block_on(c.table_exists("missing")).unwrap());
    assert_eq!(
        block_on(c.get_table("missing"))
            .unwrap_err()
            .error_class()
            .name(),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    assert_eq!(
        block_on(c.list_tables(Some("nope"), None))
            .unwrap_err()
            .error_class()
            .name(),
        "SCHEMA_NOT_FOUND"
    );
}

#[test]
fn columns_carry_partition_and_bucket_flags() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (id INT COMMENT 'key', name STRING, p STRING) USING parquet \
         PARTITIONED BY (p)",
    );
    run(&s, "CREATE TEMPORARY VIEW v AS SELECT 1 AS x, 'a' AS y");
    let c = s.catalog();
    let cols = block_on(c.list_columns("t")).unwrap();
    let got: Vec<_> = cols
        .iter()
        .map(|c| {
            (
                c.name.as_str(),
                c.data_type.as_str(),
                c.nullable,
                c.is_partition,
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("id", "int", true, false),
            ("name", "string", true, false),
            ("p", "string", true, true)
        ]
    );
    assert_eq!(cols[0].description.as_deref(), Some("key"));
    let v = block_on(c.list_columns("v")).unwrap();
    assert_eq!(
        v.iter()
            .map(|c| (c.name.as_str(), c.data_type.as_str()))
            .collect::<Vec<_>>(),
        [("x", "int"), ("y", "string")]
    );
    assert_eq!(
        block_on(c.list_columns("nope"))
            .unwrap_err()
            .error_class()
            .name(),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn functions_list_get_exists() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TEMPORARY FUNCTION twice(x INT) RETURNS INT RETURN x * 2",
    );
    let c = s.catalog();
    assert!(block_on(c.function_exists("abs")).unwrap());
    assert!(block_on(c.function_exists("TWICE")).unwrap());
    assert!(!block_on(c.function_exists("no_such_fn")).unwrap());
    // a qualified name names a stored function of that namespace, never a built-in or a temporary one
    assert!(!block_on(c.function_exists("nosuchdb.abs")).unwrap());
    assert!(!block_on(c.function_exists("default.abs")).unwrap());
    assert!(!block_on(c.function_exists("default.twice")).unwrap());
    run(
        &s,
        "CREATE FUNCTION default.stored_fn(x INT) RETURNS INT RETURN x + 1",
    );
    assert!(block_on(c.function_exists("default.stored_fn")).unwrap());
    assert!(block_on(c.function_exists("spark_catalog.default.stored_fn")).unwrap());
    let f = block_on(c.get_function("default.stored_fn")).unwrap();
    assert_eq!(f.namespace, Some(vec!["default".to_owned()]));
    assert!(!f.is_temporary);
    let f = block_on(c.get_function("abs")).unwrap();
    assert_eq!((f.catalog, f.namespace, f.is_temporary), (None, None, true));
    assert!(
        block_on(c.list_functions(Some("tw*")))
            .unwrap()
            .iter()
            .any(|f| f.name == "twice")
    );
    assert_eq!(
        block_on(c.get_function("no_such_fn"))
            .unwrap_err()
            .error_class()
            .name(),
        "UNRESOLVED_ROUTINE"
    );
}

#[test]
fn create_table_managed_with_a_schema_and_external_from_files() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let c = s.catalog();
    block_on(c.create_table(
        "m",
        Some("parquet"),
        Some("a INT, b STRING"),
        Some("a managed table"),
        &[],
    ))
    .unwrap();
    let t = block_on(c.get_table("m")).unwrap();
    assert_eq!(t.table_type, "MANAGED");
    assert_eq!(t.description.as_deref(), Some("a managed table"));
    let cols = block_on(c.list_columns("m")).unwrap();
    assert_eq!(
        cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    // the schema is a column list, not a way to add clauses to the statement
    for bad in [
        "a INT) USING csv LOCATION '/tmp/x' (b INT",
        "a INT) COMMENT 'x'",
        "a INT) OPTIONS (k 'v'",
    ] {
        let e = block_on(c.create_table("bad", Some("parquet"), Some(bad), None, &[])).unwrap_err();
        assert!(
            ["KRISHIV_UNSUPPORTED_FEATURE", "PARSE_SYNTAX_ERROR"].contains(&e.error_class().name()),
            "{bad}: {e}"
        );
        assert!(!block_on(c.table_exists("bad")).unwrap(), "{bad}");
    }
    // a second create is the statement's error
    assert_eq!(
        block_on(c.create_table("m", Some("parquet"), Some("a INT"), None, &[]))
            .unwrap_err()
            .error_class()
            .name(),
        "TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    // an external table over files written by another table: its schema comes from the files
    run(&s, "INSERT INTO m VALUES (1, 'x'), (2, 'y')");
    let location = dir.path().join("m");
    let path = location.to_str().unwrap();
    block_on(c.create_table("ext", Some("parquet"), None, None, &[("path", path)])).unwrap();
    assert_eq!(block_on(c.get_table("ext")).unwrap().table_type, "EXTERNAL");
    let rows = block_on(async {
        s.sql("SELECT concat_ws('|', a, b) FROM ext ORDER BY a")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(
        rows.iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
    let cols = block_on(c.list_columns("ext")).unwrap();
    assert_eq!(
        cols.iter()
            .map(|c| (c.name.as_str(), c.data_type.as_str()))
            .collect::<Vec<_>>(),
        [("a", "int"), ("b", "string")]
    );
}

#[test]
fn drop_views_and_refresh() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TEMPORARY VIEW tab1 AS SELECT 1 AS x");
    run(&s, "CREATE GLOBAL TEMPORARY VIEW gv AS SELECT 1 AS x");
    run(&s, "CREATE TABLE t (a INT) USING parquet");
    let c = s.catalog();
    assert!(c.drop_temp_view("tab1"));
    assert!(!c.drop_temp_view("tab1"));
    assert!(c.drop_global_temp_view("gv"));
    assert!(!c.drop_global_temp_view("gv"));
    assert!(!block_on(c.is_cached("t")).unwrap());
    block_on(c.cache_table("t")).unwrap();
    assert!(block_on(c.is_cached("t")).unwrap(), "cacheTable caches it");
    block_on(c.refresh_table("t")).unwrap();
    block_on(c.uncache_table("t")).unwrap();
    assert!(
        !block_on(c.is_cached("t")).unwrap(),
        "uncacheTable drops it"
    );
    assert_eq!(
        block_on(c.is_cached("missing"))
            .unwrap_err()
            .error_class()
            .name(),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    c.clear_cache();
    assert_eq!(
        block_on(c.refresh_table("missing"))
            .unwrap_err()
            .error_class()
            .name(),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn a_name_with_a_dot_inside_backquotes_is_one_part() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TEMPORARY VIEW `a.b` AS SELECT 1 AS x");
    let c = s.catalog();
    assert!(block_on(c.table_exists("`a.b`")).unwrap());
    assert!(!block_on(c.table_exists("a.b")).unwrap());
    let t = block_on(c.get_table("`a.b`")).unwrap();
    assert_eq!(t.name, "a.b");
}

#[test]
fn listings_come_in_sparks_order() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TEMPORARY FUNCTION aaa_first(x INT) RETURNS INT RETURN x",
    );
    let c = s.catalog();
    let names: Vec<_> = block_on(c.list_functions(None))
        .unwrap()
        .into_iter()
        .map(|f| f.name)
        .collect();
    let user = names.iter().position(|n| n == "aaa_first").unwrap();
    let abs = names.iter().position(|n| n == "abs").unwrap();
    assert!(abs < user, "built-ins come before the session's functions");
}

/// Denies `Browse` and `Select` (`Select` implies `Browse`, ADR 0046 section 4), allows the rest.
struct NoBrowse;

impl AccessPolicy for NoBrowse {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _principal: &'a Principal,
        _securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = if matches!(privilege, Privilege::Browse | Privilege::Select) {
            AccessDecision::Deny {
                reason: "browse".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

#[test]
fn the_catalog_api_shows_a_principal_no_more_than_the_sql_statements_do() {
    let dir = tempfile::tempdir().unwrap();
    let rt = EngineRuntime::builder().build().unwrap();
    let open = |policy: Option<Arc<dyn AccessPolicy>>| {
        let mut b = Session::builder()
            .runtime(Arc::clone(&rt))
            .principal(Principal::new("pat", PrincipalKind::User));
        if let Some(p) = policy {
            b = b.access_policy(p);
        }
        let s = b.build().unwrap();
        s.conf()
            .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
            .unwrap();
        s
    };
    let admin = open(None);
    run(&admin, "CREATE TABLE secret (card STRING) USING parquet");
    let s = open(Some(Arc::new(NoBrowse)));
    // the statements are refused ...
    for sql in ["SHOW TABLES", "DESCRIBE TABLE secret", "SHOW NAMESPACES"] {
        let e = block_on(s.sql(sql)).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "{sql}"
        );
    }
    // ... and so is every call of the catalog API that stands for one
    let c = s.catalog();
    let denied = |e: KrishivError| e.error_class().name() == "KRISHIV_INSUFFICIENT_PERMISSIONS";
    assert!(denied(block_on(c.list_tables(None, None)).unwrap_err()));
    assert!(denied(block_on(c.get_table("secret")).unwrap_err()));
    assert!(denied(block_on(c.table_exists("secret")).unwrap_err()));
    assert!(denied(block_on(c.table_exists("missing")).unwrap_err()));
    assert!(denied(block_on(c.list_columns("secret")).unwrap_err()));
    assert!(denied(block_on(c.list_databases(None)).unwrap_err()));
    assert!(denied(block_on(c.get_database("default")).unwrap_err()));
    assert!(denied(block_on(c.database_exists("default")).unwrap_err()));
    assert!(denied(block_on(c.list_functions(None)).unwrap_err()));
    assert!(denied(block_on(c.function_exists("abs")).unwrap_err()));
}

#[test]
fn an_external_table_over_partitioned_files_keeps_its_partition_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE src (id INT, p STRING) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "INSERT INTO src VALUES (1, 'x'), (2, 'y')");
    let location = dir.path().join("src");
    let c = s.catalog();
    block_on(c.create_table(
        "ext",
        Some("parquet"),
        None,
        None,
        &[("path", location.to_str().unwrap())],
    ))
    .unwrap();
    let cols = block_on(c.list_columns("ext")).unwrap();
    let got: Vec<_> = cols
        .iter()
        .map(|c| (c.name.as_str(), c.is_partition))
        .collect();
    assert_eq!(got, [("id", false), ("p", true)]);
}

#[test]
fn identifiers_are_parsed_and_an_empty_description_is_no_comment() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE db1");
    let c = s.catalog();
    // a quoted identifier, and `spark_catalog.db1`, name the namespace
    block_on(c.set_current_database("`db1`")).unwrap();
    assert_eq!(c.current_database(), "db1");
    block_on(c.set_current_database("default")).unwrap();
    assert_eq!(
        block_on(c.get_database("spark_catalog.db1")).unwrap().name,
        "db1"
    );
    assert!(
        block_on(c.list_tables(Some("spark_catalog.db1"), None))
            .unwrap()
            .is_empty()
    );
    block_on(c.create_table("t", Some("parquet"), Some("a INT"), Some(""), &[])).unwrap();
    assert_eq!(block_on(c.get_table("t")).unwrap().description, None);
}

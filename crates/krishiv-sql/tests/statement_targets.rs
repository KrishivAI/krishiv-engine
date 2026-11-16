// SPDX-License-Identifier: Apache-2.0
//! `statement_targets`: the privileges a statement needs on what it writes,
//! creates, drops or reads without a query, for one statement of each shape
//! (the access-control surface; the session's `access_policy` tests cover the
//! policy side). A table of statement and expected `Privilege Object(name)`
//! lines, in order.
#![expect(clippy::panic, reason = "test code")]

use krishiv_sql::ast::{DdlName, ObjectName};
use krishiv_sql::{ParserOptions, Target, TargetObject, parse_statement, statement_targets};

fn name(o: &ObjectName) -> String {
    o.parts
        .iter()
        .map(|p| p.value.as_str())
        .collect::<Vec<_>>()
        .join(".")
}

fn ddl(n: &DdlName) -> String {
    match n {
        DdlName::Parts(o) => name(o),
        DdlName::Dynamic(_) => "IDENTIFIER(..)".to_string(),
        _ => "?".to_string(),
    }
}

fn render(t: &Target<'_>) -> String {
    let object = match t.object {
        TargetObject::Table(n) => format!("Table({})", ddl(n)),
        TargetObject::ParentOf(n) => format!("ParentOf({})", ddl(n)),
        TargetObject::Namespace(n) => format!("Namespace({})", ddl(n)),
        TargetObject::ParentOfNamespace(n) => format!("ParentOfNamespace({})", ddl(n)),
        TargetObject::Function(n) => format!("Function({})", name(n)),
        TargetObject::ParentOfFunction(n) => format!("ParentOfFunction({})", name(n)),
        TargetObject::Path(p) => format!("Path({p})"),
        TargetObject::CurrentNamespace => "CurrentNamespace".to_string(),
        TargetObject::CurrentCatalog => "CurrentCatalog".to_string(),
        TargetObject::Cluster => "Cluster".to_string(),
        _ => format!("{:?}", t.object),
    };
    format!("{:?} {object}", t.privilege)
}

fn targets(sql: &str) -> Vec<String> {
    let statement =
        parse_statement(sql, &ParserOptions::default()).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    statement_targets(&statement).iter().map(render).collect()
}

#[test]
fn each_statement_shape_needs_its_privileges() {
    let cases: &[(&str, &[&str])] = &[
        ("SELECT 1", &[]),
        // `EXPLAIN FORMAT JSON` analyzes its statement and runs nothing.
        ("EXPLAIN FORMAT JSON DROP TABLE t", &[]),
        ("CREATE TABLE db.t (x INT)", &["Create ParentOf(db.t)"]),
        (
            "CREATE TABLE t (x INT) USING parquet LOCATION '/p'",
            &["Create ParentOf(t)", "Modify Path(/p)"],
        ),
        (
            "CREATE OR REPLACE TABLE t AS SELECT 1 AS x",
            &["Create ParentOf(t)", "Drop Table(t)", "Modify Table(t)"],
        ),
        (
            "CREATE TEMPORARY TABLE t (x INT) USING parquet LOCATION '/p'",
            &["Select Path(/p)"],
        ),
        (
            "CREATE TABLE t LIKE s",
            &["Create ParentOf(t)", "Select Table(s)"],
        ),
        ("DROP TABLE t", &["Drop Table(t)"]),
        (
            "ALTER TABLE t ADD COLUMNS (c INT)",
            &["Drop Table(t)", "Create ParentOf(t)"],
        ),
        (
            "ALTER TABLE t SET LOCATION '/q'",
            &["Drop Table(t)", "Create ParentOf(t)", "Modify Path(/q)"],
        ),
        (
            "ALTER NAMESPACE db SET LOCATION '/n'",
            &[
                "Drop Namespace(db)",
                "Create ParentOfNamespace(db)",
                "Modify Path(/n)",
            ],
        ),
        ("SHOW TABLES", &["Browse CurrentNamespace"]),
        ("SHOW TABLES IN db", &["Browse Namespace(db)"]),
        ("INSERT INTO t VALUES (1)", &["Modify Table(t)"]),
        (
            "CREATE FUNCTION f(x INT) RETURNS INT RETURN x + 1",
            &["Create ParentOfFunction(f)"],
        ),
        (
            "CREATE OR REPLACE FUNCTION f(x INT) RETURNS INT RETURN x + 1",
            &["Create ParentOfFunction(f)", "Drop Function(f)"],
        ),
        (
            "CREATE TEMPORARY FUNCTION f(x INT) RETURNS INT RETURN x + 1",
            &[],
        ),
        ("DROP FUNCTION f", &["Drop Function(f)"]),
        ("DROP TEMPORARY FUNCTION f", &[]),
        // A script asks for everything its statements need, up front.
        (
            "BEGIN INSERT INTO t VALUES (1); DROP TABLE u; END",
            &["Modify Table(t)", "Drop Table(u)"],
        ),
    ];
    for (sql, want) in cases {
        assert_eq!(&targets(sql), want, "{sql}");
    }
}

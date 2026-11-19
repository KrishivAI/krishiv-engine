// SPDX-License-Identifier: Apache-2.0
//! The access-control statements (P3-T062 unit G1): `GRANT`, `REVOKE`, `SHOW GRANTS` and the role
//! statements parse into [`AccessCommand`]s, classify as `ADMIN` (`SHOW GRANTS`: `READ`), fail
//! with a syntax error where malformed, and leave every statement Spark parses (identifiers named
//! `grant`, `revoke`, `role`, `grants`) as it was. A Krishiv extension: open Spark has no grammar for
//! them, so there is no Spark golden.
#![expect(clippy::panic, reason = "test code")]

use krishiv_common::error::ErrorClass;
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::Privilege;
use krishiv_plan::analyzer::access_control::{
    AccessCommand, GrantObject, GrantPrivilege, GrantPrivileges, Grantee,
};
use krishiv_sql::ast::Statement;
use krishiv_sql::{ParserOptions, StatementClass, classify, parse_statement, redact_sql};

fn parse(sql: &str) -> Statement {
    parse_statement(sql, &ParserOptions::default()).unwrap_or_else(|e| panic!("{sql}: {}", e.error))
}

fn command(sql: &str) -> AccessCommand {
    match parse(sql) {
        Statement::AccessControl(a) => a.command,
        other => panic!("{sql}: {other:?}"),
    }
}

fn names(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).to_owned()).collect()
}

fn privs(list: &[(Privilege, &[&str])]) -> GrantPrivileges {
    GrantPrivileges::List(
        list.iter()
            .map(|(p, c)| GrantPrivilege {
                privilege: *p,
                columns: names(c),
            })
            .collect(),
    )
}

#[test]
fn grant_forms() {
    assert_eq!(
        command(
            "GRANT SELECT, MODIFY ON TABLE c.s.t TO user:ana, group:`data eng` WITH GRANT OPTION"
        ),
        AccessCommand::Grant {
            privileges: privs(&[(Privilege::Select, &[]), (Privilege::Modify, &[])]),
            object: GrantObject::Table(names(&["c", "s", "t"])),
            grantees: vec![
                Grantee::User("ana".to_owned()),
                Grantee::Group("data eng".to_owned())
            ],
            with_grant_option: true,
        }
    );
    // `TABLE` is optional; column lists merge; a whole-object grant absorbs a column list.
    assert_eq!(
        command("grant select (a, b), select (b, c) on t to role:r"),
        AccessCommand::Grant {
            privileges: privs(&[(Privilege::Select, &["a", "b", "c"])]),
            object: GrantObject::Table(names(&["t"])),
            grantees: vec![Grantee::Role("r".to_owned())],
            with_grant_option: false,
        }
    );
    assert_eq!(
        command("GRANT SELECT (a), SELECT ON VIEW v TO user:'ana@example.com'"),
        AccessCommand::Grant {
            privileges: privs(&[(Privilege::Select, &[])]),
            object: GrantObject::View(names(&["v"])),
            grantees: vec![Grantee::User("ana@example.com".to_owned())],
            with_grant_option: false,
        }
    );
    for (sql, object) in [
        (
            "GRANT ALL PRIVILEGES ON CATALOG c TO user:a",
            GrantObject::Catalog("c".to_owned()),
        ),
        (
            "GRANT ALL ON SCHEMA c.s TO user:a",
            GrantObject::Namespace(names(&["c", "s"])),
        ),
        (
            "GRANT ALL ON NAMESPACE s TO user:a",
            GrantObject::Namespace(names(&["s"])),
        ),
        (
            "GRANT ALL ON DATABASE s TO user:a",
            GrantObject::Namespace(names(&["s"])),
        ),
        (
            "GRANT ALL ON FUNCTION s.f TO user:a",
            GrantObject::Function(names(&["s", "f"])),
        ),
        (
            "GRANT ALL ON VOLUME 's3://b/p' TO user:a",
            GrantObject::Path(OptionValue::from_user("path", "s3://b/p")),
        ),
        (
            "GRANT ALL ON PATH 'file:/data' TO user:a",
            GrantObject::Path(OptionValue::from_user("path", "file:/data")),
        ),
        (
            "GRANT ALL ON CONNECTION lake TO user:a",
            GrantObject::Connection("lake".to_owned()),
        ),
        (
            "GRANT ALL ON SECRET SCOPE prod TO user:a",
            GrantObject::SecretScope("prod".to_owned()),
        ),
    ] {
        assert_eq!(
            command(sql),
            AccessCommand::Grant {
                privileges: GrantPrivileges::All,
                object,
                grantees: vec![Grantee::User("a".to_owned())],
                with_grant_option: false,
            },
            "{sql}"
        );
    }
    let every = "GRANT SELECT, MODIFY, CREATE, DROP, MONITOR, USAGE, BROWSE ON t TO user:a";
    let AccessCommand::Grant {
        privileges: GrantPrivileges::List(l),
        ..
    } = command(every)
    else {
        panic!("{every}");
    };
    assert_eq!(l.len(), 7);
}

#[test]
fn revoke_show_and_role_forms() {
    assert_eq!(
        command("REVOKE GRANT OPTION FOR SELECT ON t FROM user:a, role:r;"),
        AccessCommand::Revoke {
            grant_option_only: true,
            privileges: privs(&[(Privilege::Select, &[])]),
            object: GrantObject::Table(names(&["t"])),
            grantees: vec![Grantee::User("a".to_owned()), Grantee::Role("r".to_owned())],
        }
    );
    assert_eq!(
        command("REVOKE ALL ON SCHEMA s FROM group:g"),
        AccessCommand::Revoke {
            grant_option_only: false,
            privileges: GrantPrivileges::All,
            object: GrantObject::Namespace(names(&["s"])),
            grantees: vec![Grantee::Group("g".to_owned())],
        }
    );
    assert_eq!(
        command("SHOW GRANTS"),
        AccessCommand::ShowGrants {
            object: None,
            grantee: None
        }
    );
    assert_eq!(
        command("SHOW GRANTS ON TABLE t TO user:b"),
        AccessCommand::ShowGrants {
            object: Some(GrantObject::Table(names(&["t"]))),
            grantee: Some(Grantee::User("b".to_owned())),
        }
    );
    assert_eq!(
        command("SHOW GRANTS TO role:r"),
        AccessCommand::ShowGrants {
            object: None,
            grantee: Some(Grantee::Role("r".to_owned())),
        }
    );
    assert_eq!(
        command("CREATE ROLE analysts"),
        AccessCommand::CreateRole {
            name: "analysts".to_owned()
        }
    );
    assert_eq!(
        command("DROP ROLE `analysts`"),
        AccessCommand::DropRole {
            name: "analysts".to_owned()
        }
    );
    assert_eq!(
        command("GRANT ROLE analysts TO user:a, role:r2"),
        AccessCommand::GrantRole {
            role: "analysts".to_owned(),
            grantees: vec![
                Grantee::User("a".to_owned()),
                Grantee::Role("r2".to_owned())
            ],
        }
    );
    assert_eq!(
        command("REVOKE ROLE analysts FROM group:g"),
        AccessCommand::RevokeRole {
            role: "analysts".to_owned(),
            grantees: vec![Grantee::Group("g".to_owned())],
        }
    );
}

#[test]
fn classes() {
    for sql in [
        "GRANT SELECT ON t TO user:a",
        "REVOKE SELECT ON t FROM user:a",
        "CREATE ROLE r",
        "DROP ROLE r",
        "GRANT ROLE r TO user:a",
        "REVOKE ROLE r FROM user:a",
    ] {
        assert_eq!(classify(&parse(sql)), StatementClass::Admin, "{sql}");
    }
    assert_eq!(classify(&parse("SHOW GRANTS")), StatementClass::Read);
    assert_eq!(
        classify(&parse("SHOW GRANTS ON t TO user:b")),
        StatementClass::Read
    );
}

#[test]
fn malformed_statements_are_syntax_errors() {
    for sql in [
        "GRANT",
        "GRANT SELECT ON t",
        "GRANT SELECT t TO user:a",
        "GRANT SELECT ON t TO ana",
        "GRANT SELECT ON t TO user:",
        "GRANT SELECT ON t TO user:''",
        "GRANT SELECT ON t TO user:a WITH GRANT",
        "GRANT SELECT ON t TO user:a extra",
        "GRANT FOO ON t TO user:a",
        "GRANT MANAGE ON t TO user:a",
        "GRANT OPERATE ON t TO user:a",
        "GRANT IMPERSONATE ON t TO user:a",
        "GRANT MODIFY (a) ON t TO user:a",
        "GRANT SELECT () ON t TO user:a",
        "GRANT ON t TO user:a",
        "GRANT SELECT ON PATH s3 TO user:a",
        "REVOKE SELECT ON t TO user:a",
        "REVOKE GRANT SELECT ON t FROM user:a",
        "SHOW GRANTS ON",
        "SHOW GRANTS TO",
        "CREATE ROLE",
        "DROP ROLE r s",
        "GRANT ROLE r FROM user:a",
        "REVOKE ROLE r TO user:a",
        // Never explained.
        "EXPLAIN GRANT SELECT ON t TO user:a",
        "EXPLAIN SHOW GRANTS",
    ] {
        let e = parse_statement(sql, &ParserOptions::default())
            .map(|s| panic!("{sql} parsed: {s:?}"))
            .unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::ParseSyntaxError,
            "{sql}: {}",
            e.error
        );
        assert_eq!(e.error.sql_state(), Some("42601"), "{sql}");
    }
    // A column list on a securable without columns is refused by name.
    let e = parse_statement(
        "GRANT SELECT (a) ON CATALOG c TO user:a",
        &ParserOptions::default(),
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidStatementOrClause);
}

#[test]
fn spark_statements_are_unchanged() {
    // Identifiers that are the new words, where Spark's grammar takes them, still read as before.
    for sql in [
        "SELECT 1 AS grants",
        "SELECT role, grants FROM t",
        "SELECT * FROM grants",
        "SELECT * FROM role",
        "CREATE TABLE role (grants INT)",
        "CREATE TABLE grants (role INT)",
        "DROP TABLE role",
        "DROP TABLE grants",
        "SHOW TABLES IN grants",
        "SHOW COLUMNS IN role",
        "INSERT INTO role VALUES (1)",
        "SELECT `grant`, `revoke` FROM `grant`",
    ] {
        let s = parse(sql);
        assert!(!matches!(s, Statement::AccessControl(_)), "{sql}: {s:?}");
    }
    // `SHOW <scope> FUNCTIONS` keeps its reading: `grants` there is a scope, not the statement.
    for sql in ["SHOW grants FUNCTIONS", "SHOW GRANTS.x"] {
        if let Ok(Statement::AccessControl(_)) = parse_statement(sql, &ParserOptions::default()) {
            panic!("{sql} read as SHOW GRANTS");
        }
    }
}

#[test]
fn path_credentials_never_print() {
    let sql = "GRANT SELECT ON PATH 's3://key:hunter2@b/p?sig=hunter3' TO user:a";
    let s = parse(sql);
    let shown = format!("{s:?}");
    assert!(!shown.contains("hunter"), "{shown}");
    let redacted = redact_sql(sql).to_string();
    assert!(!redacted.contains("hunter"), "{redacted}");
}

// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

/// New commands enforce the recursion limit, reported at the whole
/// statement.
#[test]
fn ddl_commands_enforce_the_depth_limit() {
    use crate::MAX_DEPTH;
    let d = ParserOptions::default();
    // A CTAS nested past the limit fails; one level less parses.
    let deep = |n: usize| {
        format!(
            "CREATE TABLE t AS SELECT {}1{}",
            "(".repeat(n),
            ")".repeat(n)
        )
    };
    let (mut ok, mut bad) = (0, MAX_DEPTH + 10);
    while ok + 1 < bad {
        let mid = (ok + bad) / 2;
        if parse_statement(&deep(mid), &d).is_ok() {
            ok = mid;
        } else {
            bad = mid;
        }
    }
    assert!(ok > 0);
    let sql = deep(bad);
    let e = parse_statement(&sql, &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::FailedToParseTooComplex);
    assert_eq!(e.span.range(), 0..sql.len());
    // A deeply nested `DEFAULT` fails the same way.
    let sql = format!(
        "CREATE TABLE t (a INT DEFAULT {}1{})",
        "(".repeat(MAX_DEPTH),
        ")".repeat(MAX_DEPTH)
    );
    let e = parse_statement(&sql, &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::FailedToParseTooComplex);
    assert_eq!(e.span.range(), 0..sql.len());
    // So does a deeply nested parenthesized query.
    let sql = format!(
        "CREATE TABLE t (SELECT {}1{})",
        "(".repeat(MAX_DEPTH),
        ")".repeat(MAX_DEPTH)
    );
    let e = parse_statement(&sql, &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::FailedToParseTooComplex);
    assert_eq!(e.span.range(), 0..sql.len());
}

/// `CREATE TABLE` errors, continued: every clause's malformed forms fail
/// where they stand.
#[test]
fn create_table_clause_errors() {
    let d = ParserOptions::default();
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let syntax = |sql: &str| {
        let e = fail(sql);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        e
    };
    // `GENERATED` needs its form.
    syntax("CREATE TABLE t (a INT GENERATED)");
    syntax("CREATE TABLE t (a INT GENERATED ALWAYS)");
    syntax("CREATE TABLE t (a INT GENERATED ALWAYS AS 1)");
    // A `COMMENT` needs its string.
    let e = syntax("CREATE TABLE t (a INT COMMENT 1)");
    assert_eq!(e.span.range(), 30..31);
    // Constraints need their shape.
    syntax("CREATE TABLE t (a INT, CONSTRAINT c FOO)");
    syntax("CREATE TABLE t (a INT, PRIMARY FOO)");
    // `OPTIONS` needs its value after `=`.
    syntax("CREATE TABLE t (a INT) OPTIONS (a = )");
    // `COMMENT`, `LOCATION` and `COLLATION` need their strings.
    syntax("CREATE TABLE t (a INT) COMMENT 1");
    syntax("CREATE TABLE t (a INT) DEFAULT COLLATION");
    // `CLUSTER BY` needs its list.
    syntax("CREATE TABLE t (a INT) CLUSTER BY a");
    // `STORED` needs `AS` or `BY`.
    syntax("CREATE TABLE t (a INT) STORED");
    // `ROW FORMAT` needs `SERDE` or `DELIMITED`.
    syntax("CREATE TABLE t (a INT) ROW FORMAT");
    syntax("CREATE TABLE t (a INT) ROW FORMAT FOO");
    // `SKEWED ... STORED` needs `AS DIRECTORIES`.
    syntax("CREATE TABLE t (a INT) SKEWED BY (a) ON (1) STORED");
    syntax("CREATE TABLE t (a INT) SKEWED BY (a) ON (1) STORED AS");
    // `AS` needs its query.
    syntax("CREATE TABLE t (a INT) AS");
    // An unclosed parenthesized query fails inside it.
    let e = syntax("CREATE TABLE t (SELECT 1");
    assert_eq!(e.span.range(), 23..24);
    // An unclosed `IDENTIFIER(` fails naming the missing `)`.
    let e = syntax("CREATE TABLE IDENTIFIER('a'");
    assert_eq!(e.span.range(), 27..27);
    // Anything else fails where it stands.
    let e = syntax("CREATE FOO");
    assert_eq!(e.span.range(), 7..10);
    let e = syntax("DROP FOO");
    assert_eq!(e.span.range(), 5..8);
    // `WITH` needs its properties.
    syntax("CREATE SCHEMA s WITH");
    // `USE` needs its name.
    let e = syntax("USE SCHEMA");
    assert_eq!(e.span.range(), 10..10);
}

/// `PARTITIONED BY` field forms.
#[test]
fn partition_field_forms() {
    use ast::{DdlName, PartitionFieldKind, Statement};
    let d = ParserOptions::default();
    let fields = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTable(t) => t.partitioned_by,
        other => panic!("{sql} is {other:?}"),
    };
    // A dotted name is an identity transform.
    let f = fields("CREATE TABLE t (a INT) PARTITIONED BY (a.b)");
    assert!(matches!(
        &f[0].kind,
        PartitionFieldKind::Transform(tr)
            if matches!(&tr.name, DdlName::Parts(n) if n.parts.len() == 2) && tr.args.is_empty()
    ));
    // A dynamic name without a type is one too.
    let f = fields("CREATE TABLE t (a INT) PARTITIONED BY (IDENTIFIER('c'))");
    assert!(matches!(
        &f[0].kind,
        PartitionFieldKind::Transform(tr)
            if matches!(&tr.name, DdlName::Dynamic(_)) && tr.args.is_empty()
    ));
    // `NOT NULL` without `NULL`, a `COMMENT` without a string, and a
    // broken type fail inside the column.
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY (a INT NOT)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 48..49);
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY (a INT COMMENT 1)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY (a array<int)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY (a 1)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 41..42);
}

/// `TBLPROPERTIES` literal values, `OPTIONS` expressions and storage
/// defaults.
#[test]
fn property_values_and_storage_defaults() {
    use ast::{RowFormatKind, Statement};
    let d = ParserOptions::default();
    let create = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTable(t) => t,
        other => panic!("{sql} is {other:?}"),
    };
    let t = create("CREATE TABLE t (a INT) TBLPROPERTIES ('a' = TRUE, 'b' = 1.5, 'c' = \"x\")");
    assert_eq!(t.tblproperties.len(), 3);
    assert!(t.tblproperties.iter().all(|p| p.value.is_some()));
    // An `OPTIONS` value without `=` reads as the next expression.
    let t = create("CREATE TABLE t (a INT) OPTIONS (a b)");
    assert_eq!(t.options.len(), 1);
    // `SORTED BY` without a direction, `ROW FORMAT SERDE` without
    // properties, and an `IDENTITY` without options.
    let t = create("CREATE TABLE t (a INT) CLUSTERED BY (a) SORTED BY (b) INTO 2 BUCKETS");
    assert_eq!(t.bucket.as_ref().unwrap().sorted_by[0].asc, None);
    let t = create("CREATE TABLE t (a INT) ROW FORMAT SERDE 'x'");
    assert!(matches!(
        t.row_format.as_ref().unwrap().kind,
        RowFormatKind::Serde(_)
    ));
    // A bucket count that is no integer fails.
    let e = parse_statement(
        "CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 99999999999999999999999 BUCKETS",
        &d,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
}

/// A parenthesized query is the CTAS query: no columns, no clauses.
#[test]
fn parenthesized_ctas_forms() {
    use ast::Statement;
    let d = ParserOptions::default();
    let t = match parse_statement("CREATE TABLE t (SELECT 1)", &d).unwrap() {
        Statement::CreateTable(t) => t,
        other => panic!("{other:?}"),
    };
    assert!(t.elements.is_empty() && t.using.is_none());
    let q = t.as_query.as_ref().unwrap();
    assert_eq!(q.span.text("CREATE TABLE t (SELECT 1)"), "(SELECT 1)");
    // Clauses do not follow one: it ends the statement.
    let e = parse_statement("CREATE TABLE t (SELECT 1) USING parquet", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // `OR REPLACE` with anything but a table or a view fails where it
    // stands; a permanent view reads as a full view statement.
    let v = match parse_statement("CREATE OR REPLACE VIEW v AS SELECT 1", &d).unwrap() {
        Statement::CreatePermanentView(v) => v,
        other => panic!("{other:?}"),
    };
    assert!(v.or_replace && !v.temporary && !v.global);
    assert_eq!(
        v.span.text("CREATE OR REPLACE VIEW v AS SELECT 1"),
        "CREATE OR REPLACE VIEW v AS SELECT 1"
    );
    // So does anything else.
    let e = parse_statement("CREATE TEMPORARY FOO", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 17..20);
}

/// `NOT NULL` without `NULL` before a `COMMENT`, and identity column
/// counts that are no integers.
#[test]
fn column_option_edge_cases() {
    let d = ParserOptions::default();
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let e = fail("CREATE TABLE t (a INT NOT COMMENT 'x')");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("CREATE TABLE t (a INT GENERATED ALWAYS AS IDENTITY (START WITH x))");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail(
        "CREATE TABLE t (a INT GENERATED ALWAYS AS IDENTITY (START WITH 99999999999999999999999))",
    );
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // Inline constraint names and references.
    let t = match parse_statement(
        "CREATE TABLE t (a INT CONSTRAINT c PRIMARY KEY, b INT REFERENCES r, c INT CONSTRAINT d CHECK (c > 0))",
        &d,
    )
    .unwrap()
    {
        ast::Statement::CreateTable(t) => t,
        other => panic!("{other:?}"),
    };
    assert_eq!(t.elements.len(), 3);
    let ast::TableElement::Column(a) = &t.elements[0] else {
        panic!("{:?}", t.elements[0]);
    };
    assert_eq!(a.constraints.len(), 1);
    assert_eq!(a.constraints[0].name.as_ref().unwrap().value, "c");
    let e = fail("CREATE TABLE t (a INT PRIMARY FOO)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
}

/// `PARTITIONED BY` restores and reports.
#[test]
fn partition_field_edge_cases() {
    use ast::{PartitionFieldKind, Statement};
    let d = ParserOptions::default();
    let fields = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTable(t) => t.partitioned_by,
        other => panic!("{sql} is {other:?}"),
    };
    // A `COLLATE`d string reads as an identity transform, which fails at
    // the `COLLATE`.
    let f = fields("CREATE TABLE t (a INT) PARTITIONED BY (a STRING)");
    assert!(matches!(&f[0].kind, PartitionFieldKind::Column(_)));
    let e = parse_statement(
        "CREATE TABLE t (a INT) PARTITIONED BY (a STRING COLLATE x)",
        &d,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // A broken type rewinds to an identity transform, which fails after
    // the name.
    let e = parse_statement("CREATE TABLE t (a INT) PARTITIONED BY (a array<int)", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // A bad name fails as one.
    let e = parse_statement("CREATE TABLE t (a INT) PARTITIONED BY (a \u{e9})", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidIdentifier);
    // An unclosed type fails at the end of the input.
    let e =
        parse_statement("CREATE TABLE t (x INT) PARTITIONED BY (a decimal(10)", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 52..52);
    // `NOT NULL` without `NULL` before a `COMMENT` fails at the `COMMENT`.
    let e = parse_statement(
        "CREATE TABLE t (a INT) PARTITIONED BY (a INT NOT COMMENT 'x')",
        &d,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // A table constraint with referenced columns.
    let t = match parse_statement(
        "CREATE TABLE t (a INT, b INT, CONSTRAINT fk FOREIGN KEY (a) REFERENCES r (id))",
        &d,
    )
    .unwrap()
    {
        Statement::CreateTable(t) => t,
        other => panic!("{other:?}"),
    };
    let ast::TableElement::Constraint(c) = &t.elements[2] else {
        panic!("{:?}", t.elements[2]);
    };
    let ast::TableConstraintKind::ForeignKey(fk) = &c.kind else {
        panic!("{c:?}");
    };
    assert_eq!(fk.ref_columns.len(), 1);
    assert_eq!(fk.table.parts.len(), 1);
}

/// The checks Spark's builder makes on a parsed `CREATE TABLE`, with the
/// error classes and order of Spark 4.2 (each case checked on Spark).
#[test]
fn create_table_builder_checks() {
    use ast::{GeneratedKind, Statement, TableElement};
    let d = ParserOptions::default();
    let fail = |sql: &str| {
        let e = parse_statement(sql, &d).unwrap_err();
        (e.error.error_class(), e.error.to_string(), e.span.range())
    };
    // A clause written twice; the first in Spark's check order wins over
    // the order in the text.
    for (clause, text) in [
        ("TBLPROPERTIES", "TBLPROPERTIES ('a' = 'b')"),
        ("OPTIONS", "OPTIONS (k 1)"),
        ("PARTITIONED BY", "PARTITIONED BY (a)"),
        ("STORED AS/BY", "STORED AS parquet"),
        ("ROW FORMAT", "ROW FORMAT DELIMITED"),
        ("COMMENT", "COMMENT 'x'"),
        ("DEFAULT COLLATION", "DEFAULT COLLATION UTF8_BINARY"),
        ("CLUSTERED BY", "CLUSTERED BY (a) INTO 2 BUCKETS"),
        ("CLUSTER BY", "CLUSTER BY (a)"),
        ("LOCATION", "LOCATION '/a'"),
    ] {
        let sql = format!("CREATE TABLE t (a INT) {text} {text}");
        let (class, message, span) = fail(&sql);
        assert_eq!(class, ErrorClass::DuplicateClauses, "{sql}");
        assert!(
            message.contains(&format!("Found duplicate clauses: {clause}.")),
            "{sql}: {message}"
        );
        assert_eq!(span, 0..sql.len());
    }
    let (_, message, _) =
        fail("CREATE TABLE t (a INT) LOCATION '/a' LOCATION '/b' COMMENT 'x' COMMENT 'y'");
    assert!(message.contains("clauses: COMMENT."), "{message}");
    let (_, message, _) = fail(
        "CREATE TABLE t (a INT) COMMENT 'x' TBLPROPERTIES ('a' = 'b') COMMENT 'y' TBLPROPERTIES ('c' = 'd')",
    );
    assert!(message.contains("clauses: TBLPROPERTIES."), "{message}");

    // A column descriptor written twice: the first repeat in the text.
    for (sql, descriptor) in [
        ("CREATE TABLE t (a INT COMMENT 'x' COMMENT 'y')", "COMMENT"),
        ("CREATE TABLE t (a INT NOT NULL NOT NULL)", "NOT NULL"),
        ("CREATE TABLE t (a INT DEFAULT 1 DEFAULT 2)", "DEFAULT"),
        (
            "CREATE TABLE t (a INT DEFAULT 1 COMMENT 'x' COMMENT 'y' DEFAULT 2)",
            "COMMENT",
        ),
        (
            "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS (1) GENERATED ALWAYS AS IDENTITY)",
            "GENERATED ALWAYS AS",
        ),
        (
            "CREATE TABLE t (a INT CHECK (a > 0) PRIMARY KEY)",
            "CONSTRAINT",
        ),
        (
            "CREATE TABLE t (IDENTIFIER('a') INT COMMENT 'x' COMMENT 'y')",
            "COMMENT",
        ),
    ] {
        let (class, message, span) = fail(sql);
        assert_eq!(
            class,
            ErrorClass::CreateTableColumnDescriptorDuplicate,
            "{sql}"
        );
        assert!(
            message.contains(&format!("column a specifies descriptor \"{descriptor}\"")),
            "{sql}: {message}"
        );
        assert_eq!(span, 0..sql.len());
    }

    // Identity options: negative counts parse; a repeated option and a
    // zero step fail, after the column's descriptors.
    let sql =
        "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (START WITH -1 INCREMENT BY - 2))";
    let Statement::CreateTable(t) = parse_statement(sql, &d).unwrap() else {
        panic!("{sql}");
    };
    let TableElement::Column(c) = &t.elements[0] else {
        panic!("{t:?}");
    };
    let Some(GeneratedKind::Identity(i)) = c.generated.as_ref().map(|g| &g.kind) else {
        panic!("{c:?}");
    };
    assert_eq!((i.start, i.step), (Some(-1), Some(-2)));
    for (sql, option) in [
        (
            "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 1 START WITH 2))",
            "START",
        ),
        (
            "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY \
             (INCREMENT BY 1 START WITH 1 INCREMENT BY 2 START WITH 3))",
            "STEP",
        ),
    ] {
        let (class, message, _) = fail(sql);
        assert_eq!(
            class,
            ErrorClass::IdentityColumnsDuplicatedSequenceGeneratorOption,
            "{sql}"
        );
        assert!(
            message.contains(&format!("generator option: {option}.")),
            "{sql}: {message}"
        );
    }
    let (class, _, _) =
        fail("CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (INCREMENT BY 0))");
    assert_eq!(class, ErrorClass::IdentityColumnsIllegalStep);
    let (class, _, _) = fail(
        "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 1 START WITH 2) \
         COMMENT 'x' COMMENT 'y')",
    );
    assert_eq!(class, ErrorClass::CreateTableColumnDescriptorDuplicate);
    // A count out of range is a syntax error, as is `10L`.
    let (class, _, _) = fail(
        "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (START WITH -99999999999999999999))",
    );
    assert_eq!(class, ErrorClass::ParseSyntaxError);

    // `SKEWED BY` and CTAS restrictions.
    let (class, message, _) = fail("CREATE TABLE t (a INT) AS SELECT 1");
    assert_eq!(class, ErrorClass::LegacyErrorTemp0035);
    assert!(
        message
            .contains("Schema may not be specified in a Create Table As Select (CTAS) statement"),
        "{message}"
    );
    let (class, message, _) = fail("CREATE TABLE t PARTITIONED BY (a INT) AS SELECT 1");
    assert_eq!(class, ErrorClass::LegacyErrorTemp0035);
    assert!(
        message.contains(
            "Partition column types may not be specified in Create Table As Select (CTAS)"
        ),
        "{message}"
    );
    let (_, message, _) = fail("CREATE TABLE t (a INT) PARTITIONED BY (b INT) AS SELECT 1");
    assert!(message.contains("Schema may not be specified"), "{message}");
    assert!(parse_statement("CREATE TABLE t PARTITIONED BY (a) AS SELECT 1 AS a", &d).is_ok());

    // Literals anywhere in the statement get the query checks.
    let bad = "DATE '2020-13-45'";
    for sql in [
        format!("CREATE TABLE t (a DATE DEFAULT {bad})"),
        format!("CREATE TABLE t (a DATE GENERATED ALWAYS AS ({bad}))"),
        format!("CREATE TABLE t (a INT CHECK ({bad} IS NULL))"),
        format!("CREATE TABLE t (a INT, CONSTRAINT c CHECK ({bad} IS NULL))"),
        format!("CREATE TABLE t (a INT) OPTIONS (k {bad})"),
        format!("CREATE TABLE t (a INT) PARTITIONED BY (bucket(2, {bad}))"),
        format!("CREATE TABLE IDENTIFIER({bad} || 'x') (a INT)"),
        format!("CREATE TABLE t (IDENTIFIER({bad} || 'x') INT)"),
        format!("CREATE TABLE t (a INT) AS SELECT {bad}"),
        format!("CREATE TABLE t (a INT DEFAULT {bad}) AS SELECT 1"),
    ] {
        assert_eq!(fail(&sql).0, ErrorClass::InvalidTypedLiteral, "{sql}");
    }

    // The order of the checks.
    for (sql, class) in [
        (
            "CREATE TABLE t (a INT COMMENT 'x' COMMENT 'y') COMMENT 'x' COMMENT 'y'",
            ErrorClass::CreateTableColumnDescriptorDuplicate,
        ),
        (
            "CREATE TABLE t (a INT) COMMENT 'x' COMMENT 'y' SKEWED BY (a) ON (1)",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TABLE t (a INT) COMMENT 'x' COMMENT 'y' AS SELECT 1",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TABLE t (a INT) OPTIONS (k DATE '2020-13-45') COMMENT 'x' COMMENT 'y'",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TABLE t SKEWED BY (a) ON (1) AS SELECT DATE '2020-13-45'",
            ErrorClass::InvalidStatementOrClause,
        ),
    ] {
        assert_eq!(fail(sql).0, class, "{sql}");
    }
}

/// With `double_quoted_identifiers`, a double-quoted word after a
/// partition column name reads as its type, as in a column definition.
#[test]
fn create_table_double_quoted_partition_type() {
    let dq = ParserOptions {
        double_quoted_identifiers: true,
        ..ParserOptions::default()
    };
    let column = parse_statement("CREATE TABLE t (a \"int\")", &dq).unwrap_err();
    let partition =
        parse_statement("CREATE TABLE t (a INT) PARTITIONED BY (b \"int\")", &dq).unwrap_err();
    assert_eq!(column.error.error_class(), ErrorClass::UnsupportedDatatype);
    assert_eq!(
        partition.error.error_class(),
        ErrorClass::UnsupportedDatatype
    );
}

/// An error deferred before a speculative parse that then succeeds is
/// still the one reported.
#[test]
fn deferred_error_survives_a_speculative_parse() {
    let d = ParserOptions::default();
    for sql in [
        "SELECT a ! IN (1), CAST(b AS STRING COLLATE c) FROM t",
        "CREATE TABLE IDENTIFIER(CASE WHEN x ! IN (1) THEN 't' END) (SELECT 1)",
    ] {
        let e = parse_statement(sql, &d).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::SyntaxDiscontinuedBangEqualsNot,
            "{sql}"
        );
        let bang = sql.find('!').unwrap();
        assert_eq!(e.span.range(), bang..bang + 1, "{sql}");
    }
}

/// `DECLARE`, `SET VARIABLE` and `DROP TEMPORARY VARIABLE`: every form,
/// with spans.
#[test]
fn variable_statements() {
    use ast::{DeclareVariable, DropVariable, SetVariableKind, Statement};
    let d = ParserOptions::default();
    let declare = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::DeclareVariable(v) => v,
        other => panic!("{sql} is {other:?}"),
    };
    // `DECLARE name type`, with and without `VARIABLE`.
    for sql in ["DECLARE var1 INT;", "DECLARE VARIABLE var1 INT"] {
        let v: DeclareVariable = declare(sql);
        assert!(!v.or_replace, "{sql}");
        assert_eq!(v.names.len(), 1, "{sql}");
        assert_eq!(v.names[0].parts[0].value, "var1", "{sql}");
        assert_eq!(
            v.data_type,
            Some(parse_data_type("INT", &d).unwrap()),
            "{sql}"
        );
        assert!(v.default.is_none(), "{sql}");
    }
    let v = declare("DECLARE var1 INT");
    assert_eq!(v.span.text("DECLARE var1 INT"), "DECLARE var1 INT");
    // `OR REPLACE`, `VAR`, qualifiers and a keyword as a name.
    let v = declare("DECLARE OR REPLACE VAR session.var1 DOUBLE");
    assert!(v.or_replace);
    assert_eq!(
        v.names[0]
            .parts
            .iter()
            .map(|p| p.value.as_str())
            .collect::<Vec<_>>(),
        ["session", "var1"]
    );
    let v = declare("DECLARE OR REPLACE VARIABLE INT DEFAULT 5.0");
    assert!(v.or_replace);
    assert_eq!(v.names[0].parts[0].value, "INT");
    assert!(v.data_type.is_none());
    assert_eq!(
        v.default
            .as_ref()
            .unwrap()
            .span
            .text("DECLARE OR REPLACE VARIABLE INT DEFAULT 5.0"),
        "5.0"
    );
    // `DEFAULT` and `=` defaults, and several names sharing both.
    let v = declare("DECLARE VARIABLE var1, var2, var3 INT DEFAULT 1 + 2");
    assert_eq!(v.names.len(), 3);
    assert_eq!(
        v.default
            .as_ref()
            .unwrap()
            .span
            .text("DECLARE VARIABLE var1, var2, var3 INT DEFAULT 1 + 2"),
        "1 + 2"
    );
    let v = declare("DECLARE var1 = 'Hello'");
    assert!(v.data_type.is_none());
    assert_eq!(
        v.default
            .as_ref()
            .unwrap()
            .span
            .text("DECLARE var1 = 'Hello'"),
        "'Hello'"
    );
    // `parse_query` takes queries only.
    assert_eq!(
        err("DECLARE var1 INT", &d).error.error_class(),
        ErrorClass::ParseSyntaxError
    );
    // `IF NOT EXISTS` is not supported: the error is at `EXISTS`.
    let e = parse_statement("DECLARE VARIABLE IF NOT EXISTS var1 INT", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 24..30);
    // Neither a type nor a default fails after parsing.
    let e = parse_statement("DECLARE var1", &d).unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::InvalidSqlSyntaxVariableTypeOrDefaultRequired
    );
    assert_eq!(e.span.range(), 0..12);
    // `SET VARIABLE` assignments.
    let set = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::SetVariable(v) => v,
        other => panic!("{sql} is {other:?}"),
    };
    let v = set("SET VARIABLE var1 = 6;");
    let SetVariableKind::Single { assignments } = &v.kind else {
        panic!("{:?}", v.kind)
    };
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].name.parts[0].value, "var1");
    assert_eq!(
        assignments[0].value.span.text("SET VARIABLE var1 = 6;"),
        "6"
    );
    assert_eq!(
        v.span.text("SET VARIABLE var1 = 6;"),
        "SET VARIABLE var1 = 6"
    );
    let v = set("SET VAR var1 = 7, var2 = 'world', system.session.var3 = pi()");
    let SetVariableKind::Single { assignments } = &v.kind else {
        panic!("{:?}", v.kind)
    };
    assert_eq!(assignments.len(), 3);
    assert_eq!(
        assignments
            .iter()
            .map(|a| a.name.parts.last().unwrap().value.as_str())
            .collect::<Vec<_>>(),
        ["var1", "var2", "var3"]
    );
    // `SET VARIABLE (v, ...) = (query)`.
    let v = set("SET VARIABLE (var1, var2) = (SELECT c1, c2 FROM t)");
    let SetVariableKind::Row { targets, query } = &v.kind else {
        panic!("{:?}", v.kind)
    };
    assert_eq!(targets.len(), 2);
    assert_eq!(**query, ok("SELECT c1, c2 FROM t"));
    assert_eq!(
        v.span
            .text("SET VARIABLE (var1, var2) = (SELECT c1, c2 FROM t)"),
        "SET VARIABLE (var1, var2) = (SELECT c1, c2 FROM t)"
    );
    // An empty target list parses and fails after parsing.
    let e = parse_statement("SET VARIABLE () = (SELECT 1)", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidSetSyntax);
    assert_eq!(e.span.range(), 0..28);
    // The target list and the query are required.
    for sql in [
        "SET VARIABLE (var1) = SELECT 1",
        "SET VARIABLE (var1) = (SELECT 1",
        "SET VARIABLE (var1 = (SELECT 1)",
        "SET VARIABLE var1",
        "SET VARIABLE var1 = ",
    ] {
        assert_eq!(
            parse_statement(sql, &d).unwrap_err().error.error_class(),
            ErrorClass::ParseSyntaxError,
            "{sql}"
        );
    }
    // Bare `VAR`/`VARIABLE` keys still read as configuration, not as
    // session variables.
    for (sql, key, value) in [
        ("SET VAR", "VAR", None),
        ("SET VARIABLE", "VARIABLE", None),
        ("SET VAR = 1", "VAR", Some("1")),
        ("SET var.foo = 1", "var.foo", Some("1")),
    ] {
        match parse_statement(sql, &d).unwrap() {
            Statement::Set(s) => match (&s.kind, value) {
                (ast::SetKind::Show { key: k, .. }, None) => assert_eq!(k, key, "{sql}"),
                (
                    ast::SetKind::Assign {
                        key: k, value: v, ..
                    },
                    Some(want),
                ) => assert_eq!((k.as_str(), v.as_str()), (key, want), "{sql}"),
                kind => panic!("{sql} is {kind:?}"),
            },
            other => panic!("{sql} is {other:?}"),
        }
    }
    // `DROP TEMPORARY VARIABLE`.
    let drop = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::DropVariable(v) => v,
        other => panic!("{sql} is {other:?}"),
    };
    let v: DropVariable = drop("DROP TEMPORARY VARIABLE var1;");
    assert!(!v.if_exists);
    assert_eq!(v.name.parts[0].value, "var1");
    assert_eq!(
        v.span.text("DROP TEMPORARY VARIABLE var1;"),
        "DROP TEMPORARY VARIABLE var1"
    );
    let v = drop("DROP TEMPORARY VAR IF EXISTS system.session.var1");
    assert!(v.if_exists);
    assert_eq!(v.name.parts.len(), 3);
    // Without `TEMPORARY`, the error is at `VARIABLE`.
    let e = parse_statement("DROP VARIABLE var1", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 5..13);
    // `TEMPORARY` needs `VAR` or `VARIABLE` after it.
    let e = parse_statement("DROP TEMPORARY VIEW v", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 15..19);
    // A quoted type name parses and fails when the type is built.
    let e = parse_statement("DECLARE v `int`", &d).unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::UnsupportedDatatype,
        "{e:?}"
    );
    // Only one keyword is the form: the next word is the name.
    let v = declare("DECLARE VAR variable INT");
    assert_eq!(v.names[0].parts[0].value, "variable");
    let v = declare("DECLARE VARIABLE var INT");
    assert_eq!(v.names[0].parts[0].value, "var");
    let ast::Statement::SetVariable(v) = parse_statement("SET VAR variable = 1", &d).unwrap()
    else {
        panic!("not a SET VARIABLE")
    };
    let SetVariableKind::Single { assignments } = &v.kind else {
        panic!("not a single assignment")
    };
    assert_eq!(assignments[0].name.parts[0].value, "variable");
    // Under `EXPLAIN`, a variable command is the explained statement.
    let ast::Statement::Explain(e) = parse_statement("EXPLAIN DECLARE var1 INT", &d).unwrap()
    else {
        panic!("not an EXPLAIN")
    };
    assert!(matches!(*e.statement, ast::Statement::DeclareVariable(_)));
}

/// `EXECUTE IMMEDIATE query [INTO ...] [USING ...]`.
#[test]
fn execute_immediate_statements() {
    use ast::{ExprKind, Statement};
    let d = ParserOptions::default();
    let run = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::ExecuteImmediate(e) => e,
        other => panic!("{sql} is {other:?}"),
    };
    let e = run("EXECUTE IMMEDIATE 'SELECT 1'");
    assert!(e.into.is_empty() && e.using.is_empty());
    let e = run("EXECUTE IMMEDIATE s INTO a, b.c USING 1, 'x' AS y, (2) z;");
    assert_eq!(e.into.len(), 2);
    let aliases: Vec<_> = e
        .using
        .iter()
        .map(|a| a.alias.as_ref().map(|i| i.value.as_str()))
        .collect();
    assert_eq!(aliases, [None, Some("y"), Some("z")]);
    // A parenthesized list is the argument list; a parenthesized
    // expression followed by an alias is one argument.
    assert_eq!(run("EXECUTE IMMEDIATE s USING (1, 'x')").using.len(), 2);
    let e = run("EXECUTE IMMEDIATE s USING (SELECT 42) AS first, 2 AS second");
    assert_eq!(e.using.len(), 2);
    assert_eq!(
        run("EXECUTE IMMEDIATE concat('a', 'b') USING :x")
            .using
            .len(),
        1
    );
    assert!(matches!(
        run("EXECUTE IMMEDIATE s USING :x").using[0].value.kind,
        ExprKind::NamedParameter(_)
    ));
    // No parenthesized `INTO` list, no empty `USING`.
    for bad in [
        "EXECUTE IMMEDIATE s INTO (a, b)",
        "EXECUTE IMMEDIATE s USING",
        "EXECUTE IMMEDIATE",
        "EXECUTE IMMEDIATE s INTO",
    ] {
        assert!(parse_statement(bad, &d).is_err(), "{bad}");
    }
}

/// `:IDENTIFIER(...)` is not a parameter marker: ANTLR fails at its argument.
#[test]
fn identifier_clause_is_not_a_named_marker() {
    let d = ParserOptions::default();
    let e = parse_statement("SELECT :IDENTIFIER('param1') FROM t", &d).unwrap_err();
    assert_eq!(e.span.start, 19);
}

/// Scripts: compound statements, control flow, declarations and handlers.
#[test]
fn script_statements() {
    use ast::{HandlerBody, HandlerCondition, ScriptStatement as S, Statement};
    let d = ParserOptions::default();
    let script =
        |sql: &str| match parse_statement(sql, &d).unwrap_or_else(|e| panic!("{sql}: {e:?}")) {
            Statement::Script(c) => *c,
            other => panic!("{sql} is {other:?}"),
        };
    assert!(script("BEGIN END").body.is_empty());
    assert!(script("BEGIN NOT ATOMIC SELECT 1; END;").body.len() == 1);
    let c = script(
        "BEGIN
           DECLARE x INT DEFAULT 1;
           DECLARE VARIABLE a, b STRING;
           DECLARE oops CONDITION FOR SQLSTATE '22012';
           DECLARE EXIT HANDLER FOR SQLSTATE VALUE '42000', oops
             BEGIN SELECT 'e'; END;
           SET x = x + 1;
           SET (a, b) = (SELECT 'p', 'q');
           IF x > 1 THEN SELECT 1; ELSEIF x = 1 THEN SELECT 2; ELSE SELECT 3; END IF;
           CASE x WHEN 1 THEN SELECT 1; WHEN 2 THEN SELECT 2; ELSE SELECT 3; END CASE;
           CASE WHEN x > 1 THEN SELECT 1; END CASE;
           l1: WHILE x < 3 DO SET x = x + 1; ITERATE l1; END WHILE l1;
           REPEAT SET x = x - 1; UNTIL x < 1 END REPEAT;
           l2: LOOP LEAVE l2; END LOOP l2;
           FOR r AS SELECT id FROM t DO SELECT r.id; END FOR;
           FOR SELECT 1 DO SELECT 2; END FOR;
           lbl: BEGIN SELECT 1; END lbl;
           INSERT INTO t VALUES (1);
         END",
    );
    let kinds: Vec<&str> = c
        .body
        .iter()
        .map(|s| match s {
            S::Declare(_) => "declare",
            S::Condition(_) => "condition",
            S::Handler(_) => "handler",
            S::Assign(_) => "assign",
            S::If(_) => "if",
            S::Case(_) => "case",
            S::While(_) => "while",
            S::Repeat(_) => "repeat",
            S::Loop(_) => "loop",
            S::For(_) => "for",
            S::Compound(_) => "compound",
            S::Sql(_) => "sql",
            _ => "other",
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "declare",
            "declare",
            "condition",
            "handler",
            "assign",
            "assign",
            "if",
            "case",
            "case",
            "while",
            "repeat",
            "loop",
            "for",
            "for",
            "compound",
            "sql"
        ]
    );
    let S::Handler(h) = &c.body[3] else { panic!() };
    assert!(!h.continue_handler && h.conditions.len() == 2);
    assert!(matches!(&h.conditions[0], HandlerCondition::SqlState(s, _) if s == "42000"));
    assert!(matches!(&h.conditions[1], HandlerCondition::Name(n) if n.value == "oops"));
    assert!(matches!(h.body, HandlerBody::Compound(_)));
    let S::While(w) = &c.body[9] else { panic!() };
    assert_eq!(w.label.as_ref().map(|l| l.value.as_str()), Some("l1"));
    assert_eq!(w.end_label.as_ref().map(|l| l.value.as_str()), Some("l1"));
    // Handlers take one statement or an assignment too.
    for body in ["SELECT 1", "SET x = 1"] {
        let c = script(&format!(
            "BEGIN DECLARE x INT DEFAULT 0; DECLARE EXIT HANDLER FOR SQLEXCEPTION {body}; END"
        ));
        assert!(matches!(c.body[1], S::Handler(_)));
    }
}

/// What a script never parses: Spark 4.2's syntax errors and the refusals the checks raise.
#[test]
fn script_syntax_errors_and_refusals() {
    let d = ParserOptions::default();
    let class = |sql: &str| parse_statement(sql, &d).unwrap_err().error.error_class();
    for bad in [
        "lbl: BEGIN END",
        "EXPLAIN BEGIN END",
        "BEGIN SELECT 1 END",
        "BEGIN SIGNAL x; END",
        "BEGIN RESIGNAL; END",
        "BEGIN GET DIAGNOSTICS x = ROW_COUNT; END",
        "BEGIN SELECT 1 INTO x; END",
        "BEGIN SET; END",
        "BEGIN SET spark.sql.ansi.enabled; END",
        "BEGIN SET TIME ZONE 'UTC'; END",
        "BEGIN RESET; END",
        "BEGIN SET VAR x = 1; END",
        "BEGIN IF 1 THEN SELECT 1; END; END",
        "BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION IF 1 THEN SELECT 1; END IF; END",
        "BEGIN END trailing",
    ] {
        assert_eq!(class(bad), ErrorClass::ParseSyntaxError, "{bad}");
    }
    assert_eq!(
        class("BEGIN DECLARE CONTINUE HANDLER FOR SQLEXCEPTION SELECT 1; END"),
        ErrorClass::UnsupportedFeatureContinueExceptionHandler
    );
    for cursor in [
        "BEGIN DECLARE c CURSOR FOR SELECT 1; END",
        "BEGIN OPEN c; END",
        "BEGIN FETCH c INTO x; END",
        "BEGIN CLOSE c; END",
    ] {
        assert_eq!(
            class(cursor),
            ErrorClass::UnsupportedFeatureSqlCursor,
            "{cursor}"
        );
    }
}

/// The structural errors of a script (design note T196, section 2.5), by class and a parameter.
#[test]
fn script_structure_is_checked_before_anything_runs() {
    let d = ParserOptions::default();
    let fails = |sql: &str| {
        let e = parse_statement(sql, &d).unwrap_err().error;
        let params: Vec<String> = e
            .message_parameters()
            .iter()
            .map(|(_, v)| v.clone())
            .collect();
        (e.error_class().name().to_owned(), e.to_string(), params)
    };
    let cases: &[(&str, &str, &str)] = &[
        ("BEGIN a: BEGIN END b; END", "LABELS_MISMATCH", "`a`"),
        (
            "BEGIN BEGIN END b; END",
            "END_LABEL_WITHOUT_BEGIN_LABEL",
            "`b`",
        ),
        (
            "BEGIN a: BEGIN a: BEGIN END; END; END",
            "LABEL_OR_FOR_VARIABLE_ALREADY_EXISTS",
            "`a`",
        ),
        (
            "BEGIN a: BEGIN FOR a AS SELECT 1 DO SELECT 1; END FOR; END; END",
            "LABEL_OR_FOR_VARIABLE_ALREADY_EXISTS",
            "`a`",
        ),
        (
            "BEGIN session: BEGIN END; END",
            "LABEL_OR_FOR_VARIABLE_NAME_FORBIDDEN",
            "`session`",
        ),
        (
            "BEGIN FOR system AS SELECT 1 DO SELECT 1; END FOR; END",
            "LABEL_OR_FOR_VARIABLE_NAME_FORBIDDEN",
            "`system`",
        ),
        (
            "BEGIN LEAVE nowhere; END",
            "INVALID_LABEL_USAGE.DOES_NOT_EXIST",
            "NOWHERE",
        ),
        (
            "BEGIN ITERATE nowhere; END",
            "INVALID_LABEL_USAGE.DOES_NOT_EXIST",
            "NOWHERE",
        ),
        (
            "BEGIN c: BEGIN ITERATE c; END; END",
            "INVALID_LABEL_USAGE.ITERATE_IN_COMPOUND",
            "C",
        ),
        (
            "BEGIN SELECT 1; DECLARE x INT; END",
            "INVALID_VARIABLE_DECLARATION.ONLY_AT_BEGINNING",
            "`x`",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 1; DECLARE x INT; END",
            "INVALID_VARIABLE_DECLARATION.ONLY_AT_BEGINNING",
            "`x`",
        ),
        (
            "BEGIN IF 1 = 1 THEN DECLARE x INT; END IF; END",
            "INVALID_VARIABLE_DECLARATION.NOT_ALLOWED_IN_SCOPE",
            "`x`",
        ),
        (
            "BEGIN SELECT 1; DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 1; END",
            "INVALID_HANDLER_DECLARATION.WRONG_PLACE_OF_DECLARATION",
            "",
        ),
        (
            "BEGIN IF 1 = 1 THEN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 1; END IF; END",
            "INVALID_HANDLER_DECLARATION.WRONG_PLACE_OF_DECLARATION",
            "",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR nosuch SELECT 1; END",
            "INVALID_HANDLER_DECLARATION.CONDITION_NOT_FOUND",
            "NOSUCH",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION, SQLSTATE '42000' SELECT 1; END",
            "INVALID_HANDLER_DECLARATION.INVALID_CONDITION_COMBINATION",
            "",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR NOT FOUND, DIVIDE_BY_ZERO SELECT 1; END",
            "INVALID_HANDLER_DECLARATION.INVALID_CONDITION_COMBINATION",
            "",
        ),
        (
            "BEGIN DECLARE c CONDITION; DECLARE EXIT HANDLER FOR c, c SELECT 1; END",
            "INVALID_HANDLER_DECLARATION.DUPLICATE_CONDITION_IN_HANDLER_DECLARATION",
            "C",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR SQLSTATE '42000', SQLSTATE '42000' SELECT 1; END",
            "INVALID_HANDLER_DECLARATION.DUPLICATE_SQLSTATE_IN_HANDLER_DECLARATION",
            "42000",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR DIVIDE_BY_ZERO SELECT 1; DECLARE EXIT HANDLER FOR DIVIDE_BY_ZERO SELECT 2; END",
            "DUPLICATE_EXCEPTION_HANDLER.CONDITION",
            "DIVIDE_BY_ZERO",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR SQLSTATE '42000' SELECT 1; DECLARE EXIT HANDLER FOR SQLSTATE '42000' SELECT 2; END",
            "DUPLICATE_EXCEPTION_HANDLER.SQLSTATE",
            "42000",
        ),
        (
            "BEGIN DECLARE c CONDITION; DECLARE c CONDITION; END",
            "DUPLICATE_CONDITION_IN_SCOPE",
            "`C`",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 1; DECLARE c CONDITION; END",
            "INVALID_ERROR_CONDITION_DECLARATION.NOT_AT_START_OF_COMPOUND_STATEMENT",
            "`c`",
        ),
        (
            "BEGIN DECLARE c CONDITION FOR SQLSTATE '0A000x'; END",
            "INVALID_SQLSTATE",
            "0A000x",
        ),
        (
            "BEGIN DECLARE c CONDITION FOR SQLSTATE '01000'; END",
            "INVALID_SQLSTATE",
            "01000",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR SQLSTATE 'XX000' SELECT 1; END",
            "INVALID_SQLSTATE",
            "XX000",
        ),
        (
            "BEGIN DECLARE EXIT HANDLER FOR SQLSTATE '4200' SELECT 1; END",
            "INVALID_SQLSTATE",
            "4200",
        ),
    ];
    for (sql, class, mentions) in cases {
        let (got, message, params) = fails(sql);
        assert_eq!(got, *class, "{sql}: {message}");
        // Exact: Spark's spelling of the parameter (quoting and case included), not a substring.
        assert!(
            mentions.is_empty() || params.iter().any(|p| p == mentions),
            "{sql}: {params:?} lacks {mentions}"
        );
    }
    // Names compare without case; a condition and an error class are both known to a handler.
    for ok in [
        "BEGIN Lbl: BEGIN LEAVE LBL; END LBL; END",
        "BEGIN l: WHILE 1 = 1 DO ITERATE L; END WHILE; END",
        "BEGIN DECLARE c CONDITION; DECLARE EXIT HANDLER FOR C SELECT 1; END",
        "BEGIN DECLARE EXIT HANDLER FOR UNRESOLVED_COLUMN, DIVIDE_BY_ZERO SELECT 1; END",
        "BEGIN DECLARE c CONDITION FOR SQLSTATE '42000'; DECLARE EXIT HANDLER FOR NOT FOUND SELECT 1; END",
        "BEGIN a: BEGIN DECLARE c CONDITION; END; b: BEGIN DECLARE c CONDITION; END; END",
        "BEGIN i: LOOP j: LOOP LEAVE i; END LOOP; END LOOP; END",
        // Spark accepts these (design note T196, unit 0 record).
        "BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION, NOT FOUND SELECT 1; END",
        "BEGIN DECLARE EXIT HANDLER FOR SQLSTATE 'abc12' SELECT 1; END",
        "BEGIN DECLARE OR REPLACE x INT; DECLARE a.y INT; END",
    ] {
        parse_statement(ok, &d).unwrap_or_else(|e| panic!("{ok}: {e:?}"));
    }
    // Case-sensitive names differ by case.
    let strict = ParserOptions {
        case_sensitive: true,
        ..d
    };
    assert!(parse_statement("BEGIN a: BEGIN END A; END", &strict).is_err());
    assert!(parse_statement("BEGIN a: BEGIN A: BEGIN END; END; END", &strict).is_ok());
}

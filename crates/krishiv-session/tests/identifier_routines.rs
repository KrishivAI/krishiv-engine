// SPDX-License-Identifier: Apache-2.0
//! Names written through `IDENTIFIER(...)` that Spark 4.2 refuses: a
//! `TIMESTAMPADD` whose unit is not a keyword, function and view names whose
//! session-catalog namespace has several parts, and a field of an
//! identifier literal. Expected classes come from Spark 4.2.0's
//! `identifier-clause.sql.out`.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

/// The error class `sql` fails with (during planning or collection).
fn class_of(s: &Session, sql: &str) -> String {
    let run = async {
        let df = s.sql(sql).await?;
        df.collect().await
    };
    match block_on(run) {
        Ok(_) => panic!("{sql}: no error"),
        Err(e) => e.error_class().name().to_owned(),
    }
}

#[test]
fn grammar_functions_without_a_keyword_unit_are_unresolved_routines() {
    let s = Session::new().unwrap();
    for sql in [
        "SELECT TIMESTAMPADD(IDENTIFIER('YEAR'), 1, DATE'2024-01-15')",
        "SELECT timestampdiff(IDENTIFIER('DAY'), DATE'2024-01-15', DATE'2024-01-16')",
    ] {
        assert_eq!(class_of(&s, sql), "UNRESOLVED_ROUTINE", "{sql}");
    }
    // The keyword forms still run.
    let df = block_on(s.sql("SELECT TIMESTAMPADD(YEAR, 1, DATE'2024-01-15')")).unwrap();
    assert_eq!(block_on(df.collect()).unwrap()[0].num_rows(), 1);
}

#[test]
fn function_names_of_three_parts_need_a_single_part_namespace() {
    let s = Session::new().unwrap();
    for sql in [
        "VALUES(IDENTIFIER('a.b.c.d')())",
        "SELECT a.b.c()",
        "SELECT spark_catalog.a.b.c(1)",
    ] {
        assert_eq!(class_of(&s, sql), "REQUIRES_SINGLE_PART_NAMESPACE", "{sql}");
    }
    assert_eq!(class_of(&s, "SELECT a.b(1)"), "UNRESOLVED_ROUTINE");
}

#[test]
fn drop_view_of_an_identifier_with_several_parts() {
    let s = Session::new().unwrap();
    assert_eq!(
        class_of(&s, "DROP VIEW IDENTIFIER('a.b.c')"),
        "REQUIRES_SINGLE_PART_NAMESPACE"
    );
}

#[test]
fn a_field_of_an_identifier_literal_is_not_a_qualified_name() {
    let s = Session::new().unwrap();
    assert_eq!(
        class_of(&s, "SELECT IDENTIFIER('t').c1 FROM VALUES(1) AS T(c1)"),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    // A table name may still start with an identifier literal.
    block_on(s.sql("CREATE TABLE t1(c1 INT) USING parquet")).unwrap();
    let df = block_on(s.sql("SELECT c1 FROM IDENTIFIER('default').t1")).unwrap();
    assert_eq!(block_on(df.collect()).unwrap().len(), 0);
}

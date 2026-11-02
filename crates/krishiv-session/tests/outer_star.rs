// SPDX-License-Identifier: Apache-2.0
//! `t.*` in a subquery's select list expands to the enclosing query's
//! columns of `t` (Spark 4.2's scalar-subquery-select golden).
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[test]
fn an_outer_star_reads_the_enclosing_relation() -> R {
    let s = Session::new()?;
    let batches = block_on(async {
        s.sql("select (select t1.* from values (2) as t2(col1)) from values (1) as t1(col1)")
            .await?
            .collect()
            .await
    })?;
    assert_eq!(batches[0].column(0).as_primitive::<Int32Type>().value(0), 1);
    let two = block_on(async {
        s.sql("select (select t1.* from values (2) as t2(col1)) from values (1, 5) as t1(col1, c2)")
            .await
    });
    let err = two.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        err.contains("SCALAR_SUBQUERY_RETURN_MORE_THAN_ONE_OUTPUT_COLUMN"),
        "{err}"
    );
    Ok(())
}

#[test]
fn a_temporary_function_named_by_identifier_takes_no_database() {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TEMPORARY FUNCTION IDENTIFIER('default.my' || 'Avg') AS 'test.org.MyAvg'",
        "DROP TEMPORARY FUNCTION IDENTIFIER('default.my' || 'Avg')",
    ] {
        let e = block_on(s.sql(sql))
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(e.contains("INVALID_TEMP_OBJ_QUALIFIER"), "{sql}: {e}");
    }
}

#[test]
fn the_legacy_identifier_clause_still_names_a_function() {
    let s = Session::new().unwrap();
    block_on(s.sql("set spark.sql.legacy.identifierClause = true")).unwrap();
    let e = block_on(s.sql("DROP TEMPORARY FUNCTION IDENTIFIER('default.my' || 'Avg')"))
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(e.contains("INVALID_TEMP_OBJ_QUALIFIER"), "{e}");
}

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

/// A relation whose columns a local `USING` join merged away still shadows
/// the enclosing query's relation of the same alias.
#[test]
fn a_using_join_relation_shadows_the_outer_one() -> R {
    let s = Session::new()?;
    let batches = block_on(async {
        s.sql(
            "select (select t1.* from values (2) as t2(k) join values (2) as t1(k) using (k)) \
             from values (1) as t1(k)",
        )
        .await?
        .collect()
        .await
    })?;
    assert_eq!(batches[0].column(0).as_primitive::<Int32Type>().value(0), 2);
    Ok(())
}

/// An outer relation with a merged-away `USING` column expands all its
/// columns: two columns in a scalar subquery are refused.
#[test]
fn an_outer_star_includes_the_using_columns_of_the_outer_query() {
    let s = Session::new().unwrap();
    let r = block_on(s.sql(
        "select (select b.*) from values (1) as a(k) join values (1, 2) as b(k, v) using (k)",
    ));
    let err = r.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        err.contains("SCALAR_SUBQUERY_RETURN_MORE_THAN_ONE_OUTPUT_COLUMN"),
        "{err}"
    );
}

/// A local struct column named like an enclosing relation wins: `t.*` expands
/// the struct's fields, not the outer relation.
#[test]
fn a_local_struct_column_wins_over_the_outer_relation() -> R {
    let s = Session::new()?;
    let batches = block_on(async {
        s.sql(
            "select (select t.* from (select named_struct('k', 2) as t) as s) \
             from values (1) as t(k)",
        )
        .await?
        .collect()
        .await
    })?;
    assert_eq!(batches[0].column(0).as_primitive::<Int32Type>().value(0), 2);
    Ok(())
}

/// The generated `_metadata` column of a file source is not part of an
/// outer star: `p.*` is just `k`.
#[test]
fn an_outer_star_over_a_file_table_skips_the_metadata_column() -> R {
    let dir = tempfile::tempdir()?;
    let s = Session::new()?;
    s.conf().set(
        "spark.sql.warehouse.dir",
        dir.path().to_str().unwrap_or_default(),
    )?;
    for sql in [
        "create table p (k int) using parquet",
        "insert into p values (1)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await })?;
    }
    let batches = block_on(async { s.sql("select (select p.*) from p").await?.collect().await })?;
    assert_eq!(batches[0].column(0).as_primitive::<Int32Type>().value(0), 1);
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

/// Rewriting `NULL IN (subquery)` (to `EXISTS(..) AND NULL`) must not
/// rename the output column: it keeps the name of the `IN` as written.
#[test]
fn a_null_in_subquery_keeps_its_written_column_name() -> R {
    let s = Session::new()?;
    let schema = block_on(async {
        s.sql("select cast(null as int) in (select 1)")
            .await?
            .schema()
            .await
    })?;
    let name = schema.field(0).name().to_lowercase();
    assert!(name.contains(" in "), "{name}");
    assert!(!name.contains("exists"), "{name}");
    Ok(())
}

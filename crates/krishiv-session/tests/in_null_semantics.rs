// SPDX-License-Identifier: Apache-2.0
//! `NULL IN (subquery)`: false over no rows, NULL over some (Spark 4.2.0's
//! in-null-semantics golden); NULL either way under
//! `spark.sql.legacy.nullInEmptyListBehavior`.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn one(s: &Session, sql: &str) -> R<String> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let c = batches[0].column(0);
        Ok(if c.is_null(0) {
            "NULL".to_owned()
        } else {
            c.as_boolean().value(0).to_string()
        })
    })
}

#[test]
fn null_in_a_subquery_depends_on_whether_it_has_rows() -> R {
    let s = Session::new()?;
    block_on(s.sql("create temporary view v(e) as values (1), (null)"))?;
    block_on(s.sql("create temporary view v_empty(e) as select e from v where false"))?;
    assert_eq!(one(&s, "select null in (select e from v_empty)")?, "false");
    assert_eq!(
        one(&s, "select null not in (select e from v_empty)")?,
        "true"
    );
    assert_eq!(one(&s, "select null in (select e from v)")?, "NULL");
    assert_eq!(one(&s, "select null not in (select e from v)")?, "NULL");
    block_on(s.sql("set spark.sql.legacy.nullInEmptyListBehavior = true"))?;
    assert_eq!(one(&s, "select null in (select e from v_empty)")?, "NULL");
    assert_eq!(
        one(&s, "select null not in (select e from v_empty)")?,
        "NULL"
    );
    Ok(())
}

#[test]
fn a_subquery_select_list_reads_the_outer_query() -> R {
    let s = Session::new()?;
    block_on(s.sql("create temporary view t(c) as values (8), (12), (null)"))?;
    let rows = |sql: &str| -> R<Vec<String>> {
        block_on(async {
            let b = s.sql(sql).await?.collect().await?;
            let mut out = Vec::new();
            for batch in &b {
                for r in 0..batch.num_rows() {
                    let v = batch.column(1);
                    out.push(if v.is_null(r) {
                        "NULL".to_owned()
                    } else {
                        format!(
                            "{}",
                            arrow_array::cast::as_primitive_array::<arrow_array::types::Int32Type>(
                                v
                            )
                            .value(r)
                        )
                    });
                }
            }
            out.sort();
            Ok(out)
        })
    };
    assert_eq!(rows("select c, (select c) from t")?, ["12", "8", "NULL"]);
    assert_eq!(
        rows("select c, (select c where c = 8) from t")?,
        ["8", "NULL", "NULL"]
    );
    Ok(())
}

#[test]
fn in_a_correlated_count_is_an_equality_with_the_count_bug_handled() -> R {
    let s = Session::new()?;
    block_on(s.sql("create temporary view t1(c1, c2) as values (0, 1), (1, 2)"))?;
    block_on(s.sql("create temporary view t2(c1, c2) as values (0, 2), (0, 3)"))?;
    let rows = |sql: &str| -> R<Vec<String>> {
        block_on(async {
            let b = s.sql(sql).await?.collect().await?;
            let mut out = Vec::new();
            for batch in &b {
                for r in 0..batch.num_rows() {
                    out.push(
                        batch
                            .columns()
                            .iter()
                            .map(|c| {
                                if c.is_null(r) {
                                    "NULL".to_owned()
                                } else if let Some(x) = c.as_boolean_opt() {
                                    x.value(r).to_string()
                                } else {
                                    c.as_primitive::<arrow_array::types::Int32Type>()
                                        .value(r)
                                        .to_string()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\t"),
                    );
                }
            }
            out.sort();
            Ok(out)
        })
    };
    assert_eq!(
        rows("select *, c1 in (select count(*) + 1 from t2 where t2.c1 = t1.c1) from t1")?,
        ["0\t1\tfalse", "1\t2\ttrue"]
    );
    assert_eq!(
        rows("select *, c1 not in (select count(*) + 1 from t2 where t2.c1 = t1.c1) from t1")?,
        ["0\t1\ttrue", "1\t2\tfalse"]
    );
    Ok(())
}

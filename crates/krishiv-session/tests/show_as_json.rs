// SPDX-License-Identifier: Apache-2.0
//! `SHOW ... AS JSON` (Spark 4.3): the expected strings are what Spark 4.3.0 returned for the same statements
//! (real run, `apache/spark:4.3.0`), so the key order is Spark's too.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn one(s: &Session, sql: &str) -> String {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let col = arrow_cast::cast(batches[0].column(0), &arrow_schema::DataType::Utf8).unwrap();
    assert_eq!(batches[0].num_rows(), 1, "{sql}");
    arrow_array::cast::AsArray::as_string::<i32>(&col)
        .value(0)
        .to_owned()
}

#[test]
fn show_tables_extended_and_partitions_as_json_match_spark() {
    let s = Session::new().unwrap();
    for q in [
        "create table tbl (id int, name string) using parquet",
        "create table pt (id int, p int) using parquet partitioned by (p)",
        "insert into pt values (1, 7)",
        "insert into pt values (2, 8)",
    ] {
        block_on(async { s.sql(q).await?.collect().await }).unwrap();
    }
    assert_eq!(
        one(&s, "show tables as json"),
        r#"{"tables":[{"name":"pt","namespace":["default"],"isTemporary":false},{"name":"tbl","namespace":["default"],"isTemporary":false}]}"#
    );
    assert_eq!(
        one(&s, "show tables in default like 'tb*' as json"),
        r#"{"tables":[{"name":"tbl","namespace":["default"],"isTemporary":false}]}"#
    );
    assert_eq!(
        one(&s, "show table extended like 'tbl' as json"),
        r#"{"tables":[{"name":"tbl","catalog":"spark_catalog","namespace":["default"],"type":"MANAGED","isTemporary":false}]}"#
    );
    assert_eq!(
        one(&s, "show partitions pt as json"),
        r#"{"partitions":[{"p":"7"},{"p":"8"}]}"#
    );
    assert_eq!(
        one(&s, "show partitions pt partition(p=7) as json"),
        r#"{"partitions":[{"p":"7"}]}"#
    );
}

fn err_class(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => "ok".to_owned(),
        Err(e) => e.error_class().name().to_owned(),
    }
}

#[test]
fn json_keeps_partition_escapes_lists_views_and_refuses_a_partition() {
    let s = Session::new().unwrap();
    for q in [
        "create table tbl (id int, name string) using parquet",
        "create table pe (id int, p string) using parquet partitioned by (p)",
        "insert into pe partition(p='a/b') values (1)",
        "insert into pe partition(p='x=y:z%') values (2)",
        "create temp view tv as select 1",
        "create view v as select * from tbl",
        "create or replace global temp view gv as select 1 as a",
    ] {
        block_on(async { s.sql(q).await?.collect().await }).unwrap();
    }
    // Spark keeps the `%XX` escapes of the partition path.
    assert_eq!(
        one(&s, "show partitions pe as json"),
        r#"{"partitions":[{"p":"a%2Fb"},{"p":"x%3Dy%3Az%25"}]}"#
    );
    // A temporary view has the session catalog's name and the type VIEW.
    assert_eq!(
        one(&s, "show table extended like 'tv' as json"),
        r#"{"tables":[{"name":"tv","catalog":"spark_catalog","namespace":[],"type":"VIEW","isTemporary":true}]}"#
    );
    assert_eq!(
        one(&s, "show table extended like 'v' as json"),
        r#"{"tables":[{"name":"v","catalog":"spark_catalog","namespace":["default"],"type":"VIEW","isTemporary":false}]}"#
    );
    assert_eq!(
        one(&s, "show table extended like 't*' as json"),
        r#"{"tables":[{"name":"tbl","catalog":"spark_catalog","namespace":["default"],"type":"MANAGED","isTemporary":false},{"name":"tv","catalog":"spark_catalog","namespace":[],"type":"VIEW","isTemporary":true}]}"#
    );
    assert_eq!(
        one(&s, "show table extended in global_temp like 'gv' as json"),
        r#"{"tables":[{"name":"gv","catalog":"spark_catalog","namespace":["global_temp"],"type":"VIEW","isTemporary":true}]}"#
    );
    assert_eq!(
        err_class(&s, "show table extended like 'pe' partition(p='a') as json"),
        "UNSUPPORTED_FEATURE.SHOW_TABLE_EXTENDED_JSON_WITH_PARTITION"
    );
}

/// The `system` tables have no session-catalog identity or stored type, so the extended JSON form is refused there.
#[test]
fn show_table_extended_as_json_over_the_system_catalog_is_refused() {
    let s = Session::new().unwrap();
    let e = block_on(async {
        s.sql("show table extended in system like 'queries' as json")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE", "{e}");
    assert!(one(&s, "show tables in system like 'queries' as json").contains("\"queries\""));
}

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

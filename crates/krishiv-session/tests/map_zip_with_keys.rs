// SPDX-License-Identifier: Apache-2.0
//! `map_zip_with` over maps with different key types casts both keys to the
//! wider type (Spark 4.2.0 `typeCoercion/native/mapZipWith` goldens).
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn result(sql: &str) -> (String, String) {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let ty = schema.field(0).data_type().to_string();
    let b = block_on(df.collect()).unwrap();
    (ty, format!("{:?}", b[0].column(0)).replace(['\n', ' '], ""))
}

#[test]
fn keys_meet_at_the_wider_type() {
    let (ty, _) =
        result("SELECT map_zip_with(map(2Y, 1Y), map(2S, 1S), (k, v1, v2) -> struct(k, v1, v2))");
    assert!(ty.contains("Int16"), "{ty}");
    let (ty, rows) = result(
        "SELECT map_zip_with(map(array(1L, 2L), array(1L, 2L)), map(array(1, 2), array(1, 2)), (k, v1, v2) -> struct(k, v1, v2))",
    );
    assert!(ty.contains("Int64"), "{ty}");
    assert!(!rows.is_empty());
    let (ty, _) = result("SELECT map_zip_with(map(2, 1), map(2L, 1L), (k, v1, v2) -> v1 + v2)");
    assert!(ty.contains("Int64"), "{ty}");
}

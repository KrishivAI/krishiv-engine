// SPDX-License-Identifier: Apache-2.0
//! Output column names of `variant_get` / `try_variant_get` (acceptance
//! SQL-41): Spark drops the type argument. Spark 4.1 to 4.3 goldens.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

#[test]
fn variant_get_column_names_match_spark() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let mut bad = Vec::new();
    for ver in ["4.1.0", "4.2.0", "4.3.0"] {
        let text = std::fs::read_to_string(
            root.join(format!("spark-{ver}/column-names-variant-get.json")),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let s = Session::new().unwrap();
        for setup in v["setup"].as_array().unwrap() {
            block_on(async {
                s.sql(setup.as_str().unwrap())
                    .await
                    .unwrap()
                    .collect()
                    .await
            })
            .unwrap();
        }
        for case in v["results"].as_array().unwrap() {
            if case.get("columns").is_none() || case["sql"].as_str().unwrap().contains("::") {
                continue;
            }
            let sql = case["sql"].as_str().unwrap();
            let want = case["columns"][0].as_str().unwrap();
            let got = block_on(async {
                let df = s.sql(sql).await?;
                df.schema().await.map(|sc| sc.field(0).name().clone())
            });
            match got {
                Ok(g) if g == want => {}
                other => bad.push(format!("{ver} {}: want {want}, got {other:?}", case["id"])),
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

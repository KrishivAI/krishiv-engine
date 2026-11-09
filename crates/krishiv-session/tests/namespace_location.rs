// SPDX-License-Identifier: Apache-2.0
//! `CREATE NAMESPACE ... LOCATION` qualifies the location as Spark does (Spark 4.2.0, local run
//! 2026-11-08: relative locations resolve against the warehouse directory and are normalised).
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

/// The `Location` row of `DESCRIBE NAMESPACE EXTENDED name`.
fn location(s: &Session, name: &str) -> String {
    let batches = block_on(async {
        s.sql(&format!("DESCRIBE NAMESPACE EXTENDED {name}"))
            .await?
            .collect()
            .await
    })
    .unwrap();
    for b in &batches {
        let (k, v) = (b.column(0), b.column(1));
        for r in 0..b.num_rows() {
            let key = k.as_string_view().value(r);
            if key == "Location" {
                return v.as_string_view().value(r).to_owned();
            }
        }
    }
    panic!("no Location row");
}

#[test]
fn a_created_namespace_stores_the_qualified_normalised_location() {
    let wh = tempfile::tempdir().unwrap();
    let wh_path = wh.path().canonicalize().unwrap();
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", wh_path.to_str().unwrap())
        .unwrap();
    let cases = [
        (
            "rel/dir//x/",
            format!("file:{}/rel/dir/x", wh_path.display()),
        ),
        ("rel/./y", format!("file:{}/rel/y", wh_path.display())),
        ("a/../b", format!("file:{}/b", wh_path.display())),
        ("/abs//z/", "file:/abs/z".to_owned()),
        ("file:/abs2//w/", "file:/abs2/w".to_owned()),
    ];
    for (i, (written, want)) in cases.iter().enumerate() {
        block_on(async {
            s.sql(&format!("CREATE NAMESPACE ns{i} LOCATION '{written}'"))
                .await?
                .collect()
                .await
        })
        .unwrap();
        assert_eq!(&location(&s, &format!("ns{i}")), want, "{written}");
    }
}

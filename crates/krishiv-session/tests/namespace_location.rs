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
        // The root slash stays (Hadoop Path, Spark 4.2.0 JVM).
        ("s3a://b/", "s3a://b/".to_owned()),
        ("s3a://b//", "s3a://b/".to_owned()),
        ("s3a://b/x//y/", "s3a://b/x/y".to_owned()),
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

/// `ALTER NAMESPACE ... SET LOCATION` qualifies like `CREATE`, an empty location is
/// `INVALID_EMPTY_LOCATION`, and a location with a scheme or an absolute path never needs the
/// warehouse directory (here one that cannot be created).
#[test]
fn set_location_qualifies_and_an_empty_location_is_refused() {
    let wh = tempfile::tempdir().unwrap();
    let wh_path = wh.path().canonicalize().unwrap();
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", wh_path.to_str().unwrap())
        .unwrap();
    let run = |q: &str| block_on(async { s.sql(q).await?.collect().await });
    run("CREATE NAMESPACE n1").unwrap();
    run("ALTER NAMESPACE n1 SET LOCATION 'rel//a/./b/'").unwrap();
    assert_eq!(
        location(&s, "n1"),
        format!("file:{}/rel/a/b", wh_path.display())
    );
    run("ALTER NAMESPACE n1 SET LOCATION 's3a://b/'").unwrap();
    assert_eq!(location(&s, "n1"), "s3a://b/");
    let err = run("CREATE NAMESPACE n2 LOCATION ''")
        .unwrap_err()
        .to_string();
    assert!(err.contains("INVALID_EMPTY_LOCATION"), "{err}");
    let err = run("ALTER NAMESPACE n1 SET LOCATION ''")
        .unwrap_err()
        .to_string();
    assert!(err.contains("INVALID_EMPTY_LOCATION"), "{err}");
}

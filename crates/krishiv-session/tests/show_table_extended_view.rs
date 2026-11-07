// SPDX-License-Identifier: Apache-2.0
//! `SHOW TABLE EXTENDED ... PARTITION (...)` on a view. Spark 4.2.0 builds
//! `UnresolvedTable(names, "SHOW TABLE EXTENDED ... PARTITION ...")` for this form
//! (`AstBuilder.visitShowTableExtended`), with `ns :+ pattern` when there is an `IN`,
//! so a view found is `EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE` with that operation, not
//! `FORBIDDEN_OPERATION`. A temporary view prints as written (the vendored golden
//! `change-column.sql.out` prints `` `temp_view` ``) and a permanent one with its
//! catalog and namespace. The exact statements here were not run on Spark; a run is
//! requested.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Result<(), String> {
    block_on(async { s.sql(sql).await?.collect().await })
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn session() -> Session {
    let s = Session::new().unwrap();
    for q in [
        "CREATE TABLE p (a INT, b INT) USING parquet PARTITIONED BY (b)",
        "CREATE VIEW v AS SELECT * FROM p",
        "CREATE TEMPORARY VIEW tv AS SELECT * FROM p",
    ] {
        run(&s, q).unwrap();
    }
    s
}

fn failure(s: &Session, q: &str) -> String {
    run(s, q).expect_err(q)
}

const OP: &str = "'SHOW TABLE EXTENDED ... PARTITION ...' expects a table but";

#[test]
fn a_permanent_view_prints_with_its_catalog_and_namespace() {
    let s = session();
    let q = "SHOW TABLE EXTENDED LIKE 'v' PARTITION (b = 1)";
    let e = failure(&s, q);
    assert!(
        e.contains("EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"),
        "{q}: {e}"
    );
    assert!(
        e.contains(&format!("{OP} `spark_catalog`.`default`.`v` is a view")),
        "{q}: {e}"
    );
    // With an IN the name is two-part and finds the permanent view the same way.
    let q = "SHOW TABLE EXTENDED IN default LIKE 'v' PARTITION (b = 1)";
    let e = failure(&s, q);
    assert!(
        e.contains(&format!("{OP} `spark_catalog`.`default`.`v` is a view")),
        "{q}: {e}"
    );
}

#[test]
fn a_temporary_view_prints_as_written_and_only_for_the_bare_form() {
    let s = session();
    let q = "SHOW TABLE EXTENDED LIKE 'tv' PARTITION (b = 1)";
    let e = failure(&s, q);
    assert!(
        e.contains("EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"),
        "{q}: {e}"
    );
    assert!(e.contains(&format!("{OP} `tv` is a view")), "{q}: {e}");
    // With an IN the lookup is the two-part name, which never finds a temporary view.
    let q = "SHOW TABLE EXTENDED IN default LIKE 'tv' PARTITION (b = 1)";
    let e = failure(&s, q);
    assert!(e.contains("TABLE_OR_VIEW_NOT_FOUND"), "{q}: {e}");
}

#[test]
fn a_temporary_view_shadows_a_table_of_the_same_name() {
    let s = session();
    run(&s, "CREATE TEMPORARY VIEW p AS SELECT 1 AS a, 2 AS b").unwrap();
    let q = "SHOW TABLE EXTENDED LIKE 'p' PARTITION (b = 1)";
    let e = failure(&s, q);
    assert!(e.contains(&format!("{OP} `p` is a view")), "{q}: {e}");
    // With an IN the two-part name skips the temporary view and finds the table.
    let q = "SHOW TABLE EXTENDED IN default LIKE 'p' PARTITION (b = 1)";
    let e = failure(&s, q);
    assert!(e.contains("PARTITIONS_NOT_FOUND"), "{q}: {e}");
}

#[test]
fn a_name_in_capitals_finds_the_view_too() {
    let s = session();
    let q = "SHOW TABLE EXTENDED LIKE 'TV' PARTITION (b = 1)";
    let e = failure(&s, q);
    assert!(
        e.contains("EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"),
        "{q}: {e}"
    );
    // A temporary view prints as typed, case included.
    assert!(e.contains(&format!("{OP} `TV` is a view")), "{q}: {e}");
}

#[test]
fn a_table_with_no_such_partition_is_still_partitions_not_found() {
    let s = session();
    let q = "SHOW TABLE EXTENDED LIKE 'p' PARTITION (b = 1)";
    let e = failure(&s, q);
    assert!(e.contains("PARTITIONS_NOT_FOUND"), "{q}: {e}");
}

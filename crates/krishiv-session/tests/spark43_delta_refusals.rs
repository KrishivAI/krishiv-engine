// SPDX-License-Identifier: Apache-2.0
//! What a Delta table answers to a time travel clause and to `WITH (...)` options on `DELETE`/`UPDATE`:
//! delta-spark reads a version and takes the options, so Krishiv refuses by its own name
//! (`KRISHIV_UNSUPPORTED_FEATURE`), never with a Spark class claiming Spark refuses it.
#![cfg(all(test, feature = "delta"))]

use futures::executor::block_on;
use krishiv_session::Session;

fn run(session: &Session, sql: &str) -> Result<(), String> {
    block_on(async { session.sql(sql).await?.collect().await })
        .map(|_| ())
        .map_err(|e| e.error_class().name().to_owned())
}

#[test]
fn a_delta_table_refuses_time_travel_and_dml_options_by_its_own_name() {
    let dir = tempfile::tempdir().expect("temp dir");
    let s = Session::new().expect("session");
    run(
        &s,
        &format!(
            "create table d (id int) using delta location '{}'",
            dir.path().display()
        ),
    )
    .expect("create");
    run(&s, "insert into d values (1)").expect("insert");
    for sql in [
        "select * from d version as of 1",
        "select * from d@v1",
        "delete from d with ('k' = 'v') where id = 1",
        "update d with ('k' = 'v') set id = 2",
    ] {
        assert_eq!(
            run(&s, sql),
            Err("KRISHIV_UNSUPPORTED_FEATURE".to_owned()),
            "{sql}"
        );
    }
    // Without the options the same statements run.
    run(&s, "delete from d where id = 1").expect("delete");
}

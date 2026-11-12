// SPDX-License-Identifier: Apache-2.0
//! The session hooks the Spark Connect server uses (design P3-T011 §3.13):
//! `Session::from_plan_with` (ExecutePlan's operation id), `RuntimeConf::entries`
//! (Config `GetAll`) and `Session::clone_session` (CloneSession).
#![cfg(test)]

use std::sync::Arc;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringArray, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

/// The rows of `query` as text, one string per row (columns joined by `,`).
fn rows(s: &Session, query: &str) -> Vec<String> {
    let batches: Vec<RecordBatch> = block_on(block_on(s.sql(query)).unwrap().collect()).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    let any = c.as_any();
                    if c.is_null(i) {
                        "NULL".to_owned()
                    } else if let Some(a) = any.downcast_ref::<Int32Array>() {
                        a.value(i).to_string()
                    } else if let Some(a) = any.downcast_ref::<Int64Array>() {
                        a.value(i).to_string()
                    } else if let Some(a) = any.downcast_ref::<StringArray>() {
                        a.value(i).to_owned()
                    } else if let Some(a) = any.downcast_ref::<StringViewArray>() {
                        a.value(i).to_owned()
                    } else {
                        panic!("unexpected column type {}", c.data_type())
                    }
                })
                .collect();
            out.push(cells.join(","));
        }
    }
    out
}

fn run(s: &Session, query: &str) {
    block_on(block_on(s.sql(query)).unwrap().collect()).unwrap();
}

fn error_class(s: &Session, query: &str) -> String {
    let e = match block_on(s.sql(query)) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).unwrap_err(),
    };
    e.error_class().name().to_owned()
}

#[test]
fn entries_list_the_set_keys_sorted_as_set_does() {
    let s = Session::new().unwrap();
    s.conf().set("zz.app.flag", "1").unwrap();
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    s.conf().set("aa.app.flag", "2").unwrap();
    let entries = s.conf().entries();
    let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    assert_eq!(keys, sorted);
    assert!(entries.contains(&("aa.app.flag".to_owned(), "2".to_owned())));
    assert!(entries.contains(&("zz.app.flag".to_owned(), "1".to_owned())));
    assert!(
        entries
            .iter()
            .any(|(k, v)| k.ends_with("caseSensitive") && v == "true"),
        "{entries:?}"
    );
    // Defaults are not listed; the list is what `SET` shows.
    assert!(!entries.iter().any(|(k, _)| k.ends_with("ansi.enabled")));
    let listed: Vec<String> = entries.iter().map(|(k, v)| format!("{k},{v}")).collect();
    assert_eq!(rows(&s, "SET"), listed);
    s.conf().unset("zz.app.flag").unwrap();
    assert!(!s.conf().entries().iter().any(|(k, _)| k == "zz.app.flag"));
}

#[test]
fn a_clone_starts_as_a_copy_and_then_goes_its_own_way() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    run(&s, "CREATE TEMPORARY VIEW v AS SELECT a * 10 AS b FROM t");
    run(
        &s,
        "CREATE TEMPORARY FUNCTION twice(x INT) RETURNS INT RETURN x * 2",
    );
    run(&s, "DECLARE VARIABLE k INT DEFAULT 7");
    run(&s, "CREATE NAMESPACE hooks_ns");
    run(&s, "USE hooks_ns");
    s.conf().set("my.app.flag", "on").unwrap();

    let c = s.clone_session().unwrap();
    assert_eq!(c.conf().get("my.app.flag").as_deref(), Some("on"));
    assert_eq!(c.conf().entries(), s.conf().entries());
    assert_eq!(rows(&c, "SELECT sum(a) FROM t"), ["3"]);
    assert_eq!(rows(&c, "SELECT b FROM v ORDER BY b"), ["10", "20"]);
    assert_eq!(rows(&c, "SELECT twice(k)"), ["14"]);
    assert_eq!(rows(&c, "SELECT current_database()"), ["hooks_ns"]);
    assert_eq!(
        rows(&c, "SELECT current_user()"),
        rows(&s, "SELECT current_user()")
    );

    // Changes to the clone stay in the clone ...
    c.conf().set("my.app.flag", "off").unwrap();
    run(&c, "DROP VIEW v");
    run(&c, "SET VAR k = 8");
    run(&c, "USE default");
    run(&c, "CREATE TEMPORARY VIEW only_clone AS SELECT 1 AS x");
    assert_eq!(s.conf().get("my.app.flag").as_deref(), Some("on"));
    assert_eq!(rows(&s, "SELECT b FROM v ORDER BY b"), ["10", "20"]);
    assert_eq!(rows(&s, "SELECT k"), ["7"]);
    assert_eq!(rows(&s, "SELECT current_database()"), ["hooks_ns"]);
    assert_eq!(
        error_class(&s, "SELECT * FROM only_clone"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    // ... and the source's stay in the source.
    run(
        &s,
        "CREATE OR REPLACE TEMPORARY FUNCTION twice(x INT) RETURNS INT RETURN x * 3",
    );
    run(&s, "CREATE TEMPORARY VIEW only_source AS SELECT 1 AS x");
    assert_eq!(rows(&c, "SELECT twice(k)"), ["16"]);
    assert_eq!(
        error_class(&c, "SELECT * FROM only_source"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    // RESET goes back to the starting configuration, not the clone point.
    run(&c, "RESET");
    assert_eq!(c.conf().get("my.app.flag"), None);
    assert_eq!(s.conf().get("my.app.flag").as_deref(), Some("on"));
    // Permanent objects are shared, as across `new_session`.
    run(&c, "CREATE TABLE hooks_shared (x INT) USING parquet");
    assert_eq!(rows(&s, "SELECT count(*) FROM default.hooks_shared"), ["0"]);
}

#[test]
fn cached_relations_are_not_copied() {
    let s = Session::new().unwrap();
    run(&s, "CREATE TEMPORARY VIEW r AS SELECT id FROM range(1000)");
    run(&s, "CACHE TABLE r");
    assert!(s.cache_reserved() > 0);
    let c = s.clone_session().unwrap();
    assert_eq!(rows(&c, "SELECT count(*) FROM r"), ["1000"]);
    assert_eq!(c.cache_reserved(), 0);
}

#[test]
fn a_closed_session_is_not_cloned() {
    let s = Session::new().unwrap();
    s.close();
    let e = s.clone_session().unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");
}

#[cfg(feature = "unstable-ir")]
mod from_plan_with {
    use futures::executor::block_on;
    use krishiv_plan::Plan;
    use krishiv_session::{QueryOptions, Session};

    use super::rows;

    const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";

    fn plan(s: &Session, query: &str) -> Plan {
        let df = block_on(s.sql(query)).unwrap();
        Plan::try_new((**df.plan()).clone()).unwrap()
    }

    #[test]
    fn the_operation_id_names_the_query_and_is_held_by_the_dataframe() {
        let s = Session::new().unwrap();
        let p = plan(&s, "SELECT 'from-plan' AS tag");
        let options = QueryOptions::new().operation_id(OP);
        let df = s.from_plan_with(&p, options.clone()).unwrap();
        let e = s.from_plan_with(&p, options.clone()).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "INVALID_HANDLE.OPERATION_ALREADY_EXISTS"
        );
        let e = block_on(s.sql_with("SELECT 1", options.clone())).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "INVALID_HANDLE.OPERATION_ALREADY_EXISTS"
        );
        block_on(df.collect()).unwrap();
        drop(df);
        let e = s.from_plan_with(&p, options).unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_HANDLE.OPERATION_ABANDONED");
        assert!(
            rows(&s, "SELECT query_id FROM system.queries").contains(&OP.to_owned()),
            "the plan's query is recorded under the operation id"
        );
    }

    #[test]
    fn a_bad_id_or_a_closed_session_is_refused() {
        let s = Session::new().unwrap();
        let p = plan(&s, "SELECT 1");
        let e = s
            .from_plan_with(&p, QueryOptions::new().operation_id("not-a-uuid"))
            .unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_HANDLE.FORMAT");
        s.close();
        let e = s.from_plan_with(&p, QueryOptions::new()).unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");
    }

    #[test]
    fn a_deadline_limits_the_plan_query() {
        let s = Session::new().unwrap();
        let p = plan(&s, "SELECT count(*) FROM range(0, 20000000000)");
        let df = s
            .from_plan_with(
                &p,
                QueryOptions::new().deadline(std::time::Duration::from_millis(50)),
            )
            .unwrap();
        let e = block_on(df.collect()).unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_QUERY_TIMEOUT");
    }
}

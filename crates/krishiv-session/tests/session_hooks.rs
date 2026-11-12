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
        entries.contains(&("spark.sql.caseSensitive".to_owned(), "true".to_owned())),
        "{entries:?}"
    );
    // A registered key set through its Spark alias is listed under its
    // canonical name, once.
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    let entries = s.conf().entries();
    let ansi: Vec<&(String, String)> = entries
        .iter()
        .filter(|(k, _)| k.ends_with("ansi.enabled"))
        .collect();
    assert_eq!(
        ansi,
        [&(
            krishiv_common::config::ANSI_ENABLED.key.to_owned(),
            "false".to_owned()
        )]
    );
    s.conf().unset("spark.sql.ansi.enabled").unwrap();
    let entries = s.conf().entries();
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
    // RESET goes back to the starting configuration, not the clone point
    // (Spark v4.3.0 `ResetCommand`, SetCommand.scala:220-238: clear, then
    // the shared state's conf and the session's initial options, which
    // `cloneSession` passes empty, SparkSession.scala:286-298).
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

/// The first query context of the error `query` fails with.
fn fragment(s: &Session, query: &str) -> (String, usize, usize) {
    let e = match block_on(s.sql(query)) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).unwrap_err(),
    };
    let c = e
        .query_contexts()
        .first()
        .cloned()
        .unwrap_or_else(|| panic!("no context: {e}"));
    (c.fragment, c.start, c.stop)
}

/// The clone hands out SQL source ids after the source's: a function
/// created in the clone does not share an id with a copied one, so an error
/// inside its body points into its own statement.
#[test]
fn errors_in_a_function_created_in_a_clone_point_into_its_own_text() {
    let s = Session::new().unwrap();
    for i in 0..4 {
        run(
            &s,
            &format!(
                "CREATE TEMPORARY FUNCTION srcfn{i}(x INT) RETURNS INT RETURN x + {i} + \
                 length('a long body of text so that a span of another statement fits in it')"
            ),
        );
    }
    let c = s.clone_session().unwrap();
    let fresh = Session::new().unwrap();
    let mut seen = Vec::new();
    for session in [&fresh, &c] {
        run(session, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS a");
        run(
            session,
            "CREATE TEMPORARY FUNCTION zzg(y INT) RETURNS INT RETURN (SELECT max(a) + y FROM tv)",
        );
        run(session, "DROP VIEW tv");
        seen.push(fragment(session, "SELECT zzg(1)"));
    }
    assert_eq!(seen[0].0, "tv", "{seen:?}");
    assert_eq!(seen[1], seen[0]);
    // The copied functions still answer in the clone.
    assert_eq!(rows(&c, "SELECT srcfn3(1)"), ["70"]);
}

/// Spark's `cloneSession` passes empty initial session options
/// (SparkSession.scala:286-298 at v4.3.0, the same at v3.5.0, v4.0.0,
/// v4.1.0 and v4.2.0), and `RESET` restores the shared state's
/// configuration plus those options (SetCommand.scala:220-238): a clone's
/// `RESET` goes back to the runtime's configuration, without the options
/// the source's builder added. The source's own `RESET` keeps them.
#[test]
fn reset_in_a_clone_goes_back_to_the_runtime_configuration() {
    let rt = krishiv_session::EngineRuntime::builder()
        .config("my.runtime.flag", "rt")
        .build()
        .unwrap();
    let s = Session::builder()
        .runtime(Arc::clone(&rt))
        .config("my.builder.flag", "b")
        .build()
        .unwrap();
    s.conf().set("my.app.flag", "on").unwrap();
    let c = s.clone_session().unwrap();
    assert_eq!(c.conf().get("my.builder.flag").as_deref(), Some("b"));
    run(&c, "RESET my.builder.flag");
    assert_eq!(c.conf().get("my.builder.flag"), None);
    run(&c, "SET my.builder.flag = again");
    run(&c, "RESET");
    assert_eq!(c.conf().get("my.runtime.flag").as_deref(), Some("rt"));
    assert_eq!(c.conf().get("my.builder.flag"), None);
    assert_eq!(c.conf().get("my.app.flag"), None);
    run(&s, "RESET");
    assert_eq!(s.conf().get("my.builder.flag").as_deref(), Some("b"));
    assert_eq!(s.conf().get("my.runtime.flag").as_deref(), Some("rt"));
}

/// The clone of a session with a memory limit gets a pool of its own with
/// that limit (the key is fixed when a session is built, so the source's
/// value is the one it was built with), and keeps it across `RESET`.
#[test]
fn a_clone_of_a_limited_session_has_its_own_pool_with_the_same_limit() {
    let rt = krishiv_session::EngineRuntime::builder()
        .config("krishiv.memory.limit", "100m")
        .build()
        .unwrap();
    let s = Session::builder()
        .runtime(Arc::clone(&rt))
        .config("krishiv.session.memoryLimit", "48m")
        .build()
        .unwrap();
    let c = s.clone_session().unwrap();
    run(&c, "RESET");
    assert_eq!(
        c.conf().get("krishiv.session.memoryLimit"),
        s.conf().get("krishiv.session.memoryLimit")
    );
    let pool = |s: &Session| {
        rows(
            s,
            "SELECT path, limit_bytes FROM system.memory \
             WHERE kind = 'pool' AND (parent IS NULL OR parent = 'process')",
        )
    };
    let (source, clone) = (pool(&s), pool(&c));
    assert_eq!(source.len(), 1, "{source:?}");
    assert_eq!(clone.len(), 1, "{clone:?}");
    assert!(clone[0].starts_with("process/session:"), "{clone:?}");
    assert_ne!(clone, source);
    assert!(clone[0].ends_with(&format!(",{}", 48 << 20)), "{clone:?}");
}

fn py_def(name: &str) -> krishiv_session::PythonUdfDef {
    krishiv_session::PythonUdfDef {
        name: name.to_owned(),
        eval_type: krishiv_plan::PythonEvalType::ArrowBatched,
        return_type: krishiv_types::DataType::Long,
        deterministic: true,
        thread_safe: true,
    }
}

/// Python UDFs registered before the clone resolve in it; later ones stay
/// in the session that registered them; versions continue from the copy.
#[test]
fn python_udfs_are_copied_at_the_clone_point() {
    let s = Session::new().unwrap();
    s.register_python_udf(py_def("py_before"));
    let c = s.clone_session().unwrap();
    s.register_python_udf(py_def("py_source"));
    c.register_python_udf(py_def("py_clone"));
    assert!(block_on(c.sql("SELECT py_before(1)")).is_ok());
    assert!(block_on(c.sql("SELECT py_source(1)")).is_err());
    assert!(block_on(s.sql("SELECT py_clone(1)")).is_err());
    assert_eq!(c.register_python_udf(py_def("py_before")).version, 2);
    assert_eq!(s.register_python_udf(py_def("py_before")).version, 2);
}

/// The current catalog and the path are copied; a global temporary view
/// is visible in the clone.
#[test]
fn the_catalog_path_and_global_views_are_copied() {
    let s = Session::new().unwrap();
    run(&s, "CREATE GLOBAL TEMPORARY VIEW hooks_gv AS SELECT 5 AS g");
    run(&s, "SET spark.sql.path.enabled = true");
    run(&s, "SET PATH = system.session, system.builtin");
    run(&s, "USE SYSTEM");
    let c = s.clone_session().unwrap();
    assert_eq!(rows(&c, "SELECT current_catalog()"), ["system"]);
    assert_eq!(
        rows(&c, "SELECT current_path()"),
        rows(&s, "SELECT current_path()")
    );
    assert_eq!(rows(&c, "SELECT g FROM global_temp.hooks_gv"), ["5"]);
}

/// A DataFrame view pins a snapshot of the memory table it reads: dropping
/// the table in the clone leaves the view readable in both sessions.
#[test]
fn a_copied_dataframe_view_keeps_its_snapshot() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![4, 5]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    block_on(s.table("t").create_or_replace_temp_view("dv")).unwrap();
    let c = s.clone_session().unwrap();
    run(&c, "DROP TABLE t");
    assert_eq!(rows(&c, "SELECT sum(a) FROM dv"), ["9"]);
    assert_eq!(rows(&s, "SELECT sum(a) FROM dv"), ["9"]);
    assert_eq!(rows(&s, "SELECT sum(a) FROM t"), ["9"]);
}

/// Cloning while another thread creates and drops views gives a usable
/// clone each time: every copied view reads.
#[test]
fn a_clone_during_concurrent_ddl_is_usable() {
    let s = Session::new().unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ddl = {
        let (s, stop) = (s.clone(), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut i = 0_u32;
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                run(
                    &s,
                    &format!(
                        "CREATE OR REPLACE TEMPORARY VIEW cv{} AS SELECT {i} AS x",
                        i % 4
                    ),
                );
                run(&s, &format!("DROP VIEW IF EXISTS cv{}", (i + 2) % 4));
                i = i.wrapping_add(1);
            }
        })
    };
    for _ in 0..50 {
        let c = s.clone_session().unwrap();
        for name in c.temp_table_names() {
            assert_eq!(rows(&c, &format!("SELECT count(*) FROM {name}")), ["1"]);
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Release);
    ddl.join().unwrap();
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

    /// A result stream of the plan's DataFrame holds the id after the
    /// DataFrame is dropped.
    #[test]
    fn a_live_result_stream_holds_the_operation_id() {
        let s = Session::new().unwrap();
        let p = plan(&s, "SELECT id FROM range(10)");
        let options = QueryOptions::new().operation_id(OP);
        let df = s.from_plan_with(&p, options.clone()).unwrap();
        let stream = block_on(df.execute_stream()).unwrap();
        drop(df);
        let e = s.from_plan_with(&p, options.clone()).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "INVALID_HANDLE.OPERATION_ALREADY_EXISTS"
        );
        drop(stream);
        let e = s.from_plan_with(&p, options).unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_HANDLE.OPERATION_ABANDONED");
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
        // A grouped aggregate over a huge range: no statistic answers it.
        let p = plan(
            &s,
            "SELECT id % 7 AS k, max(id) FROM range(0, 20000000000) GROUP BY id % 7",
        );
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

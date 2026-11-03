// SPDX-License-Identifier: Apache-2.0
//! The `system` catalog (P2-T126 unit 1).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

/// The first column of `sql` as strings (the query casts to STRING itself).
fn strings(sql: &str) -> Vec<String> {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let b = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for batch in &b {
        let c = batch.column(0);
        if let Some(v) = c.as_string_view_opt() {
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        } else {
            let v = c.as_string::<i32>();
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        }
    }
    out
}

fn error(sql: &str) -> String {
    let s = Session::new().unwrap();
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => format!("{sql}: it succeeded"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn capabilities_lists_the_build() {
    let r = strings(
        "SELECT concat_ws('|', kind, name, value) FROM system.capabilities \
         WHERE kind = 'build' ORDER BY name",
    );
    assert_eq!(r[0], "build|engine|krishiv");
    let r = strings("SELECT value FROM system.capabilities WHERE name = 'system_tables'");
    assert_eq!(r, ["1"]);
}

#[test]
fn the_catalog_name_is_matched_without_case() {
    let r = strings("SELECT worker_id FROM SYSTEM.Capabilities LIMIT 1");
    assert_eq!(r, ["local"]);
}

#[test]
fn an_unknown_system_table_is_not_found() {
    let e = error("SELECT * FROM system.nope");
    assert!(e.contains("TABLE_OR_VIEW_NOT_FOUND"), "{e}");
}

#[test]
fn nothing_can_be_created_or_dropped_in_system() {
    for sql in [
        "CREATE DATABASE system",
        "CREATE SCHEMA IF NOT EXISTS system",
        "DROP DATABASE system",
        "CREATE DATABASE spark_catalog.system",
        "DROP SCHEMA SPARK_CATALOG.System",
        "CREATE TABLE system.x (a INT) USING parquet",
        "CREATE TABLE system.capabilities (a INT) USING parquet",
        "DROP TABLE system.capabilities",
        "INSERT INTO system.capabilities VALUES ('a', 'b', 'c', 'd', 'e')",
        "CREATE VIEW system.v AS SELECT 1",
    ] {
        let e = error(sql);
        assert!(e.contains("KRISHIV_RESERVED_NAME"), "{sql}: {e}");
    }
}

#[test]
fn memory_lists_the_pool_tree() {
    let r = strings("SELECT path FROM system.memory WHERE kind = 'pool' AND parent IS NULL");
    assert_eq!(r, ["process"]);
    // Rows are the tree as the scan was planned: before the reading query's own pool exists.
    let r = strings(
        "SELECT CAST(sum(reserved_bytes) >= 0 AS STRING) FROM system.memory WHERE kind = 'consumer'",
    );
    assert_eq!(r, ["true"]);
    let r = strings("SELECT worker_id FROM system.memory LIMIT 1");
    assert_eq!(r, ["local"]);
}

#[test]
fn a_view_over_a_system_table_ignores_a_session_table_of_the_same_name() {
    let s = Session::new().unwrap();
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "x",
        arrow_schema::DataType::Int32,
        false,
    )]));
    s.register_batches("capabilities", schema, vec![]).unwrap();
    let df = block_on(s.sql("SELECT name FROM system.capabilities")).unwrap();
    block_on(df.create_or_replace_temp_view("v")).unwrap();
    let rows = block_on(async {
        s.sql("SELECT count(*) FROM v WHERE name = 'engine'")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert!(rows[0].num_rows() == 1);
}

fn run_all(s: &Session, sql: &str) -> Result<(), String> {
    block_on(async { s.sql(sql).await?.collect().await })
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn strings_in(s: &Session, sql: &str) -> Vec<String> {
    let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for batch in &b {
        let c = batch.column(0);
        if let Some(v) = c.as_string_view_opt() {
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        } else {
            let v = c.as_string::<i32>();
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        }
    }
    out
}

#[test]
fn queries_lists_finished_and_failed_statements_with_their_text() {
    let s = Session::new().unwrap();
    run_all(&s, "SELECT 1 AS one, 'secret-literal' AS s").unwrap();
    run_all(&s, "SELECT * FROM range(3)").unwrap();
    let ids = strings_in(
        &s,
        "SELECT query_id FROM system.queries ORDER BY submitted_at",
    );
    assert!(ids.len() >= 2, "{ids:?}");
    // Every id is a UUIDv7 and they differ.
    let mut seen = std::collections::BTreeSet::new();
    for id in &ids {
        assert_eq!(id.len(), 36, "{id}");
        assert_eq!(&id[14..15], "7", "{id}");
        assert!(seen.insert(id.clone()), "{id}");
    }
    let states = strings_in(&s, "SELECT DISTINCT state FROM system.queries");
    assert!(
        states
            .iter()
            .all(|x| ["finished", "failed", "running", "cancelled"].contains(&x.as_str())),
        "{states:?}"
    );
    let texts = strings_in(
        &s,
        "SELECT query_text FROM system.queries WHERE query_text LIKE 'SELECT 1 AS one%'",
    );
    assert_eq!(texts.len(), 1, "{texts:?}");
    // Literals are kept (`queryText.literals = keep`, the design default).
    assert!(texts[0].contains("secret-literal"), "{}", texts[0]);
    let walls = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.queries WHERE state = 'finished' AND wall_ns > 0 AND ended_at >= submitted_at",
    );
    assert!(walls[0].parse::<u32>().unwrap() >= 1, "{walls:?}");
}

#[test]
fn operators_lists_what_a_finished_query_ran() {
    let s = Session::new().unwrap();
    run_all(
        &s,
        "SELECT id % 3 AS k, count(*) FROM range(1000) GROUP BY id % 3",
    )
    .unwrap();
    let n = strings_in(&s, "SELECT CAST(count(*) AS STRING) FROM system.operators");
    assert!(n[0].parse::<u32>().unwrap() >= 2, "{n:?}");
    // Rows join to their query, and an aggregate saw all 1,000 input rows.
    let rows = strings_in(
        &s,
        "SELECT CAST(max(o.rows_in) AS STRING) FROM system.operators o JOIN system.queries q ON o.query_id = q.query_id \
         WHERE q.query_text LIKE '%GROUP BY%'",
    );
    assert_eq!(rows, ["1000"], "{rows:?}");
    let roles = strings_in(
        &s,
        "SELECT DISTINCT role FROM system.operators ORDER BY role",
    );
    assert!(
        roles
            .iter()
            .all(|r| ["source", "operator", "sink"].contains(&r.as_str())),
        "{roles:?}"
    );
}

#[test]
fn queries_records_statements_refused_before_they_ran() {
    let s = Session::new().unwrap();
    assert!(run_all(&s, "SELECT nope FROM range(1)").is_err());
    assert!(run_all(&s, "SELECT 1 +").is_err());
    let rows = strings_in(
        &s,
        "SELECT concat_ws('|', state, error_class) FROM system.queries WHERE state NOT IN ('finished', 'running') ORDER BY submitted_at",
    );
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert!(rows[0].starts_with("failed|UNRESOLVED_COLUMN"), "{rows:?}");
    assert!(rows[1].starts_with("failed|PARSE_SYNTAX_ERROR"), "{rows:?}");
}

#[test]
fn queries_carries_the_query_wide_counters() {
    let s = Session::new().unwrap();
    run_all(
        &s,
        "SELECT id % 7 AS k, count(*) FROM range(100000) GROUP BY id % 7",
    )
    .unwrap();
    let r = strings_in(
        &s,
        "SELECT concat_ws('|', workload_class, CAST(pipelines >= 1 AS STRING), CAST(workers >= 1 AS STRING), \
         CAST(cpu_ns >= 0 AS STRING), CAST(spill_bytes >= 0 AS STRING)) FROM system.queries WHERE state = 'finished'",
    );
    assert_eq!(r, ["interactive|true|true|true|true"], "{r:?}");
}

#[test]
fn queries_name_the_session_and_its_principal() {
    use krishiv_session::{Principal, PrincipalKind};
    let a = Session::builder()
        .principal(Principal::new("etl", PrincipalKind::Service))
        .build()
        .unwrap();
    let b = Session::new().unwrap();
    for s in [&a, &b] {
        run_all(s, "SELECT 1").unwrap();
        run_all(s, "SELECT 2").unwrap();
    }
    let rows = |s: &Session| {
        strings_in(
            s,
            "SELECT concat_ws('|', principal, session_id) FROM system.queries ORDER BY submitted_at",
        )
    };
    let (ra, rb) = (rows(&a), rows(&b));
    assert!(ra.len() >= 2 && ra.iter().all(|r| r == &ra[0]), "{ra:?}");
    assert!(
        ra[0].starts_with("etl|") && !rb[0].starts_with("etl|"),
        "{ra:?} {rb:?}"
    );
    assert_ne!(ra[0], rb[0]);
    assert_eq!(ra[0].len(), "etl|".len() + 36);
}

#[test]
fn queries_carry_the_trace_of_the_inbound_header() {
    let s = Session::new().unwrap();
    run_all(&s, "SELECT 1").unwrap();
    s.set_trace_context("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        .unwrap();
    run_all(&s, "SELECT 2").unwrap();
    s.clear_trace_context();
    run_all(&s, "SELECT 3").unwrap();
    let r = strings_in(
        &s,
        "SELECT coalesce(trace_id, '-') FROM system.queries ORDER BY submitted_at",
    );
    assert_eq!(
        &r[..3],
        ["-", "4bf92f3577b34da6a3ce929d0e0e4736", "-"],
        "{r:?}"
    );
}

#[test]
fn queries_name_the_statement_class() {
    let s = Session::new().unwrap();
    run_all(&s, "SELECT 1 AS one").unwrap();
    let r = strings_in(
        &s,
        "SELECT statement_class FROM system.queries WHERE query_text LIKE 'SELECT 1 AS one%'",
    );
    assert_eq!(r, ["READ"], "{r:?}");
}

#[test]
fn caches_counts_the_relation_cache_and_the_file_metadata_cache() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let path = dir.path().join("t");
    block_on(async {
        s.sql("SELECT id FROM range(100)")
            .await?
            .write()
            .mode("overwrite")
            .format("parquet")
            .save(path.to_str().unwrap())
            .await
    })
    .unwrap();
    // A file read loads (and then re-uses) the file's metadata.
    for _ in 0..2 {
        run_all(
            &s,
            &format!("SELECT count(*) FROM parquet.`{}`", path.display()),
        )
        .unwrap();
    }
    run_all(&s, "CACHE TABLE c AS SELECT id FROM range(10)").unwrap();
    let r = strings_in(
        &s,
        "SELECT concat_ws('|', cache, CAST(entries AS STRING), CAST(hits >= 0 AS STRING)) FROM system.caches ORDER BY cache",
    );
    assert_eq!(r.len(), 2, "{r:?}");
    assert!(
        r[0].starts_with("file_metadata|") && r[1].starts_with("relation|"),
        "{r:?}"
    );
    let n: u64 = r[0].split('|').nth(1).unwrap().parse().unwrap();
    assert!(n >= 1, "file metadata entries: {r:?}");
    let n: u64 = r[1].split('|').nth(1).unwrap().parse().unwrap();
    assert!(n >= 1, "relation entries: {r:?}");
}

#[test]
fn metrics_lists_the_engine_series_with_labels() {
    let s = Session::new().unwrap();
    run_all(&s, "SELECT count(*) FROM range(10)").unwrap();
    let r = strings_in(
        &s,
        "SELECT concat_ws('|', kind, name, CAST(size(labels) >= 0 AS STRING)) FROM system.metrics \
         WHERE name = 'krishiv.build.info'",
    );
    assert_eq!(r.len(), 1, "{r:?}");
    assert!(r[0].starts_with("gauge|krishiv.build.info|true"), "{r:?}");
    // The label map holds the build info; a histogram row carries its buckets.
    let labels = strings_in(
        &s,
        "SELECT CAST(size(labels) AS STRING) FROM system.metrics WHERE name = 'krishiv.build.info'",
    );
    assert!(labels[0].parse::<u32>().unwrap() >= 1, "{labels:?}");
    let h = strings_in(
        &s,
        "SELECT CAST(size(buckets) AS STRING) FROM system.metrics WHERE kind = 'histogram' LIMIT 1",
    );
    if let Some(n) = h.first() {
        assert!(n.parse::<u32>().unwrap() >= 1, "{h:?}");
    }
}

#[test]
fn operators_carry_pipelines_extras_and_workers() {
    let s = Session::new().unwrap();
    run_all(
        &s,
        "SELECT id % 3 AS k, count(*) FROM range(1000) GROUP BY id % 3",
    )
    .unwrap();
    let r = strings_in(
        &s,
        "SELECT concat_ws('|', CAST(size(pipelines) >= 1 AS STRING), CAST(size(extra) >= 0 AS STRING), CAST(workers AS STRING)) \
         FROM system.operators LIMIT 1",
    );
    assert_eq!(r, ["true|true|1"], "{r:?}");
    // Some operator reports an extra counter (an aggregate's groups, say).
    let n = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.operators WHERE size(extra) > 0",
    );
    assert!(n[0].parse::<u32>().unwrap() >= 1, "{n:?}");
}

#[test]
fn queries_carry_the_session_tags() {
    let s = Session::new().unwrap();
    s.set_tag("team", "etl").unwrap();
    s.set_tag("job", "nightly").unwrap();
    run_all(&s, "SELECT 1").unwrap();
    s.remove_tag("job");
    run_all(&s, "SELECT 2").unwrap();
    let r = strings_in(
        &s,
        "SELECT concat_ws(',', CAST(size(tags) AS STRING), tags['team'], coalesce(tags['job'], '-')) \
         FROM system.queries ORDER BY submitted_at",
    );
    assert_eq!(&r[..2], ["2,etl,nightly", "1,etl,-"], "{r:?}");
    assert!(s.set_tag("", "x").is_err());
    assert!(s.set_tag("k", &"v".repeat(257)).is_err());
    for i in 0..31 {
        s.set_tag(&format!("t{i}"), "v").unwrap();
    }
    assert!(s.set_tag("one-too-many", "v").is_err());
    s.set_tag("team", "other").unwrap();
}

#[test]
fn queries_count_result_rows_and_io() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let out = dir.path().join("o");
    block_on(async {
        s.sql("SELECT id FROM range(5000)")
            .await?
            .write()
            .mode("overwrite")
            .format("parquet")
            .save(out.to_str().unwrap())
            .await
    })
    .unwrap();
    run_all(
        &s,
        &format!("SELECT id FROM parquet.`{}` WHERE id < 7", out.display()),
    )
    .unwrap();
    let r = strings_in(
        &s,
        "SELECT concat_ws('|', CAST(rows_produced AS STRING), CAST(bytes_read > 0 AS STRING)) \
         FROM system.queries WHERE state = 'finished' AND query_text LIKE '%WHERE id < 7%'",
    );
    assert_eq!(r, ["7|true"], "{r:?}");
}

#[test]
fn an_eager_failure_is_one_row_not_two() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT 1 / 0 FROM range(2)"));
    drop(df);
    let before = strings_in(&s, "SELECT CAST(count(*) AS STRING) FROM system.queries");
    // `EXPLAIN ANALYZE` runs the query inside `sql`; a failure there already has its row.
    let _ = block_on(s.sql("EXPLAIN ANALYZE SELECT CAST('x' AS INT) FROM range(2)"));
    let after = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.queries WHERE query_text LIKE 'EXPLAIN ANALYZE SELECT CAST%'",
    );
    assert!(
        after[0].parse::<u32>().unwrap() <= 1,
        "{before:?} {after:?}"
    );
}

#[test]
fn overlapping_refused_statements_each_get_a_row() {
    let s = Session::new().unwrap();
    // Statements of one session refused while others run: none of them is hidden by another's row.
    std::thread::scope(|scope| {
        for i in 0..8 {
            let s = s.clone();
            scope.spawn(move || {
                let sql = if i % 2 == 0 {
                    format!("SELECT nope_{i} FROM range(1)")
                } else {
                    format!("SELECT {i} +")
                };
                assert!(run_all(&s, &sql).is_err());
            });
        }
    });
    let n = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.queries WHERE state = 'failed' \
         AND (query_text LIKE 'SELECT nope_%' OR query_text LIKE 'SELECT _ +')",
    );
    assert_eq!(n, ["8"], "{n:?}");
}

#[test]
fn explain_analyze_and_commands_show_their_statement_in_the_history() {
    let s = Session::new().unwrap();
    run_all(&s, "EXPLAIN ANALYZE SELECT 1 AS probe").unwrap();
    let r = strings_in(
        &s,
        "SELECT concat_ws('|', state, statement_class) FROM system.queries \
         WHERE query_text LIKE 'EXPLAIN ANALYZE SELECT 1 AS probe%'",
    );
    // The explained query that ran, and the one-row result frame that was collected.
    assert_eq!(r.len(), 2, "{r:?}");
    assert!(r.iter().all(|row| row == "finished|READ"), "{r:?}");
    // Neither is an anonymous row.
    let anonymous = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.queries \
         WHERE query_text IS NULL AND state = 'finished' AND statement_class IS NULL",
    );
    assert_eq!(anonymous, ["0"], "{anonymous:?}");
    // A command's result frame shows its statement when it runs.
    run_all(&s, "SHOW TABLES").unwrap();
    let shown = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.queries WHERE query_text LIKE 'SHOW TABLES%'",
    );
    assert_eq!(shown, ["1"], "{shown:?}");
}

#[test]
fn a_result_cache_hit_has_its_own_row_in_the_history() {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.resultCache.enabled", "true").unwrap();
    s.set_tag("team", "etl").unwrap();
    for _ in 0..2 {
        run_all(&s, "SELECT 1 AS cached_probe").unwrap();
    }
    let rows = strings_in(
        &s,
        "SELECT concat_ws('|', state, tags['team']) FROM system.queries \
         WHERE query_text LIKE 'SELECT 1 AS cached_probe%'",
    );
    assert_eq!(rows, ["finished|etl", "finished|etl"], "{rows:?}");
}

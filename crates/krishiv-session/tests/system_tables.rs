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
fn queries_records_an_action_that_fails_in_physical_planning() {
    let s = Session::new().unwrap();
    // Analyzes, so `sql` succeeds; the planner has no window form of `approx_count_distinct` yet.
    let q = "SELECT approx_count_distinct(id) OVER (PARTITION BY id % 2) FROM range(4)";
    let df = block_on(s.sql(q)).unwrap();
    assert!(block_on(df.collect()).is_err());
    let rows = strings_in(
        &s,
        "SELECT concat_ws('|', state, query_text) FROM system.queries WHERE state NOT IN ('finished', 'running')",
    );
    assert_eq!(rows.len(), 1, "one row, no duplicate: {rows:?}");
    assert!(
        rows[0].starts_with("failed|SELECT approx_count_distinct"),
        "{rows:?}"
    );
}

#[test]
fn queries_records_an_action_whose_reanalysis_fails() {
    let s = Session::new().unwrap();
    run_all(&s, "CREATE TEMPORARY VIEW v AS SELECT id FROM range(3)").unwrap();
    let df = block_on(s.sql("SELECT id FROM v")).unwrap();
    run_all(&s, "DROP VIEW v").unwrap();
    // The catalog changed, so the action analyzes again, and fails.
    assert!(block_on(df.collect()).is_err());
    let rows = strings_in(
        &s,
        "SELECT concat_ws('|', state, error_class, query_text) FROM system.queries WHERE state NOT IN ('finished', 'running')",
    );
    assert_eq!(rows.len(), 1, "one row, no duplicate: {rows:?}");
    assert!(
        rows[0].starts_with("failed|TABLE_OR_VIEW_NOT_FOUND|SELECT id FROM v"),
        "{rows:?}"
    );
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
        "SELECT concat_ws('|', CAST(rows_produced AS STRING), CAST(bytes_read > 0 AS STRING), \
         CAST(rows_read >= 7 AS STRING)) \
         FROM system.queries WHERE state = 'finished' AND query_text LIKE '%WHERE id < 7%'",
    );
    assert_eq!(r, ["7|true|true"], "{r:?}");
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

#[test]
fn the_result_cache_probe_does_not_count_a_relation_cache_hit_twice() {
    let s = Session::new().unwrap();
    // A relation in the cache: a frame that asked for it, run once.
    let cached = block_on(s.sql("SELECT id FROM range(10)")).unwrap().cache();
    block_on(cached.count()).unwrap();
    let hits = |s: &Session| -> u64 {
        strings_in(
            s,
            "SELECT CAST(hits AS STRING) FROM system.caches WHERE cache = 'relation'",
        )[0]
        .parse()
        .unwrap()
    };
    // A second frame over the same plan reads it: one execution is one hit, with the result cache
    // on or off (its eligibility probe is not a lookup).
    let run_again = |s: &Session| {
        let before = hits(s);
        let df = block_on(s.sql("SELECT id FROM range(10)")).unwrap();
        assert_eq!(
            block_on(df.collect())
                .unwrap()
                .iter()
                .map(arrow_array::RecordBatch::num_rows)
                .sum::<usize>(),
            10
        );
        hits(s) - before
    };
    let without = run_again(&s);
    s.conf().set("krishiv.resultCache.enabled", "true").unwrap();
    let with = run_again(&s);
    assert_eq!(
        without, 1,
        "relation cache hits for one run without the result cache"
    );
    assert_eq!(
        with, 1,
        "relation cache hits for one run with the result cache"
    );
}

#[test]
fn query_text_honours_the_max_bytes_and_literals_settings() {
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.system.queryText.literals", "mask")
        .unwrap();
    run_all(&s, "SELECT 'canary-7f3a' AS c, 424242 AS n").unwrap();
    let texts = strings_in(
        &s,
        "SELECT query_text FROM system.queries WHERE query_text LIKE 'SELECT ? AS c%'",
    );
    assert_eq!(texts, ["SELECT ? AS c , ? AS n"]);
    let leaked = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.queries \
         WHERE query_text LIKE '%canary-7f3a%' OR query_text LIKE '%424242%'",
    );
    // The statement that reads the table names the canary itself; only that row may.
    assert!(leaked[0].parse::<u32>().unwrap() <= 1, "{leaked:?}");
    // Back to `keep`, and a short limit: the text is cut and says so.
    s.conf()
        .set("krishiv.system.queryText.literals", "keep")
        .unwrap();
    s.conf()
        .set("krishiv.system.queryText.maxBytes", "12")
        .unwrap();
    run_all(&s, "SELECT 'cutting-edge-literal' AS c").unwrap();
    let cut = strings_in(
        &s,
        "SELECT concat(query_text, '|', CAST(query_text_truncated AS STRING)) FROM system.queries \
         WHERE query_text LIKE 'SELECT %cutt'",
    );
    assert_eq!(cut, ["SELECT 'cutt|true"]);
}

#[test]
fn queries_carry_the_sqlstate_of_a_failure_and_are_never_partial_in_one_process() {
    let s = Session::new().unwrap();
    run_all(&s, "SELECT 1").unwrap();
    assert!(run_all(&s, "SELECT * FROM no_such_table_xyz").is_err());
    let failed = strings_in(
        &s,
        "SELECT concat(error_class, '|', sqlstate) FROM system.queries \
         WHERE error_class = 'TABLE_OR_VIEW_NOT_FOUND'",
    );
    assert_eq!(failed, ["TABLE_OR_VIEW_NOT_FOUND|42P01"]);
    let ok = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.queries WHERE state = 'finished' AND sqlstate IS NOT NULL",
    );
    assert_eq!(ok, ["0"]);
    let partial = strings_in(
        &s,
        "SELECT CAST(count(*) AS STRING) FROM system.queries WHERE partial",
    );
    assert_eq!(partial, ["0"]);
}

/// Column `index` of the rows `sql` returns.
fn column_of(s: &Session, sql: &str, index: usize) -> Vec<String> {
    let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for batch in &b {
        let c = batch.column(index);
        if let Some(v) = c.as_string_view_opt() {
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        } else if let Some(v) = c.as_string_opt::<i32>() {
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        } else {
            let v = c.as_string::<i64>();
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        }
    }
    out
}

#[test]
fn show_tables_in_system_lists_the_tables() {
    let s = Session::new().unwrap();
    let mut names = column_of(&s, "SHOW TABLES IN system", 1);
    names.sort();
    assert_eq!(
        names,
        [
            "caches",
            "capabilities",
            "memory",
            "metrics",
            "operators",
            "queries"
        ]
    );
    assert_eq!(
        column_of(&s, "SHOW TABLES IN SYSTEM LIKE 'q*'", 1),
        ["queries"]
    );
    // `system` is still not a database of the session catalog.
    let dbs = column_of(&s, "SHOW DATABASES", 0);
    assert!(
        !dbs.iter().any(|d| d.eq_ignore_ascii_case("system")),
        "{dbs:?}"
    );
}

fn run(s: &Session, sql: &str) -> Result<Vec<arrow_array::RecordBatch>, String> {
    block_on(async { s.sql(sql).await?.collect().await }).map_err(|e| e.to_string())
}

#[test]
fn use_system_switches_the_current_catalog() {
    let s = Session::new().unwrap();
    assert_eq!(
        column_of(&s, "SELECT current_catalog()", 0),
        ["spark_catalog"]
    );
    run(&s, "USE SYSTEM").unwrap();
    assert_eq!(column_of(&s, "SELECT current_catalog()", 0), ["system"]);
    // A bare name now reads the system catalog.
    assert!(!column_of(&s, "SELECT name FROM capabilities LIMIT 1", 0).is_empty());
    // Using the current catalog again is a no-op; a namespace of it is not found.
    run(&s, "USE system").unwrap();
    let e = run(&s, "USE x").unwrap_err();
    assert!(
        e.contains("SCHEMA_NOT_FOUND") && e.contains("`system`.`x`"),
        "{e}"
    );
    let e = run(&s, "USE spark_catalog.a.b").unwrap_err();
    assert!(e.contains("`spark_catalog`.`a`.`b`"), "{e}");
    // Back to the session catalog, in its default database.
    run(&s, "USE spark_catalog").unwrap();
    assert_eq!(
        column_of(&s, "SELECT current_catalog()", 0),
        ["spark_catalog"]
    );
    assert_eq!(column_of(&s, "SELECT current_schema()", 0), ["default"]);
}

#[test]
fn a_temp_view_still_resolves_under_system() {
    let s = Session::new().unwrap();
    run(&s, "CREATE TEMPORARY VIEW mine AS SELECT 7 AS n").unwrap();
    run(&s, "CREATE TEMPORARY VIEW queries AS SELECT 1 AS only_col").unwrap();
    run(&s, "USE system").unwrap();
    assert_eq!(
        column_of(&s, "SELECT CAST(n AS STRING) FROM mine", 0),
        ["7"]
    );
    // The temp view wins over the system table of the same name.
    assert_eq!(
        column_of(&s, "SELECT CAST(only_col AS STRING) FROM queries", 0),
        ["1"]
    );
    // A bare persistent name that is not a system table is not found.
    let e = run(&s, "SELECT * FROM nosuch").unwrap_err();
    assert!(e.contains("TABLE_OR_VIEW_NOT_FOUND"), "{e}");
}

#[test]
fn show_catalogs_lists_both_and_follows_use() {
    let s = Session::new().unwrap();
    assert_eq!(
        column_of(&s, "SHOW CATALOGS", 0),
        ["spark_catalog", "system"]
    );
    assert_eq!(column_of(&s, "SHOW CATALOGS LIKE 'sys*'", 0), ["system"]);
    run(&s, "USE system").unwrap();
    assert_eq!(column_of(&s, "SHOW CATALOGS", 0).len(), 2);
}

#[test]
fn a_finished_query_started_no_earlier_than_it_was_submitted() {
    let s = Session::new().unwrap();
    run_all(&s, "SELECT 1").unwrap();
    let r = strings_in(
        &s,
        "SELECT CAST(started_at >= submitted_at AS STRING) FROM system.queries \
         WHERE state = 'finished' AND query_text = 'SELECT 1'",
    );
    assert_eq!(r, ["true"], "{r:?}");
}

#[test]
fn a_query_answered_from_the_result_cache_is_flagged() {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.resultCache.enabled", "true").unwrap();
    run_all(&s, "SELECT id FROM range(5) WHERE id < 4").unwrap();
    run_all(&s, "SELECT id FROM range(5) WHERE id < 4").unwrap();
    let r = strings_in(
        &s,
        "SELECT CAST(result_cache_hit AS STRING) FROM system.queries \
         WHERE state = 'finished' AND query_text LIKE '%id < 4%' ORDER BY submitted_at, query_id",
    );
    assert_eq!(r, ["false", "true"], "{r:?}");
}

#[test]
fn a_file_write_records_the_rows_it_wrote() {
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("w");
    block_on(async {
        s.sql("SELECT id FROM range(321)")
            .await?
            .write()
            .mode("overwrite")
            .format("parquet")
            .save(out.to_str().unwrap())
            .await
    })
    .unwrap();
    let r = strings_in(
        &s,
        "SELECT CAST(rows_written AS STRING) FROM system.queries \
         WHERE state = 'finished' AND rows_written > 0",
    );
    assert_eq!(r, ["321"], "{r:?}");
}

#[test]
fn a_file_write_records_the_files_it_wrote() {
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("w");
    block_on(async {
        s.sql("SELECT id FROM range(50)")
            .await?
            .write()
            .mode("overwrite")
            .format("parquet")
            .save(out.to_str().unwrap())
            .await
    })
    .unwrap();
    let r = strings_in(
        &s,
        "SELECT CAST(files_written AS STRING) FROM system.queries \
         WHERE state = 'finished' AND files_written > 0",
    );
    assert_eq!(r.len(), 1, "{r:?}");
    assert!(r[0].parse::<i64>().unwrap() >= 1, "{r:?}");
}

#[test]
fn a_failed_query_records_its_message_and_masking_hides_literals() {
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.system.queryText.literals", "mask")
        .unwrap();
    let _ = block_on(async {
        s.sql("SELECT * FROM missing_canary_table")
            .await?
            .collect()
            .await
    });
    let r = strings_in(
        &s,
        "SELECT error_message FROM system.queries WHERE error_message IS NOT NULL",
    );
    assert_eq!(r.len(), 1, "{r:?}");
    assert!(r[0].contains("missing_canary_table"), "{r:?}");
    assert!(r[0].contains("TABLE_OR_VIEW_NOT_FOUND"), "{r:?}");
}

#[test]
fn an_error_message_with_a_string_literal_is_masked_when_literals_are_masked() {
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.ansi.enabled", "true").unwrap();
    s.conf()
        .set("krishiv.system.queryText.literals", "mask")
        .unwrap();
    let _ = block_on(async {
        s.sql("SELECT CAST('canary99x' AS INT)")
            .await?
            .collect()
            .await
    });
    let r = strings_in(
        &s,
        "SELECT error_message FROM system.queries WHERE error_message IS NOT NULL",
    );
    assert_eq!(r.len(), 1, "{r:?}");
    assert!(!r[0].contains("canary99x"), "{r:?}");
}

#[test]
fn a_scan_records_the_files_it_opened() {
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("r");
    block_on(async {
        s.sql("SELECT id FROM range(40)")
            .await?
            .write()
            .mode("overwrite")
            .format("parquet")
            .save(out.to_str().unwrap())
            .await
    })
    .unwrap();
    let q = format!("SELECT count(*) FROM parquet.`{}`", out.display());
    block_on(async { s.sql(&q).await?.collect().await }).unwrap();
    let r = strings_in(
        &s,
        "SELECT CAST(files_read AS STRING) FROM system.queries WHERE files_read > 0",
    );
    assert_eq!(r.len(), 1, "{r:?}");
}

#[test]
fn use_system_shows_no_default_database() {
    let s = Session::new().unwrap();
    block_on(async { s.sql("USE system").await.unwrap().collect().await.unwrap() });
    let cur = column_of(&s, "SHOW CURRENT NAMESPACE", 0);
    assert_eq!(cur, ["system"]);
    assert_eq!(column_of(&s, "SHOW CURRENT NAMESPACE", 1), [""]);
    let mut names = column_of(&s, "SHOW TABLES", 1);
    names.sort();
    assert_eq!(
        names,
        [
            "caches",
            "capabilities",
            "memory",
            "metrics",
            "operators",
            "queries"
        ]
    );
}

#[test]
fn use_namespace_under_system_finds_no_namespace() {
    let s = Session::new().unwrap();
    run(&s, "USE system").unwrap();
    for sql in ["USE NAMESPACE x", "USE NAMESPACE system"] {
        let e = run(&s, sql).unwrap_err();
        assert!(e.contains("SCHEMA_NOT_FOUND"), "{sql}: {e}");
    }
    // The failed USE leaves the current catalog alone.
    assert_eq!(column_of(&s, "SELECT current_catalog()", 0), ["system"]);
}

#[test]
fn analyze_table_in_system_is_reserved() {
    let s = Session::new().unwrap();
    run(&s, "USE system").unwrap();
    for sql in [
        "ANALYZE TABLE queries COMPUTE STATISTICS",
        "ANALYZE TABLE system.queries COMPUTE STATISTICS FOR ALL COLUMNS",
    ] {
        let e = run(&s, sql).unwrap_err();
        assert!(e.contains("KRISHIV_RESERVED_NAME"), "{sql}: {e}");
    }
}

#[test]
fn namespace_listings_under_system_fail_closed() {
    let s = Session::new().unwrap();
    run(&s, "USE system").unwrap();
    for sql in ["SHOW VIEWS", "SHOW TABLE EXTENDED LIKE '*'"] {
        let e = run(&s, sql).unwrap_err();
        assert!(e.contains("SCHEMA_NOT_FOUND"), "{sql}: {e}");
    }
}

#[test]
fn creating_under_system_is_refused_but_a_temp_view_survives_the_switch_back() {
    let s = Session::new().unwrap();
    run(&s, "USE system").unwrap();
    for sql in [
        "CREATE VIEW v1 AS SELECT 1 AS n",
        "CREATE TABLE tt1 (a INT)",
    ] {
        let e = run(&s, sql).unwrap_err();
        assert!(e.contains("KRISHIV_RESERVED_NAME"), "{sql}: {e}");
    }
    run(&s, "CREATE TEMPORARY VIEW t1 AS SELECT 2 AS n").unwrap();
    run(&s, "USE spark_catalog").unwrap();
    assert_eq!(column_of(&s, "SELECT CAST(n AS STRING) FROM t1", 0), ["2"]);
}

#[test]
fn metadata_commands_on_a_bare_name_do_not_read_the_session_catalog_under_system() {
    let s = Session::new().unwrap();
    run(&s, "CREATE TABLE t (a INT)").unwrap();
    run(&s, "USE system").unwrap();
    for sql in [
        "DESCRIBE TABLE t",
        "SHOW COLUMNS IN t",
        "SHOW CREATE TABLE t",
    ] {
        let e = run(&s, sql).unwrap_err();
        assert!(e.contains("TABLE_OR_VIEW_NOT_FOUND"), "{sql}: {e}");
    }
    // The qualified name still reaches the session catalog's table.
    assert!(run(&s, "DESCRIBE TABLE spark_catalog.default.t").is_ok());
}

#[test]
fn a_failure_message_in_the_history_is_the_client_form() {
    let s = Session::new().unwrap();
    assert!(run(&s, "SELECT * FROM nosuch_table_here").is_err());
    let m = column_of(
        &s,
        "SELECT error_message FROM system.queries WHERE error_class IS NOT NULL",
        0,
    );
    assert!(
        !m.is_empty() && m.iter().all(|x| !x.contains("Caused by")),
        "{m:?}"
    );
}

#[test]
fn a_bare_drop_of_a_function_or_view_does_not_reach_the_session_catalog_under_system() {
    let s = Session::new().unwrap();
    run(&s, "CREATE VIEW v AS SELECT 1 AS n").unwrap();
    run(&s, "CREATE FUNCTION f() RETURNS INT RETURN 1").unwrap();
    run(&s, "USE system").unwrap();
    for sql in ["DROP VIEW v", "DROP FUNCTION f"] {
        assert!(run(&s, sql).is_err(), "{sql}");
    }
    run(&s, "USE spark_catalog").unwrap();
    assert_eq!(column_of(&s, "SELECT CAST(n AS STRING) FROM v", 0), ["1"]);
    assert_eq!(column_of(&s, "SELECT CAST(f() AS STRING)", 0), ["1"]);
}

#[test]
fn a_replayed_query_masks_literals_by_the_session_policy() {
    let s = Session::new().unwrap();
    run(&s, "SET krishiv.resultCache.enabled=true").unwrap();
    run(&s, "SET krishiv.system.queryText.literals=mask").unwrap();
    run(&s, "CREATE TABLE rc (a INT)").unwrap();
    run(&s, "INSERT INTO rc VALUES (1)").unwrap();
    run(&s, "SELECT a FROM rc WHERE a = 12345").unwrap();
    run(&s, "SELECT a FROM rc WHERE a = 12345").unwrap();
    let texts = column_of(&s, "SELECT query_text FROM system.queries", 0);
    assert!(texts.iter().all(|t| !t.contains("12345")), "{texts:?}");
}

#[test]
fn describe_lists_the_columns_of_a_system_table() {
    let s = Session::new().unwrap();
    let qualified = column_of(&s, "DESCRIBE TABLE system.memory", 0);
    assert!(!qualified.is_empty());
    assert_eq!(column_of(&s, "DESC system.memory", 0), qualified);
    assert_eq!(column_of(&s, "DESCRIBE system.MEMORY", 0), qualified);
    let types = column_of(&s, "DESCRIBE system.queries", 1);
    assert!(types.iter().any(|t| t == "string"), "{types:?}");
    run(&s, "USE system").unwrap();
    assert_eq!(column_of(&s, "DESCRIBE memory", 0), qualified);
    let e = run(&s, "DESCRIBE EXTENDED system.memory").unwrap_err();
    assert!(
        e.contains("UNSUPPORTED") || e.contains("not supported"),
        "{e}"
    );
    let e = run(&s, "DESCRIBE system.nosuch").unwrap_err();
    assert!(e.contains("TABLE_OR_VIEW_NOT_FOUND"), "{e}");
}

#[test]
fn a_qualified_stored_view_is_not_read_through_the_system_catalog() {
    let s = Session::new().unwrap();
    run(&s, "CREATE VIEW v AS SELECT 'secret' AS value").unwrap();
    assert_eq!(column_of(&s, "SELECT value FROM default.v", 0), ["secret"]);
    run(&s, "USE system").unwrap();
    let e = run(&s, "SELECT value FROM default.v").unwrap_err();
    assert!(e.contains("TABLE_OR_VIEW_NOT_FOUND"), "{e}");
    // The catalog-qualified name still reaches it.
    assert_eq!(
        column_of(&s, "SELECT value FROM spark_catalog.default.v", 0),
        ["secret"]
    );
}

#[test]
fn temp_views_win_over_system_tables_in_describe_and_metadata() {
    let s = Session::new().unwrap();
    run(&s, "CREATE TEMPORARY VIEW queries AS SELECT 1 AS only_col").unwrap();
    run(&s, "CREATE TEMPORARY VIEW mine AS SELECT 2 AS other").unwrap();
    run(&s, "USE system").unwrap();
    assert_eq!(column_of(&s, "DESCRIBE queries", 0), ["only_col"]);
    assert_eq!(column_of(&s, "DESCRIBE mine", 0), ["other"]);
}

#[test]
fn a_system_only_query_has_a_start_time() {
    let s = Session::new().unwrap();
    run(&s, "SELECT * FROM system.memory").unwrap();
    let started = column_of(
        &s,
        "SELECT CAST(started_at IS NOT NULL AS STRING) FROM system.queries WHERE query_text LIKE '%system.memory%' AND query_text NOT LIKE '%started_at%'",
        0,
    );
    assert_eq!(started, ["true"]);
}

#[test]
fn use_system_dot_name_is_a_missing_schema_of_system() {
    let s = Session::new().unwrap();
    let e = run(&s, "USE system.x").unwrap_err();
    assert!(
        e.contains("SCHEMA_NOT_FOUND") && e.contains("`system`"),
        "{e}"
    );
}

#[test]
fn query_text_is_cut_at_a_character_boundary() {
    let s = Session::new().unwrap();
    run(&s, "SET krishiv.system.queryText.maxBytes=40").unwrap();
    run(&s, "SELECT 'ééééééééééééééééééééééééééééééé' AS wide").unwrap();
    let texts = column_of(&s, "SELECT query_text FROM system.queries", 0);
    assert!(texts.iter().all(|t| t.len() <= 40), "{texts:?}");
    assert!(
        texts.iter().any(|t| t.starts_with("SELECT 'é")),
        "{texts:?}"
    );
}

#[test]
fn namespace_qualified_names_do_not_reach_the_session_catalog_under_system() {
    let s = Session::new().unwrap();
    run(&s, "CREATE TABLE t (a INT)").unwrap();
    run(&s, "CREATE FUNCTION f() RETURNS INT RETURN 1").unwrap();
    run(&s, "USE system").unwrap();
    for sql in [
        "DESCRIBE TABLE default.t",
        "SHOW COLUMNS IN default.t",
        "SHOW CREATE TABLE default.t",
        "DROP FUNCTION default.f",
    ] {
        assert!(run(&s, sql).is_err(), "{sql}");
    }
    run(&s, "USE spark_catalog").unwrap();
    assert_eq!(column_of(&s, "SELECT CAST(f() AS STRING)", 0), ["1"]);
    assert!(run(&s, "DESCRIBE TABLE default.t").is_ok());
}

#[test]
fn a_masked_error_message_carries_no_value_of_the_failed_statement() {
    let s = Session::new().unwrap();
    run(&s, "SET krishiv.system.queryText.literals=mask").unwrap();
    run(&s, "SET spark.sql.ansi.enabled=true").unwrap();
    assert!(run(&s, "SELECT CAST(123456789 AS TINYINT)").is_err());
    let m = column_of(
        &s,
        "SELECT error_message FROM system.queries WHERE error_class IS NOT NULL",
        0,
    );
    assert!(!m.is_empty());
    assert!(m.iter().all(|x| !x.contains("123456789")), "{m:?}");
}

// SPDX-License-Identifier: Apache-2.0
//! `checkpoint` and `localCheckpoint` (design P3-T083 §3.3): the lineage is cut, a missing
//! directory is Spark's error, the snapshot's files go with it, `system.caches` shows it.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

/// A session whose warehouse is `dir`, with the managed table `src(id)` holding 1, 2, 3.
fn session(dir: &std::path::Path, pairs: &[(&str, &str)]) -> Session {
    let s = Session::builder()
        .config("spark.sql.warehouse.dir", dir.join("wh").to_str().unwrap())
        .build()
        .unwrap();
    for (k, v) in pairs {
        s.conf().set(k, v).unwrap();
    }
    run(&s, "CREATE TABLE src (id INT) USING parquet");
    run(&s, "INSERT INTO src VALUES (1), (2), (3)");
    s
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap();
}

fn ids(df: &krishiv_session::DataFrame) -> Vec<i32> {
    let mut out: Vec<i32> = block_on(df.collect())
        .unwrap()
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    out.sort_unstable();
    out
}

fn files(dir: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| {
            let p = e.path();
            if p.is_dir() {
                files(&p)
            } else {
                usize::from(p.extension().is_some_and(|x| x == "spill"))
            }
        })
        .sum()
}

#[test]
fn a_reliable_checkpoint_needs_a_directory_eager_or_not() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    for eager in [true, false] {
        let e = block_on(s.table("src").checkpoint(eager)).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp3016, "{eager}");
        assert!(
            e.to_string()
                .contains("Checkpoint directory has not been set in the SparkContext"),
            "{e}"
        );
    }
}

#[test]
fn a_local_checkpoint_does_not_see_later_writes_a_cache_does() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let cp = block_on(s.table("src").local_checkpoint(true, None)).unwrap();
    let cached = s.table("src").cache();
    assert_eq!(ids(&cached), vec![1, 2, 3]);
    run(&s, "INSERT INTO src VALUES (4)");
    assert_eq!(ids(&cp), vec![1, 2, 3], "the lineage is cut");
    assert_eq!(ids(&cached), vec![1, 2, 3, 4], "the cache is invalidated");
    assert!(!block_on(cp.is_cached()).unwrap());
    run(&s, "CLEAR CACHE");
    assert_eq!(ids(&cp), vec![1, 2, 3], "CLEAR CACHE does not reach it");
    // A frame derived from the checkpoint reads the same rows.
    let derived =
        cp.filter(krishiv_session::functions::col("id").gt(krishiv_session::functions::lit(1)));
    assert_eq!(ids(&derived), vec![2, 3]);
}

#[test]
fn a_lazy_checkpoint_takes_its_rows_at_the_first_action() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    run(&s, "INSERT INTO src VALUES (4)");
    let derived =
        cp.filter(krishiv_session::functions::col("id").gt(krishiv_session::functions::lit(1)));
    assert_eq!(ids(&derived), vec![2, 3, 4], "the first action's rows");
    run(&s, "INSERT INTO src VALUES (5)");
    assert_eq!(ids(&cp), vec![1, 2, 3, 4], "fixed by the first action");
}

#[test]
fn the_krishiv_key_sets_the_checkpoint_directory() {
    let dir = tempfile::tempdir().unwrap();
    let cp_dir = dir.path().join("checkpoints");
    let s = session(
        dir.path(),
        &[("krishiv.checkpoint.dir", cp_dir.to_str().unwrap())],
    );
    let cp = block_on(s.table("src").checkpoint(true)).unwrap();
    assert_eq!(files(&cp_dir), 1);
    assert_eq!(ids(&cp), vec![1, 2, 3]);
}

#[test]
fn a_reliable_checkpoint_writes_under_the_directory_and_deletes_with_the_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let cp_dir = dir.path().join("checkpoints");
    let s = session(
        dir.path(),
        &[("spark.checkpoint.dir", cp_dir.to_str().unwrap())],
    );
    let cp = block_on(s.table("src").checkpoint(true)).unwrap();
    assert_eq!(files(&cp_dir), 1);
    run(&s, "INSERT INTO src VALUES (4)");
    assert_eq!(ids(&cp), vec![1, 2, 3]);
    let derived = cp.limit(10);
    drop(cp);
    assert_eq!(files(&cp_dir), 1, "a derived frame still owns the snapshot");
    assert_eq!(ids(&derived), vec![1, 2, 3]);
    drop(derived);
    assert_eq!(files(&cp_dir), 0, "the files go with the snapshot");
}

#[test]
fn system_caches_has_a_checkpoint_row() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let row = |s: &Session| -> String {
        let b = block_on(async {
            s.sql("SELECT concat_ws('|', CAST(entries AS STRING), CAST(memory_bytes > 0 AS STRING)) FROM system.caches WHERE cache = 'checkpoint'")
                .await?
                .collect()
                .await
        })
        .unwrap();
        let c = b[0].column(0);
        c.as_string_view_opt()
            .map(|v| v.value(0).to_owned())
            .unwrap_or_else(|| c.as_string::<i32>().value(0).to_owned())
    };
    assert_eq!(row(&s), "0|false");
    let cp = block_on(s.table("src").local_checkpoint(true, None)).unwrap();
    assert_eq!(row(&s), "1|true");
    drop(cp);
    assert_eq!(row(&s), "0|false");
}

/// `f` run on a thread of its own, failing the test when it does not end in 60 seconds.
fn within_a_minute<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .expect("the call hung")
}

/// A `cache()` of a frame standing on a lazy checkpoint registers without a hang: registering
/// analyzes the frame, which takes the checkpoint's rows by running a query, which reads the
/// registry again (review B1).
#[test]
fn caching_a_lazy_checkpoint_does_not_hang() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    let cached = cp.cache();
    let (n, is_cached) = within_a_minute(move || {
        let n = block_on(cached.count()).unwrap();
        (n, block_on(cached.is_cached()).unwrap())
    });
    assert_eq!(n, 3);
    assert!(is_cached);
    // A frame derived from the checkpoint, cached, then read.
    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    let derived =
        cp.filter(krishiv_session::functions::col("id").gt(krishiv_session::functions::lit(1)));
    let derived = derived.cache();
    let rows = within_a_minute(move || ids(&derived));
    assert_eq!(rows, vec![2, 3]);
}

#[test]
fn a_frame_with_no_columns_checkpoints_with_its_rows() {
    let dir = tempfile::tempdir().unwrap();
    let cp_dir = dir.path().join("checkpoints");
    let s = session(
        dir.path(),
        &[("spark.checkpoint.dir", cp_dir.to_str().unwrap())],
    );
    for reliable in [false, true] {
        let df = block_on(s.sql("SELECT * EXCEPT (id) FROM src")).unwrap();
        let cp = if reliable {
            block_on(df.checkpoint(true))
        } else {
            block_on(df.local_checkpoint(true, None))
        }
        .unwrap();
        assert_eq!(block_on(cp.count()).unwrap(), 3);
    }
}

/// A cached lazy checkpoint answers `schema()` first, and two cached frames sharing one lazy
/// checkpoint both read: taking the rows must not wait on a registration that waits on them.
#[test]
fn registrations_sharing_a_lazy_checkpoint_do_not_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    let cached = cp.cache();
    let schema_fields = within_a_minute(move || {
        let n = block_on(cached.schema()).unwrap().fields().len();
        (n, block_on(cached.count()).unwrap())
    });
    assert_eq!(schema_fields, (1, 3));

    let cp = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    let a = cp.limit(10).cache();
    let b = cp.limit(20).cache();
    let counts =
        within_a_minute(move || (block_on(a.count()).unwrap(), block_on(b.count()).unwrap()));
    assert_eq!(counts, (3, 3));
}

#[test]
fn a_temp_view_of_a_lazy_file_checkpoint_keeps_the_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let path = dir.path().join("rows.parquet");
    let path = path.to_str().unwrap().to_owned();
    block_on(s.table("src").write().mode("overwrite").parquet(&path)).unwrap();
    let df = s.read().parquet([path.as_str()]);
    let cp = block_on(df.local_checkpoint(false, None)).unwrap();
    block_on(cp.create_or_replace_temp_view("snap")).unwrap();
    block_on(
        s.table("src")
            .filter(krishiv_session::functions::col("id").gt(krishiv_session::functions::lit(0)))
            .write()
            .mode("append")
            .parquet(&path),
    )
    .unwrap();
    assert_eq!(
        ids(&block_on(s.sql("SELECT * FROM snap")).unwrap()),
        vec![1, 2, 3]
    );
}

/// The shape an iterative graph algorithm (GraphFrames, label propagation) runs: each round joins
/// the state with the edges and checkpoints the result. The checkpoint cuts the lineage, so the
/// plan of round 50 is no bigger than the plan of round 2, and the rows stay right.
fn iterate(s: &Session, eager: bool, local: bool, rounds: usize) -> (Vec<usize>, Vec<i32>) {
    use krishiv_session::{ExplainMode, JoinType};
    run(s, "CREATE TABLE edges (id INT, nxt INT) USING parquet");
    run(s, "INSERT INTO edges VALUES (1, 2), (2, 3), (3, 1)");
    let edges = s.table("edges");
    let mut state = s.table("src");
    let mut sizes = Vec::new();
    for _ in 0..rounds {
        let joined = state
            .join_using(&edges, ["id"], JoinType::Inner)
            .select([krishiv_session::functions::col("nxt").alias("id")]);
        state = if local {
            block_on(joined.local_checkpoint(eager, None)).unwrap()
        } else {
            block_on(joined.checkpoint(eager)).unwrap()
        };
        // A lazy checkpoint takes its rows at the first action: run one every round, as a loop
        // that reads its state (a convergence test, a count) does.
        assert_eq!(block_on(state.count()).unwrap(), 3);
        sizes.push(
            block_on(state.explain(ExplainMode::Extended))
                .unwrap()
                .len(),
        );
    }
    (sizes, ids(&state))
}

/// Runs `iterate` for 50 rounds in `modes` and checks the plan stays bounded and the rows right.
fn assert_bounded(modes: &[(bool, bool)]) {
    for &(eager, local) in modes {
        let dir = tempfile::tempdir().unwrap();
        let s = session(
            dir.path(),
            &[(
                "spark.checkpoint.dir",
                dir.path().join("cp").to_str().unwrap(),
            )],
        );
        let (sizes, rows) = within_a_minute(move || iterate(&s, eager, local, 50));
        let first = sizes[1];
        let last = sizes[49];
        assert!(
            last <= first + first / 10,
            "eager={eager} local={local}: the plan grew from {first} to {last} characters: {sizes:?}"
        );
        assert_eq!(rows, vec![1, 2, 3], "eager={eager} local={local}");
    }
}

/// `checkpoint()` and `localCheckpoint()` (eager, the default GraphFrames uses) cut the lineage.
#[test]
fn fifty_rounds_of_join_and_eager_checkpoint_keep_the_plan_bounded() {
    assert_bounded(&[(true, true), (true, false)]);
}

/// Spark's lazy checkpoint truncates the plan at once (`LogicalRDD`): so does this one, its plan is
/// a leaf standing for the snapshot until an action takes the rows.
#[test]
fn fifty_rounds_of_join_and_lazy_checkpoint_keep_the_plan_bounded() {
    assert_bounded(&[(false, true), (false, false)]);
}

/// A round that joins the state with a frame derived from the state (a self-join) must not double
/// what the frame holds each time: 30 lazy rounds finish at once, and the rows stay right.
#[test]
fn thirty_lazy_self_join_rounds_do_not_double_what_the_frames_hold() {
    use krishiv_session::JoinType;
    use krishiv_session::functions::col;
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let ids_of = ids;
    let rows = within_a_minute(move || {
        let mut state = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
        for _ in 0..30 {
            let other = state.select([col("id").alias("other")]);
            let joined = state
                .join(&other, col("id").eq(col("other")), JoinType::Inner)
                .select([col("id")]);
            state = block_on(joined.local_checkpoint(false, None)).unwrap();
        }
        ids_of(&state)
    });
    assert_eq!(rows, vec![1, 2, 3]);
}

/// Once a lazy checkpoint has taken its rows it no longer holds the frame it came from.
#[test]
fn a_lazy_checkpoint_lets_go_of_its_source_once_its_rows_are_taken() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    let mut state = block_on(s.table("src").local_checkpoint(false, None)).unwrap();
    for _ in 0..5 {
        state = block_on(state.limit(10).local_checkpoint(false, None)).unwrap();
        assert_eq!(block_on(state.count()).unwrap(), 3);
    }
    let b = block_on(async {
        s.sql("SELECT CAST(entries AS STRING) FROM system.caches WHERE cache = 'checkpoint'")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let c = b[0].column(0);
    let held = c
        .as_string_view_opt()
        .map(|a| a.value(0).to_owned())
        .or_else(|| c.as_string_opt::<i32>().map(|a| a.value(0).to_owned()))
        .unwrap();
    assert_eq!(held, "1");
}

/// How many entries `system.caches` lists for `cache`.
fn cache_entries(s: &Session, cache: &str) -> i64 {
    let b = block_on(async {
        s.sql(&format!(
            "SELECT CAST(entries AS BIGINT) FROM system.caches WHERE cache = '{cache}'"
        ))
        .await?
        .collect()
        .await
    })
    .unwrap();
    b[0].column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0)
}

/// GraphFrames' Pregel loop (`checkpoint(eager = false)` every second round, `persist` the new
/// state, `unpersist` the one from the round before last): the plan stays bounded, the rows are right at the
/// end, and the cache holds only the live frames. The plan-size measurement is `explain(Extended)`, which
/// analyzes the frame and so takes each pending lazy checkpoint of an even round (its snapshot collects the
/// source frame) before the next round starts: this test does not cover lineage that stays lazy across rounds,
/// it covers the plan and cache bounds with that snapshot taken each time.
#[test]
fn a_pregel_loop_of_checkpoints_and_persists_stays_bounded_when_each_round_is_explained() {
    use krishiv_session::{ExplainMode, JoinType, StorageLevel};
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path(), &[]);
    run(&s, "CREATE TABLE edges (id INT, nxt INT) USING parquet");
    run(&s, "INSERT INTO edges VALUES (1, 2), (2, 3), (3, 1)");
    // No wall-clock bound here: the 50 rounds plan and run queries, and the test runner's own timeout
    // catches a hang.
    let (sizes, rows, live) = {
        let edges = s.table("edges");
        let mut state = s.table("src");
        let mut sizes = Vec::new();
        let mut previous: Vec<krishiv_session::DataFrame> = Vec::new();
        for round in 1..=50 {
            let joined = state
                .join_using(&edges, ["id"], JoinType::Inner)
                .select([krishiv_session::functions::col("nxt").alias("id")]);
            state = if round % 2 == 0 {
                block_on(joined.local_checkpoint(false, None)).unwrap()
            } else {
                joined
            };
            state = state.persist(StorageLevel::MEMORY_AND_DISK);
            sizes.push(
                block_on(state.explain(ExplainMode::Extended))
                    .unwrap()
                    .len(),
            );
            previous.push(state.clone());
            if previous.len() > 2 {
                let old = previous.remove(0);
                block_on(old.unpersist()).unwrap();
            }
        }
        let rows = ids(&state);
        let live = cache_entries(&s, "relation");
        (sizes, rows, live)
    };
    // The plan is flat: the longest of the last ten rounds is no longer than the longest of the
    // first four (a checkpoint round is short, the round after it is the long one).
    let early = *sizes[..4].iter().max().unwrap();
    let late = *sizes[40..].iter().max().unwrap();
    assert!(
        late <= early + early / 10,
        "the plan grew from {early} to {late} characters: {sizes:?}"
    );
    assert_eq!(rows, vec![1, 2, 3]);
    // The last two persisted frames are live (and the checkpoint the final round took, at most one more): a
    // count above three means unpersisted frames stay cached, none at all means the live ones were dropped.
    assert!(
        (1..=3).contains(&live),
        "{live} cache entries are live: expected the last persisted frames, 1 to 3"
    );
}

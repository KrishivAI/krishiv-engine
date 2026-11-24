// SPDX-License-Identifier: Apache-2.0
//! PY-054 and PY-075: a killed write's leftovers and a write that cannot start.

use std::sync::Arc;

use bytes::Bytes;
use krishiv_common::error::ErrorClass;
use object_store::local::LocalFileSystem;
use object_store::path::Path;

use super::{Env, SaveMode, TempDir};

/// PY-075: a write under a file fails as a write (`TASK_WRITE_FAILED`, not "Reading the file"), and the
/// OS error is the error's cause.
#[test]
fn a_write_that_cannot_start_names_the_os_error_not_a_read() {
    let tmp = TempDir::new("py075");
    std::fs::write(tmp.0.join("f.csv"), b"hi").unwrap();
    let env = Env::new(
        Arc::new(LocalFileSystem::new_with_prefix(&tmp.0).unwrap()),
        "file://",
    );
    let e = env
        .begin("f.csv/sub", SaveMode::Overwrite, "job-1")
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TaskWriteFailed, "{e}");
    let cause = e.cause().expect("the OS error is the cause").to_string();
    assert!(cause.contains("os error"), "{cause}");
}

/// Two uncommitted jobs under `t` (after one committed append): job-2's record names a vanished
/// process, job-3's names this process's parent, which is alive. Then `mode` writes job-4.
fn sweep_after_dead_writer(mode: SaveMode) -> Vec<String> {
    let me = crate::commit::writers::current().expect("/proc says who this process is");
    let env = Env::memory();
    env.write_logged("t", SaveMode::Append, "job-1", 4, true);
    for job in ["job-2", "job-3"] {
        env.write_logged("t", SaveMode::Append, job, 3, false);
    }
    let machine_id = me.lines().next().unwrap();
    let parent = std::os::unix::process::parent_id();
    let started = crate::commit::writers::start_time(parent).expect("the parent's start time");
    // PID 4194304 is above the Linux maximum, so it never runs.
    for (job, pid, start) in [
        ("job-2", 4_194_304, "1".to_owned()),
        ("job-3", parent, started),
    ] {
        let record = crate::commit::writers::record_path(&Path::from("t"), job);
        let body = format!("{machine_id}\n{pid}\n{start}");
        env.rt
            .block_on(env.io.put_object(&env.store, &record, Bytes::from(body)))
            .unwrap();
    }
    env.write_logged("t", mode, "job-4", 2, true);
    env.names("t")
}

/// PY-054: a write whose process died leaves a record; the next write deletes that job's unlogged
/// files at once (inside the grace), keeps a live write's, and removes the dead record.
#[test]
fn a_write_sweeps_the_unlogged_files_of_a_dead_writer_inside_the_grace() {
    if crate::commit::writers::current().is_none() {
        return;
    }
    let on_store = sweep_after_dead_writer(SaveMode::Overwrite);
    assert!(
        !on_store.iter().any(|n| n.contains("job-2")),
        "{on_store:?}"
    );
    assert!(on_store.iter().any(|n| n.contains("job-3")), "{on_store:?}");
}

/// The same in any mode: an append sweeps a dead writer's files and record, and nothing of a live
/// writer's, so only the liveness decision tells job-2 from job-3.
#[test]
fn an_append_sweeps_the_unlogged_files_of_a_dead_writer_and_only_those() {
    if crate::commit::writers::current().is_none() {
        return;
    }
    let on_store = sweep_after_dead_writer(SaveMode::Append);
    assert!(
        !on_store.iter().any(|n| n.contains("job-2")),
        "{on_store:?}"
    );
    assert!(
        on_store.iter().any(|n| n.contains("job-3.writer")),
        "{on_store:?}"
    );
    assert!(
        on_store.iter().filter(|n| n.contains("job-3")).count() > 1,
        "{on_store:?}"
    );
}

/// PY-054 (kill -9 mid-file): the local store stages an upload as `<name>#<n>` and hides it from
/// listings, so a killed write leaves a partial file that only a directory read finds. The next
/// write deletes a dead writer's staging files and keeps a live writer's.
#[test]
fn a_write_deletes_the_hidden_staging_files_of_a_dead_writer_only() {
    let Some(me) = crate::commit::writers::current() else {
        return;
    };
    let tmp = TempDir::new("py054-staging");
    let env = Env::new(Arc::new(LocalFileSystem::new()), "file");
    let dir = tmp.0.join("t");
    std::fs::create_dir_all(dir.join("p=1")).unwrap();
    let dead = dir.join("part-00000-job-2-c000.snappy.parquet#1");
    let nested = dir.join("p=1/part-00001-job-2_1-c001.snappy.parquet#2");
    let live = dir.join("part-00000-job-3-c000.snappy.parquet#1");
    let other = dir.join("part-00000-job-22-c000.snappy.parquet#1");
    for f in [&dead, &nested, &live, &other] {
        std::fs::write(f, b"partial").unwrap();
    }
    let machine_id = me.lines().next().unwrap();
    let parent = std::os::unix::process::parent_id();
    let started = crate::commit::writers::start_time(parent).expect("the parent's start time");
    let rel = tmp.0.strip_prefix("/").unwrap().join("t");
    let rel = rel.to_str().unwrap();
    for (job, pid, start) in [
        ("job-2", 4_194_304, "1".to_owned()),
        ("job-3", parent, started),
    ] {
        let record = crate::commit::writers::record_path(&Path::from(rel), job);
        let body = format!("{machine_id}\n{pid}\n{start}");
        env.rt
            .block_on(env.io.put_object(&env.store, &record, Bytes::from(body)))
            .unwrap();
    }
    env.write_logged(rel, SaveMode::Overwrite, "job-4", 2, true);
    assert!(
        !dead.exists() && !nested.exists(),
        "dead writer's files stay"
    );
    assert!(live.exists(), "a live writer's staging file was deleted");
    assert!(other.exists(), "another job's staging file was deleted");
}

const CHILD_DIR: &str = "KRISHIV_TEST_KILLED_WRITER_DIR";

/// The writer process of [`a_write_killed_with_sigkill_leaves_nothing_a_directory_read_finds`]: it
/// starts a logged overwrite (which records who it is), leaves a staging file as an upload in
/// flight does, says so, and waits to be killed. Without the variable it does nothing.
#[test]
fn killed_writer_child() {
    let Ok(dir) = std::env::var(CHILD_DIR) else {
        return;
    };
    let env = Env::new(Arc::new(LocalFileSystem::new()), "file");
    let rel = dir.trim_start_matches('/');
    let w = env
        .begin(rel, SaveMode::Overwrite, "job-killed")
        .unwrap()
        .unwrap();
    let _w = env.rt.block_on(w.logged(&env.io)).unwrap();
    let dir = std::path::PathBuf::from(&dir);
    std::fs::write(
        dir.join("part-00000-job-killed-c000.snappy.parquet#1"),
        b"partial",
    )
    .unwrap();
    std::fs::write(dir.join("ready.flag"), b"1").unwrap();
    std::thread::sleep(std::time::Duration::from_secs(600));
}

/// PY-054 with a real `kill -9`: the writer is a separate process, killed while its file is in
/// flight; the next overwrite leaves only its own files in the directory.
#[test]
fn a_write_killed_with_sigkill_leaves_nothing_a_directory_read_finds() {
    if crate::commit::writers::current().is_none() || std::env::var(CHILD_DIR).is_ok() {
        return;
    }
    let tmp = TempDir::new("py054-sigkill");
    let dir = tmp.0.join("t");
    std::fs::create_dir_all(&dir).unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "--nocapture",
            "commit::tests::dead_writers::killed_writer_child",
        ])
        .env(CHILD_DIR, &dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let flag = dir.join("ready.flag");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !flag.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "the child never got going"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    child.kill().unwrap(); // SIGKILL
    child.wait().unwrap();
    std::fs::remove_file(&flag).unwrap();

    let env = Env::new(Arc::new(LocalFileSystem::new()), "file");
    let rel = dir.to_str().unwrap().trim_start_matches('/');
    env.write_logged(rel, SaveMode::Overwrite, "job-next", 2, true);
    let mut left: Vec<String> = walk(&dir);
    left.sort();
    assert!(
        !left
            .iter()
            .any(|n| n.contains("job-killed") || n.contains('#')),
        "{left:?}"
    );
}

fn walk(dir: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        if e.file_type().unwrap().is_dir() {
            out.extend(walk(&e.path()));
        } else {
            out.push(e.file_name().to_string_lossy().into_owned());
        }
    }
    out
}

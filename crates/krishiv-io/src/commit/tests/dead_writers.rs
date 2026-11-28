// SPDX-License-Identifier: Apache-2.0
//! PY-054 and PY-075: a killed write's leftovers and a write that cannot start.

use std::sync::Arc;

use bytes::Bytes;
use krishiv_common::error::ErrorClass;
use object_store::local::LocalFileSystem;
use object_store::path::Path;

use super::{Env, SaveMode, TempDir};
use crate::SettledLocalFileSystem;
use crate::io::{IoConfig, IoScheduler, StoreId};

/// The local store as a session registers it: with the sweeper of its hidden staging files.
fn local_env() -> Env {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let sched = IoScheduler::new(IoConfig::default());
    let store = Arc::new(SettledLocalFileSystem::new());
    sched.register_store(StoreId::new("file"), Arc::clone(&store) as _);
    sched.register_staging_sweeper(StoreId::new("file"), store);
    Env {
        rt,
        io: sched.context(),
        store: StoreId::new("file"),
    }
}

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

/// PY-054 (kill -9 mid-file): the local store stages an upload as `.<name>#<n>` (`<name>#<n>` before R1-V-2) and hides it from
/// listings, so a killed write leaves a partial file that only a directory read finds. The next
/// write deletes a dead writer's staging files and keeps a live writer's.
#[test]
fn a_write_deletes_the_hidden_staging_files_of_a_dead_writer_only() {
    let Some(me) = crate::commit::writers::current() else {
        return;
    };
    let tmp = TempDir::new("py054-staging");
    let env = local_env();
    let dir = tmp.0.join("t");
    std::fs::create_dir_all(dir.join("p=1")).unwrap();
    let dead = dir.join("part-00000-job-2-c000.snappy.parquet#1");
    let nested = dir.join("p=1/part-00001-job-2_1-c001.snappy.parquet#2");
    let live = dir.join("part-00000-job-3-c000.snappy.parquet#1");
    let other = dir.join("part-00000-job-22-c000.snappy.parquet#1");
    // The name the store stages an upload under now: hidden from Spark by the leading dot.
    let hidden = dir.join(".part-00002-job-2-c002.snappy.parquet#1");
    for f in [&dead, &nested, &live, &other, &hidden] {
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
        !dead.exists() && !nested.exists() && !hidden.exists(),
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
    let env = local_env();
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
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            panic!("the child never got going");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    child.kill().unwrap(); // SIGKILL
    child.wait().unwrap();
    std::fs::remove_file(&flag).unwrap();

    let env = local_env();
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

/// PY-054 in a container: the writer died with its PID namespace, so its record cannot be judged.
/// An overwrite deletes the staging files older than the grace whatever job they name, and keeps
/// a fresh one (an upload in flight).
#[test]
fn an_overwrite_sweeps_old_staging_files_of_a_writer_nobody_can_judge() {
    let tmp = TempDir::new("py054-grace");
    let env = local_env();
    let dir = tmp.0.join("t");
    std::fs::create_dir_all(dir.join("p=1")).unwrap();
    let old = dir.join("part-00000-job-9-c000.snappy.parquet#1");
    let nested = dir.join("p=1/part-00001-job-9_1-c001.snappy.parquet#3");
    let fresh = dir.join("part-00000-job-8-c000.snappy.parquet#1");
    for f in [&old, &nested, &fresh] {
        std::fs::write(f, b"partial").unwrap();
    }
    let long_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(3 * 3600);
    for f in [&old, &nested] {
        std::fs::File::options()
            .write(true)
            .open(f)
            .unwrap()
            .set_modified(long_ago)
            .unwrap();
    }
    let rel = dir.to_str().unwrap().trim_start_matches('/');
    env.write_logged(rel, SaveMode::Overwrite, "job-next", 2, true);
    assert!(!old.exists() && !nested.exists(), "old staging files stay");
    assert!(fresh.exists(), "a fresh staging file was deleted");
}

/// An append never sweeps staging files by age: only a dead writer's go.
#[test]
fn an_append_leaves_old_staging_files_of_unknown_writers() {
    let tmp = TempDir::new("py054-append");
    let env = local_env();
    let dir = tmp.0.join("t");
    std::fs::create_dir_all(&dir).unwrap();
    let old = dir.join("part-00000-job-9-c000.snappy.parquet#1");
    std::fs::write(&old, b"partial").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&old)
        .unwrap()
        .set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(3 * 3600))
        .unwrap();
    let rel = dir.to_str().unwrap().trim_start_matches('/');
    env.write_logged(rel, SaveMode::Append, "job-next", 2, true);
    assert!(old.exists());
}

/// The store that staged the files sweeps them: a store with no sweeper (here an in-memory one)
/// is left alone, and the sweep reports clean.
#[test]
fn a_store_without_a_sweeper_has_nothing_to_sweep() {
    let env = Env::memory();
    assert!(
        env.rt.block_on(
            env.io
                .sweep_staging(&env.store, &Path::from("t"), |_, _| true)
        )
    );
}

/// Staging files of an anchored write are swept descriptor-relative, links are not followed.
#[cfg(unix)]
#[test]
fn the_anchored_store_sweeps_staging_files_and_follows_no_link() {
    use crate::io::StagingSweep;
    let tmp = TempDir::new("py054-anchored");
    let dir = tmp.0.join("t");
    let elsewhere = tmp.0.join("elsewhere");
    std::fs::create_dir_all(dir.join("p=1")).unwrap();
    std::fs::create_dir_all(&elsewhere).unwrap();
    let victim = elsewhere.join("part-00000-job-2-c000.parquet#1");
    std::fs::write(&victim, b"x").unwrap();
    std::os::unix::fs::symlink(&elsewhere, dir.join("link")).unwrap();
    let mine = dir.join("p=1/part-00000-job-2-c000.parquet#1");
    let plain = dir.join("p=1/part-00000-job-2-c000.parquet");
    std::fs::write(&mine, b"x").unwrap();
    std::fs::write(&plain, b"x").unwrap();
    let store = crate::BeneathStore::anchor(&dir).unwrap();
    let rel = Path::from(dir.to_str().unwrap().trim_start_matches('/'));
    store
        .sweep_staging(&rel, &|name, _| name.contains("job-2"))
        .unwrap();
    assert!(!mine.exists(), "the staging file stays");
    assert!(plain.exists(), "a finished file was deleted");
    assert!(victim.exists(), "a file behind a link was deleted");
}

/// A sweep that could not remove a file keeps the dead writer's record, so the next write looks
/// again (and drops it once the sweep is clean).
#[cfg(unix)]
#[test]
fn a_sweep_that_failed_keeps_the_dead_writers_record() {
    use std::os::unix::fs::PermissionsExt;
    let Some(me) = crate::commit::writers::current() else {
        return;
    };
    let tmp = TempDir::new("py054-keep");
    let env = local_env();
    let dir = tmp.0.join("t");
    let locked = dir.join("p=1");
    std::fs::create_dir_all(&locked).unwrap();
    std::fs::write(locked.join("part-00000-job-2-c000.parquet#1"), b"x").unwrap();
    // Readable, not writable: the file cannot be removed.
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();
    if std::fs::File::create(locked.join("probe")).is_ok() {
        // Root writes anywhere: the failure cannot be made.
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        return;
    }
    let rel = dir.to_str().unwrap().trim_start_matches('/');
    let record = crate::commit::writers::record_path(&Path::from(rel), "job-2");
    let body = format!("{}\n4194304\n1", me.lines().next().unwrap());
    env.rt
        .block_on(env.io.put_object(&env.store, &record, Bytes::from(body)))
        .unwrap();
    env.write_logged(rel, SaveMode::Append, "job-4", 2, true);
    let kept = dir.join("_krishiv_log/job-2.writer").exists();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(kept, "the record of a sweep that failed was dropped");
}

/// The anchored sweep of a directory that is gone is clean; one that cannot unlink a staging file
/// reports the failure (after trying the rest).
#[cfg(unix)]
#[test]
fn the_anchored_sweep_reports_a_file_it_cannot_remove_and_skips_a_gone_directory() {
    use crate::io::StagingSweep;
    use std::os::unix::fs::PermissionsExt;
    let tmp = TempDir::new("py054-anchored-fail");
    let dir = tmp.0.join("t");
    let locked = dir.join("p=1");
    std::fs::create_dir_all(&locked).unwrap();
    let staged = locked.join("part-00000-job-2-c000.parquet#1");
    std::fs::write(&staged, b"x").unwrap();
    let store = crate::BeneathStore::anchor(&dir).unwrap();
    let rel = Path::from(dir.to_str().unwrap().trim_start_matches('/'));
    let gone = Path::from(format!("{}/missing", rel.as_ref()));
    store.sweep_staging(&gone, &|_, _| true).unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();
    let rooted = std::fs::File::create(locked.join("probe")).is_ok();
    let result = store.sweep_staging(&rel, &|_, _| true);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    if !rooted {
        assert!(result.is_err(), "a file that stayed was reported clean");
        assert!(staged.exists());
    }
}

/// A confined store refuses to sweep a directory outside its roots, and reports a directory it
/// cannot read.
#[cfg(unix)]
#[test]
fn the_confined_sweep_refuses_outside_roots_and_reports_an_unreadable_directory() {
    use crate::io::StagingSweep;
    use std::os::unix::fs::PermissionsExt;
    let tmp = TempDir::new("py054-confined");
    let root = std::fs::canonicalize(&tmp.0).unwrap();
    let inside = root.join("in");
    let outside = TempDir::new("py054-confined-out");
    std::fs::create_dir_all(&inside).unwrap();
    std::fs::write(outside.0.join("part-00000-job-2-c000.parquet#1"), b"x").unwrap();
    let store = SettledLocalFileSystem::confined(vec![root]);
    let as_path = |p: &std::path::Path| Path::from(p.to_str().unwrap().trim_start_matches('/'));
    let refused = store
        .sweep_staging(
            &as_path(&std::fs::canonicalize(&outside.0).unwrap()),
            &|_, _| true,
        )
        .unwrap_err();
    assert_eq!(refused.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(outside.0.join("part-00000-job-2-c000.parquet#1").exists());

    std::fs::set_permissions(&inside, std::fs::Permissions::from_mode(0o000)).unwrap();
    let rooted = std::fs::read_dir(&inside).is_ok();
    let result = store.sweep_staging(&as_path(&inside), &|_, _| true);
    std::fs::set_permissions(&inside, std::fs::Permissions::from_mode(0o755)).unwrap();
    if !rooted {
        assert!(
            result.is_err(),
            "an unreadable directory was reported clean"
        );
    }
}

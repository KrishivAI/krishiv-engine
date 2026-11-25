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

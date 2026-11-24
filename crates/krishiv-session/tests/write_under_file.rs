// SPDX-License-Identifier: Apache-2.0
//! PY-075: a write that cannot start (the path is under a file, or on a full device) fails as a
//! write, `TASK_WRITE_FAILED`, with the OS reason as its cause, never as "Reading the file".
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn fail(target: &str, mode: &str) -> krishiv_common::error::KrishivError {
    let s = Session::new().unwrap();
    block_on(async {
        s.sql("SELECT 1 AS a")
            .await?
            .write()
            .format("parquet")
            .mode(mode)
            .save(target)
            .await
    })
    .expect_err("the write cannot succeed")
}

#[test]
fn a_write_under_a_file_is_a_write_failure() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("f.csv"), b"x").unwrap();
    let target = tmp.path().join("f.csv/sub");
    for mode in ["overwrite", "append", "error", "ignore"] {
        let e = fail(target.to_str().unwrap(), mode);
        assert_eq!(e.error_class(), ErrorClass::TaskWriteFailed, "{mode}: {e}");
        assert!(!e.to_string().contains("Reading the file"), "{mode}: {e}");
    }
}

#[test]
#[cfg(target_os = "linux")]
fn a_write_to_dev_full_is_a_write_failure() {
    let e = fail("/dev/full/x", "overwrite");
    assert_eq!(e.error_class(), ErrorClass::TaskWriteFailed, "{e}");
}

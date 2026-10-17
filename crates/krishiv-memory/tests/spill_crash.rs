// SPDX-License-Identifier: Apache-2.0
//! A process killed in the middle of a spill leaves its files behind; the
//! next start of the spill directory removes them (and only them).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, SpillConfig, SpillManager, TreePool};
use krishiv_types::batch::Batch;

const DIR_VAR: &str = "KRISHIV_SPILL_CRASH_DIR";

fn files_below(dir: &Path, suffix: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| {
            let p = e.path();
            if p.is_dir() {
                files_below(&p, suffix)
            } else {
                usize::from(p.to_string_lossy().ends_with(suffix))
            }
        })
        .sum()
}

/// Kills the child if an assertion fails before the test does.
struct KillOnDrop(std::process::Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// The doomed process: spills forever. Does nothing unless the parent test
/// started it with `KRISHIV_SPILL_CRASH_DIR` set.
#[test]
fn spilling_child() {
    let Some(dir) = std::env::var_os(DIR_VAR) else {
        return;
    };
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let manager = SpillManager::open(&SpillConfig {
            dir: PathBuf::from(dir),
            ..SpillConfig::default()
        })?;
        let scope = manager.scope()?;
        let pool = TreePool::new(&MemoryConfig::default()).child("q", None);
        let id = pool.register("spill", None);
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let record = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..10_000))],
        )?;
        let batch = Batch::new(record);
        let mut writer = scope.writer(schema, Reservation::new(Arc::clone(&pool), id))?;
        loop {
            writer.write(&batch)?;
        }
    })();
    // Only reached if spilling failed; the parent then fails its own checks.
    drop(result);
}

#[test]
fn files_of_a_killed_process_are_removed_at_the_next_start() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("spill-crash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "spilling_child", "--test-threads=1"])
        .env(DIR_VAR, &root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(KillOnDrop)
        .unwrap();

    // Wait until it is mid-spill: a temp file has bytes in it.
    let deadline = Instant::now() + Duration::from_secs(60);
    while files_below(&root, ".spill.tmp") == 0 {
        assert!(Instant::now() < deadline, "child never started spilling");
        assert!(child.0.try_wait().unwrap().is_none(), "child exited early");
        std::thread::sleep(Duration::from_millis(5));
    }
    child.0.kill().unwrap(); // SIGKILL: no destructors run
    child.0.wait().unwrap();
    assert_eq!(
        files_below(&root, ".spill.tmp"),
        1,
        "the dead process left its file"
    );

    let manager = SpillManager::open(&SpillConfig {
        dir: root.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    assert_eq!(manager.stale_sessions_removed(), 1);
    assert_eq!(files_below(&root, ".spill.tmp"), 0);
    // Only this process's session directory and lock remain.
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 2);
    drop(manager);
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    let _ = std::fs::remove_dir_all(&root);
}

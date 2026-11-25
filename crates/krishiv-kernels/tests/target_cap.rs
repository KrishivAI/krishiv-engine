// SPDX-License-Identifier: Apache-2.0
//! `KRISHIV_KERNEL_TARGET` caps the process's kernel target (read once by `Target::best`, so each case
//! runs this test binary again as a child process with the variable set).

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use std::process::Command;

use krishiv_kernels::{KernelCtx, TARGET_CAP_ENV, Target};

/// Set in the child only: the target `best` must then pick.
const EXPECT_VAR: &str = "KRISHIV_TEST_EXPECT_KERNEL_TARGET";

/// The child's side: with the cap set by the parent, `best` and every new context use the expected target.
#[test]
fn capped_child() {
    let Ok(expected) = std::env::var(EXPECT_VAR) else {
        return; // Not started by `cap_lowers_the_process_target`.
    };
    let expected = Target::from_name(&expected).unwrap();
    assert_eq!(Target::best(), expected);
    assert_eq!(KernelCtx::new().target(), expected);
    // The cap leaves the detection, and forcing a supported target for a test, alone.
    assert!(KernelCtx::with_target(Target::detected()).is_ok());
}

fn run_child(cap: &str, expected: Target) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "capped_child", "--test-threads=1"])
        .env(TARGET_CAP_ENV, cap)
        .env(EXPECT_VAR, expected.name())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cap {cap:?}, expected {expected:?}:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cap_lowers_the_process_target() {
    let detected = Target::detected();
    for cap in Target::ALL {
        // A cap above the CPU's target never raises it.
        run_child(cap.name(), detected.capped(cap));
    }
    run_child("BASELINE", Target::Baseline);
    // No cap, and a value naming no target, leave the detected target.
    run_child("", detected);
    run_child("avx512", detected);
}

// SPDX-License-Identifier: Apache-2.0
//! Idle epochs: once the process has stayed idle, each thread that checks
//! calls the release hook once for that epoch. The hook and the epochs are
//! process-wide, so this binary holds one test.

use std::sync::atomic::{AtomicUsize, Ordering};

use krishiv_memory::{idle_epoch, note_idle, release_if_due_here, set_release_hook};

static CALLS: AtomicUsize = AtomicUsize::new(0);

fn hook() {
    CALLS.fetch_add(1, Ordering::SeqCst);
}

#[test]
fn each_thread_releases_once_per_idle_epoch() {
    // Without a hook an idle epoch still moves, but nothing is called.
    note_idle();
    assert!(!release_if_due_here(), "no hook installed");
    assert!(set_release_hook(hook));

    let before = idle_epoch();
    note_idle();
    assert_eq!(idle_epoch(), before + 1);
    assert!(release_if_due_here(), "a new idle epoch");
    assert!(!release_if_due_here(), "once per epoch");
    assert_eq!(CALLS.load(Ordering::SeqCst), 1);

    // Another thread keeps its own record: it releases for the same epoch.
    let other = std::thread::spawn(|| (release_if_due_here(), release_if_due_here()))
        .join()
        .unwrap();
    assert_eq!(other, (true, false));
    assert_eq!(CALLS.load(Ordering::SeqCst), 2);

    // Two epochs before a check: one call.
    note_idle();
    note_idle();
    assert!(release_if_due_here());
    assert!(!release_if_due_here());
    assert_eq!(CALLS.load(Ordering::SeqCst), 3);
}

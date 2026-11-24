// SPDX-License-Identifier: Apache-2.0
//! Which writers of a directory are still alive (PY-054).
//!
//! A logged write records who it is in `_krishiv_log/<job>.writer` before any task writes a
//! file, and removes the record at its commit. A write that was killed leaves the record behind; a
//! later write that finds a record of a dead process on this machine deletes that job's unlogged
//! files at once, instead of waiting for the one-hour grace that guards a concurrent write's files.
//! A record that cannot be judged (another machine, another PID namespace, no `/proc`) is never
//! dead: the grace still applies to it.

use object_store::path::Path;

use crate::commit_log::LOG_DIR;

/// The suffix of a record's name; it never parses as a log version (`<n>.json`).
const SUFFIX: &str = ".writer";

/// The record of the write `job` under `root`.
pub(super) fn record_path(root: &Path, job: &str) -> Path {
    log_dir(root).join(format!("{job}{SUFFIX}"))
}

/// The directory of the records under `root`: the commit log's own (a record is deleted at the commit,
/// and a directory of their own would be left behind empty).
pub(super) fn log_dir(root: &Path) -> Path {
    root.clone().join(LOG_DIR)
}

/// The job a record is named for.
pub(super) fn job_of(record: &Path) -> Option<&str> {
    record.filename()?.strip_suffix(SUFFIX)
}

/// This process as a record body: boot id, PID namespace, PID and start time, so a recycled PID
/// is not mistaken for the writer. `None` where `/proc` does not say.
pub(super) fn current() -> Option<String> {
    let pid = std::process::id();
    Some(format!("{}\n{pid}\n{}", machine()?, start_time(pid)?))
}

/// Whether the process a record names is gone from this machine. `false` when the record is
/// another machine's or namespace's, or cannot be read: only a certain death counts.
pub(super) fn is_dead(record: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(record) else {
        return false;
    };
    let mut lines = text.lines();
    let (Some(machine_id), Some(pid), Some(started)) = (lines.next(), lines.next(), lines.next())
    else {
        return false;
    };
    if machine().as_deref() != Some(machine_id) {
        return false;
    }
    let Ok(pid) = pid.parse::<u32>() else {
        return false;
    };
    // Not our own record: a write of this process is alive until it commits or fails.
    if pid == std::process::id() {
        return false;
    }
    let exists = rustix::process::Pid::from_raw(i32::try_from(pid).unwrap_or(0))
        .map(rustix::process::test_kill_process);
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok();
    judged_dead(exists, stat.as_deref(), started)
}

/// The decision of [`is_dead`]. Liveness comes from the `kill(pid, 0)` result, never from what
/// `/proc` shows (`hidepid=2` hides another user's live process): `ESRCH` is death, any other
/// answer (`EPERM` included) is a live process. Only a live process whose start time can be read
/// and differs from the record's is a recycled PID. An unreadable stat counts as alive.
fn judged_dead(exists: Option<rustix::io::Result<()>>, stat: Option<&str>, started: &str) -> bool {
    match exists {
        None => false,
        Some(Err(e)) if e == rustix::io::Errno::SRCH => true,
        Some(_) => stat.and_then(start_of).is_some_and(|s| s != started),
    }
}

/// The boot id and PID namespace: two processes with the same pair share a PID space.
fn machine() -> Option<String> {
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
    let ns = std::fs::read_link("/proc/self/ns/pid").ok()?;
    Some(format!("{}:{}", boot.trim(), ns.display()))
}

pub(super) fn start_time(pid: u32) -> Option<String> {
    start_of(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?).map(str::to_owned)
}

/// Field 22 of `/proc/<pid>/stat` (clock ticks after boot at which the process started). The
/// command name in field 2 may hold spaces and parentheses: fields count from the last `)`.
fn start_of(stat: &str) -> Option<&str> {
    stat.rsplit_once(')')?.1.split_whitespace().nth(19)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_start_time_is_field_22_even_when_the_command_holds_parentheses() {
        let stat = "1234 (a (b) c) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 99 23 24";
        assert_eq!(start_of(stat), Some("99"));
    }

    #[test]
    fn a_record_of_a_running_process_is_alive_and_one_of_another_machine_is_not_judged() {
        let Some(me) = current() else { return };
        assert!(!is_dead(me.as_bytes()));
        let other = me.replacen(':', "x:", 1);
        assert!(!is_dead(other.as_bytes()));
        assert!(!is_dead(b"garbage"));
    }

    #[test]
    fn a_record_of_a_vanished_process_is_dead() {
        let Some(me) = current() else { return };
        let machine_id = me.lines().next().unwrap();
        // PID 4194304 is above the Linux maximum, so it never runs.
        let record = format!("{machine_id}\n4194304\n1");
        assert!(is_dead(record.as_bytes()));
        // The PID of a live process with another start time: a recycled PID.
        let parent = std::os::unix::process::parent_id();
        let recycled = format!("{machine_id}\n{parent}\n0");
        assert!(is_dead(recycled.as_bytes()));
    }

    #[test]
    fn a_process_that_cannot_be_signalled_or_read_is_alive() {
        let eperm = Some(Err(rustix::io::Errno::PERM));
        // `hidepid=2`: kill says the process exists, /proc shows nothing.
        assert!(!judged_dead(eperm, None, "5"));
        assert!(!judged_dead(Some(Ok(())), None, "5"));
        assert!(judged_dead(Some(Err(rustix::io::Errno::SRCH)), None, "5"));
        assert!(!judged_dead(None, None, "5"));
        let stat = "1 (a) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 99 23";
        assert!(judged_dead(Some(Ok(())), Some(stat), "5"));
        assert!(!judged_dead(Some(Ok(())), Some(stat), "99"));
    }
}

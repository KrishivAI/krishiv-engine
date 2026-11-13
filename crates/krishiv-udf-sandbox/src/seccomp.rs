// SPDX-License-Identifier: Apache-2.0
//! The worker's seccomp filters (note §4.2 item 5).
//!
//! A default-allow **denylist**: CPython, numpy and pandas need too many system calls for a reviewed
//! allowlist, and Landlock covers the file side regardless. Two filters, because one seccompiler filter has a
//! single match action:
//! - `EPERM` for the denied calls: every `socket()` (or every non-TCP one with a port allowlist), process
//!   creation other than threads, tracing and other processes' memory, `io_uring`, `bpf`, `perf_event_open`,
//!   mounts and namespaces, keyrings, `userfaultfd`, kernel modules and `kexec`, `open_by_handle_at`, and
//!   signals to any other process;
//! - `ENOSYS` for `clone3`, whose flags live in memory a filter cannot read, so glibc falls back to `clone`,
//!   whose flags the first filter checks.
//!
//! When several filters match, the kernel applies the most restrictive action, so a call one filter allows
//! and the other refuses is refused.

use std::collections::BTreeMap;

use seccompiler::{
    BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter,
    SeccompRule, TargetArch,
};

use crate::{Network, SandboxError};

/// What the filters allow beyond the base denylist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FilterSpec<'a> {
    /// The worker's pid: the only process it may signal.
    pub(crate) own_pid: u32,
    pub(crate) network: &'a Network,
    pub(crate) allow_fork: bool,
}

/// The two BPF programs, in the order they are installed.
pub(crate) fn programs(spec: &FilterSpec<'_>) -> Result<Vec<BpfProgram>, SandboxError> {
    let arch = target_arch()?;
    let deny = filter(
        deny_rules(spec)?,
        SeccompAction::Errno(errno(libc::EPERM)),
        arch,
    )?;
    let mut no_clone3 = BTreeMap::new();
    no_clone3.insert(libc::SYS_clone3, Vec::new());
    let nosys = filter(no_clone3, SeccompAction::Errno(errno(libc::ENOSYS)), arch)?;
    Ok(vec![deny, nosys])
}

fn errno(e: i32) -> u32 {
    e.unsigned_abs()
}

fn target_arch() -> Result<TargetArch, SandboxError> {
    TargetArch::try_from(std::env::consts::ARCH).map_err(|e| {
        SandboxError::unavailable(format!(
            "seccomp filters are not built for {}: {e}",
            std::env::consts::ARCH
        ))
    })
}

fn filter(
    rules: BTreeMap<i64, Vec<SeccompRule>>,
    on_match: SeccompAction,
    arch: TargetArch,
) -> Result<BpfProgram, SandboxError> {
    SeccompFilter::new(rules, SeccompAction::Allow, on_match, arch)
        .and_then(BpfProgram::try_from)
        .map_err(|e| SandboxError::unavailable(format!("building the seccomp filter failed: {e}")))
}

fn condition(
    arg: u8,
    len: SeccompCmpArgLen,
    op: SeccompCmpOp,
    value: u64,
) -> Result<SeccompCondition, SandboxError> {
    SeccompCondition::new(arg, len, op, value)
        .map_err(|e| SandboxError::unavailable(format!("building a seccomp condition failed: {e}")))
}

fn rule(conditions: Vec<SeccompCondition>) -> Result<SeccompRule, SandboxError> {
    SeccompRule::new(conditions)
        .map_err(|e| SandboxError::unavailable(format!("building a seccomp rule failed: {e}")))
}

/// A rule matching when the first argument (a pid) is not the worker's own.
fn other_pid(own_pid: u32) -> Result<Vec<SeccompRule>, SandboxError> {
    Ok(vec![rule(vec![condition(
        0,
        SeccompCmpArgLen::Dword,
        SeccompCmpOp::Ne,
        u64::from(own_pid),
    )?])?])
}

/// System calls denied unconditionally (an empty rule list matches every call).
const ALWAYS_DENIED: &[i64] = &[
    libc::SYS_ptrace,
    libc::SYS_process_vm_readv,
    libc::SYS_process_vm_writev,
    libc::SYS_io_uring_setup,
    libc::SYS_io_uring_enter,
    libc::SYS_io_uring_register,
    libc::SYS_bpf,
    libc::SYS_perf_event_open,
    libc::SYS_mount,
    libc::SYS_umount2,
    libc::SYS_pivot_root,
    libc::SYS_chroot,
    libc::SYS_unshare,
    libc::SYS_setns,
    libc::SYS_keyctl,
    libc::SYS_add_key,
    libc::SYS_request_key,
    libc::SYS_userfaultfd,
    libc::SYS_kexec_load,
    libc::SYS_kexec_file_load,
    libc::SYS_init_module,
    libc::SYS_finit_module,
    libc::SYS_delete_module,
    libc::SYS_open_by_handle_at,
    libc::SYS_tkill,
    libc::SYS_pidfd_send_signal,
];

fn deny_rules(spec: &FilterSpec<'_>) -> Result<BTreeMap<i64, Vec<SeccompRule>>, SandboxError> {
    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
    for &call in ALWAYS_DENIED {
        rules.insert(call, Vec::new());
    }
    // Signals: only to the worker itself (Landlock ABI 6 scopes them as well).
    rules.insert(libc::SYS_kill, other_pid(spec.own_pid)?);
    rules.insert(libc::SYS_tgkill, other_pid(spec.own_pid)?);
    rules.insert(libc::SYS_rt_sigqueueinfo, other_pid(spec.own_pid)?);
    rules.insert(libc::SYS_rt_tgsigqueueinfo, other_pid(spec.own_pid)?);

    // Process creation: threads only, unless an administrator allowed fork.
    if !spec.allow_fork {
        let clone_flags = af(libc::CLONE_THREAD);
        rules.insert(
            libc::SYS_clone,
            vec![rule(vec![condition(
                0,
                SeccompCmpArgLen::Qword,
                SeccompCmpOp::MaskedEq(clone_flags),
                0,
            )?])?],
        );
        #[cfg(target_arch = "x86_64")]
        {
            rules.insert(libc::SYS_fork, Vec::new());
            rules.insert(libc::SYS_vfork, Vec::new());
        }
    }

    // Sockets: none at all, or TCP streams over IPv4/IPv6 when ports are open (Landlock limits the ports).
    match spec.network {
        Network::Denied => {
            rules.insert(libc::SYS_socket, Vec::new());
        }
        Network::TcpConnect(_) => {
            let dword = SeccompCmpArgLen::Dword;
            let not_inet = rule(vec![
                condition(0, dword.clone(), SeccompCmpOp::Ne, af(libc::AF_INET))?,
                condition(0, dword.clone(), SeccompCmpOp::Ne, af(libc::AF_INET6))?,
            ])?;
            // `type` carries SOCK_NONBLOCK/SOCK_CLOEXEC in its high bits; the low four are the socket type.
            let mut socket_rules = vec![not_inet];
            for kind in [
                libc::SOCK_DGRAM,
                libc::SOCK_RAW,
                libc::SOCK_RDM,
                libc::SOCK_SEQPACKET,
                libc::SOCK_DCCP,
                // SOCK_PACKET (10, obsolete): libc deprecates the constant, the kernel still accepts the type.
                10,
            ] {
                socket_rules.push(rule(vec![condition(
                    1,
                    dword.clone(),
                    SeccompCmpOp::MaskedEq(0xf),
                    af(kind),
                )?])?);
            }
            rules.insert(libc::SYS_socket, socket_rules);
        }
    }
    Ok(rules)
}

fn af(value: i32) -> u64 {
    u64::from(value.unsigned_abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_compile_for_every_policy() -> Result<(), SandboxError> {
        let tcp = Network::TcpConnect(vec![443]);
        for (network, allow_fork) in [
            (&Network::Denied, false),
            (&Network::Denied, true),
            (&tcp, false),
        ] {
            let progs = programs(&FilterSpec {
                own_pid: 1234,
                network,
                allow_fork,
            })?;
            assert_eq!(progs.len(), 2);
            assert!(progs.iter().all(|p| !p.is_empty()));
        }
        Ok(())
    }

    #[test]
    fn fork_rule_follows_the_opt_out() -> Result<(), SandboxError> {
        let strict = deny_rules(&FilterSpec {
            own_pid: 1,
            network: &Network::Denied,
            allow_fork: false,
        })?;
        let open = deny_rules(&FilterSpec {
            own_pid: 1,
            network: &Network::Denied,
            allow_fork: true,
        })?;
        assert!(strict.contains_key(&libc::SYS_clone));
        assert!(!open.contains_key(&libc::SYS_clone));
        // The socket rule and the base denylist stay with fork allowed.
        assert!(open.contains_key(&libc::SYS_socket));
        assert!(open.contains_key(&libc::SYS_ptrace));
        Ok(())
    }
}

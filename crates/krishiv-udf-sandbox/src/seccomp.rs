// SPDX-License-Identifier: Apache-2.0
//! The worker's seccomp filters (note §4.2 item 5).
//!
//! A default-allow **denylist**: CPython, numpy and pandas need too many system calls for a reviewed
//! allowlist, and Landlock covers the file side regardless. Two filters, because one seccompiler filter has a
//! single match action:
//! - (x86_64 only, installed first) `ENOSYS` for every x32 system call: the kernel reports them with the
//!   x86_64 architecture and bit 30 set in the number, so a denylist keyed on plain numbers would match none
//!   of them on a kernel with x32 enabled;
//! - `EPERM` for the denied calls: every `socket()` (or, with a port allowlist, everything but plain TCP
//!   streams over IPv4/IPv6: MPTCP and SCTP sockets bypass Landlock's TCP rules), process creation other than
//!   threads, tracing and other processes' memory, `io_uring`, `bpf`, `perf_event_open`, mounts and
//!   namespaces, keyrings, `userfaultfd`, kernel modules and `kexec`, `open_by_handle_at`, path `truncate`
//!   (Landlock restricts it only from ABI 3) and read-only opens with `O_TRUNC`, file metadata changes (mode,
//!   owner, extended attributes, times: Landlock has no right for them), clearing the parent-death signal,
//!   datagram socket pairs, terminal input injection (`TIOCSTI`, `TIOCLINUX`), signals to
//!   any other process, and changing any other process's limits, priority, scheduling, CPU affinity, I/O
//!   priority or memory placement (the kernel allows all of these between processes of one uid, and
//!   Landlock has no hook for them);
//! - `ENOSYS` for `clone3` and `openat2`, whose flags live in memory a filter cannot read, so callers fall back
//!   to `clone` and `openat`, whose flags the `EPERM` filter checks.
//!
//! When several filters match, the kernel applies the most restrictive action, so a call one filter allows
//! and another refuses is refused.

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

/// The BPF programs, in the order they are installed.
pub(crate) fn programs(spec: &FilterSpec<'_>) -> Result<Vec<BpfProgram>, SandboxError> {
    let arch = target_arch()?;
    let mut programs = Vec::with_capacity(3);
    #[cfg(target_arch = "x86_64")]
    programs.push(no_x32());
    let deny = filter(
        deny_rules(spec)?,
        SeccompAction::Errno(errno(libc::EPERM)),
        arch,
    )?;
    let mut no_clone3 = BTreeMap::new();
    no_clone3.insert(libc::SYS_clone3, Vec::new());
    no_clone3.insert(libc::SYS_openat2, Vec::new());
    let nosys = filter(no_clone3, SeccompAction::Errno(errno(libc::ENOSYS)), arch)?;
    programs.push(deny);
    programs.push(nosys);
    Ok(programs)
}

/// Bit 30 of an x86_64 system call number marks the x32 ABI (`__X32_SYSCALL_BIT`).
#[cfg(target_arch = "x86_64")]
const X32_SYSCALL_BIT: u32 = 0x4000_0000;

/// A hand-built filter (seccompiler has no x32 check): load the system call number; if it has the x32 bit,
/// return `ENOSYS`, otherwise allow. Other architectures (i386 through `int 0x80`) are killed by the
/// seccompiler filters' own architecture check, so this one needs none.
#[cfg(target_arch = "x86_64")]
fn no_x32() -> BpfProgram {
    use seccompiler::sock_filter;
    // Classic BPF opcodes (linux/bpf_common.h) and seccomp return values (linux/seccomp.h): kernel ABI.
    const LD_W_ABS: u16 = 0x20; // BPF_LD (0x00) | BPF_W (0x00) | BPF_ABS (0x20)
    const JMP_JGE_K: u16 = 0x35; // BPF_JMP (0x05) | BPF_JGE (0x30) | BPF_K (0x00)
    const RET_K: u16 = 0x06; // BPF_RET | BPF_K
    const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
    const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
    // `struct seccomp_data` starts with `int nr`.
    const NR_OFFSET: u32 = 0;
    let insn = |code, jt, jf, k| sock_filter { code, jt, jf, k };
    vec![
        insn(LD_W_ABS, 0, 0, NR_OFFSET),
        insn(JMP_JGE_K, 0, 1, X32_SYSCALL_BIT),
        insn(RET_K, 0, 0, SECCOMP_RET_ERRNO | errno(libc::ENOSYS)),
        insn(RET_K, 0, 0, SECCOMP_RET_ALLOW),
    ]
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

/// A rule matching when argument `arg` (a pid) is neither 0 (the caller) nor the worker's own pid. A call
/// naming one of the worker's other threads by tid is refused too; numpy and OpenBLAS pass 0.
fn neither_self_nor_zero(arg: u8, own_pid: u32) -> Result<SeccompRule, SandboxError> {
    rule(vec![
        condition(arg, SeccompCmpArgLen::Dword, SeccompCmpOp::Ne, 0)?,
        condition(
            arg,
            SeccompCmpArgLen::Dword,
            SeccompCmpOp::Ne,
            u64::from(own_pid),
        )?,
    ])
}

/// `setpriority(which, who, ..)` and `ioprio_set(which, who, ..)`: allowed only for one process
/// (`which == process_kind`) that is the worker itself; a process group or a user (every process of the uid,
/// the server included) is refused.
fn self_only_who(process_kind: u64, own_pid: u32) -> Result<Vec<SeccompRule>, SandboxError> {
    Ok(vec![
        rule(vec![condition(
            0,
            SeccompCmpArgLen::Dword,
            SeccompCmpOp::Ne,
            process_kind,
        )?])?,
        neither_self_nor_zero(1, own_pid)?,
    ])
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
    // Path truncation: Landlock restricts it only from ABI 3 (Linux 6.2). `ftruncate` of a file the worker
    // opened for writing (only in its private directory) still works.
    libc::SYS_truncate,
    // File metadata changes: Landlock has no right for mode, owner, extended attributes (ACLs) or times, so a
    // worker of the server's uid could `chmod 0` a server directory it cannot even open, or change it through
    // a descriptor it opened read-only (review threads on #717). Refused everywhere, the private directory
    // included: `shutil.copymode`/`copystat` and `os.utime` fail there with EPERM.
    libc::SYS_fchmod,
    libc::SYS_fchmodat,
    FCHMODAT2,
    libc::SYS_fchown,
    libc::SYS_fchownat,
    libc::SYS_setxattr,
    libc::SYS_lsetxattr,
    libc::SYS_fsetxattr,
    SETXATTRAT,
    libc::SYS_removexattr,
    libc::SYS_lremovexattr,
    libc::SYS_fremovexattr,
    REMOVEXATTRAT,
    libc::SYS_utimensat,
    #[cfg(target_arch = "x86_64")]
    libc::SYS_chmod,
    #[cfg(target_arch = "x86_64")]
    libc::SYS_chown,
    #[cfg(target_arch = "x86_64")]
    libc::SYS_lchown,
    #[cfg(target_arch = "x86_64")]
    libc::SYS_utime,
    #[cfg(target_arch = "x86_64")]
    libc::SYS_utimes,
    #[cfg(target_arch = "x86_64")]
    libc::SYS_futimesat,
];

/// `fchmodat2` (Linux 6.6), `setxattrat` and `removexattrat` (Linux 6.13): one number on every architecture
/// (the unified table), which `libc` does not define for all targets yet.
const FCHMODAT2: i64 = 452;
const SETXATTRAT: i64 = 463;
const REMOVEXATTRAT: i64 = 466;
/// `PR_SET_PDEATHSIG`.
const PR_SET_PDEATHSIG: u64 = 1;
/// `O_ACCMODE` and `O_TRUNC` (the same on x86_64 and aarch64).
const O_ACCMODE: u64 = 0o3;
const O_TRUNC: u64 = 0o1000;

/// `PRIO_PROCESS` (`setpriority`) and `IOPRIO_WHO_PROCESS` (`ioprio_set`).
const PRIO_PROCESS: u64 = 0;
const IOPRIO_WHO_PROCESS: u64 = 1;
/// `TIOCSTI` and `TIOCLINUX` (asm-generic ioctls, the same on x86_64 and aarch64): pushing input into, or
/// controlling, a terminal the worker inherited.
const TERMINAL_INJECTION: [u64; 2] = [0x5412, 0x541C];

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

    // Other processes' limits, priority, scheduling and memory: the kernel allows these between processes of
    // one uid (a worker could set the server's RLIMIT_CPU to one second), so only the worker itself (pid 0 or
    // its own pid) may be named.
    for call in [
        libc::SYS_prlimit64,
        libc::SYS_sched_setaffinity,
        libc::SYS_sched_setscheduler,
        libc::SYS_sched_setparam,
        libc::SYS_sched_setattr,
        libc::SYS_migrate_pages,
    ] {
        rules.insert(call, vec![neither_self_nor_zero(0, spec.own_pid)?]);
    }
    rules.insert(
        libc::SYS_setpriority,
        self_only_who(PRIO_PROCESS, spec.own_pid)?,
    );
    rules.insert(
        libc::SYS_ioprio_set,
        self_only_who(IOPRIO_WHO_PROCESS, spec.own_pid)?,
    );
    rules.insert(
        libc::SYS_ioctl,
        TERMINAL_INJECTION
            .iter()
            .map(|&request| {
                rule(vec![condition(
                    1,
                    SeccompCmpArgLen::Dword,
                    SeccompCmpOp::Eq,
                    request,
                )?])
            })
            .collect::<Result<_, _>>()?,
    );

    // The parent-death signal armed by the trampoline must stay: clearing it (`prctl(PR_SET_PDEATHSIG, 0)`
    // through ctypes) would let the worker outlive the server.
    rules.insert(
        libc::SYS_prctl,
        vec![rule(vec![condition(
            0,
            SeccompCmpArgLen::Dword,
            SeccompCmpOp::Eq,
            PR_SET_PDEATHSIG,
        )?])?],
    );
    // `open(O_RDONLY | O_TRUNC)` truncates a file the worker may only read (Landlock checks the truncate right
    // only from ABI 3, and a read-only open needs no write right). Any other truncating open needs the write
    // right, which Landlock grants only in the private directory. `openat2` keeps its flags in memory a filter
    // cannot read: `ENOSYS` (second filter), so callers fall back to `openat`.
    rules.insert(libc::SYS_openat, vec![read_only_truncate(2)?]);
    #[cfg(target_arch = "x86_64")]
    rules.insert(libc::SYS_open, vec![read_only_truncate(1)?]);

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

    // Socket pairs: only connection-oriented ones. A datagram pair (SOCK_RAW is a datagram pair for AF_UNIX)
    // can `sendto` or reconnect to any pathname socket the uid may write, outside every other rule (Landlock
    // scopes pathname sockets only from ABI 9).
    rules.insert(
        libc::SYS_socketpair,
        [libc::SOCK_DGRAM, libc::SOCK_RAW, libc::SOCK_RDM, 10]
            .into_iter()
            .map(|kind| {
                rule(vec![condition(
                    1,
                    SeccompCmpArgLen::Dword,
                    SeccompCmpOp::MaskedEq(0xf),
                    af(kind),
                )?])
            })
            .collect::<Result<_, _>>()?,
    );

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
            // Landlock's TCP rules check only plain TCP sockets, so the protocol must be 0 (the default, TCP
            // for a stream) or IPPROTO_TCP: MPTCP (262), SCTP (132) and others would connect anywhere.
            let not_tcp = rule(vec![
                condition(2, dword.clone(), SeccompCmpOp::Ne, 0)?,
                condition(2, dword.clone(), SeccompCmpOp::Ne, af(libc::IPPROTO_TCP))?,
            ])?;
            // `type` carries SOCK_NONBLOCK/SOCK_CLOEXEC in its high bits; the low four are the socket type.
            let mut socket_rules = vec![not_inet, not_tcp];
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

/// A rule matching a read-only open that truncates: flags (argument `arg`) with no write access and `O_TRUNC`.
fn read_only_truncate(arg: u8) -> Result<SeccompRule, SandboxError> {
    rule(vec![condition(
        arg,
        SeccompCmpArgLen::Dword,
        SeccompCmpOp::MaskedEq(O_ACCMODE | O_TRUNC),
        O_TRUNC,
    )?])
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
            let expected = if cfg!(target_arch = "x86_64") { 3 } else { 2 };
            assert_eq!(progs.len(), expected);
            assert!(progs.iter().all(|p| !p.is_empty()));
        }
        Ok(())
    }

    /// The x32 filter is installed first and refuses exactly the numbers with bit 30 set.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x32_calls_are_refused_first() -> Result<(), SandboxError> {
        let progs = programs(&FilterSpec {
            own_pid: 1,
            network: &Network::Denied,
            allow_fork: false,
        })?;
        assert_eq!(progs.first(), Some(&no_x32()));
        let x32 = no_x32();
        assert_eq!(x32.len(), 4);
        assert_eq!(x32[1].k, X32_SYSCALL_BIT);
        assert_eq!(x32[2].k & 0xffff, 38, "ENOSYS");
        Ok(())
    }

    #[test]
    fn other_processes_cannot_be_reconfigured() -> Result<(), SandboxError> {
        let rules = deny_rules(&FilterSpec {
            own_pid: 7,
            network: &Network::Denied,
            allow_fork: true,
        })?;
        for call in [
            libc::SYS_prlimit64,
            libc::SYS_setpriority,
            libc::SYS_sched_setaffinity,
            libc::SYS_ioprio_set,
            libc::SYS_truncate,
            libc::SYS_ioctl,
        ] {
            assert!(rules.contains_key(&call), "{call}");
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

// SPDX-License-Identifier: Apache-2.0
//! What a sandboxed worker may do, and its encoding as trampoline arguments.
//!
//! The server builds a [`SandboxSpec`] and starts its own executable as the trampoline with
//! [`SandboxSpec::to_args`]; the trampoline decodes them with [`SandboxSpec::from_args`]. Both sides are the
//! same binary, so the encoding needs no version: it is never stored and never crosses a release.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use crate::SandboxError;

/// The confinement of one UDF worker process (`design/P3-T014-connect-udfs-sandbox.md` §4.2).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SandboxSpec {
    /// The Python interpreter to `exec`, an absolute path. It is the only program the worker may execute
    /// (Landlock's execute right), apart from the dynamic loader that starts it.
    pub interpreter: PathBuf,
    /// Arguments after the interpreter (for example `-I`, `-m`, the worker module).
    pub args: Vec<OsString>,
    /// The whole environment of the worker; nothing of the server's environment is inherited.
    pub env: Vec<(OsString, OsString)>,
    /// Paths the worker may read (files, or directories with everything beneath them). Missing paths are
    /// skipped: an interpreter's `sys.path` routinely names entries that do not exist.
    pub read_paths: Vec<PathBuf>,
    /// The worker's private directory, the only path it may write. It must exist.
    pub private_dir: PathBuf,
    /// Resource limits.
    pub limits: Limits,
    /// Network access; [`Network::Denied`] unless an administrator opened ports.
    pub network: Network,
    /// Allow `fork` and `vfork` (`krishiv.python.worker.sandbox.allowFork`, an admin opt-out). Landlock and
    /// the socket rule still apply, so a forked process still cannot execute other programs or connect out.
    pub allow_fork: bool,
    /// The server's pid. The trampoline exits if its parent is no longer this process when it starts (the
    /// server died before the parent-death signal was armed).
    pub parent_pid: Option<u32>,
}

/// Resource limits applied with `setrlimit` (`None` leaves the limit as inherited).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Limits {
    /// `RLIMIT_AS` in bytes: an allocation past it fails in the worker, never in the server.
    pub address_space: Option<u64>,
    /// `RLIMIT_FSIZE` in bytes: the largest file the worker may write in its private directory.
    pub file_size: Option<u64>,
    /// `RLIMIT_NOFILE`.
    pub open_files: u64,
    /// `RLIMIT_CPU` in seconds (`krishiv.python.worker.cpuSeconds`; Spark has no such limit).
    pub cpu_seconds: Option<u64>,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            address_space: None,
            file_size: Some(DEFAULT_FILE_SIZE),
            open_files: DEFAULT_OPEN_FILES,
            cpu_seconds: None,
        }
    }
}

/// Default `RLIMIT_FSIZE`: 1 GiB (note §4.2 item 6).
pub const DEFAULT_FILE_SIZE: u64 = 1 << 30;
/// Default `RLIMIT_NOFILE` (note §4.2 item 6).
pub const DEFAULT_OPEN_FILES: u64 = 256;

/// What network access a worker has.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Network {
    /// No socket can be created (`socket()` fails with `EPERM` for every family); only the sockets the server
    /// hands over work.
    #[default]
    Denied,
    /// TCP stream sockets may be created and may connect to these ports only (Landlock ABI 4 or later); binding
    /// stays denied, and so do UDP and every other socket type.
    TcpConnect(Vec<u16>),
}

impl SandboxSpec {
    /// A spec that runs `interpreter` with `private_dir` as its only writable path, the default limits and no
    /// network. Add read paths, arguments and environment before use.
    pub fn new(interpreter: impl Into<PathBuf>, private_dir: impl Into<PathBuf>) -> Self {
        SandboxSpec {
            interpreter: interpreter.into(),
            private_dir: private_dir.into(),
            ..SandboxSpec::default()
        }
    }

    /// The system paths every Python worker reads: shared libraries, the loader cache, time zones, locale
    /// data, CA certificates, and the two device files Python opens (note §4.2 item 4). Missing ones are
    /// skipped when the sandbox is entered.
    pub fn system_read_paths() -> Vec<PathBuf> {
        [
            "/usr/lib",
            "/usr/lib64",
            "/lib",
            "/lib64",
            "/usr/share/zoneinfo",
            "/usr/share/locale",
            "/etc/ld.so.cache",
            "/etc/localtime",
            "/etc/ssl",
            "/dev/null",
            "/dev/urandom",
        ]
        .into_iter()
        .map(PathBuf::from)
        .collect()
    }

    /// Encodes the spec as trampoline arguments, decoded by [`SandboxSpec::from_args`].
    pub fn to_args(&self) -> Vec<OsString> {
        let mut out: Vec<OsString> = Vec::new();
        let mut flag = |name: &str, value: &OsStr| {
            out.push(OsString::from(name));
            out.push(value.to_owned());
        };
        flag("--interpreter", self.interpreter.as_os_str());
        flag("--dir", self.private_dir.as_os_str());
        for a in &self.args {
            flag("--arg", a);
        }
        for p in &self.read_paths {
            flag("--read", p.as_os_str());
        }
        let limits = &self.limits;
        let number = |n: u64| OsString::from(n.to_string());
        if let Some(n) = limits.address_space {
            flag("--as", &number(n));
        }
        if let Some(n) = limits.file_size {
            flag("--fsize", &number(n));
        }
        flag("--nofile", &number(limits.open_files));
        if let Some(n) = limits.cpu_seconds {
            flag("--cpu", &number(n));
        }
        if let Network::TcpConnect(ports) = &self.network {
            flag("--tcp", &OsString::from(join_ports(ports)));
        }
        if let Some(pid) = self.parent_pid {
            flag("--parent", &number(u64::from(pid)));
        }
        if self.allow_fork {
            out.push(OsString::from("--allow-fork"));
        }
        // Environment pairs last: each is a flag followed by two values.
        for (k, v) in &self.env {
            out.push(OsString::from("--env"));
            out.push(k.clone());
            out.push(v.clone());
        }
        out
    }

    /// Decodes trampoline arguments made by [`SandboxSpec::to_args`].
    ///
    /// # Errors
    /// [`SandboxError::BadSpec`] for an unknown flag, a missing value, a bad number, or a missing interpreter
    /// or private directory.
    pub fn from_args(args: impl IntoIterator<Item = OsString>) -> Result<Self, SandboxError> {
        let mut spec = SandboxSpec {
            limits: Limits {
                file_size: None,
                ..Limits::default()
            },
            ..SandboxSpec::default()
        };
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let flag = flag
                .into_string()
                .map_err(|f| SandboxError::BadSpec(format!("a flag is not UTF-8: {f:?}")))?;
            if flag == "--allow-fork" {
                spec.allow_fork = true;
                continue;
            }
            let mut value = || {
                args.next()
                    .ok_or_else(|| SandboxError::BadSpec(format!("{flag} has no value")))
            };
            match flag.as_str() {
                "--interpreter" => spec.interpreter = value()?.into(),
                "--dir" => spec.private_dir = value()?.into(),
                "--arg" => spec.args.push(value()?),
                "--read" => spec.read_paths.push(value()?.into()),
                "--as" => spec.limits.address_space = Some(parse_number(&flag, &value()?)?),
                "--fsize" => spec.limits.file_size = Some(parse_number(&flag, &value()?)?),
                "--nofile" => spec.limits.open_files = parse_number(&flag, &value()?)?,
                "--cpu" => spec.limits.cpu_seconds = Some(parse_number(&flag, &value()?)?),
                "--tcp" => spec.network = Network::TcpConnect(parse_ports(&value()?)?),
                "--parent" => {
                    let pid = parse_number(&flag, &value()?)?;
                    spec.parent_pid = Some(u32::try_from(pid).map_err(|_| {
                        SandboxError::BadSpec(format!("--parent {pid} is not a pid"))
                    })?);
                }
                "--env" => {
                    let key = value()?;
                    let val = value()?;
                    spec.env.push((key, val));
                }
                other => return Err(SandboxError::BadSpec(format!("unknown flag {other}"))),
            }
        }
        if spec.interpreter.as_os_str().is_empty() {
            return Err(SandboxError::BadSpec("no --interpreter".to_owned()));
        }
        if spec.private_dir.as_os_str().is_empty() {
            return Err(SandboxError::BadSpec("no --dir".to_owned()));
        }
        Ok(spec)
    }
}

fn join_ports(ports: &[u16]) -> String {
    ports
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_number(flag: &str, value: &OsStr) -> Result<u64, SandboxError> {
    value
        .to_str()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| SandboxError::BadSpec(format!("{flag} {value:?} is not a number")))
}

fn parse_ports(value: &OsStr) -> Result<Vec<u16>, SandboxError> {
    let text = value
        .to_str()
        .ok_or_else(|| SandboxError::BadSpec(format!("--tcp {value:?} is not UTF-8")))?;
    if text.is_empty() {
        return Ok(Vec::new());
    }
    text.split(',')
        .map(|p| {
            p.parse()
                .map_err(|_| SandboxError::BadSpec(format!("--tcp port {p:?} is not a port")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_common::error::ErrorClass;

    fn full_spec() -> SandboxSpec {
        let mut spec = SandboxSpec::new("/usr/bin/python3", "/tmp/w1");
        spec.args = vec!["-I".into(), "-c".into(), "print(1)".into()];
        spec.env = vec![("A".into(), "b=c".into()), ("EMPTY".into(), "".into())];
        spec.read_paths = vec!["/usr/lib".into(), "/opt/py lib".into()];
        spec.limits = Limits {
            address_space: Some(1 << 30),
            file_size: Some(4096),
            open_files: 64,
            cpu_seconds: Some(5),
        };
        spec.network = Network::TcpConnect(vec![443, 8080]);
        spec.allow_fork = true;
        spec.parent_pid = Some(4242);
        spec
    }

    #[test]
    fn spec_round_trips_through_args() -> Result<(), SandboxError> {
        for spec in [
            full_spec(),
            SandboxSpec::new("/p", "/d"),
            SandboxSpec {
                network: Network::TcpConnect(Vec::new()),
                ..SandboxSpec::new("/p", "/d")
            },
        ] {
            assert_eq!(SandboxSpec::from_args(spec.to_args())?, spec);
        }
        Ok(())
    }

    #[test]
    fn bad_args_are_internal_errors() {
        let cases: [&[&str]; 6] = [
            &["--interpreter"],
            &["--interpreter", "/p", "--dir", "/d", "--nofile", "many"],
            &["--interpreter", "/p", "--dir", "/d", "--tcp", "80,http"],
            &[
                "--interpreter",
                "/p",
                "--dir",
                "/d",
                "--parent",
                "99999999999",
            ],
            &["--interpreter", "/p", "--dir", "/d", "--bogus", "x"],
            &["--dir", "/d"],
        ];
        for case in cases {
            let args = case.iter().map(OsString::from);
            match SandboxSpec::from_args(args) {
                Err(e) => assert_eq!(e.error_class(), ErrorClass::InternalError, "{case:?}: {e}"),
                Ok(spec) => panic!("{case:?} decoded to {spec:?}"),
            }
        }
        let no_dir = SandboxSpec::from_args(["--interpreter", "/p"].map(OsString::from));
        assert!(matches!(no_dir, Err(SandboxError::BadSpec(m)) if m.contains("--dir")));
    }

    #[test]
    fn defaults_follow_the_note() {
        let spec = SandboxSpec::new("/p", "/d");
        assert_eq!(spec.limits.file_size, Some(1 << 30));
        assert_eq!(spec.limits.open_files, 256);
        assert_eq!(spec.network, Network::Denied);
        assert!(!spec.allow_fork);
        assert!(SandboxSpec::system_read_paths().contains(&PathBuf::from("/dev/null")));
    }
}

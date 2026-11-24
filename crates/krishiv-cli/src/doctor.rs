// SPDX-License-Identifier: Apache-2.0
//! `krishiv doctor` (P3-T070): checks a `krishiv server` configuration without starting the
//! server, and says what to change.
//!
//! It parses the same `--conf` settings with the same code `krishiv server` runs at startup
//! ([`crate::server::load_settings`], [`crate::server::check_host`]), so a configuration the
//! doctor passes is one the server accepts, and the other way round. On top of that it looks at
//! what the server would find at run time: the secret and certificate files, the ports, the
//! spill, cache and checkpoint directories, and the local path and object-store allow lists.
//! Nothing is written outside a probe file in a directory that exists, removed at once, and
//! nothing touches the network: a check that needs a remote answer is `skipped`, with the
//! reason.
//!
//! Exit status: 0 when no check failed (warnings allowed), 1 when one did, 2 for a usage error.

use std::net::{IpAddr, SocketAddr, TcpListener};
use std::path::Path;
use std::sync::Arc;

use krishiv_common::config::{
    CHECKPOINT_DIR, CONNECT_PORT, Config, ConfigValue, IO_CACHE_PATH, IO_LOCAL_ALLOWED_ROOTS,
    IO_OBJECT_STORE_ALLOWED_PREFIXES, SPILL_DIR,
};
use krishiv_connect_server::host;
use krishiv_session::auth::{ServerAuth, ServerTls};
use krishiv_session::{AllowAll, PathPrefixPolicy};

use crate::server::{ServerArgs, Settings, check_host, load_settings, resolve_host};

/// The command's arguments: the ones `krishiv server` takes, and the output form.
#[derive(Debug, clap::Args)]
pub(crate) struct DoctorArgs {
    #[command(flatten)]
    server: ServerArgs,
    /// Print the report as JSON: `{"ok": bool, "checks": [{"name", "status", "detail", "fix"}]}`.
    #[arg(long)]
    json: bool,
}

/// How one check came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Status {
    /// Fine.
    Ok,
    /// Works, but probably not what was meant.
    Warn,
    /// `krishiv server` would refuse to start, or fail at run time.
    Fail,
    /// Not checked: the reason is the detail.
    Skipped,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Warn => "warn",
            Self::Fail => "FAIL",
            Self::Skipped => "skipped",
        }
    }
}

/// One line of the report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Check {
    pub(crate) name: &'static str,
    pub(crate) status: Status,
    pub(crate) detail: String,
    /// What to change, for a failure or a warning.
    pub(crate) fix: Option<String>,
}

impl Check {
    fn new(name: &'static str, status: Status, detail: impl Into<String>) -> Self {
        Self {
            name,
            status,
            detail: detail.into(),
            fix: None,
        }
    }
    fn ok(name: &'static str, detail: impl Into<String>) -> Self {
        Self::new(name, Status::Ok, detail)
    }
    fn skipped(name: &'static str, detail: impl Into<String>) -> Self {
        Self::new(name, Status::Skipped, detail)
    }
    fn fail(name: &'static str, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            fix: Some(fix.into()),
            ..Self::new(name, Status::Fail, detail)
        }
    }
    fn warn(name: &'static str, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            fix: Some(fix.into()),
            ..Self::new(name, Status::Warn, detail)
        }
    }
}

/// Runs the command; the process exit code.
#[expect(clippy::print_stdout, reason = "CLI binary: the report is its output")]
pub(crate) fn run(args: &DoctorArgs) -> i32 {
    let checks = diagnose(&args.server);
    let failed = checks.iter().any(|c| c.status == Status::Fail);
    if args.json {
        println!("{}", render_json(&checks));
    } else {
        print!("{}", render_text(&checks));
    }
    i32::from(failed)
}

/// Every check, in the order the server's own startup meets them.
pub(crate) fn diagnose(args: &ServerArgs) -> Vec<Check> {
    let mut out = Vec::new();
    let settings = match load_settings(args.conf()) {
        Ok(s) => {
            out.push(Check::ok(
                "config",
                format!("{} settings parsed", args.conf().len()),
            ));
            s
        }
        Err(e) => {
            out.push(Check::fail(
                "config",
                e,
                "fix the setting named above (`krishiv sql -e 'SET -v'` lists the keys)",
            ));
            // Nothing below can say anything about a configuration that does not parse.
            for name in DEPENDENT {
                out.push(Check::skipped(name, "the configuration does not parse"));
            }
            return out;
        }
    };
    let host = match resolve_host(args.host(), &settings.config) {
        Ok(h) => h,
        Err(e) => {
            out.push(Check::fail(
                "listener",
                e,
                "set --host or krishiv.connect.address to an IP address",
            ));
            // Same rows as every other failure path: what needs the listener address is skipped.
            for name in DEPENDENT.iter().skip(1) {
                out.push(Check::skipped(
                    name,
                    "the listener address does not resolve",
                ));
            }
            return out;
        }
    };
    out.push(listener(host, &settings));
    out.push(authentication(host, &settings));
    out.push(tls(&settings));
    out.push(ports(host, args, &settings.config));
    out.push(user(args));
    out.push(directories(&settings.config));
    out.push(paths(&settings));
    out.push(Check::skipped(
        "object store",
        "credentials and reachability are not probed (no network from the doctor); the allowed \
         prefixes are checked under `paths`",
    ));
    out.push(Check::skipped(
        "catalogs",
        "declared catalogs (krishiv.catalog.<name>.*) are checked when the server starts, before either listener binds",
    ));
    out
}

/// The checks that need a parsed configuration.
const DEPENDENT: [&str; 9] = [
    "listener",
    "authentication",
    "tls",
    "ports",
    "user",
    "directories",
    "paths",
    "object store",
    "catalogs",
];

fn listener(host: IpAddr, settings: &Settings) -> Check {
    match check_host(host, settings) {
        Ok(()) => Check::ok("listener", format!("{host}: authentication and TLS fit")),
        Err(e) => Check::fail(
            "listener",
            e,
            "bind a loopback address, or configure authentication and TLS as the message says",
        ),
    }
}

fn authentication(host: IpAddr, settings: &Settings) -> Check {
    if !settings.auth.is_configured() {
        return if host.is_loopback() {
            Check::warn(
                "authentication",
                "off: any local process can run SQL as the server",
                "set krishiv.auth.tokens.<id>.{secret,principal,role} or krishiv.auth.oidc.*",
            )
        } else {
            // `listener` already failed this; say it here too, where the reader looks.
            Check::fail(
                "authentication",
                "off on a non-loopback address",
                "set krishiv.auth.tokens.<id>.{secret,principal,role} or krishiv.auth.oidc.*",
            )
        };
    }
    // `ServerAuth::new` starts the OIDC key fetcher, which calls the issuer at once: the doctor
    // does not use the network, so it builds the authenticator without the OIDC part.
    let mut offline = settings.auth.clone();
    let oidc = offline.oidc.is_on() || offline.impersonation.signed_assertion;
    offline.oidc = Default::default();
    offline.impersonation.signed_assertion = false;
    match ServerAuth::new(&offline) {
        Ok(_) if oidc => Check::ok(
            "authentication",
            "configured, secrets resolve; the OIDC and assertion issuers and their keys are not contacted",
        ),
        Ok(_) => Check::ok("authentication", "configured, secrets resolve"),
        Err(e) => Check::fail(
            "authentication",
            e.to_string(),
            "make each secret an `env:NAME` or a `file:` readable by this user and by no one else \
             (mode 0600)",
        ),
    }
}

fn tls(settings: &Settings) -> Check {
    if !settings.tls.is_on() {
        return Check::ok("tls", "off (plaintext listeners)");
    }
    match ServerTls::new(&settings.tls, &settings.auth.mtls) {
        Ok(_) => Check::ok("tls", "certificate and key load; the key is private"),
        Err(e) => Check::fail(
            "tls",
            e.to_string(),
            "give krishiv.server.tls.certFile a PEM chain and keyFile its key, mode 0600 or 0400",
        ),
    }
}

/// Can the listeners bind? A port of 0 picks a free one and is not checked.
fn ports(host: IpAddr, args: &ServerArgs, config: &Config) -> Check {
    let connect = args.connect_port().or_else(|| {
        config
            .get(&CONNECT_PORT)
            .and_then(ConfigValue::as_int)
            .and_then(|p| u16::try_from(p).ok())
    });
    let Some(connect) = connect else {
        return Check::fail(
            "ports",
            "krishiv.connect.port is not a port number",
            "set krishiv.connect.port to 0-65535",
        );
    };
    if connect != 0 && connect == args.port() {
        return Check::fail(
            "ports",
            format!("the Spark Connect and admin ports are both {connect}"),
            "give --connect-port and --port different values",
        );
    }
    let mut busy = Vec::new();
    for (what, port) in [("Spark Connect", connect), ("admin", args.port())] {
        if port == 0 {
            continue;
        }
        if let Err(e) = TcpListener::bind(SocketAddr::new(host, port)) {
            busy.push(format!("{what} port {port} cannot be bound: {e}"));
        }
    }
    if busy.is_empty() {
        Check::ok(
            "ports",
            format!("{host}:{connect} and {host}:{} are free", args.port()),
        )
    } else {
        Check::fail(
            "ports",
            busy.join("; "),
            "stop the process that holds the port, or pass --connect-port / --port",
        )
    }
}

fn user(args: &ServerArgs) -> Check {
    match (host::is_root(), args.allow_root()) {
        (false, _) => Check::ok("user", "not root"),
        (true, true) => Check::warn(
            "user",
            "root, allowed by --allow-root",
            "run the server as an unprivileged user",
        ),
        (true, false) => Check::fail(
            "user",
            "running as root: the server refuses to start",
            "run as an unprivileged user, or pass --allow-root",
        ),
    }
}

/// The spill, block-cache and checkpoint directories, when set.
fn directories(config: &Config) -> Check {
    let mut problems = Vec::new();
    let mut seen = Vec::new();
    for entry in [&SPILL_DIR, &IO_CACHE_PATH, &CHECKPOINT_DIR] {
        let Some(dir) = config.get(entry).and_then(ConfigValue::as_str) else {
            continue;
        };
        seen.push(entry.key);
        // The spill root must also be one the spill manager accepts (owner, mode, lock-safe file
        // system), or the server refuses it at start.
        let accepted = if entry.key == SPILL_DIR.key {
            krishiv_memory::check_spill_root(Path::new(dir))
        } else {
            Ok(())
        };
        if let Err(e) = accepted.and_then(|()| writable_dir(Path::new(dir))) {
            problems.push(format!("{} = {dir}: {e}", entry.key));
        }
    }
    if !problems.is_empty() {
        return Check::fail(
            "directories",
            problems.join("; "),
            "point the setting at a directory this user can write, on a local disk",
        );
    }
    if seen.is_empty() {
        Check::ok(
            "directories",
            "spill and cache directories use their defaults",
        )
    } else {
        Check::ok("directories", format!("{} are writable", seen.join(", ")))
    }
}

/// A directory that exists and takes a file, or a missing one whose nearest existing ancestor
/// does (it is created at start).
fn writable_dir(dir: &Path) -> Result<(), String> {
    if dir.is_dir() {
        // `create_new` refuses an existing name, a planted symbolic link included; the name is not
        // predictable.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let probe = dir.join(format!(
            ".krishiv-doctor-{}-{nanos:08x}",
            std::process::id()
        ));
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)
            .map_err(|e| format!("cannot write: {e}"))?;
        let _ = std::fs::remove_file(&probe);
        return Ok(());
    }
    if dir.exists() {
        return Err("exists and is not a directory".to_owned());
    }
    // The spill manager creates missing ancestors, so the nearest existing one must take a file.
    match dir.parent() {
        Some(parent) if parent.as_os_str().is_empty() => writable_dir(Path::new(".")),
        Some(parent) => match parent
            .ancestors()
            // A relative path ends in the empty ancestor: the working directory.
            .map(|p| {
                if p.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    p
                }
            })
            .find(|p| p.exists())
        {
            Some(ancestor) if ancestor.is_dir() => writable_dir(ancestor),
            Some(ancestor) => Err(format!("{} is not a directory", ancestor.display())),
            None => Err("no parent directory of it exists".to_owned()),
        },
        None => Err("it has no parent directory".to_owned()),
    }
}

/// The local roots and object-store prefixes a server with authentication lets paths read.
fn paths(settings: &Settings) -> Check {
    let list = |entry| -> Vec<String> {
        settings
            .config
            .get(entry)
            .and_then(ConfigValue::as_str)
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    };
    let roots = list(&IO_LOCAL_ALLOWED_ROOTS);
    let prefixes = list(&IO_OBJECT_STORE_ALLOWED_PREFIXES);
    // The server's own parser: what it refuses at startup, the doctor refuses.
    if let Err(e) = PathPrefixPolicy::from_config(&settings.config, Arc::new(AllowAll)) {
        return Check::fail(
            "paths",
            e.to_string(),
            "fix krishiv.io.local.allowedRoots / krishiv.io.objectStore.allowedPrefixes",
        );
    }
    if !settings.auth.is_configured() {
        return Check::ok(
            "paths",
            "no authentication: the allow lists are not used and paths are not confined",
        );
    }
    if roots.is_empty() && prefixes.is_empty() {
        return Check::warn(
            "paths",
            "authentication is on and neither list is set: no table can be read or written by path",
            "set krishiv.io.local.allowedRoots and/or krishiv.io.objectStore.allowedPrefixes",
        );
    }
    Check::ok(
        "paths",
        format!(
            "{} local root(s), {} object-store prefix(es)",
            roots.len(),
            prefixes.len()
        ),
    )
}

/// The report as plain text: one line per check, the fix under a failure or warning.
pub(crate) fn render_text(checks: &[Check]) -> String {
    let width = checks.iter().map(|c| c.name.len()).max().unwrap_or(0);
    let mut out = String::new();
    for c in checks {
        out.push_str(&format!(
            "{:<7} {:<width$}  {}\n",
            c.status.as_str(),
            c.name,
            c.detail
        ));
        if let Some(fix) = &c.fix {
            out.push_str(&format!("{:<7} {:<width$}  fix: {fix}\n", "", ""));
        }
    }
    let failed = checks.iter().filter(|c| c.status == Status::Fail).count();
    let warned = checks.iter().filter(|c| c.status == Status::Warn).count();
    out.push_str(&match (failed, warned) {
        (0, 0) => "all checks passed\n".to_owned(),
        (0, w) => format!("no failures, {w} warning(s)\n"),
        (f, w) => format!("{f} check(s) failed, {w} warning(s)\n"),
    });
    out
}

/// The report as JSON.
pub(crate) fn render_json(checks: &[Check]) -> String {
    let rows: Vec<serde_json::Value> = checks
        .iter()
        .map(|c| {
            serde_json::json!({
                "name": c.name,
                "status": c.status.as_str().to_lowercase(),
                "detail": c.detail,
                "fix": c.fix,
            })
        })
        .collect();
    let ok = !checks.iter().any(|c| c.status == Status::Fail);
    serde_json::json!({ "ok": ok, "checks": rows }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(conf: &[(&str, &str)]) -> ServerArgs {
        ServerArgs::for_test(conf)
    }

    fn find<'a>(checks: &'a [Check], name: &str) -> &'a Check {
        checks
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no check named {name}: {checks:?}"))
    }

    #[test]
    fn the_defaults_pass() {
        let checks = diagnose(&args(&[]));
        // Run as root, the `user` check fails without `--allow-root`: that is its job.
        let root = host::is_root();
        assert!(
            checks
                .iter()
                .all(|c| c.status != Status::Fail || (root && c.name == "user")),
            "{}",
            render_text(&checks)
        );
        assert_eq!(
            find(&checks, "user").status,
            if root { Status::Fail } else { Status::Ok }
        );
        assert_eq!(find(&checks, "authentication").status, Status::Warn);
    }

    /// The incoherent configurations of the threat model (T-A1, T-A2) and the settings the
    /// server refuses, each with the check that names it.
    #[test]
    fn incoherent_configurations_fail_the_check_that_names_them() {
        let token = (
            "krishiv.auth.tokens.a.secret",
            "env:KRISHIV_DOCTOR_TEST_SECRET",
        );
        let who = ("krishiv.auth.tokens.a.principal", "alice");
        let role = ("krishiv.auth.tokens.a.role", "admin");
        // (settings, host, check that fails, text the failure holds)
        type Case<'a> = (&'a [(&'a str, &'a str)], &'a str, &'a str, &'a str);
        let table: [Case<'_>; 4] = [
            (&[], "0.0.0.0", "listener", "needs an authentication method"),
            (&[token, who, role], "0.0.0.0", "listener", "need TLS"),
            (
                &[("krishiv.server.tls.certFile", "/nonexistent/cert.pem")],
                "127.0.0.1",
                "config",
                "keyFile",
            ),
            (
                &[("krishiv.ui.enabled", "maybe")],
                "127.0.0.1",
                "config",
                "krishiv.ui.enabled",
            ),
        ];
        for (conf, host, check, text) in table {
            let mut a = args(conf);
            a.set_host(host);
            let checks = diagnose(&a);
            let c = find(&checks, check);
            assert_eq!(c.status, Status::Fail, "{conf:?}: {}", render_text(&checks));
            assert!(c.detail.contains(text), "{conf:?}: {}", c.detail);
        }
    }

    /// The mode matrix (T070): every registered key that has a typed value refuses a value of
    /// the wrong shape at startup, and the refusal names the key. A key whose type accepts any
    /// string has no such value and is skipped.
    #[test]
    fn every_typed_key_refuses_a_value_of_the_wrong_shape() {
        use krishiv_common::config::{REGISTRY, ValueType};
        let mut checked = 0;
        for entry in REGISTRY {
            let bad = match entry.value_type {
                ValueType::Bool => "maybe",
                ValueType::Int | ValueType::SignedSize | ValueType::Int32 { .. } => "twelve",
                ValueType::PositiveInt => "0",
                ValueType::Size => "12 parsecs",
                ValueType::Duration | ValueType::Millis => "soon",
                ValueType::TimeZone => "+99:99:99:99",
                ValueType::Enum(_) | ValueType::ExactEnum(_) => "__no_such_value__",
                ValueType::String => continue,
            };
            let conf = vec![(entry.key.to_owned(), bad.to_owned())];
            let Err(e) = load_settings(&conf) else {
                panic!("{} = {bad:?} was accepted at startup", entry.key);
            };
            assert!(
                e.contains(entry.key),
                "{}: the error does not name the key: {e}",
                entry.key
            );
            checked += 1;
        }
        assert!(checked > 50, "only {checked} typed keys were checked");
    }

    #[test]
    fn a_world_readable_secret_file_is_named() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let secret = dir.path().join("token");
        std::fs::write(&secret, b"s3cret-value-long-enough-for-a-token").expect("write");
        std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o644)).expect("chmod");
        let file = format!("file:{}", secret.to_str().expect("utf8"));
        let conf = [
            ("krishiv.auth.tokens.a.secret", file.as_str()),
            ("krishiv.auth.tokens.a.principal", "alice"),
            ("krishiv.auth.tokens.a.role", "admin"),
        ];
        let checks = diagnose(&args(&conf));
        let c = find(&checks, "authentication");
        assert_eq!(c.status, Status::Fail, "{}", render_text(&checks));
        assert!(c.detail.contains("group or others"), "{}", c.detail);
        std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        let checks = diagnose(&args(&conf));
        assert_eq!(find(&checks, "authentication").status, Status::Ok);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_spill_root_the_server_refuses_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::set_permissions(
            dir.path(),
            <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o777),
        )
        .expect("chmod");
        let conf = [("krishiv.spill.dir", dir.path().to_str().expect("utf8"))];
        let checks = diagnose(&args(&conf));
        let c = find(&checks, "directories");
        assert_eq!(c.status, Status::Fail, "{}", render_text(&checks));
        assert!(c.detail.contains("group or others"), "{}", c.detail);
        std::fs::set_permissions(
            dir.path(),
            <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700),
        )
        .expect("chmod");
        assert_eq!(
            find(&diagnose(&args(&conf)), "directories").status,
            Status::Ok
        );
    }

    #[test]
    fn a_relative_spill_dir_under_the_working_directory_passes() {
        let checks = diagnose(&args(&[("krishiv.spill.dir", "no-such-dir/nested/spill")]));
        assert_eq!(find(&checks, "directories").status, Status::Ok);
    }

    #[test]
    fn a_directory_that_cannot_be_written_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("a-file");
        std::fs::write(&file, b"").expect("write");
        let checks = diagnose(&args(&[(
            "krishiv.spill.dir",
            file.to_str().expect("utf8"),
        )]));
        let c = find(&checks, "directories");
        assert_eq!(c.status, Status::Fail);
        assert!(c.detail.contains("not a directory"), "{}", c.detail);
        // A missing directory is created with its missing ancestors, under a writable one.
        let missing = dir.path().join("no/such/parent");
        let checks = diagnose(&args(&[(
            "krishiv.spill.dir",
            missing.to_str().expect("utf8"),
        )]));
        assert_eq!(find(&checks, "directories").status, Status::Ok);
        // Under a file it cannot be created.
        let under_file = file.join("x/y");
        let checks = diagnose(&args(&[(
            "krishiv.spill.dir",
            under_file.to_str().expect("utf8"),
        )]));
        let c = find(&checks, "directories");
        assert_eq!(c.status, Status::Fail);
        assert!(c.detail.contains("is not a directory"), "{}", c.detail);
    }

    #[test]
    fn allow_lists_are_checked() {
        // The server's own parser reports the first list that is wrong.
        for (key, value) in [
            ("krishiv.io.local.allowedRoots", "relative/dir"),
            ("krishiv.io.objectStore.allowedPrefixes", "ftp://x"),
        ] {
            let checks = diagnose(&args(&[(key, value)]));
            let c = find(&checks, "paths");
            assert_eq!(c.status, Status::Fail, "{key}: {}", render_text(&checks));
            assert!(c.detail.contains(key), "{key}: {}", c.detail);
        }
    }

    #[test]
    fn a_taken_port_fails() {
        let held = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = held.local_addr().expect("addr").port();
        let mut a = args(&[]);
        a.set_ports(port, 0);
        let checks = diagnose(&a);
        let c = find(&checks, "ports");
        assert_eq!(c.status, Status::Fail);
        assert!(c.detail.contains(&port.to_string()), "{}", c.detail);
    }

    #[test]
    fn the_report_is_stable() {
        let checks = vec![
            Check::ok("config", "2 settings parsed"),
            Check::fail(
                "listener",
                "refusing to serve on 0.0.0.0",
                "bind a loopback address",
            ),
            Check::skipped("catalogs", "none declared"),
        ];
        assert_eq!(
            render_text(&checks),
            "ok      config    2 settings parsed\n\
             FAIL    listener  refusing to serve on 0.0.0.0\n\
             \x20       \x20         fix: bind a loopback address\n\
             skipped catalogs  none declared\n\
             1 check(s) failed, 0 warning(s)\n"
        );
        assert_eq!(
            render_json(&checks),
            r#"{"checks":[{"detail":"2 settings parsed","fix":null,"name":"config","status":"ok"},{"detail":"refusing to serve on 0.0.0.0","fix":"bind a loopback address","name":"listener","status":"fail"},{"detail":"none declared","fix":null,"name":"catalogs","status":"skipped"}],"ok":false}"#
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
//! `krishiv server` (design P3-T001): one process, one engine runtime, two
//! listeners on one tokio runtime: Spark Connect (gRPC, port 15002) and the
//! admin server (HTTP, port 8080: `/healthz`, `/readyz`, `/metrics` and,
//! with the `ui` feature and `krishiv.ui.enabled`, the web UI). With
//! `krishiv.auth.tokens.*` configured the admin routes take those principals
//! and the ADR 0046 roles (P3-T002); without, a random token printed once.
//!
//! Before anything binds: the address, authentication and TLS must fit
//! together (`check_listener`, design P3-T002 §3.5: a non-loopback address
//! needs an authentication method, a bearer method there needs TLS, client
//! certificates need TLS), uid 0 is refused without `--allow-root`, and
//! core dumps are turned off unless `krishiv.server.allowCoreDumps`. With
//! `krishiv.server.tls.certFile`/`keyFile` both listeners speak TLS; on
//! SIGHUP (Unix) the certificate and key are read again, and a failed read
//! keeps the old ones serving. On SIGTERM or SIGINT the server drains
//! (`/readyz` 503, new RPCs `UNAVAILABLE`), gives running queries
//! `krishiv.server.shutdownTimeout`, closes every session, and returns; the
//! engine runtime is dropped before `main` exits, which removes the spill
//! directory. A second signal exits at once with 130.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use krishiv_common::config::{
    CONNECT_ADDRESS, CONNECT_PORT, Config, ConfigValue, METRICS_ENABLED, SERVER_ALLOW_CORE_DUMPS,
    SERVER_METRICS_PUBLIC, UI_ENABLED,
};
use krishiv_connect_server::{
    ConnectAuth, ConnectServer, ConnectServerConfig, SessionLimits, SessionManager, host,
    role_sessions,
};
use krishiv_server::guard::{Guard, TokenGuard};
use krishiv_server::session_guard::SessionGuard;
use krishiv_session::auth::{
    AuthConfig, ServerAuth, ServerTls, TlsConfig, check_listener, is_auth_key,
};
use krishiv_session::{EngineRuntime, Session};
use tokio::net::TcpListener;
use tokio::sync::watch;

/// The command's arguments.
#[derive(Debug, clap::Args)]
pub(crate) struct ServerArgs {
    /// Admin/UI HTTP port (0 picks a free port).
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Spark Connect port (0 picks a free port); default `krishiv.connect.port`, 15002.
    #[arg(long = "connect-port")]
    connect_port: Option<u16>,
    /// Address both listeners bind; default `krishiv.connect.address`, loopback. A
    /// non-loopback address needs authentication (`krishiv.auth.*`), and TLS
    /// (`krishiv.server.tls.*`) when that is a bearer token or OIDC.
    #[arg(long)]
    host: Option<IpAddr>,
    /// Run even as uid 0.
    #[arg(long)]
    allow_root: bool,
    /// Engine setting, `KEY=VALUE` (repeatable).
    #[arg(long = "conf", value_parser = crate::common::key_value)]
    conf: Vec<(String, String)>,
}

/// Runs the server; the process exit code.
#[expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")]
pub(crate) fn run(args: &ServerArgs) -> i32 {
    match serve(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("krishiv server: {e}");
            1
        }
    }
}

fn serve(args: &ServerArgs) -> Result<(), String> {
    let mut config = Config::new();
    let mut builder = EngineRuntime::builder();
    // Authentication and TLS settings (secret references and key files included) are read once
    // here and never become engine configuration: a session could otherwise list them with `SET`.
    let listener_settings = || {
        args.conf
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .filter(|(k, _)| is_listener_key(k))
    };
    let auth_config = AuthConfig::from_settings(listener_settings()).map_err(|e| e.to_string())?;
    let tls_config = TlsConfig::from_settings(listener_settings()).map_err(|e| e.to_string())?;
    for (k, v) in &args.conf {
        if is_listener_key(k) {
            continue;
        }
        let warning = config.set(k, v).map_err(|e| e.to_string())?;
        // A Spark Connect setting Krishiv does not read would be ignored
        // silently: refuse it, naming the replacement where there is one.
        if warning.is_some() && k.starts_with("spark.connect.") {
            let hint = if k == "spark.connect.session.manager.defaultSessionTimeout" {
                " (use krishiv.connect.session.idleTimeout; 0s means never, Spark's -1)"
            } else {
                ""
            };
            return Err(format!("{k} is not supported by krishiv server{hint}"));
        }
        builder = builder.config(k, v);
    }
    let host = match args.host {
        Some(h) => h,
        None => config
            .get(&CONNECT_ADDRESS)
            .and_then(ConfigValue::as_str)
            .unwrap_or("127.0.0.1")
            .parse()
            .map_err(|_| "krishiv.connect.address is not an IP address".to_owned())?,
    };
    // ADR 0046 §6, design §3.5: before either listener binds (both bind `host`).
    check_listener(SocketAddr::new(host, 0), &auth_config, &tls_config).map_err(|e| {
        format!(
            "refusing to serve on {host}: {e}. Bind a loopback address (--host 127.0.0.1), or \
             configure authentication (krishiv.auth.*) and, for tokens or OIDC, TLS \
             (krishiv.server.tls.certFile and keyFile)."
        )
    })?;
    let tls = ServerTls::new(&tls_config, &auth_config.mtls).map_err(|e| e.to_string())?;
    let auth = ServerAuth::new(&auth_config).map_err(|e| e.to_string())?;
    if host::is_root() && !args.allow_root {
        return Err("refusing to run as root; pass --allow-root to override".to_owned());
    }
    let core_dumps = flag(&config, &SERVER_ALLOW_CORE_DUMPS, false);
    if !core_dumps {
        host::disable_core_dumps().map_err(|e| format!("cannot turn core dumps off: {e}"))?;
    }
    let connect_port = match args.connect_port {
        Some(p) => p,
        None => config
            .get(&CONNECT_PORT)
            .and_then(ConfigValue::as_int)
            .and_then(|p| u16::try_from(p).ok())
            .ok_or("krishiv.connect.port is not a port number")?,
    };
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("krishiv-server")
        .build()
        .map_err(|e| e.to_string())?;
    let result = tokio.block_on(async {
        let runtime = builder.build().map_err(|e| e.to_string())?;
        run_listeners(
            runtime,
            &config,
            auth,
            tls,
            SocketAddr::new(host, connect_port),
            SocketAddr::new(host, args.port),
        )
        .await
    });
    // Every handle on the engine runtime went with the async block; dropping
    // the tokio runtime ends the tasks that were left.
    drop(tokio);
    result
}

/// A setting `krishiv server` reads itself: authentication (`krishiv.auth.*`, Spark's token key)
/// and the listeners' TLS (`krishiv.server.tls.*`).
fn is_listener_key(key: &str) -> bool {
    is_auth_key(key) || key.starts_with("krishiv.server.tls.")
}

fn flag(config: &Config, entry: &krishiv_common::config::ConfigEntry, default: bool) -> bool {
    config
        .get(entry)
        .and_then(ConfigValue::as_bool)
        .unwrap_or(default)
}

#[expect(
    clippy::print_stdout,
    reason = "CLI binary: the addresses and the token are its output"
)]
async fn run_listeners(
    runtime: Arc<EngineRuntime>,
    config: &Config,
    auth: ServerAuth,
    tls: Option<ServerTls>,
    connect_addr: SocketAddr,
    admin_addr: SocketAddr,
) -> Result<(), String> {
    let base = Session::builder()
        .runtime(Arc::clone(&runtime))
        .build()
        .map_err(|e| e.to_string())?;
    let authenticated = auth.is_enabled();
    let auth = Arc::new(auth);
    // One denial counter for the guards and every caller's session (ADR 0046 §8).
    let denials = auth.denials();
    let limits = SessionLimits::from_config(config);
    let manager = Arc::new(if authenticated {
        SessionManager::new(
            role_sessions(base, config, Arc::clone(&denials)).map_err(|e| e.to_string())?,
            limits,
        )
    } else {
        SessionManager::from_base(base, limits)
    });
    let connect_listener = TcpListener::bind(connect_addr)
        .await
        .map_err(|e| format!("cannot bind the Spark Connect port {connect_addr}: {e}"))?;
    let admin_listener = TcpListener::bind(admin_addr)
        .await
        .map_err(|e| format!("cannot bind the admin port {admin_addr}: {e}"))?;
    let connect_bound = connect_listener.local_addr().map_err(|e| e.to_string())?;
    let admin_bound = admin_listener.local_addr().map_err(|e| e.to_string())?;

    // With authentication configured the admin routes take the same principals as Connect;
    // without it, a random token printed once (loopback only).
    let token = krishiv_server::internal::random_token().map_err(|e| e.to_string())?;
    let guard: Arc<dyn Guard> = if authenticated {
        Arc::new(SessionGuard::new(Arc::clone(&auth) as Arc<_>).with_denials(Arc::clone(&denials)))
    } else {
        Arc::new(TokenGuard::new(&token))
    };
    let ready = Arc::new(AtomicBool::new(false));
    let app = admin_router(
        &runtime,
        config,
        guard,
        authenticated.then(|| Arc::clone(&auth)),
        Arc::clone(&ready),
        admin_bound,
    )?;
    let mut connect = ConnectServer::new(
        Arc::clone(&manager),
        ConnectServerConfig::from_config(config),
    );
    if authenticated {
        connect = connect.with_auth(ConnectAuth::new(auth).with_denials(denials));
    }
    if let Some(tls) = &tls {
        connect = connect.with_tls(&tls.server_config());
    }

    let (stop_tx, stop_rx) = watch::channel(false);
    // Installed before the addresses are printed: a SIGTERM right after
    // readiness must start the drain, not kill the process.
    let handlers = Signals::install()
        .map_err(|e| format!("cannot install the SIGTERM/SIGINT handlers: {e}"))?;
    let signals = tokio::spawn(watch_signals(handlers, Arc::clone(&ready), stop_tx));
    let stopped = |mut rx: watch::Receiver<bool>| async move {
        let _ = rx.wait_for(|stop| *stop).await;
    };
    let max_admin = krishiv_server::admin::HttpLimits::default().max_connections;
    let admin = match &tls {
        Some(tls) => {
            let config = tls.server_config();
            let shutdown = stopped(stop_rx.clone());
            tokio::spawn(async move {
                krishiv_server::tls::serve_capped_tls(
                    admin_listener,
                    app,
                    max_admin,
                    &config,
                    shutdown,
                )
                .await
            })
        }
        None => tokio::spawn(krishiv_server::admin::serve_capped(
            admin_listener,
            app,
            max_admin,
            stopped(stop_rx.clone()),
        )),
    };
    // The SIGHUP handler is installed here, before the addresses are printed: a SIGHUP after
    // readiness must reload, not end the process (SIGHUP's default action).
    let reloads = tls
        .clone()
        .map(|tls| tokio::spawn(reload_on_hangup(tls, Hangups::install())));
    let (sc_suffix, http) = if tls.is_some() {
        ("/;use_ssl=true", "https")
    } else {
        ("", "http")
    };
    println!("krishiv server: Spark Connect on sc://{connect_bound}{sc_suffix}");
    println!("krishiv server: admin on {http}://{admin_bound}");
    let ui = flag(config, &UI_ENABLED, true) && cfg!(feature = "ui");
    if authenticated {
        println!(
            "krishiv server: Spark Connect and the admin routes need a configured credential \
             (krishiv.auth.*)"
        );
    } else if ui {
        println!("open {http}://{admin_bound}/?token={token}");
    } else if metrics_need_token(config, admin_bound) {
        // `/metrics` is behind the admin token: print it once.
        println!("krishiv server: admin token {token}");
    }
    ready.store(true, Ordering::Release);
    let served = connect
        .serve(connect_listener, stopped(stop_rx))
        .await
        .map_err(|e| e.to_string());
    ready.store(false, Ordering::Release);
    signals.abort();
    if let Some(reloads) = reloads {
        reloads.abort();
    }
    // The admin listener stopped accepting with the same signal; requests in
    // flight get `ADMIN_DRAIN` to finish. Open event streams never end on
    // their own, so whatever is left then is aborted.
    let mut admin = admin;
    if tokio::time::timeout(ADMIN_DRAIN, &mut admin).await.is_err() {
        admin.abort();
        let _ = admin.await;
    }
    drop(manager);
    drop(runtime);
    served
}

fn admin_router(
    runtime: &Arc<EngineRuntime>,
    config: &Config,
    guard: Arc<dyn Guard>,
    auth: Option<Arc<ServerAuth>>,
    ready: Arc<AtomicBool>,
    bound: SocketAddr,
) -> Result<axum::Router, String> {
    let authenticated = auth.is_some();
    let metrics_on = flag(config, &METRICS_ENABLED, false);
    // With authentication configured every scrape authenticates (ADR 0046 §3c: a viewer
    // service principal); `krishiv.server.metrics.public` opens it only without.
    let public = metrics_on && !authenticated && !metrics_need_token(config, bound);
    let snapshot_runtime = Arc::clone(runtime);
    let metrics = krishiv_server::metrics::metrics_route_guarded(
        metrics_on,
        // With authentication on, the auth counters ride along (ADR 0046 §8).
        Arc::new(move || {
            let mut snapshot = snapshot_runtime.metrics();
            if let Some(auth) = &auth {
                snapshot.merge(&auth.metrics());
            }
            snapshot
        }),
        (!public).then(|| Arc::clone(&guard)),
    );
    let base = if cfg!(feature = "ui") && flag(config, &UI_ENABLED, true) {
        let source = krishiv_server::session_source::SessionSource::new(Arc::clone(runtime))
            .map_err(|e| e.to_string())?;
        krishiv_server::internal::router_with_ui_guard(Arc::new(source), guard)
    } else {
        krishiv_server::api_router()
    };
    let app = base
        .merge(krishiv_server::admin::health_routes(ready))
        .merge(metrics);
    Ok(krishiv_server::admin::with_limits(
        app,
        krishiv_server::admin::HttpLimits::default(),
    ))
}

/// Whether `/metrics` is on and behind the admin token: it is open only on
/// a loopback address, and only with `krishiv.server.metrics.public`.
fn metrics_need_token(config: &Config, bound: SocketAddr) -> bool {
    flag(config, &METRICS_ENABLED, false)
        && !(flag(config, &SERVER_METRICS_PUBLIC, false) && bound.ip().is_loopback())
}

/// How long admin requests in flight may take once the server stops.
const ADMIN_DRAIN: Duration = Duration::from_secs(2);

/// SIGHUP, registered when installed (tokio registers a handler only when its stream is
/// created). `None` elsewhere than Unix, or when it cannot be installed.
struct Hangups {
    #[cfg(unix)]
    signal: Option<tokio::signal::unix::Signal>,
}

impl Hangups {
    #[expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")]
    fn install() -> Self {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let signal = signal(SignalKind::hangup()).ok();
            if signal.is_none() {
                eprintln!("krishiv server: cannot install the SIGHUP handler; TLS reload is off");
            }
            Self { signal }
        }
        #[cfg(not(unix))]
        {
            Self {}
        }
    }
}

/// Reads the TLS certificate and key again on every SIGHUP (Unix; elsewhere it waits forever).
/// New handshakes get the new material; a failed read is logged and the old material serves on.
#[expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")]
async fn reload_on_hangup(tls: ServerTls, hangups: Hangups) {
    #[cfg(unix)]
    {
        let Some(mut hangups) = hangups.signal else {
            return;
        };
        while hangups.recv().await.is_some() {
            let tls = tls.clone();
            match tokio::task::spawn_blocking(move || tls.reload()).await {
                Ok(Ok(())) => eprintln!("krishiv server: TLS certificate reloaded"),
                Ok(Err(e)) => {
                    eprintln!(
                        "krishiv server: TLS reload failed, the old certificate serves on: {e}"
                    );
                }
                Err(e) => eprintln!("krishiv server: TLS reload did not finish: {e}"),
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (tls, hangups);
        std::future::pending::<()>().await;
    }
}

/// The first SIGTERM or SIGINT starts the drain; a second exits at once.
#[expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")]
async fn watch_signals(mut signals: Signals, ready: Arc<AtomicBool>, stop: watch::Sender<bool>) {
    signals.next().await;
    ready.store(false, Ordering::Release);
    eprintln!("krishiv server: shutting down (a second signal exits at once)");
    let _ = stop.send(true);
    signals.next().await;
    eprintln!("krishiv server: second signal, exiting");
    std::process::exit(130);
}

/// SIGTERM and SIGINT, registered when installed (tokio registers a
/// handler only when its stream is created, not when it is first polled).
struct Signals {
    #[cfg(unix)]
    term: tokio::signal::unix::Signal,
    #[cfg(unix)]
    int: tokio::signal::unix::Signal,
}

impl Signals {
    #[cfg(unix)]
    fn install() -> std::io::Result<Self> {
        use tokio::signal::unix::{SignalKind, signal};
        Ok(Self {
            term: signal(SignalKind::terminate())?,
            int: signal(SignalKind::interrupt())?,
        })
    }

    #[cfg(not(unix))]
    fn install() -> std::io::Result<Self> {
        Ok(Self {})
    }

    async fn next(&mut self) {
        #[cfg(unix)]
        tokio::select! {
            _ = self.term.recv() => {}
            _ = self.int.recv() => {}
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(pairs: &[(&str, &str)]) -> Config {
        let mut c = Config::new();
        for (k, v) in pairs {
            c.set(k, v).unwrap();
        }
        c
    }

    #[test]
    fn core_dumps_stay_off_unless_allowed() {
        assert!(!flag(&config(&[]), &SERVER_ALLOW_CORE_DUMPS, false));
        let c = config(&[("krishiv.server.allowCoreDumps", "true")]);
        assert!(flag(&c, &SERVER_ALLOW_CORE_DUMPS, false));
    }

    #[test]
    fn public_metrics_skip_the_token_only_on_loopback() {
        let lo: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let any: SocketAddr = "0.0.0.0:8080".parse().unwrap();
        let on = config(&[("krishiv.metrics.enabled", "true")]);
        assert!(metrics_need_token(&on, lo));
        let public = config(&[
            ("krishiv.metrics.enabled", "true"),
            ("krishiv.server.metrics.public", "true"),
        ]);
        assert!(!metrics_need_token(&public, lo));
        assert!(metrics_need_token(&public, any));
        assert!(!metrics_need_token(&config(&[]), lo));
    }
}

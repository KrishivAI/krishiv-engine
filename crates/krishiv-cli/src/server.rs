// SPDX-License-Identifier: Apache-2.0
//! `krishiv server` (design P3-T001): one process, one engine runtime, two
//! listeners on one tokio runtime: Spark Connect (gRPC, port 15002) and the
//! admin server (HTTP, port 8080: `/healthz`, `/readyz`, `/metrics` and,
//! with the `ui` feature and `krishiv.ui.enabled`, the web UI behind a token
//! printed once at start).
//!
//! Before anything binds: a non-loopback address is refused (no
//! authentication exists until P3-T002), uid 0 is refused without
//! `--allow-root`, and core dumps are turned off unless
//! `krishiv.server.allowCoreDumps`. On SIGTERM or SIGINT the server drains
//! (`/readyz` 503, new RPCs `UNAVAILABLE`), gives running queries
//! `krishiv.server.shutdownTimeout`, closes every session, and returns; the
//! engine runtime is dropped before `main` exits, which removes the spill
//! directory. A second signal exits at once with 130.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use krishiv_common::config::{
    CONNECT_ADDRESS, CONNECT_PORT, Config, ConfigValue, METRICS_ENABLED, SERVER_ALLOW_CORE_DUMPS,
    SERVER_METRICS_PUBLIC, UI_ENABLED,
};
use krishiv_connect_server::{
    ConnectServer, ConnectServerConfig, SessionLimits, SessionManager, host,
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
    /// non-loopback address needs authentication, which this build cannot configure yet.
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
    for (k, v) in &args.conf {
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
    // ADR 0046 §6: before either listener binds.
    if host::needs_auth(host) {
        return Err(format!(
            "refusing to serve on {host} without authentication: a non-loopback address needs \
             authentication configured (P3-T002, not in this build). Bind a loopback address \
             (--host 127.0.0.1) and put an authenticated tunnel or proxy in front of it."
        ));
    }
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

fn flag(config: &Config, entry: &krishiv_common::config::ConfigEntry, default: bool) -> bool {
    config
        .get(entry)
        .and_then(ConfigValue::as_bool)
        .unwrap_or(default)
}

#[expect(
    clippy::print_stdout,
    reason = "CLI binary: the addresses and token are its output"
)]
async fn run_listeners(
    runtime: Arc<EngineRuntime>,
    config: &Config,
    connect_addr: SocketAddr,
    admin_addr: SocketAddr,
) -> Result<(), String> {
    let base = Session::builder()
        .runtime(Arc::clone(&runtime))
        .build()
        .map_err(|e| e.to_string())?;
    let manager = Arc::new(SessionManager::from_base(
        base,
        SessionLimits::from_config(config),
    ));
    let connect_listener = TcpListener::bind(connect_addr)
        .await
        .map_err(|e| format!("cannot bind the Spark Connect port {connect_addr}: {e}"))?;
    let admin_listener = TcpListener::bind(admin_addr)
        .await
        .map_err(|e| format!("cannot bind the admin port {admin_addr}: {e}"))?;
    let connect_bound = connect_listener.local_addr().map_err(|e| e.to_string())?;
    let admin_bound = admin_listener.local_addr().map_err(|e| e.to_string())?;

    let token = krishiv_server::internal::random_token().map_err(|e| e.to_string())?;
    let ready = Arc::new(AtomicBool::new(false));
    let app = admin_router(&runtime, config, &token, Arc::clone(&ready), admin_bound)?;
    let connect = ConnectServer::new(
        Arc::clone(&manager),
        ConnectServerConfig::from_config(config),
    );

    let (stop_tx, stop_rx) = watch::channel(false);
    let signals = tokio::spawn(watch_signals(Arc::clone(&ready), stop_tx));
    let stopped = |mut rx: watch::Receiver<bool>| async move {
        let _ = rx.wait_for(|stop| *stop).await;
    };
    let admin = tokio::spawn(krishiv_server::admin::serve_capped(
        admin_listener,
        app,
        krishiv_server::admin::HttpLimits::default().max_connections,
        stopped(stop_rx.clone()),
    ));
    println!("krishiv server: Spark Connect on sc://{connect_bound}");
    println!("krishiv server: admin on http://{admin_bound}");
    let ui = flag(config, &UI_ENABLED, true) && cfg!(feature = "ui");
    if ui {
        println!("open http://{admin_bound}/?token={token}");
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
    // The admin listener stopped with the same signal; its open event
    // streams end with the process.
    admin.abort();
    let _ = admin.await;
    drop(manager);
    drop(runtime);
    served
}

fn admin_router(
    runtime: &Arc<EngineRuntime>,
    config: &Config,
    token: &str,
    ready: Arc<AtomicBool>,
    bound: SocketAddr,
) -> Result<axum::Router, String> {
    let metrics_on = flag(config, &METRICS_ENABLED, false);
    let public = metrics_on && !metrics_need_token(config, bound);
    let snapshot_runtime = Arc::clone(runtime);
    let metrics = krishiv_server::metrics::metrics_route(
        metrics_on,
        Arc::new(move || snapshot_runtime.metrics()),
        (!public).then_some(token),
    );
    let base = if cfg!(feature = "ui") && flag(config, &UI_ENABLED, true) {
        let source = krishiv_server::session_source::SessionSource::new(Arc::clone(runtime))
            .map_err(|e| e.to_string())?;
        krishiv_server::internal::router_with_ui(Arc::new(source), token)
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

/// The first SIGTERM or SIGINT starts the drain; a second exits at once.
#[expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")]
async fn watch_signals(ready: Arc<AtomicBool>, stop: watch::Sender<bool>) {
    signal().await;
    ready.store(false, Ordering::Release);
    eprintln!("krishiv server: shutting down (a second signal exits at once)");
    let _ = stop.send(true);
    signal().await;
    eprintln!("krishiv server: second signal, exiting");
    std::process::exit(130);
}

async fn signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = term.recv() => {}
                    _ = tokio::signal::ctrl_c() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

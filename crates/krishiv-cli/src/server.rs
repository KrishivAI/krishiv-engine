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
//!
//! Declared catalogs (design P3-T067 §2.2, §2.3): the `krishiv.catalog.<name>.*` and
//! `spark.sql.catalog.*` settings are read here, never stored as engine configuration, and handed
//! to the runtime in server mode with the factories this build carries (`iceberg-rest` with the
//! `catalog-iceberg-rest` feature). The runtime parses and checks them before either listener
//! binds (types without a factory, literal secrets, endpoint addresses the egress rule refuses,
//! unresolved references): any refusal stops startup with its error. Sessions can never declare
//! or change one.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use krishiv_common::config::{
    CONNECT_ADDRESS, CONNECT_PORT, Config, ConfigValue, IO_EGRESS_PRIVATE_ADDRESSES,
    METRICS_ENABLED, SERVER_ALLOW_CORE_DUMPS, SERVER_METRICS_PUBLIC,
    SERVER_READINESS_FAILURE_THRESHOLD, SERVER_READINESS_INTERVAL, SERVER_READINESS_REQUIRED,
    SERVER_READINESS_TIMEOUT, SERVER_SHUTDOWN_TIMEOUT, UI_ALLOWED_HOSTS, UI_ALLOWED_ORIGINS,
    UI_EMBED_HIDE_ATTRIBUTION, UI_EMBED_LINK_TTL, UI_ENABLED,
};
use krishiv_connect_server::{
    ConnectAuth, ConnectServer, ConnectServerConfig, ExtraProvider, SessionFactory, SessionLimits,
    SessionManager, host, provider_sessions,
};
use krishiv_server::guard::{Guard, TokenGuard};
use krishiv_server::session_guard::SessionGuard;
use krishiv_server::udf_workers::UdfWorkersProvider;
use krishiv_session::auth::{
    AuthConfig, Authenticated, ServerAuth, ServerTls, TlsConfig, check_listener, is_auth_key,
};
use krishiv_session::catalog_decl::{DeclMode, is_declaration_key};
use krishiv_session::catalog_registry::CatalogSetup;
use krishiv_session::{EngineRuntime, Session, Transport};
use tokio::net::TcpListener;
use tokio::sync::watch;

/// Each authenticated caller's session factory: the role matrix, joined by the `external`
/// provider when `krishiv.access.provider` lists it (ADR 0049 §1: `external`, `roles+external`).
/// Unset, the role matrix alone (the server's caller sessions carry no grants). `grants` is not
/// wired into the server yet, so any list naming it is refused at startup rather than run without
/// the grants it asks for.
fn caller_sessions(
    base: Session,
    config: &Config,
    denials: Arc<krishiv_session::auth::DeniedCounts>,
) -> Result<Box<SessionFactory>, String> {
    use krishiv_server::access_external::{ExternalAccess, ExternalConfig};
    use krishiv_session::grants::{AccessProvider, access_providers};
    // Unset is `roles` here (not the authenticated default `grants`): that is what the sessions
    // below enforce.
    let providers = access_providers(config, false).map_err(|e| e.to_string())?;
    if providers.contains(&AccessProvider::Grants) {
        return Err(
            "krishiv.access.provider: the grants provider is not available in `krishiv server` yet, so \
             a list naming grants is refused; use `roles`, `external` or `roles+external`"
                .to_owned(),
        );
    }
    if !providers.contains(&AccessProvider::External) {
        return provider_sessions(base, config, denials, None).map_err(|e| e.to_string());
    }
    // One provider per engine runtime: the Connect sessions and the UI editor of one server share
    // its channel, version and counts (review m8). The configuration is read and checked on every
    // call; a second server in the process (another runtime) gets its own provider (review D6).
    let external = ExternalConfig::from_config(config).map_err(|e| e.to_string())?;
    let access = shared_external(base.runtime(), &external, || {
        ExternalAccess::new(&external).map_err(|e| e.to_string())
    })?;
    let extra = ExtraProvider {
        policy: Arc::new(move || {
            Arc::new(access.policy()) as Arc<dyn krishiv_session::AccessPolicy>
        }),
        roles_decide_data: providers.contains(&AccessProvider::Roles),
    };
    provider_sessions(base, config, denials, Some(extra)).map_err(|e| e.to_string())
}

/// The `external` provider of `runtime` with the settings `config`, made by `make` the first
/// time that runtime asks with them: one runtime asked with other settings (another server over
/// it) gets a provider of its own (G6 review r3). Entries of dropped runtimes go at the next call.
fn shared_external(
    runtime: &Arc<krishiv_session::EngineRuntime>,
    config: &krishiv_server::access_external::ExternalConfig,
    make: impl FnOnce() -> Result<Arc<krishiv_server::access_external::ExternalAccess>, String>,
) -> Result<Arc<krishiv_server::access_external::ExternalAccess>, String> {
    type Entry = (
        std::sync::Weak<krishiv_session::EngineRuntime>,
        krishiv_server::access_external::ExternalConfig,
        Arc<krishiv_server::access_external::ExternalAccess>,
    );
    static SHARED: std::sync::Mutex<Vec<Entry>> = std::sync::Mutex::new(Vec::new());
    let mut shared = SHARED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    shared.retain(|(r, _, _)| r.strong_count() > 0);
    if let Some((_, _, access)) = shared
        .iter()
        .find(|(r, c, _)| c == config && r.upgrade().is_some_and(|r| Arc::ptr_eq(&r, runtime)))
    {
        return Ok(Arc::clone(access));
    }
    let access = make()?;
    shared.push((Arc::downgrade(runtime), config.clone(), Arc::clone(&access)));
    Ok(access)
}

/// A server without authentication has no principal for a policy service to decide about.
fn refuse_unauthenticated_providers(config: &Config) -> Result<(), String> {
    use krishiv_session::grants::{AccessProvider, access_providers};
    let providers = access_providers(config, false).map_err(|e| e.to_string())?;
    // `grants` and `external` both decide about a principal; without authentication the server
    // would run every caller as the base session and enforce neither (review: grants on an
    // unauthenticated server started silently).
    for provider in [AccessProvider::Grants, AccessProvider::External] {
        if providers.contains(&provider) {
            return Err(format!(
                "krishiv.access.provider: the {} provider needs authentication (krishiv.auth.*)",
                provider.as_str()
            ));
        }
    }
    Ok(())
}

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

impl ServerArgs {
    /// `--host`.
    pub(crate) fn host(&self) -> Option<IpAddr> {
        self.host
    }
    /// `--connect-port`.
    pub(crate) fn connect_port(&self) -> Option<u16> {
        self.connect_port
    }
    /// `--port`.
    pub(crate) fn port(&self) -> u16 {
        self.port
    }
    /// The `--conf` settings.
    pub(crate) fn conf(&self) -> &[(String, String)] {
        &self.conf
    }
    /// `--allow-root`.
    pub(crate) fn allow_root(&self) -> bool {
        self.allow_root
    }
    /// The settings of a test: default ports, no host, no `--allow-root`.
    #[cfg(test)]
    pub(crate) fn for_test(conf: &[(&str, &str)]) -> Self {
        Self {
            port: 0,
            connect_port: Some(0),
            host: None,
            allow_root: false,
            conf: conf
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        }
    }
    #[cfg(test)]
    pub(crate) fn set_host(&mut self, host: &str) {
        self.host = host.parse().ok();
    }
    #[cfg(test)]
    pub(crate) fn set_ports(&mut self, connect: u16, admin: u16) {
        self.connect_port = Some(connect);
        self.port = admin;
    }
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

/// The settings of `krishiv server`, parsed and checked. `krishiv doctor` runs the same code, so
/// the two cannot disagree about what a configuration means.
pub(crate) struct Settings {
    pub(crate) config: Config,
    pub(crate) auth: AuthConfig,
    pub(crate) tls: TlsConfig,
    /// The declared catalogs' setup (declarations are the registry's, not configuration).
    pub(crate) catalogs: CatalogSetup,
    /// The settings the engine runtime receives: everything but the listener's own and the
    /// catalog declarations, the default-catalog key under its one spelling.
    pub(crate) engine: Vec<(String, String)>,
}

/// Parses every `--conf` setting: the listener's own (authentication, TLS) into their typed
/// forms, the rest into the engine configuration.
///
/// `krishiv.auth.*` and `krishiv.server.tls.*` settings (secret references and key files
/// included) are read once here and never become engine configuration: a session could otherwise
/// list them with `SET`.
pub(crate) fn load_settings(conf: &[(String, String)]) -> Result<Settings, String> {
    let mut config = Config::new();
    let listener_settings = || {
        conf.iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .filter(|(k, _)| is_listener_key(k))
    };
    let auth = AuthConfig::from_settings(listener_settings()).map_err(|e| e.to_string())?;
    let tls = TlsConfig::from_settings(listener_settings()).map_err(|e| e.to_string())?;
    let mut catalogs = catalog_setup();
    let mut engine = Vec::new();
    for (k, v) in conf {
        if is_listener_key(k) {
            continue;
        }
        // Declarations (they may hold secret references) are the registry's, not configuration.
        if is_declaration_key(k) {
            catalogs = catalogs.setting(k, v);
            continue;
        }
        // Spark's default-catalog key in any letter case, like the declaration keys (review
        // 58a2 N3): it names the default to the registry and stays engine configuration under
        // its one spelling.
        let k = if k.eq_ignore_ascii_case(SPARK_DEFAULT_CATALOG) {
            catalogs = catalogs.setting(k, v);
            SPARK_DEFAULT_CATALOG
        } else {
            k.as_str()
        };
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
        engine.push((k.to_owned(), v.clone()));
    }
    Ok(Settings {
        config,
        auth,
        tls,
        catalogs,
        engine,
    })
}

/// The address both listeners bind: `--host`, else `krishiv.connect.address`.
pub(crate) fn resolve_host(host: Option<IpAddr>, config: &Config) -> Result<IpAddr, String> {
    match host {
        Some(h) => Ok(h),
        None => config
            .get(&CONNECT_ADDRESS)
            .and_then(ConfigValue::as_str)
            .unwrap_or("127.0.0.1")
            .parse()
            .map_err(|_| "krishiv.connect.address is not an IP address".to_owned()),
    }
}

/// ADR 0046 §6, design §3.5: the check before either listener binds (both bind `host`).
pub(crate) fn check_host(host: IpAddr, settings: &Settings) -> Result<(), String> {
    check_listener(SocketAddr::new(host, 0), &settings.auth, &settings.tls).map_err(|e| {
        format!(
            "refusing to serve on {host}: {e}. Bind a loopback address (--host 127.0.0.1), or \
             configure authentication (krishiv.auth.*) and, for tokens or OIDC, TLS \
             (krishiv.server.tls.certFile and keyFile)."
        )
    })
}

fn serve(args: &ServerArgs) -> Result<(), String> {
    let settings = load_settings(&args.conf)?;
    let mut builder = EngineRuntime::builder();
    let host = resolve_host(args.host, &settings.config)?;
    check_host(host, &settings)?;
    // The `krishiv.auth.*` settings as given (references, never resolved), for the admin `GetConfig`.
    let startup_auth: Vec<(String, String)> = args
        .conf
        .iter()
        .filter(|(k, _)| is_auth_key(k))
        .cloned()
        .collect();
    let Settings {
        config,
        auth: auth_config,
        tls: tls_config,
        catalogs,
        engine,
    } = settings;
    for (k, v) in &engine {
        builder = builder.config(k, v);
    }
    builder = builder.catalogs(catalogs);
    // The egress policy (design P3-T067 §5.1): a server allows only globally routable
    // destinations unless the admin chose otherwise.
    if !args
        .conf
        .iter()
        .any(|(k, _)| k == IO_EGRESS_PRIVATE_ADDRESSES.key)
    {
        builder = builder.config(IO_EGRESS_PRIVATE_ADDRESSES.key, "deny");
    }
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
            startup_auth,
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

/// Keeps persisted Python UDFs in the SQLite catalog named by `krishiv.connect.pythonFunctions.catalog` (P3-T072c).
///
/// # Errors
/// The file cannot be opened, or the key is set in a build without the `python-function-catalog` feature.
fn with_function_catalog(
    manager: SessionManager,
    config: &Config,
) -> Result<SessionManager, String> {
    let Some(path) = config
        .get(&krishiv_common::config::CONNECT_PYTHON_FUNCTIONS_CATALOG)
        .and_then(krishiv_common::config::ConfigValue::as_str)
    else {
        return Ok(manager);
    };
    #[cfg(feature = "python-function-catalog")]
    {
        let catalog = krishiv_catalogs::sqlite::SqliteCatalog::open(
            "python_functions",
            std::path::Path::new(path),
            false,
        )
        .map_err(|e| format!("cannot open the Python function catalog: {e}"))?;
        Ok(
            manager.with_function_store(Arc::new(
                krishiv_connect_server::SqliteFunctionStore::new(Arc::new(catalog)),
            )),
        )
    }
    #[cfg(not(feature = "python-function-catalog"))]
    {
        let _ = (manager, path);
        Err("krishiv.connect.pythonFunctions.catalog needs a build with the `python-function-catalog` feature".to_owned())
    }
}

/// The declared catalogs' setup in server mode, with the factories this build compiles in.
fn catalog_setup() -> CatalogSetup {
    let setup = CatalogSetup::new(DeclMode::Server);
    #[cfg(feature = "catalog-iceberg-rest")]
    let setup = setup.factory(Arc::new(krishiv_catalogs::declared::IcebergRestFactory));
    setup
}

/// A setting `krishiv server` reads itself: authentication (`krishiv.auth.*`, Spark's token key)
/// and the listeners' TLS (`krishiv.server.tls.*`).
pub(crate) fn is_listener_key(key: &str) -> bool {
    is_auth_key(key) || key.starts_with("krishiv.server.tls.")
}

/// `GetSqlInfo`'s server facts (design P3-T101 §5): the version, the parser's keywords and the
/// functions Krishiv implements of Spark's math (numeric), string, datetime and misc (system)
/// documentation groups.
#[cfg(feature = "flight-sql")]
fn flight_server_info() -> krishiv_flight_sql::ServerInfo {
    use krishiv_functions::registry::{FunctionGroup, builtins, spark_function_groups};
    let implemented: std::collections::HashSet<&str> =
        builtins().functions().iter().map(|d| d.name).collect();
    let mut info = krishiv_flight_sql::ServerInfo {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        keywords: krishiv_sql::SQL_KEYWORDS
            .iter()
            .map(|(k, _)| (*k).to_owned())
            .collect(),
        ..krishiv_flight_sql::ServerInfo::default()
    };
    for (name, group) in spark_function_groups().filter(|(n, _)| implemented.contains(n)) {
        let list = match group {
            FunctionGroup::Math => &mut info.numeric_functions,
            FunctionGroup::String => &mut info.string_functions,
            FunctionGroup::Datetime => &mut info.datetime_functions,
            FunctionGroup::Misc => &mut info.system_functions,
            _ => continue,
        };
        list.push(name.to_owned());
    }
    info
}

/// The Flight SQL service for `connect`'s listener and its sweep task, unless
/// `krishiv.flightSql.enabled` is off (design P3-T101 §2, owner item 6: on by default when
/// compiled in).
#[cfg(feature = "flight-sql")]
fn flight_sql(
    config: &Config,
    manager: &Arc<SessionManager>,
    connect: &ConnectServer,
    auth: ConnectAuth,
) -> Result<
    Option<(
        krishiv_flight_sql::FlightSqlServer,
        krishiv_flight_sql::SweepGuard,
    )>,
    String,
> {
    if !flag(config, &krishiv_common::config::FLIGHT_SQL_ENABLED, true) {
        return Ok(None);
    }
    let info = flight_server_info();
    let flight = krishiv_flight_sql::FlightSqlServer::new(
        Arc::clone(manager),
        Arc::new(auth),
        connect.draining(),
        connect.response_budget().clone(),
        &info,
        krishiv_flight_sql::FlightSqlConfig::from_config(config),
    )
    .map_err(|e| e.to_string())?;
    let interval = ConnectServerConfig::from_config(config).maintenance_interval;
    let sweeper = flight.spawn_sweeper(interval);
    Ok(Some((flight, sweeper)))
}

fn flag(config: &Config, entry: &krishiv_common::config::ConfigEntry, default: bool) -> bool {
    config
        .get(entry)
        .and_then(ConfigValue::as_bool)
        .unwrap_or(default)
}

fn duration(
    config: &Config,
    entry: &krishiv_common::config::ConfigEntry,
    default: Duration,
) -> Duration {
    config
        .get(entry)
        .and_then(ConfigValue::as_duration)
        .unwrap_or(default)
}

#[expect(
    clippy::print_stdout,
    reason = "CLI binary: the addresses and the token are its output"
)]
async fn run_listeners(
    runtime: Arc<EngineRuntime>,
    config: &Config,
    startup_auth: Vec<(String, String)>,
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
    let manager = if authenticated {
        SessionManager::new(caller_sessions(base, config, Arc::clone(&denials))?, limits)
    } else {
        refuse_unauthenticated_providers(config)?;
        SessionManager::from_base(base, limits)
    };
    // One Python UDF worker pool per server (P3-T014): every Connect session's UDFs run on it, in
    // workers of their own owner, once the session passed the `Usage` check on `System("udf")`.
    // Starting it writes the worker package (off the async workers); the listed interpreters are inspected
    // on a thread of the pool's, so the listeners below open without waiting for Python to import its
    // libraries; the first UDF call waits for that inspection.
    #[cfg(not(target_os = "linux"))]
    let udf_workers: Option<UdfWorkersProvider> = None;
    #[cfg(target_os = "linux")]
    let (manager, udf_workers) = {
        let pool_config = krishiv_udf_pool::PoolConfig::from_config(config);
        let pool = tokio::task::spawn_blocking(move || krishiv_udf_pool::Pool::new(pool_config))
            .await
            .map_err(|e| format!("cannot start the Python UDF worker pool: {e}"))?
            .map_err(|e| format!("cannot start the Python UDF worker pool: {e}"))?;
        // The console's Resources page lists the workers through a view that does not keep the pool.
        let udf_workers = Some(udf_workers_of(pool.monitor()));
        (manager.with_udf_host(Arc::new(pool)), udf_workers)
    };
    let manager = with_function_catalog(manager, config)?;
    let manager = Arc::new(manager);
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
    // `/readyz` answers from the cached checks (design P3-T066 §5.2); the serving flag is
    // read on every probe, so a drain flips it at once.
    // `auth` always fails readiness; `krishiv.server.readiness.required` adds shared dependencies.
    let mut required = vec!["auth".to_owned()];
    required.extend(
        config
            .get(&SERVER_READINESS_REQUIRED)
            .and_then(ConfigValue::as_str)
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(str::to_owned),
    );
    // Only the checks that are wired can be required; a name with no check would never fail.
    if let Some(unknown) = required.iter().find(|n| n.as_str() != "auth") {
        return Err(format!(
            "krishiv.server.readiness.required names `{unknown}`, which has no readiness check yet"
        ));
    }
    let interval = duration(config, &SERVER_READINESS_INTERVAL, Duration::from_secs(10));
    if interval < Duration::from_secs(1) {
        return Err("krishiv.server.readiness.interval must be at least 1s".to_owned());
    }
    let threshold = config
        .get(&SERVER_READINESS_FAILURE_THRESHOLD)
        .and_then(ConfigValue::as_int)
        .map_or(Ok(3), u32::try_from)
        .map_err(|_| "krishiv.server.readiness.failureThreshold is too large".to_owned())?;
    let readiness = krishiv_server::readiness::Readiness::with_policy(
        Arc::clone(&ready),
        vec![Arc::new(krishiv_server::readiness::AuthReady(Arc::clone(&auth))) as Arc<_>],
        interval,
        duration(config, &SERVER_READINESS_TIMEOUT, Duration::from_secs(2)),
        threshold,
        required,
    );
    let checker = readiness.spawn();
    // The drain controller (design P3-T066 §5.1) decides when the listeners stop: once held work
    // is done or the grace period and the stop bound passed. The Connect server then has nothing
    // left to wait for, so its own grace is zero.
    let grace = duration(config, &SERVER_SHUTDOWN_TIMEOUT, Duration::from_secs(30));
    let drain = Arc::new(krishiv_server::drain::DrainController::new(Arc::clone(
        &ready,
    )));
    let (stop_tx, stop_rx) = watch::channel(false);
    let drainer = Arc::new(Drainer {
        drain: Arc::clone(&drain),
        runtime: Arc::clone(&runtime),
        manager: Arc::clone(&manager),
        stop: stop_tx,
        driving: AtomicBool::new(false),
        extra_held: std::sync::OnceLock::new(),
        change: std::sync::Mutex::new(()),
    });
    let ops = Arc::new(drainer.ops(Arc::clone(&readiness), grace));
    let late_metrics = LateMetrics::default();
    let app = admin_router(
        &runtime,
        config,
        guard,
        authenticated.then(|| Arc::clone(&auth)),
        &ops,
        startup_auth,
        admin_bound,
        tls.is_some(),
        Arc::clone(&late_metrics),
        udf_workers,
    )?;
    let mut connect_config = ConnectServerConfig::from_config(config);
    connect_config.shutdown_timeout = Duration::ZERO;
    let mut connect = ConnectServer::new(Arc::clone(&manager), connect_config)
        .with_draining(drain.refusing_flag());
    // One `ConnectAuth` per service on the listener, over the one `ServerAuth` (P3-T101 §2).
    let connect_auth = || {
        if authenticated {
            ConnectAuth::new(Arc::clone(&auth) as Arc<_>)
                .with_denials(Arc::clone(&denials))
                .with_impersonation(Arc::clone(&auth))
        } else {
            ConnectAuth::off()
        }
    };
    if authenticated {
        connect = connect.with_auth(connect_auth());
    }
    if let Some(tls) = &tls {
        connect = connect.with_tls(&tls.server_config());
    }
    #[cfg(feature = "flight-sql")]
    let flight = flight_sql(config, &manager, &connect, connect_auth())?;
    #[cfg(feature = "flight-sql")]
    if let Some((flight, _)) = &flight {
        connect = connect.with_routes(flight.routes());
        // `krishiv.flight_sql.requests` in the scrape (design §7, review m6).
        let counted = flight.clone();
        let _ = late_metrics.set(Box::new(move || counted.metrics()));
        // The drain waits for unfetched tickets and running fetches too (design P3-T066 §5.1).
        let held = flight.clone();
        let _ = drainer.extra_held.set(Box::new(move || held.held()));
    }

    // Installed before the addresses are printed: a SIGTERM right after
    // readiness must start the drain, not kill the process.
    let handlers = Signals::install()
        .map_err(|e| format!("cannot install the SIGTERM/SIGINT handlers: {e}"))?;
    let signals = tokio::spawn(watch_signals(handlers, Arc::clone(&drainer), grace));
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
    #[cfg(feature = "flight-sql")]
    if flight.is_some() {
        println!("krishiv server: Arrow Flight SQL on the same port");
    }
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
    // A SIGTERM during start-up may already have started a drain: do not undo its readiness.
    drainer.mark_serving_unless_draining(&ready);
    let served = connect
        .serve(connect_listener, stopped(stop_rx))
        .await
        .map_err(|e| e.to_string());
    ready.store(false, Ordering::Release);
    // `serve` closed every session: Flight's table and handles go too (design §16 m7); the sweep
    // task ends with its guard.
    #[cfg(feature = "flight-sql")]
    if let Some((flight, sweeper)) = flight {
        flight.close_all();
        drop(sweeper);
    }
    signals.abort();
    if let Some(checker) = checker {
        checker.abort();
    }
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

/// What the console's Resources page reads of the Python UDF worker pool: its workers now, and nothing once
/// the pool is gone (the server is stopping).
#[cfg(target_os = "linux")]
fn udf_workers_of(pool: krishiv_udf_pool::PoolMonitor) -> UdfWorkersProvider {
    use krishiv_server::udf_workers::{UdfWorkerOwner, UdfWorkers};
    let n = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
    Arc::new(move || {
        pool.snapshot().map(|now| UdfWorkers {
            live: n(now.live),
            idle: n(now.idle),
            max_total: n(now.max_total),
            max_per_session: n(now.max_per_session),
            max_per_tenant: n(now.max_per_tenant),
            owners: now
                .owners
                .into_iter()
                .map(|o| UdfWorkerOwner {
                    principal: o.owner.principal,
                    session_id: o.owner.session,
                    tenant: o.owner.tenant,
                    live: n(o.live),
                    idle: n(o.idle),
                })
                .collect(),
        })
    })
}

/// Samples another service adds to `/metrics` once it exists (the Flight SQL service is built
/// after the admin router).
type LateMetrics =
    Arc<std::sync::OnceLock<Box<dyn Fn() -> krishiv_session::MetricsSnapshot + Send + Sync>>>;

#[expect(
    clippy::too_many_arguments,
    reason = "the admin listener's parts, each built by `run`"
)]
fn admin_router(
    runtime: &Arc<EngineRuntime>,
    config: &Config,
    guard: Arc<dyn Guard>,
    auth: Option<Arc<ServerAuth>>,
    ops: &Arc<krishiv_server::admin_v1::DrainOps>,
    startup_auth: Vec<(String, String)>,
    bound: SocketAddr,
    tls: bool,
    late: LateMetrics,
    udf_workers: Option<UdfWorkersProvider>,
) -> Result<axum::Router, String> {
    let authenticated = auth.is_some();
    let denials = auth.as_ref().map(|a| a.denials());
    let metrics_on = flag(config, &METRICS_ENABLED, false);
    // With authentication configured every scrape authenticates (ADR 0046 §3c: a viewer
    // service principal); `krishiv.server.metrics.public` opens it only without.
    let public = metrics_on && !authenticated && !metrics_need_token(config, bound);
    // Denials of the admin API count into the same counters as every other surface.
    let admin_denials = auth.as_ref().map_or_else(
        || Arc::new(krishiv_session::auth::DeniedCounts::default()),
        |a| a.denials(),
    );
    let snapshot_runtime = Arc::clone(runtime);
    let metrics = krishiv_server::metrics::metrics_route_guarded(
        metrics_on,
        // With authentication on, the auth counters ride along (ADR 0046 §8).
        Arc::new(move || {
            let mut snapshot = snapshot_runtime.metrics();
            if let Some(auth) = &auth {
                snapshot.merge(&auth.metrics());
            }
            if let Some(more) = late.get() {
                snapshot.merge(&more());
            }
            snapshot
        }),
        (!public).then(|| Arc::clone(&guard)),
    );
    // `krishiv.admin.v1`: gRPC and REST on the admin listener, behind the same guard.
    let allowed_hosts = config
        .get(&krishiv_common::config::SERVER_ALLOWED_HOSTS)
        .and_then(krishiv_common::config::ConfigValue::as_str)
        .map(|hosts| {
            hosts
                .split(',')
                .map(str::trim)
                .filter(|h| !h.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let alias_guard = Arc::clone(&guard);
    let admin_v1 = krishiv_server::admin_v1::router(
        Arc::new(
            krishiv_server::admin_v1::SessionAdmin::new(Arc::clone(runtime))
                .with_denials(admin_denials)
                .with_startup_auth(startup_auth)
                .with_ops(Arc::clone(ops)),
        ),
        Arc::clone(&guard),
        krishiv_server::admin_v1::AdminConfig::for_listener(bound.ip(), allowed_hosts),
    );
    let base = if cfg!(feature = "ui") && flag(config, &UI_ENABLED, true) {
        let source = krishiv_server::session_source::SessionSource::new(Arc::clone(runtime))
            .map_err(|e| e.to_string())?;
        let mut source = with_stored_history(source, config);
        if let Some(read) = udf_workers {
            source = source.with_udf_workers(read);
        }
        if let Some(denials) = denials {
            // The editor runs a statement as the authenticated caller, in the session a Spark Connect session of
            // that caller gets: the same role matrix and path confinement.
            let base = Session::builder()
                .runtime(Arc::clone(runtime))
                .build()
                .map_err(|e| e.to_string())?;
            // The editor's sessions are UI sessions: their denials count under the `ui` surface, where the
            // statement arrived (P3-T101 §2 H3).
            let factory = caller_sessions(base, config, denials)?;
            source = source.with_caller_sessions(Arc::new(move |caller: &Authenticated| {
                let session = factory(caller, Transport::Ui)?;
                // `system.sessions` shows the transport the denials count under.
                session.set_client_info(Transport::Ui, None, None, None);
                Ok(session)
            }));
        }
        // The `Host` allowlist (DNS rebinding, T-W5): on a loopback address the loopback names and
        // `krishiv.ui.allowedHosts`; on another address only when names are configured. The environment variable
        // `KRISHIV_UI_ALLOWED_HOSTS` adds names to either.
        let configured = config
            .get(&UI_ALLOWED_HOSTS)
            .and_then(ConfigValue::as_str)
            .unwrap_or("")
            .to_owned();
        let from_env = std::env::var("KRISHIV_UI_ALLOWED_HOSTS").unwrap_or_default();
        // Who may frame the embeddable views (T071): invalid origins stop the start, as a bad host would not.
        let origins = krishiv_server::embed::parse_origins(
            config
                .get(&UI_ALLOWED_ORIGINS)
                .and_then(ConfigValue::as_str)
                .unwrap_or(""),
        )?;
        let embed = krishiv_server::embed::EmbedPolicy::new(
            origins,
            config
                .get(&UI_EMBED_HIDE_ATTRIBUTION)
                .and_then(ConfigValue::as_bool)
                .unwrap_or(false),
            match config.get(&UI_EMBED_LINK_TTL) {
                Some(ConfigValue::Duration(d)) => *d,
                _ => match UI_EMBED_LINK_TTL.default {
                    Some(ConfigValue::Duration(d)) => d,
                    _ => std::time::Duration::from_secs(300),
                },
            },
        );
        let policy = krishiv_server::pages::UiPolicy::for_listener(
            bound.ip().is_loopback(),
            configured.split(',').chain(from_env.split(',')),
        )
        // The session cookie holds the bearer token: over TLS it never travels in the clear.
        .secure_cookie(tls)
        .with_embed(embed);
        krishiv_server::internal::router_with_ui_policy(Arc::new(source), guard, policy)
    } else {
        krishiv_server::api_router()
    };
    let app = base
        .merge(krishiv_server::admin::health_routes_cached(Arc::clone(
            &ops.readiness,
        )))
        .merge(admin_v1)
        .merge(metrics);
    // With authentication configured the deprecated version alias needs a credential like
    // `GetVersion` (design P3-T066 D6, ADR 0046 section 3c).
    let app = if authenticated {
        krishiv_server::admin_v1::guard_version_alias(app, alias_guard)
    } else {
        app
    };
    Ok(krishiv_server::admin::with_limits(
        app,
        krishiv_server::admin::HttpLimits::default(),
    ))
}

/// `source` with the history of earlier runs of the server, read from the store this run writes to
/// (`krishiv.history.path`). A store that does not read costs the old history, not the server.
#[expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")]
fn with_stored_history(
    source: krishiv_server::session_source::SessionSource,
    config: &Config,
) -> krishiv_server::session_source::SessionSource {
    let Some(dir) = config
        .get(&krishiv_common::config::HISTORY_PATH)
        .and_then(ConfigValue::as_str)
        .filter(|p| !p.is_empty())
    else {
        return source;
    };
    match krishiv_session::read_profile_history(std::path::Path::new(dir)) {
        Ok(stored) => source.with_stored_history(&stored),
        Err(e) => {
            eprintln!(
                "warning: the stored query history was not read ({e}); the console lists this run only"
            );
            source
        }
    }
}

/// Whether `/metrics` is on and behind the admin token: it is open only on
/// a loopback address, and only with `krishiv.server.metrics.public`.
fn metrics_need_token(config: &Config, bound: SocketAddr) -> bool {
    flag(config, &METRICS_ENABLED, false)
        && !(flag(config, &SERVER_METRICS_PUBLIC, false) && bound.ip().is_loopback())
}

/// Spark's default-catalog key (`krishiv.catalog.default` is a declaration key).
const SPARK_DEFAULT_CATALOG: &str = "spark.sql.defaultCatalog";

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

/// The first SIGTERM or SIGINT starts the drain (`Drain { grace_period = shutdownTimeout,
/// exit_when_drained = true }`, design P3-T066 §5.1a); a second exits at once.
#[expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")]
async fn watch_signals(mut signals: Signals, drainer: Arc<Drainer>, grace: Duration) {
    signals.next().await;
    eprintln!("krishiv server: draining (a second signal exits at once)");
    signal_action(&format!(
        "grace {}s, exit_when_drained true",
        grace.as_secs()
    ));
    drainer.start(grace, true);
    signals.next().await;
    eprintln!("krishiv server: second signal, exiting");
    signal_action("second signal, exit at once");
    std::process::exit(130);
}

/// The action record (`krishiv::admin_action`) for a drain the operating system asked for, so the
/// log shows every drain, not only the ones made through `admin.v1` (design P3-T066 §5.6). The
/// caller is the process's own signal handler: no principal, no secret.
fn signal_action(target: &str) {
    krishiv_server::admin_v1::action::record(&krishiv_server::admin_v1::action::Action {
        principal: "signal",
        role: "none",
        auth_method: "none",
        rpc: "Drain",
        transport: "signal",
        target,
        own: false,
        request_id: "",
        result: "ok",
        error_class: "",
        replay: false,
    });
}

/// What starts, drives and undoes a drain: the signal handler and the admin API's `Drain` /
/// `ResumeServing` share it, so there is one driver.
pub(crate) struct Drainer {
    drain: Arc<krishiv_server::drain::DrainController>,
    runtime: Arc<EngineRuntime>,
    manager: Arc<SessionManager>,
    stop: watch::Sender<bool>,
    driving: AtomicBool,
    /// Held work beyond the manager's (Flight SQL's unfetched tickets and running fetches),
    /// set once the Flight service exists (design P3-T066 §5.1).
    extra_held: std::sync::OnceLock<Box<dyn Fn() -> usize + Send + Sync>>,
    /// Serialises `start` and `resume`: the controller and the engine flag change together.
    change: std::sync::Mutex<()>,
}

impl Drainer {
    /// Starts a drain, or shortens a running one, and makes sure one task drives it: new work is
    /// refused at once, held work gets `grace`, what is left is cancelled with
    /// `CancelReason::Drain`, and the listeners stop (with `exit`) when the controller says so.
    pub(crate) fn start(self: &Arc<Self>, grace: Duration, exit: bool) {
        {
            let _changing = self
                .change
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.runtime.set_draining(true);
            self.drain.start(std::time::Instant::now(), grace, exit);
            // Under the lock the driver retires with, so a driver that is just finishing and
            // this call cannot both assume the other one handles the request.
            if !self.driving.swap(true, Ordering::AcqRel) {
                tokio::spawn(Arc::clone(self).drive());
            }
        }
    }

    /// Back to serving; refused once the process is exiting.
    fn resume(&self) -> Result<(), krishiv_server::drain::Exiting> {
        let _changing = self
            .change
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.drain.resume()?;
        self.runtime.set_draining(false);
        Ok(())
    }

    /// Sets the readiness flag unless a drain has started, serialized with `start`/`resume`.
    fn mark_serving_unless_draining(&self, ready: &AtomicBool) {
        let _changing = self
            .change
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.drain.status().state == krishiv_server::drain::DrainState::Serving {
            ready.store(true, Ordering::Release);
        }
    }

    fn load(&self) -> (i64, i64) {
        let queries = i64::try_from(self.manager.running_queries()).unwrap_or(i64::MAX);
        let sessions = i64::try_from(self.manager.counts().live).unwrap_or(i64::MAX);
        (queries, sessions)
    }

    /// What the admin API's `Drain`, `ResumeServing`, `GetDrainStatus` and `GetReadiness` act on.
    pub(crate) fn ops(
        self: &Arc<Self>,
        readiness: Arc<krishiv_server::readiness::Readiness>,
        default_grace: Duration,
    ) -> krishiv_server::admin_v1::DrainOps {
        let (start, resume, load) = (Arc::clone(self), Arc::clone(self), Arc::clone(self));
        krishiv_server::admin_v1::DrainOps {
            drain: Arc::clone(&self.drain),
            readiness,
            start: Box::new(move |grace, exit| start.start(grace, exit)),
            resume: Box::new(move || resume.resume()),
            load: Box::new(move || load.load()),
            default_grace,
        }
    }

    /// One decision of the drain: the load sample, the controller's step and its effect (the
    /// cancellation, the retirement of the driver) happen under `change`, so a `resume` or a
    /// new `start` cannot slip between the sample and what it decides.
    fn drive_once(&self) -> Option<krishiv_server::drain::DrainStep> {
        use krishiv_server::drain::{DrainState, DrainStep};
        let _changing = self
            .change
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let held = self.manager.busy() + self.extra_held.get().map_or(0, |f| f());
        let step = self.drain.step(std::time::Instant::now(), held);
        match step {
            DrainStep::Wait => {
                if self.drain.status().state == DrainState::Serving {
                    // Resumed: retire. A later `start` sees the flag cleared and spawns the
                    // next driver itself.
                    self.driving.store(false, Ordering::Release);
                    return None;
                }
            }
            DrainStep::CancelRemaining => {
                let n = self
                    .runtime
                    .cancel_all(krishiv_session::CancelReason::Drain);
                Self::report_cancelled(n);
            }
            DrainStep::Drained { exit } => {
                self.driving.store(false, Ordering::Release);
                if exit {
                    let _ = self.stop.send(true);
                }
                return None;
            }
        }
        Some(step)
    }

    #[expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")]
    fn report_cancelled(n: usize) {
        eprintln!("krishiv server: drain deadline, cancelled {n} queries");
    }

    async fn drive(self: Arc<Self>) {
        while self.drive_once().is_some() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
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

    /// `GetSqlInfo` lists the parser's keywords and the implemented functions by group (review m9).
    #[cfg(feature = "flight-sql")]
    #[test]
    fn flight_server_info_lists_keywords_and_functions() {
        let info = flight_server_info();
        assert!(info.keywords.iter().any(|k| k == "SELECT"));
        for (list, name) in [
            (&info.numeric_functions, "abs"),
            (&info.string_functions, "upper"),
            (&info.datetime_functions, "date_add"),
            (&info.system_functions, "current_user"),
        ] {
            assert!(list.iter().any(|f| f == name), "{name} in {list:?}");
        }
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

    /// G6 review r3: one runtime asked with other `external` settings gets a provider of its own;
    /// the same settings share one.
    #[tokio::test]
    async fn the_external_provider_is_shared_per_runtime_and_settings() {
        use krishiv_server::access_external::{ExternalAccess, ExternalConfig};
        let runtime = EngineRuntime::builder().build().unwrap();
        let settings = |endpoint: &str| {
            ExternalConfig::from_config(&config(&[("krishiv.access.external.endpoint", endpoint)]))
                .unwrap()
        };
        let a = settings("http://127.0.0.1:1");
        let b = settings("http://127.0.0.1:2");
        let get = |c: &ExternalConfig| {
            shared_external(&runtime, c, || {
                ExternalAccess::new(c).map_err(|e| e.to_string())
            })
            .unwrap()
        };
        let first = get(&a);
        assert!(Arc::ptr_eq(&first, &get(&a)));
        assert!(!Arc::ptr_eq(&first, &get(&b)));
        // Another runtime with the same settings gets its own too.
        let other = EngineRuntime::builder().build().unwrap();
        let theirs = shared_external(&other, &a, || {
            ExternalAccess::new(&a).map_err(|e| e.to_string())
        })
        .unwrap();
        assert!(!Arc::ptr_eq(&first, &theirs));
    }

    /// Design P3-T066 §5.1: a drain refuses new queries at once, gives a stuck one the grace
    /// period, cancels it with the drain reason (`KRISHIV_QUERY_KILLED`) and then stops.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_drain_cancels_a_stuck_query_at_the_deadline_and_then_stops() {
        let runtime = EngineRuntime::builder().build().unwrap();
        let base = Session::builder()
            .runtime(Arc::clone(&runtime))
            .build()
            .unwrap();
        let manager = Arc::new(SessionManager::from_base(
            base,
            krishiv_connect_server::SessionLimits::default(),
        ));
        // The query runs in a session the manager holds, as every Connect query does.
        let caller = krishiv_session::auth::Authenticated::new(
            krishiv_common::principal::Principal::local_user(),
            None,
            krishiv_session::auth::Method::None,
        );
        let entry = manager
            .get_or_create(&caller, "00112233-4455-6677-8899-aabbccddeeff", None, "u")
            .unwrap();
        let session = entry.session().clone();
        let df = session
            .sql("SELECT count(*) FROM range(0, 100000000000, 1, 4)")
            .await
            .unwrap();
        let query = tokio::spawn(async move { df.collect().await });
        let registered = tokio::time::timeout(Duration::from_secs(20), async {
            while session.running_queries() == 0 {
                if query.is_finished() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(registered.is_ok(), "the query never registered");
        assert!(
            !query.is_finished(),
            "the query ended early: {:?}",
            query.await
        );
        let ready = Arc::new(AtomicBool::new(true));
        let drain = Arc::new(krishiv_server::drain::DrainController::new(Arc::clone(
            &ready,
        )));
        let (stop_tx, mut stop_rx) = watch::channel(false);
        let drainer = Arc::new(Drainer {
            drain: Arc::clone(&drain),
            runtime: Arc::clone(&runtime),
            manager: Arc::clone(&manager),
            stop: stop_tx,
            driving: AtomicBool::new(false),
            extra_held: std::sync::OnceLock::new(),
            change: std::sync::Mutex::new(()),
        });
        drainer.start(Duration::from_millis(300), true);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!ready.load(Ordering::SeqCst), "readiness flips first");
        let refused = session.sql("SELECT 1").await.unwrap().collect().await;
        assert_eq!(
            refused.unwrap_err().error_class().name(),
            "KRISHIV_SERVER_DRAINING"
        );
        tokio::time::timeout(Duration::from_secs(20), stop_rx.wait_for(|stop| *stop))
            .await
            .expect("the drain finishes")
            .unwrap();
        let e = tokio::time::timeout(Duration::from_secs(20), query)
            .await
            .expect("the cancelled query ends")
            .unwrap()
            .unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_QUERY_KILLED");
    }
}

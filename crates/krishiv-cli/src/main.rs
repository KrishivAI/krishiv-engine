// SPDX-License-Identifier: Apache-2.0
//! Krishiv command line binary (layer L7).

// `deny` rather than `forbid`: the allocator test must call a C API (see `tests`).
#![deny(unsafe_code)]
#![cfg_attr(
    feature = "mcp",
    expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")
)]

mod archive;
mod bundle;
mod check_compat;
mod common;
mod complete;
mod deadline;
#[cfg(feature = "server")]
mod doctor;
mod editor;
mod format;
mod highlight;
mod history;
mod repl;
#[cfg(feature = "server")]
mod server;
mod sql;
#[cfg(feature = "ui")]
mod ui;

use clap::Parser;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// `--version` output: the crate version plus build info the bench harness reads
/// (`allocator=<name>`, one `key=value` per line).
#[cfg(feature = "mimalloc")]
const LONG_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\nallocator=mimalloc");
/// `--version` output: the crate version plus build info the bench harness reads
/// (`allocator=<name>`, one `key=value` per line).
#[cfg(not(feature = "mimalloc"))]
const LONG_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\nallocator=system");

/// Krishiv: a Spark-compatible batch and streaming engine.
#[derive(Parser)]
#[command(name = "krishiv", version, long_version = LONG_VERSION)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Report Spark functions in `.py`, `.sql` and `.ipynb` files that Krishiv does not fully support.
    #[command(name = "check-compat")]
    CheckCompat(check_compat::CheckArgs),
    /// Run SQL from `-e`, `-f` or stdin and print the result.
    Sql(sql::SqlArgs),
    /// Collect a redacted diagnostic archive.
    ///
    /// Per-statement failures are recorded in the bundle and the exit status
    /// stays 0; only I/O or setup failures exit non-zero (unlike
    /// `krishiv sql`, which exits 1 on a query error).
    #[command(name = "debug-bundle")]
    DebugBundle(bundle::BundleArgs),
    /// Serve read-only MCP tools over stdio (design P2-T095).
    #[cfg(feature = "mcp")]
    Mcp {
        /// Directory whose parquet files are the tables; files outside it cannot be read.
        #[arg(long)]
        data: std::path::PathBuf,
        /// Run even as uid 0.
        #[arg(long)]
        allow_root: bool,
    },
    /// Run Krishiv single-node: serve Spark Connect (port 15002) and the admin API (port 8080; the web UI with `ui`).
    #[cfg(feature = "server")]
    Server(server::ServerArgs),
    /// Check a `krishiv server` configuration without starting it: settings, authentication,
    /// TLS files, ports, directories and path allow lists; exit 1 when a check fails.
    #[cfg(feature = "server")]
    Doctor(doctor::DoctorArgs),
    /// Open stored query profiles in the web UI with no server and no engine (`krishiv.history.path` stores).
    #[cfg(feature = "ui")]
    Ui(ui::UiArgs),
}

/// `KRISHIV_LOG=json` turns on JSON log records on stderr (query, session and trace ids on each;
/// at most `KRISHIV_LOG_RATE` events per callsite per second, default 50). Anything else: off.
#[cfg(feature = "telemetry-logs")]
#[expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")]
fn install_logs_from_env() {
    if std::env::var("KRISHIV_LOG").is_ok_and(|v| v.eq_ignore_ascii_case("json")) {
        let rate = std::env::var("KRISHIV_LOG_RATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);
        if let Err(e) = krishiv_session::logging::install_json_logs(rate) {
            eprintln!("krishiv: JSON logs not enabled: {e}");
        }
    }
}

fn main() -> std::process::ExitCode {
    // First, so a CPU older than the build targets gets a message instead of SIGILL.
    // `run` holds everything else: the compiler may use the build's CPU level in the
    // prologue of a function with large locals, which must not come before the check.
    krishiv_common::cpu::check_or_exit();
    // `krishiv __udf-worker ...`: the Python UDF worker's sandbox trampoline (P3-T014b). Before
    // anything starts a thread: it confines this thread and replaces the process, returning only
    // on failure.
    #[cfg(any(feature = "connect", feature = "server"))]
    if let Some(code) = krishiv_udf_pool::run_trampoline_if_requested() {
        return code;
    }
    krishiv::register_build_features(&bundle::feature_states());
    #[cfg(feature = "telemetry-logs")]
    install_logs_from_env();
    #[cfg(feature = "mimalloc")]
    install_allocator_policy();
    run();
    std::process::ExitCode::SUCCESS
}

/// Allocator policy is this binary's (it owns the global allocator): under
/// memory pressure, threads return freed memory to the OS. With a configured
/// limit that is not generous, mimalloc also purges promptly (set before
/// `main`, see `krishiv_alloc`). Every limited root pool keeps the resident
/// set within its reservations plus an allowance, the default limit (80% of
/// the container) included: without the probe, memory the pools do not
/// track took a 4 GB container over its cgroup (acceptance SQL-34).
#[cfg(feature = "mimalloc")]
fn install_allocator_policy() {
    krishiv_memory::set_release_hook(krishiv_alloc::install());
    krishiv_memory::set_resident_probe(krishiv_memory::process_resident_bytes);
}

#[inline(never)]
fn run() {
    #[cfg(feature = "tokio-console")]
    krishiv::maybe_init_tokio_console();
    let cli = Cli::parse();
    match cli.command {
        Command::CheckCompat(args) => std::process::exit(check_compat::run(&args)),
        Command::Sql(args) => std::process::exit(sql::run(&args)),
        Command::DebugBundle(args) => std::process::exit(bundle::run(&args)),
        #[cfg(feature = "mcp")]
        Command::Mcp { data, allow_root } => std::process::exit(run_mcp(data, allow_root)),
        // Returns only after the engine runtime is dropped (spill directory removed).
        #[cfg(feature = "server")]
        Command::Server(args) => std::process::exit(server::run(&args)),
        #[cfg(feature = "server")]
        Command::Doctor(args) => std::process::exit(doctor::run(&args)),
        #[cfg(feature = "ui")]
        Command::Ui(args) => std::process::exit(ui::run(&args)),
    }
}

/// `krishiv mcp`: stdio until the client closes it. Exit 0 ok, 1 failure.
#[cfg(feature = "mcp")]
fn run_mcp(data: std::path::PathBuf, allow_root: bool) -> i32 {
    if krishiv_mcp::gate::is_root() && !allow_root {
        eprintln!("krishiv mcp: refusing to run as root; pass --allow-root to override");
        return 1;
    }
    if !data.is_dir() {
        eprintln!("krishiv mcp: --data {} is not a directory", data.display());
        return 1;
    }
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("krishiv mcp: {e}");
            return 1;
        }
    };
    let server = krishiv_mcp::KrishivMcpServer::with_tools(krishiv_mcp::McpConfig::new(data));
    let result = runtime.block_on(async {
        let running = server.serve_stdio().await.map_err(|e| e.to_string())?;
        running
            .waiting()
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("krishiv mcp: {e}");
            1
        }
    }
}

#[cfg(all(test, feature = "mimalloc"))]
#[expect(
    unsafe_code,
    reason = "queries the mimalloc heap through its C API to prove it backs the global allocator"
)]
mod tests {
    #[test]
    fn mimalloc_is_the_global_allocator() {
        let boxed = Box::new([0_u8; 64]);
        let ptr: *const u8 = boxed.as_ptr();
        // SAFETY: `mi_is_in_heap_region` only compares the address with mimalloc's
        // arenas; it never dereferences `ptr`, which is live and valid anyway.
        let in_mimalloc = unsafe { libmimalloc_sys::mi_is_in_heap_region(ptr.cast()) };
        assert!(in_mimalloc, "global allocations must come from mimalloc");
        // SAFETY: `mi_version` takes no arguments and only returns a constant.
        assert!(unsafe { libmimalloc_sys::mi_version() } > 0);
    }

    /// The probe and the hook are installed whatever the limit: this test
    /// process has no limit below `krishiv_alloc`'s generous threshold, the
    /// case that ran without the guard before (acceptance SQL-34).
    #[test]
    fn the_allocator_policy_installs_the_guard_at_a_generous_limit() {
        assert!(
            !krishiv_alloc::bounded(),
            "this test needs a process started without a limit below 4 GiB: unset \
             KRISHIV_MEMORY_LIMIT and run outside a cgroup smaller than that"
        );
        super::install_allocator_policy();
        assert!(krishiv_memory::release_hook_installed());
        assert!(krishiv_memory::resident_probe_installed());
    }
}

// SPDX-License-Identifier: Apache-2.0
//! Krishiv command line binary (layer L7).

// `deny` rather than `forbid`: the allocator test must call a C API (see `tests`).
#![deny(unsafe_code)]
#![cfg_attr(
    feature = "ui",
    expect(
        clippy::print_stdout,
        clippy::print_stderr,
        reason = "CLI binary: stdout/stderr are its interface"
    )
)]
#![cfg_attr(
    all(feature = "mcp", not(feature = "ui")),
    expect(clippy::print_stderr, reason = "CLI binary: stderr is its interface")
)]

mod archive;
mod bundle;
mod common;
mod complete;
mod deadline;
mod format;
mod highlight;
mod history;
mod repl;
mod sql;

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
    /// Serve the web UI and admin API.
    #[cfg(feature = "ui")]
    Server {
        /// Port to listen on (0 picks a free port).
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Address to bind; loopback by default.
        #[arg(long, default_value = "127.0.0.1")]
        host: std::net::IpAddr,
    },
}

fn main() {
    // First, so a CPU older than the build targets gets a message instead of SIGILL.
    // `run` holds everything else: the compiler may use the build's CPU level in the
    // prologue of a function with large locals, which must not come before the check.
    krishiv_common::cpu::check_or_exit();
    // Allocator policy is this binary's (it owns the global allocator):
    // under memory pressure, threads return freed memory to the OS. With a
    // configured limit that is not generous, mimalloc also purges promptly
    // (set before `main`, see `krishiv_alloc`).
    #[cfg(feature = "mimalloc")]
    {
        krishiv_memory::set_release_hook(krishiv_alloc::install());
        // Under that bounded policy the resident set follows the pools, and
        // the pools keep it within their reservations plus an allowance.
        if krishiv_alloc::bounded() {
            krishiv_memory::set_resident_probe(krishiv_memory::process_resident_bytes);
        }
    }
    run();
}

#[inline(never)]
fn run() {
    #[cfg(feature = "tokio-console")]
    krishiv::maybe_init_tokio_console();
    let cli = Cli::parse();
    match cli.command {
        Command::Sql(args) => std::process::exit(sql::run(&args)),
        Command::DebugBundle(args) => std::process::exit(bundle::run(&args)),
        #[cfg(feature = "mcp")]
        Command::Mcp { data, allow_root } => std::process::exit(run_mcp(data, allow_root)),
        #[cfg(feature = "ui")]
        Command::Server { port, host } => {
            let addr = std::net::SocketAddr::new(host, port);
            let result = krishiv_server::run_blocking(addr, |a| {
                println!("krishiv server listening on http://{a}");
            });
            if let Err(e) = result {
                eprintln!("krishiv server: {e}");
                std::process::exit(1);
            }
        }
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
}

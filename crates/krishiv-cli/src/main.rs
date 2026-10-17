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
    #[cfg(feature = "ui")]
    #[command(subcommand)]
    command: Option<Command>,
}

#[cfg(feature = "ui")]
#[derive(clap::Subcommand)]
enum Command {
    /// Serve the web UI and admin API.
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
    let cli = Cli::parse();
    #[cfg(feature = "ui")]
    if let Some(Command::Server { port, host }) = cli.command {
        let addr = std::net::SocketAddr::new(host, port);
        let result = krishiv_server::run_blocking(addr, |a| {
            println!("krishiv server listening on http://{a}");
        });
        if let Err(e) = result {
            eprintln!("krishiv server: {e}");
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "ui"))]
    let _ = cli;
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

// SPDX-License-Identifier: Apache-2.0
//! `krishiv ui --profiles <dir>` (P3-T050): the web UI over the profiles a history store holds
//! (`krishiv.history.path`), with no engine and no Spark Connect listener. Loopback only, behind a start-up
//! token printed once; nothing live is shown and nothing can be cancelled.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use krishiv_server::session_source::ProfilesSource;

/// The command's arguments.
#[derive(Debug, clap::Args)]
pub(crate) struct UiArgs {
    /// The history directory (`krishiv.history.path`) whose stored profiles to open.
    #[arg(long)]
    profiles: PathBuf,
    /// The port to serve on (0 picks a free port).
    #[arg(long, default_value_t = 0)]
    port: u16,
}

/// Serves until the process is stopped; the exit code.
#[expect(clippy::print_stdout, clippy::print_stderr, reason = "CLI binary")]
pub(crate) fn run(args: &UiArgs) -> i32 {
    let go = || -> Result<(), String> {
        if !args.profiles.is_dir() {
            return Err(format!("{} is not a directory", args.profiles.display()));
        }
        let source = ProfilesSource::open(&args.profiles).map_err(|e| e.to_string())?;
        let token = krishiv_server::internal::random_token().map_err(|e| e.to_string())?;
        let app = krishiv_server::internal::router_with_ui(Arc::new(source), &token);
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), args.port);
        krishiv_server::run_blocking_with(addr, app, |bound| {
            println!("krishiv ui: open http://{bound}/?token={token}");
        })
        .map_err(|e| e.to_string())
    };
    match go() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("krishiv ui: {e}");
            1
        }
    }
}

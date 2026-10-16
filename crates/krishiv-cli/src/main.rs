//! Krishiv command line binary (layer L7).

#![forbid(unsafe_code)]

use clap::Parser;

/// Krishiv: a Spark-compatible batch and streaming engine.
#[derive(Parser)]
#[command(name = "krishiv", version)]
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

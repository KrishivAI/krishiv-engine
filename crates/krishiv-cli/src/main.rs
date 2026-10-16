//! Krishiv command line binary (layer L7).

#![forbid(unsafe_code)]

use clap::Parser;

/// Krishiv: a Spark-compatible batch and streaming engine.
#[derive(Parser)]
#[command(name = "krishiv", version)]
struct Cli {}

fn main() {
    let _cli = Cli::parse();
}

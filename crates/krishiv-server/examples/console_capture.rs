// SPDX-License-Identifier: Apache-2.0
//! Serves the web UI over an embedded engine runtime after running a set of TPC-H queries through it,
//! so the pages show real query history, plans and resources (used by the website's console captures).
//!
//! `cargo run --release -p krishiv-server --features session --example console_capture -- DATA_DIR [PORT]`
//!
//! `DATA_DIR` holds the TPC-H Parquet files (`lineitem.parquet`, ...). Prints `URL=<url with token>` once the
//! queries have run, then serves until killed.
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    clippy::unwrap_used,
    reason = "a command-line tool"
)]

use std::sync::Arc;

use krishiv_server::session_source::EmbeddedUi;
use krishiv_session::{EngineRuntime, Session};

const TABLES: [&str; 8] = [
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let data = args.next().expect("usage: console_capture DATA_DIR [PORT]");
    let port: u16 = args.next().map_or(0, |p| p.parse().unwrap());
    let rt = EngineRuntime::builder().build().unwrap();
    let s = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    for t in TABLES {
        let sql =
            format!("CREATE TEMPORARY VIEW {t} AS SELECT * FROM parquet.`{data}/{t}.parquet`");
        s.sql(&sql).await.unwrap().collect().await.unwrap();
    }
    let queries = std::env::var("CONSOLE_QUERIES").unwrap_or_default();
    for path in queries.split(':').filter(|p| !p.is_empty()) {
        let text = std::fs::read_to_string(path).unwrap();
        let df = s.sql(&text).await.unwrap();
        df.collect().await.unwrap();
    }
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let ui = EmbeddedUi::start(Arc::clone(&rt), addr).unwrap();
    println!("URL={}", ui.url());
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    }
}

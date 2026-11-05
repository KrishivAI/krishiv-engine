// SPDX-License-Identifier: Apache-2.0
//! Runs the TPC-H queries over Parquet tables through the facade API and writes every result as JSON
//! text (the Rust entry point of the P2a correctness check, `bench/scripts/p2a_correctness.py`).
//!
//! `cargo run --release -p krishiv --example tpch_correctness -- DATA_DIR QUERIES_DIR OUT.json [qNN...]`
//!
//! `DATA_DIR` holds `<table>.parquet`, `QUERIES_DIR` holds `qNN.sql`. The output maps each query to
//! `{"rows": [[cell, ...], ...], "seconds": s}` or `{"error": "..."}`; a cell is its text as the CLI prints
//! it, `null` for SQL NULL.
#![expect(
    clippy::print_stderr,
    clippy::unwrap_used,
    reason = "a command-line example reports on stderr and stops on a broken setup"
)]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use krishiv::Session;
use krishiv::agent::{cell, session_zone};
use krishiv::arrow::array::Array;

const TABLES: [&str; 8] = [
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

fn json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn run_query(
    session: &Session,
    zone: &krishiv::agent::ZoneRef,
    sql: &str,
) -> Result<Vec<Vec<Option<String>>>, krishiv::Error> {
    let df = futures::executor::block_on(session.sql(sql))?;
    let mut rows = Vec::new();
    for batch in futures::executor::block_on(df.collect())? {
        for i in 0..batch.num_rows() {
            rows.push(
                batch
                    .columns()
                    .iter()
                    .map(|c| (!c.is_null(i)).then(|| cell(c.as_ref(), i, "", zone)))
                    .collect(),
            );
        }
    }
    Ok(rows)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [data, queries, out, only @ ..] = args.as_slice() else {
        eprintln!("usage: tpch_correctness DATA_DIR QUERIES_DIR OUT.json [qNN...]");
        std::process::exit(2);
    };
    let session = Session::new().unwrap();
    let zone = session_zone(None).unwrap();
    for table in TABLES {
        let path = PathBuf::from(data).join(format!("{table}.parquet"));
        let df = session.read().parquet([path.to_str().unwrap()]);
        futures::executor::block_on(df.create_or_replace_temp_view(table)).unwrap();
    }
    let mut ids: Vec<String> = (1..=22).map(|n| format!("q{n:02}")).collect();
    if !only.is_empty() {
        ids.retain(|id| only.contains(id));
    }
    let mut json = String::from("{");
    for (n, id) in ids.iter().enumerate() {
        let sql =
            std::fs::read_to_string(PathBuf::from(queries).join(format!("{id}.sql"))).unwrap();
        let start = Instant::now();
        let result = run_query(&session, &zone, &sql);
        let seconds = start.elapsed().as_secs_f64();
        if n > 0 {
            json.push(',');
        }
        json_str(&mut json, id);
        json.push(':');
        match result {
            Ok(rows) => {
                json.push_str("{\"rows\":[");
                for (r, row) in rows.iter().enumerate() {
                    if r > 0 {
                        json.push(',');
                    }
                    json.push('[');
                    for (c, v) in row.iter().enumerate() {
                        if c > 0 {
                            json.push(',');
                        }
                        match v {
                            Some(s) => json_str(&mut json, s),
                            None => json.push_str("null"),
                        }
                    }
                    json.push(']');
                }
                let _ = write!(json, "],\"seconds\":{seconds}}}");
                eprintln!("{id}: {} rows in {seconds:.2}s", rows.len());
            }
            Err(e) => {
                json.push_str("{\"error\":");
                json_str(&mut json, &e.to_string());
                json.push('}');
                eprintln!("{id}: ERROR {e}");
            }
        }
    }
    json.push('}');
    std::fs::write(out, json).unwrap();
}

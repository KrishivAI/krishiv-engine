// SPDX-License-Identifier: Apache-2.0
//! `krishiv server` built with `flight-sql` serves Arrow Flight SQL on its Spark Connect port
//! (design P3-T101 §2), unless `krishiv.flightSql.enabled=false`.
#![cfg(test)]
#![cfg(feature = "flight-sql")]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

use arrow_array::{Array, Int32Array};
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use tonic::transport::Channel;

struct Running {
    child: Child,
    connect: String,
    _stdout: std::io::Lines<BufReader<std::process::ChildStdout>>,
}

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A loopback server without authentication; `flight` is what it printed about Flight SQL.
fn start(spill: &std::path::Path, conf: &[&str]) -> (Running, bool) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .args([
            "server",
            "--allow-root",
            "--port",
            "0",
            "--connect-port",
            "0",
        ])
        .arg("--conf")
        .arg(format!("krishiv.spill.dir={}", spill.display()))
        .args(conf.iter().flat_map(|c| ["--conf", c]))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn krishiv server");
    let mut lines = BufReader::new(child.stdout.take().expect("stdout")).lines();
    let (mut connect, mut flight, mut admin) = (None, false, false);
    while connect.is_none() || !admin {
        let line = lines.next().expect("server output").expect("utf-8");
        if let Some(a) = line.strip_prefix("krishiv server: Spark Connect on sc://") {
            connect = Some(a.to_owned());
        }
        flight |= line == "krishiv server: Arrow Flight SQL on the same port";
        admin |= line.starts_with("krishiv server: admin on ");
    }
    let running = Running {
        child,
        connect: connect.expect("connect"),
        _stdout: lines,
    };
    (running, flight)
}

async fn client(addr: &str) -> FlightSqlServiceClient<Channel> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    FlightSqlServiceClient::new(channel)
}

#[tokio::test(flavor = "multi_thread")]
async fn flight_sql_answers_on_the_connect_port() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (server, flight) = start(dir.path(), &[]);
    assert!(flight, "the server says it serves Flight SQL");
    let mut c = client(&server.connect).await;
    let info = c
        .execute("SELECT 41 + 1 AS answer".to_owned(), None)
        .await
        .expect("execute");
    let ticket = info.endpoint[0].ticket.clone().expect("ticket");
    let batches: Vec<_> = c
        .do_get(ticket)
        .await
        .expect("do_get")
        .try_collect()
        .await
        .expect("rows");
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("INT");
    assert_eq!((col.len(), col.value(0)), (1, 42));
}

#[tokio::test(flavor = "multi_thread")]
async fn flight_sql_disabled_answers_unimplemented() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (server, flight) = start(dir.path(), &["krishiv.flightSql.enabled=false"]);
    assert!(!flight);
    let mut c = client(&server.connect).await;
    let e = c
        .execute("SELECT 1".to_owned(), None)
        .await
        .expect_err("no Flight service");
    let msg = e.to_string();
    // tonic renders `Code::Unimplemented` as its description.
    assert!(msg.contains("not implemented"), "{msg}");
}

/// `adbc_flight_sql_tpch_q1` (design §11): TPC-H q1 through the ADBC Flight SQL driver equals
/// the Spark Connect run, types included. Needs `uv` and the network for the driver: runs only
/// with `KRISHIV_TEST_ADBC=1` (`tests/scripts/flight_sql_adbc.py`).
#[test]
fn adbc_flight_sql_tpch_q1() {
    if std::env::var_os("KRISHIV_TEST_ADBC").is_none() {
        eprintln!("skipped: set KRISHIV_TEST_ADBC=1 (needs uv) to run the ADBC end-to-end test");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (server, flight) = start(dir.path(), &[]);
    assert!(flight);
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/scripts/flight_sql_adbc.py");
    let out = Command::new("uv")
        .args(["run", "--quiet", "--script"])
        .arg(&script)
        .args(["--addr", &server.connect])
        .output()
        .expect("uv runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}\n{stderr}");
    eprintln!("{stdout}");
}

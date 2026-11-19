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
    /// The admin listener's address and its token, when `start_with_admin_token` asked for them.
    admin: String,
    admin_token: String,
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
    start_inner(spill, conf, false)
}

/// [`start`], also reading the admin address and the token the server prints once (only with
/// `krishiv.metrics.enabled=true` or the `ui` feature: pass the former).
fn start_with_admin_token(spill: &std::path::Path, conf: &[&str]) -> (Running, bool) {
    start_inner(spill, conf, true)
}

fn start_inner(spill: &std::path::Path, conf: &[&str], want_token: bool) -> (Running, bool) {
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
    let (mut admin_addr, mut token) = (String::new(), String::new());
    while connect.is_none() || !admin || (want_token && token.is_empty()) {
        let line = lines.next().expect("server output").expect("utf-8");
        if let Some(a) = line.strip_prefix("krishiv server: Spark Connect on sc://") {
            connect = Some(a.to_owned());
        }
        flight |= line == "krishiv server: Arrow Flight SQL on the same port";
        if let Some(a) = line.strip_prefix("krishiv server: admin on http://") {
            admin = true;
            a.clone_into(&mut admin_addr);
        }
        if let Some(t) = line.strip_prefix("krishiv server: admin token ") {
            t.clone_into(&mut token);
        } else if let Some((_, t)) = line.split_once("/?token=") {
            t.clone_into(&mut token);
        }
    }
    let running = Running {
        child,
        connect: connect.expect("connect"),
        admin: admin_addr,
        admin_token: token,
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

/// `method path` of the admin API with the admin token; the status line and the body.
fn admin_call(server: &Running, method: &str, path: &str, body: &str) -> String {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(&server.admin).expect("connect admin");
    s.set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .expect("timeout");
    write!(
        s,
        "{method} /api/admin/v1{path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        server.admin_token,
        body.len()
    )
    .expect("write");
    let mut out = String::new();
    s.read_to_string(&mut out).expect("read");
    out
}

/// Design P3-T066 §5.1, Flight SQL row, through the admin API: after `Drain` a new statement is
/// refused with `KRISHIV_SERVER_DRAINING` (not `UNAVAILABLE`); a stored command result issued
/// before is still fetched; a query ticket that has not started is a query "admitted when it
/// runs" (§5.1), so its first `DoGet` is refused the same way; after `ResumeServing` new statements run.
#[tokio::test(flavor = "multi_thread")]
async fn flight_sql_follows_the_admin_drain() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (server, _) = start_with_admin_token(dir.path(), &["krishiv.metrics.enabled=true"]);
    let mut c = client(&server.connect).await;
    let stored = c
        .execute("SHOW TABLES".to_owned(), None)
        .await
        .expect("a command before the drain");
    let stored = stored.endpoint[0].ticket.clone().expect("ticket");
    let unstarted = c
        .execute("SELECT 7 AS x".to_owned(), None)
        .await
        .expect("a query before the drain");
    let unstarted = unstarted.endpoint[0].ticket.clone().expect("ticket");
    let reply = admin_call(&server, "POST", "/drain", "{}");
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    let refused = |msg: String| {
        assert!(msg.contains("KRISHIV_SERVER_DRAINING"), "{msg}");
        assert!(!msg.contains("Unavailable"), "{msg}");
    };
    refused(format!(
        "{:?}",
        c.execute("SELECT 8".to_owned(), None)
            .await
            .expect_err("new work while draining")
    ));
    let rows: Vec<_> = c
        .do_get(stored)
        .await
        .expect("a stored result is fetched while draining")
        .try_collect()
        .await
        .expect("rows");
    drop(rows);
    let e = match c.do_get(unstarted.clone()).await {
        Err(e) => format!("{e:?}"),
        Ok(stream) => format!(
            "{:?}",
            stream.try_collect::<Vec<_>>().await.expect_err("refused")
        ),
    };
    refused(e);
    let reply = admin_call(&server, "DELETE", "/drain", "");
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    // The refused fetch used up the ticket (one fetch per cookie-less ticket); the client
    // reconnects, as the refusal says, and a new statement runs after the resume.
    let again = c
        .execute("SELECT 9 AS x".to_owned(), None)
        .await
        .expect("a statement after the resume");
    let batches: Vec<_> = c
        .do_get(again.endpoint[0].ticket.clone().expect("ticket"))
        .await
        .expect("fetch after the resume")
        .try_collect()
        .await
        .expect("rows");
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

const ADMIN_TOKEN: &str = "adbc-admin-token-0123456789abcdef0123";

/// The conf of a token-authenticated server (one admin token, its secret in a 0600 file), plus
/// TLS from the session crate's test fixtures when `tls`; and the CA path to trust.
#[cfg(unix)]
fn auth_conf(dir: &std::path::Path, tls: bool) -> (Vec<String>, Option<std::path::PathBuf>) {
    use std::os::unix::fs::PermissionsExt;
    let secret = dir.join("admin.token");
    std::fs::write(&secret, ADMIN_TOKEN).expect("write secret");
    std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let mut conf = vec![
        format!("krishiv.auth.tokens.adbc.secret=file:{}", secret.display()),
        "krishiv.auth.tokens.adbc.principal=adbc".to_owned(),
        "krishiv.auth.tokens.adbc.role=admin".to_owned(),
        // The confinement of server mode (ADR 0046 §3a): q1 reads the generated file there.
        format!("krishiv.io.local.allowedRoots={}", data_dir(dir).display()),
    ];
    if !tls {
        return (conf, None);
    }
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../krishiv-session/tests/fixtures/tls");
    let (cert, key) = (dir.join("server.pem"), dir.join("server.key"));
    std::fs::copy(fixtures.join("server.pem"), &cert).expect("cert");
    std::fs::copy(fixtures.join("server.key"), &key).expect("key");
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    conf.push(format!("krishiv.server.tls.certFile={}", cert.display()));
    conf.push(format!("krishiv.server.tls.keyFile={}", key.display()));
    (conf, Some(fixtures.join("ca.pem")))
}

/// Where the script writes `lineitem.parquet` (made first: an allowed root must exist).
fn data_dir(dir: &std::path::Path) -> std::path::PathBuf {
    let d = dir.join("data");
    std::fs::create_dir_all(&d).expect("data dir");
    d.canonicalize().expect("canonical data dir")
}

/// TPC-H q1 at SF 0.01's row count through the ADBC Flight SQL driver with a bearer token equals
/// the Spark Connect run, types included (design §0, §11, §16 m7; `tests/scripts/flight_sql_adbc.py`).
#[cfg(unix)]
fn adbc_q1(tls: bool) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (conf, ca) = auth_conf(dir.path(), tls);
    let conf: Vec<&str> = conf.iter().map(String::as_str).collect();
    let (server, flight) = start(dir.path(), &conf);
    assert!(flight);
    let addr = server
        .connect
        .split(';')
        .next()
        .expect("address")
        .trim_end_matches('/')
        .to_owned();
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/scripts/flight_sql_adbc.py");
    let mut cmd = Command::new("uv");
    cmd.args(["run", "--quiet", "--script"])
        .arg(&script)
        .args(["--addr", &addr, "--token", ADMIN_TOKEN])
        .arg("--data")
        .arg(data_dir(dir.path()));
    if let Some(ca) = &ca {
        cmd.arg("--ca").arg(ca);
    }
    let out = cmd.output().expect("uv runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}\n{stderr}");
    eprintln!("{stdout}");
}

/// Plaintext loopback with token auth. Needs `uv` and the network for the drivers, so it is
/// ignored by default: `cargo nextest run -p krishiv-cli --features flight-sql --test flight_sql
/// --run-ignored only`.
#[cfg(unix)]
#[test]
#[ignore = "needs uv and the network (ADBC and PySpark drivers): --run-ignored only"]
fn adbc_flight_sql_tpch_q1() {
    adbc_q1(false);
}

/// The same over TLS (the test CA).
#[cfg(unix)]
#[test]
#[ignore = "needs uv and the network (ADBC and PySpark drivers): --run-ignored only"]
fn adbc_flight_sql_tpch_q1_tls() {
    adbc_q1(true);
}

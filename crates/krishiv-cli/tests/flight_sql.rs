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
    // Without either, the server never prints a token and the read below would wait for ever.
    assert!(
        !want_token || cfg!(feature = "ui") || conf.contains(&"krishiv.metrics.enabled=true"),
        "the admin token is printed only with krishiv.metrics.enabled=true or the `ui` feature"
    );
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

/// Design P3-T066 §5.1, Flight SQL row, through the admin API. After `Drain` (not exiting): a new
/// statement is refused with `KRISHIV_SERVER_DRAINING` (not `UNAVAILABLE`); the drain stays
/// `draining` while a ticket issued before it is unfetched; a stored command result and a query
/// ticket issued before are fetched (the query runs although the runtime refuses new ones); once
/// both are fetched the drain is `drained`; after `ResumeServing` statements work again.
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
    let e = c
        .execute("SELECT 8".to_owned(), None)
        .await
        .expect_err("new work while draining");
    let msg = format!("{e:?}");
    assert!(msg.contains("KRISHIV_SERVER_DRAINING"), "{msg}");
    assert!(!msg.contains("Unavailable"), "{msg}");
    // The two tickets are held work: the drain does not finish while they are unfetched.
    std::thread::sleep(std::time::Duration::from_millis(400));
    let status = admin_call(&server, "GET", "/drain", "");
    assert!(status.contains("\"draining\""), "{status}");
    let _rows: Vec<_> = c
        .do_get(stored)
        .await
        .expect("a stored result is fetched while draining")
        .try_collect()
        .await
        .expect("rows");
    let batches: Vec<_> = c
        .do_get(unstarted)
        .await
        .expect("a query ticket is fetched while draining")
        .try_collect()
        .await
        .expect("rows");
    assert_eq!(
        batches
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let status = admin_call(&server, "GET", "/drain", "");
        if status.contains("\"drained\"") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "never drained: {status}"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let reply = admin_call(&server, "DELETE", "/drain", "");
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
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
    assert_eq!(
        batches
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );
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

/// TPC-H q1..q22 over SF 0.01 (`tpchgen-cli`), each through ADBC and through Spark Connect on the
/// same server; the rows and Arrow types must agree.
#[cfg(unix)]
#[test]
#[ignore = "needs uv, the network and tpchgen-cli (cargo install tpchgen-cli): --run-ignored only"]
fn adbc_flight_sql_tpch_all() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tpch = data_dir(dir.path()).join("tpch");
    let gen_status = Command::new("tpchgen-cli")
        .args(["parquet", "-s", "0.01", "--output-dir"])
        .arg(&tpch)
        .status()
        .expect("tpchgen-cli runs (cargo install tpchgen-cli)");
    assert!(gen_status.success());
    let (conf, _) = auth_conf(dir.path(), false);
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
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = Command::new("uv")
        .args(["run", "--quiet", "--script"])
        .arg(root.join("tests/scripts/flight_sql_adbc.py"))
        .args(["--addr", &addr, "--token", ADMIN_TOKEN])
        .arg("--tpch-dir")
        .arg(&tpch)
        .arg("--queries")
        .arg(root.join("bench/queries/tpch"))
        .output()
        .expect("uv runs");
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

/// TPC-H q1 over a `lineitem` of SF 0.01's row count, generated in a CTE (seeded by row id, so
/// no file and no session state): `{cutoff}` is the shipdate bound, a literal or `?`.
const JDBC_Q1: &str = "WITH base AS (
  SELECT id, 1 + pmod(id * 13, 50) AS qty,
         date_add(DATE '1992-01-02', CAST(pmod(id * 101, 2526) AS INT)) AS ship
  FROM range(60175)),
lineitem AS (
  SELECT CASE WHEN ship <= DATE '1995-06-17' THEN IF(id % 2 = 0, 'R', 'A') ELSE 'N' END
           AS l_returnflag,
         IF(ship <= DATE '1995-06-17', 'F', 'O') AS l_linestatus,
         CAST(qty AS DECIMAL(15,2)) AS l_quantity,
         CAST(qty * (900 + pmod(id * 7919, 1100)) / 100 AS DECIMAL(15,2)) AS l_extendedprice,
         CAST(pmod(id * 31, 11) / 100 AS DECIMAL(15,2)) AS l_discount,
         CAST(pmod(id * 17, 9) / 100 AS DECIMAL(15,2)) AS l_tax,
         ship AS l_shipdate
  FROM base)
SELECT l_returnflag, l_linestatus,
       sum(l_quantity) AS sum_qty,
       sum(l_extendedprice) AS sum_base_price,
       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
       avg(l_quantity) AS avg_qty,
       avg(l_extendedprice) AS avg_price,
       avg(l_discount) AS avg_disc,
       count(*) AS count_order
FROM lineitem
WHERE l_shipdate <= {cutoff}
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus";

/// q1's `DATE '1998-12-01' - INTERVAL '90' DAY`.
const JDBC_CUTOFF: &str = "1998-09-02";

/// One result as `FlightSqlJdbc.java` prints it: `(label, java.sql.Types code, precision,
/// scale)` per column (precision and scale 0 but for decimals), and the rows as text.
#[derive(Debug, Default, PartialEq, Eq)]
struct Printed {
    columns: Vec<(String, i32, i32, i32)>,
    rows: Vec<String>,
}

/// q1 through the Rust Flight SQL client, printed as the Java program prints it.
async fn q1_by_rust_client(addr: &str, token: &str) -> Printed {
    use arrow_array::{Decimal128Array, Int64Array, StringArray};
    use arrow_schema::DataType;
    let mut c = client(addr).await;
    c.set_header("authorization", format!("Bearer {token}"));
    let sql = JDBC_Q1.replace("{cutoff}", &format!("DATE '{JDBC_CUTOFF}'"));
    let info = c.execute(sql, None).await.expect("q1 info");
    let ticket = info.endpoint[0].ticket.clone().expect("ticket");
    let batches: Vec<arrow_array::RecordBatch> = c
        .do_get(ticket)
        .await
        .expect("q1 do_get")
        .try_collect()
        .await
        .expect("q1 rows");
    let schema = batches.first().expect("a batch").schema();
    let mut out = Printed::default();
    for f in schema.fields() {
        // java.sql.Types: VARCHAR 12, BIGINT -5, DECIMAL 3.
        let col = match f.data_type() {
            DataType::Utf8 => (12, 0, 0),
            DataType::Int64 => (-5, 0, 0),
            DataType::Decimal128(p, s) => (3, i32::from(*p), i32::from(*s)),
            other => panic!("q1 has no {other} column"),
        };
        out.columns.push((f.name().clone(), col.0, col.1, col.2));
    }
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|a| {
                    if a.is_null(r) {
                        return "\\N".to_owned();
                    }
                    let any = a.as_any();
                    if let Some(s) = any.downcast_ref::<StringArray>() {
                        s.value(r).to_owned()
                    } else if let Some(i) = any.downcast_ref::<Int64Array>() {
                        i.value(r).to_string()
                    } else if let Some(d) = any.downcast_ref::<Decimal128Array>() {
                        d.value_as_string(r)
                    } else {
                        panic!("q1 has no {} column", a.data_type())
                    }
                })
                .collect();
            out.rows.push(cells.join("\t"));
        }
    }
    out
}

/// The program's output, split into its parts.
#[derive(Debug, Default)]
struct JdbcRun {
    statement: Printed,
    prepared: Printed,
    update: Option<u64>,
    tables: Option<u64>,
    types: Vec<String>,
    catalogs: Vec<String>,
    schemas: Vec<String>,
    columns: Vec<String>,
    /// q1's rows over a `user=`/`password=` URL (Basic, design §3.1).
    basic: Option<usize>,
    /// The URL with another principal's name was refused.
    basic_wrong_refused: bool,
}

fn parse_jdbc(stdout: &str) -> JdbcRun {
    let mut run = JdbcRun::default();
    let mut current: Option<&mut Printed> = None;
    for line in stdout.lines() {
        if let Some(name) = line.strip_prefix("QUERY ") {
            current = Some(match name {
                "statement" => &mut run.statement,
                "prepared" => &mut run.prepared,
                other => panic!("unknown query {other}"),
            });
        } else if let Some(col) = line.strip_prefix("COL ") {
            let p: Vec<&str> = col.split('|').collect();
            let n = |i: usize| p[i].parse::<i32>().expect("number");
            let printed = current.as_mut().expect("a query first");
            // Precision and scale are compared for decimals (java.sql.Types 3) only.
            let (precision, scale) = if n(1) == 3 { (n(2), n(3)) } else { (0, 0) };
            printed
                .columns
                .push((p[0].to_owned(), n(1), precision, scale));
        } else if let Some(row) = line.strip_prefix("ROW ") {
            current
                .as_mut()
                .expect("a query first")
                .rows
                .push(row.to_owned());
        } else if let Some(n) = line.strip_prefix("UPDATE ") {
            run.update = n.parse().ok();
        } else if let Some(n) = line.strip_prefix("TABLES ") {
            run.tables = n.parse().ok();
        } else if let Some(t) = line.strip_prefix("TYPES ") {
            run.types = t.split(',').map(str::to_owned).collect();
        } else if let Some(t) = line.strip_prefix("CATALOGS ") {
            run.catalogs = t.split(',').map(str::to_owned).collect();
        } else if let Some(t) = line.strip_prefix("SCHEMAS ") {
            run.schemas = t.split(',').map(str::to_owned).collect();
        } else if let Some(t) = line.strip_prefix("COLUMNS ") {
            run.columns = t.split(',').map(str::to_owned).collect();
        } else if let Some(n) = line.strip_prefix("BASIC ") {
            run.basic = n.parse().ok();
        } else if line == "BASIC_WRONG refused" {
            run.basic_wrong_refused = true;
        }
    }
    run
}

/// TPC-H q1 through the Arrow Flight SQL JDBC driver (arrow-java 19) with a bearer token equals
/// the Rust Flight SQL client's answer, rows and decimal types (precision and scale from the
/// Flight SQL column metadata); the parameterized prepared q1 takes the query path (m11) and
/// binds `setDate` into the `DATE` parameter its analysis gives the marker (§4.4, amended
/// 2026-11-20); each `executeUpdate` runs once (an update without a count answers `0`: the driver
/// reads `-1` as a result set and would run it again), `setInt` binds the `INSERT ... VALUES`
/// cell's `INT` column; `getTables`, `getCatalogs`, `getSchemas` and `getColumns` answer; a
/// `user=<principal>&password=<token>` URL (Basic, owner item 5) runs q1 and another user name
/// with the token is refused (design §0, §3.1, §4.3, §4.4, §5, §11, §17;
/// `tests/scripts/flight_sql_jdbc.sh`). Needs Java 21 or Docker, and the network the first time
/// (the driver jar is cached, checked against its pinned SHA-256), so it is `--run-ignored only`:
/// the nightly all-features job runs it in a step of its own.
#[cfg(unix)]
#[test]
#[ignore = "needs Java 21 or Docker, and the network on a cold cache (JDBC driver): --run-ignored only"]
fn jdbc_flight_sql_tpch_q1() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (mut conf, _) = auth_conf(dir.path(), false);
    let wh = data_dir(dir.path()).join("wh");
    std::fs::create_dir_all(&wh).expect("warehouse dir");
    conf.push(format!("spark.sql.warehouse.dir={}", wh.display()));
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
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let expected = rt.block_on(q1_by_rust_client(&addr, ADMIN_TOKEN));
    assert!(expected.rows.len() >= 3, "q1 read nothing: {expected:?}");
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/scripts/flight_sql_jdbc.sh");
    let url = format!("jdbc:arrow-flight-sql://{addr}?useEncryption=false&token={ADMIN_TOKEN}");
    // Basic (owner item 5): the token as the password, its principal `adbc` as the user name.
    let basic = format!(
        "jdbc:arrow-flight-sql://{addr}?useEncryption=false&user=adbc&password={ADMIN_TOKEN}"
    );
    let wrong = format!(
        "jdbc:arrow-flight-sql://{addr}?useEncryption=false&user=someone&password={ADMIN_TOKEN}"
    );
    let out = Command::new("sh")
        .arg(&script)
        .arg(&url)
        .arg(JDBC_Q1.replace("{cutoff}", &format!("DATE '{JDBC_CUTOFF}'")))
        .arg(JDBC_Q1.replace("{cutoff}", "?"))
        .arg(JDBC_CUTOFF)
        .arg(&basic)
        .arg(&wrong)
        .output()
        .expect("the JDBC script runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}\n{stderr}");
    let run = parse_jdbc(&stdout);
    assert_eq!(
        run.statement, expected,
        "JDBC Statement q1 vs the Rust client"
    );
    assert_eq!(
        run.prepared, expected,
        "JDBC prepared q1 (m11) vs the Rust client"
    );
    assert_eq!(run.update, Some(3), "each executeUpdate ran once\n{stdout}");
    assert_eq!(run.tables, Some(1), "getTables lists the table\n{stdout}");
    // arrow-java 19 does not implement `getTypeInfo` (neither `ArrowDatabaseMetadata` nor
    // `ArrowFlightMetaImpl` overrides it): Avatica's default answers no rows and the driver never
    // sends `GetXdbcTypeInfo`, which krishiv-flight-sql's own tests cover. Seeing rows here means
    // a driver that asks: then check them.
    assert!(
        run.types.iter().all(String::is_empty),
        "the driver now reads GetXdbcTypeInfo: assert INT, DECIMAL, STRING, TIMESTAMP in {:?}",
        run.types
    );
    assert!(
        run.catalogs.iter().any(|c| c == "spark_catalog"),
        "getCatalogs: {:?}",
        run.catalogs
    );
    assert!(
        run.schemas.iter().any(|s| s == "default"),
        "getSchemas: {:?}",
        run.schemas
    );
    assert_eq!(run.columns.len(), 1, "getColumns: {:?}", run.columns);
    assert_eq!(
        run.basic,
        Some(expected.rows.len()),
        "q1 over a user/password URL\n{stdout}"
    );
    assert!(
        run.basic_wrong_refused,
        "another principal's name with the token is refused\n{stdout}"
    );
    assert!(
        run.columns[0].starts_with("n:"),
        "getColumns: {:?}",
        run.columns
    );
    eprintln!(
        "JDBC q1: {} rows, {} columns, equal to the Rust client",
        expected.rows.len(),
        expected.columns.len()
    );
}

/// TPC-H q1..q22 through the Arrow Flight SQL JDBC driver: each runs and returns as many rows and
/// columns as the Rust Flight SQL client reads for the same text (design P3-T101 §11; the ADBC
/// counterpart is `adbc_flight_sql_tpch_all`, which compares values with Spark Connect).
#[cfg(unix)]
#[test]
#[ignore = "needs Java 21 or Docker, the network on a cold cache and tpchgen-cli: --run-ignored only"]
fn jdbc_flight_sql_tpch_all() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tpch = data_dir(dir.path()).join("tpch");
    let gen_status = Command::new("tpchgen-cli")
        .args(["parquet", "-s", "0.01", "--output-dir"])
        .arg(&tpch)
        .status()
        .expect("tpchgen-cli runs (cargo install tpchgen-cli)");
    assert!(gen_status.success());
    let (conf, _) = auth_conf(dir.path(), false);
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
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let queries = root.join("bench/queries/tpch");
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let expected: Vec<(usize, usize)> = (1..=22)
        .map(|n| {
            let text =
                std::fs::read_to_string(queries.join(format!("q{n:02}.sql"))).expect("query");
            rt.block_on(counts_by_rust_client(&addr, &tpch_text(&text, &tpch)))
        })
        .collect();
    let url = format!("jdbc:arrow-flight-sql://{addr}?useEncryption=false&token={ADMIN_TOKEN}");
    let out = Command::new("sh")
        .arg(root.join("tests/scripts/flight_sql_jdbc.sh"))
        .arg("tpch")
        .arg(&url)
        .arg(&queries)
        .arg(&tpch)
        .output()
        .expect("the JDBC script runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}\n{stderr}");
    assert_eq!(
        parse_tpch_counts(&stdout),
        expected,
        "JDBC vs the Rust client\n{stdout}"
    );
}

/// A TPC-H query with each table a CTE over its Parquet file, spliced into the query's own `WITH`.
fn tpch_text(sql: &str, dir: &std::path::Path) -> String {
    let ctes = [
        "part", "supplier", "partsupp", "customer", "orders", "lineitem", "nation", "region",
    ]
    .iter()
    .map(|t| {
        format!(
            "{t} AS (SELECT * FROM parquet.`{}`)",
            dir.join(format!("{t}.parquet")).display()
        )
    })
    .collect::<Vec<_>>()
    .join(", ");
    let body = sql.trim().trim_end_matches(';').trim();
    match body.get(..4).filter(|w| w.eq_ignore_ascii_case("with")) {
        Some(_) => format!("WITH {ctes}, {}", body[4..].trim_start()),
        None => format!("WITH {ctes} {body}"),
    }
}

/// The row and column counts the Rust Flight SQL client reads for `sql`.
async fn counts_by_rust_client(addr: &str, sql: &str) -> (usize, usize) {
    let mut c = client(addr).await;
    c.set_header("authorization", format!("Bearer {ADMIN_TOKEN}"));
    let info = c.execute(sql.to_owned(), None).await.expect("info");
    let ticket = info.endpoint[0].ticket.clone().expect("ticket");
    let batches: Vec<arrow_array::RecordBatch> = c
        .do_get(ticket)
        .await
        .expect("do_get")
        .try_collect()
        .await
        .expect("rows");
    let columns = batches
        .first()
        .map_or(0, arrow_array::RecordBatch::num_columns);
    (
        batches.iter().map(arrow_array::RecordBatch::num_rows).sum(),
        columns,
    )
}

/// `(rows, columns)` of each `TPCH q<n> <rows> <columns>` line, in order.
fn parse_tpch_counts(out: &str) -> Vec<(usize, usize)> {
    out.lines()
        .filter_map(|l| l.strip_prefix("TPCH q"))
        .filter_map(|l| {
            let mut p = l.split(' ');
            p.next()?;
            Some((p.next()?.parse().ok()?, p.next()?.parse().ok()?))
        })
        .collect()
}

#[test]
fn jdbc_tpch_counts_parse() {
    assert_eq!(
        parse_tpch_counts("TPCH q1 4 10\nnoise\nTPCH q2 100 8\n"),
        [(4, 10), (100, 8)]
    );
}

#[test]
fn jdbc_output_parses() {
    let run = parse_jdbc(
        "QUERY statement\nCOL a|12|0|0\nROW x\nQUERY prepared\nCOL a|12|0|0\nROW x\n\
         UPDATE 3\nTABLES 1\nTYPES INT,STRING\nCATALOGS spark_catalog\nSCHEMAS default,db\n\
         COLUMNS n:INTEGER\nBASIC 3\nBASIC_WRONG refused\n",
    );
    assert_eq!((run.basic, run.basic_wrong_refused), (Some(3), true));
    assert_eq!(run.statement, run.prepared);
    assert_eq!(run.statement.rows, ["x"]);
    assert_eq!((run.update, run.tables), (Some(3), Some(1)));
    assert_eq!(run.types, ["INT", "STRING"]);
    assert_eq!(run.catalogs, ["spark_catalog"]);
    assert_eq!(run.schemas, ["default", "db"]);
    assert_eq!(run.columns, ["n:INTEGER"]);
}

/// The one string cell of a Flight SQL query on `server`.
async fn one_string(server: &Running, sql: &str) -> String {
    let mut c = client(&server.connect).await;
    let info = c.execute(sql.to_owned(), None).await.expect("execute");
    let ticket = info.endpoint[0].ticket.clone().expect("ticket");
    let batches: Vec<arrow_array::RecordBatch> = c
        .do_get(ticket)
        .await
        .expect("do_get")
        .try_collect()
        .await
        .expect("rows");
    let col = batches[0].column(0);
    if let Some(s) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
        return s.value(0).to_owned();
    }
    col.as_any()
        .downcast_ref::<arrow_array::StringViewArray>()
        .expect("a string column")
        .value(0)
        .to_owned()
}

/// Code health I-1 over what `krishiv server` itself builds (its runtime is server mode): a client's
/// `${env:..}` stays as written, and the operator's startup key turns Spark's substitution back on.
#[tokio::test(flavor = "multi_thread")]
async fn the_served_runtime_never_reads_the_environment_for_a_client() {
    let path = std::env::var("PATH").expect("PATH is set");
    let dir = tempfile::tempdir().expect("tempdir");
    let (server, _) = start(dir.path(), &[]);
    assert_eq!(
        one_string(&server, "SELECT '${env:PATH}' AS p").await,
        "${env:PATH}"
    );
    drop(server);
    let dir = tempfile::tempdir().expect("tempdir");
    let (server, _) = start(
        dir.path(),
        &["krishiv.server.sql.substituteEnvironment=true"],
    );
    assert_eq!(one_string(&server, "SELECT '${env:PATH}' AS p").await, path);
}

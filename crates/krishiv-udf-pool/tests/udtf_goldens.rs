// SPDX-License-Identifier: Apache-2.0
//! PySpark 4.2.0's answers for Python table functions (UDTFs, eval types 300, 301, 302), replayed through a
//! [`Session`] whose UDTFs run on the worker pool (PR-26c-4).
//!
//! The cases and their answers are `python/tests/pyspark_udf/udtf/{cases.py,goldens.json}`, a real local PySpark
//! 4.2.0 run. `export_cases.py` runs each case against a recording stand-in with `pyspark`'s decorators (what a
//! Spark Connect client sends) and gives the UDTFs it registers, cloudpickled, and the SQL it runs. Its output is
//! committed (`tests/goldens/udtf_cases.json.gz`, regenerated with a full PySpark as `export_cases.py` says), so
//! the replay needs only the workers' interpreter; it fails when `cases.py` or `export_cases.py` changed since the
//! export (their SHA-256 are recorded) or the workers run another CPython minor version than the export did. This
//! test registers the UDTFs on a session (`Session::register_python_udtf`, the pool's `SessionUdfs::register`),
//! runs the SQL and compares the schema and rows (sorted, as the case orders them by every column), or the error's
//! condition and the worker's `[CLASS] message` (or the traceback line). Cases built with DataFrame calls are
//! Spark Connect's relation path, not replayed here; definition errors are the client's (`pyspark` raises them
//! before anything reaches the server) and are checked against the golden as the client gave them.
//!
//! `KNOWN_GAPS` lists the cases that differ, with the reason; a listed case that matches fails, so the list
//! cannot go stale. `INCOMPATIBLE` lists those Krishiv answers differently on purpose (`known-differences.md`).
//!
//! The interpreter: `KRISHIV_TEST_UDF_PYTHON` runs the workers; without it the test passes after a note unless
//! `KRISHIV_UDF_POOL_REQUIRED` is set.
#![cfg(target_os = "linux")]
#![expect(
    clippy::print_stderr,
    clippy::expect_used,
    clippy::panic,
    reason = "the skip note and the case tally; test helpers sit outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::DataType as ArrowType;
use krishiv_plan::function::PythonEvalType;
use krishiv_session::{PythonUdtfDef, Session};
use krishiv_types::DataType;
use krishiv_udf_pool::{Launcher, Owner, Pool, PoolConfig, PythonPayload, SessionUdfs};
use serde_json::Value;

/// Cases that differ from Spark 4.2.0, with the reason.
const KNOWN_GAPS: &[(&str, &str)] = &[(
    "analyze/udtf300/argument_array",
    "the analyzer does not fold `array(1, 2)`, so `analyze` sees it as no constant (`value` None, `isConstantExpression` false)",
)];

/// Cases Krishiv answers differently on purpose (`docs/concepts/known-differences.md`, "Python table
/// functions"): Spark 4.2.0 fails them with a JVM `NullPointerException` (a `terminate` row joined with no input
/// row); Krishiv opens no UDTF instance for an input without rows, so they return no rows.
const INCOMPATIBLE: &[(&str, &str)] = &[
    (
        "rows/udtf300/empty_lateral_input",
        "Spark's NullPointerException",
    ),
    (
        "rows/udtf301/empty_lateral_input",
        "Spark's NullPointerException",
    ),
    (
        "rows/udtf300/empty_table_input",
        "Spark's NullPointerException",
    ),
    (
        "rows/udtf301/empty_table_input",
        "Spark's NullPointerException",
    ),
    (
        "rows/udtf302/empty_table_input",
        "Spark's NullPointerException",
    ),
    // A JVM object's `toString` (`[B@<identity hash>`, `java.util.GregorianCalendar[...]`) in a string column:
    // Krishiv gives the value's Python `str`.
    (
        "mistyped/udtf300/bytes_for_string",
        "the JVM's toString of a byte[]",
    ),
    (
        "mistyped/udtf300/date_for_string",
        "the JVM's toString of a Calendar",
    ),
    // The JVM cannot unpickle a `Row` inside a 300 result row (`PickleException`, no class); Krishiv has no
    // JVM, the `Row` is an `Object[]` to `makeFromJava` and the `INT` column NULL.
    (
        "errors/udtf300/table_argument_to_scalar_eval",
        "the JVM's PickleException",
    ),
    // `None` in a non-nullable column of a 302 result: Spark's JVM fails reading the Arrow batch
    // (`IllegalStateException: Value at index is null`, no class); the pool refuses the batch as unlike the
    // declared schema (`PYTHON_EXCEPTION`).
    (
        "mistyped/udtf302/none_for_not_null",
        "the JVM's IllegalStateException",
    ),
];

/// The child entry: run as the trampoline when started with `--` and spec arguments.
#[test]
fn worker_child() {
    let args: Vec<OsString> = std::env::args_os().collect();
    let Some(split) = args.iter().position(|a| a == "--") else {
        return;
    };
    let spec: Vec<OsString> = args.get(split + 1..).unwrap_or_default().to_vec();
    if spec.is_empty() {
        return;
    }
    let code = krishiv_udf_sandbox::trampoline_main(spec);
    let failed = code == ExitCode::from(krishiv_udf_sandbox::EXIT_SANDBOX_FAILED);
    std::process::exit(if failed { 125 } else { 1 });
}

fn interpreter() -> Option<PathBuf> {
    let Some(worker) = std::env::var_os("KRISHIV_TEST_UDF_PYTHON") else {
        assert!(
            std::env::var_os("KRISHIV_UDF_POOL_REQUIRED").is_none(),
            "KRISHIV_UDF_POOL_REQUIRED is set but KRISHIV_TEST_UDF_PYTHON is not"
        );
        eprintln!("note: KRISHIV_TEST_UDF_PYTHON is not set; the UDTF replay did not run");
        return None;
    };
    Some(PathBuf::from(worker))
}

fn pool(python: &Path) -> Pool {
    let mut config = PoolConfig::default();
    config.executables = vec![python.display().to_string()];
    config.launcher = Launcher {
        program: std::env::current_exe().expect("test binary"),
        args: ["worker_child", "--exact", "--nocapture", "--"]
            .map(OsString::from)
            .to_vec(),
    };
    let pool = Pool::new(config).expect("pool");
    assert!(pool.refusal().is_none(), "{:?}", pool.refusal());
    pool
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The case sources whose SHA-256 the export records.
const SOURCES: [&str; 2] = ["cases.py", "export_cases.py"];

/// The committed `export_cases.py` record of every case (`tests/goldens/udtf_cases.json.gz`), after checking
/// it is the export of today's sources (hashed by `python`, the workers' interpreter) for `python`'s CPython
/// minor version and the session's `arrow_default`.
fn export(python: &Path, arrow_default: &str) -> serde_json::Map<String, Value> {
    use std::io::Read as _;
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/udtf_cases.json.gz");
    let mut text = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(&path).expect("the committed export"))
        .read_to_string(&mut text)
        .expect("gzip");
    let mut export: Value = serde_json::from_str(&text).expect("json");
    let regenerate = "regenerate it as python/tests/pyspark_udf/udtf/export_cases.py says";
    let dir = repo().join("python/tests/pyspark_udf/udtf");
    for source in SOURCES {
        let out = Command::new(python)
            .args([
                "-I",
                "-c",
                "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest(), end='')",
            ])
            .arg(dir.join(source))
            .output()
            .expect("hash");
        assert_eq!(
            export["sources"][source],
            String::from_utf8_lossy(&out.stdout).as_ref(),
            "{source} changed since the committed export: {regenerate}"
        );
    }
    assert_eq!(
        export["python"],
        python_version(python).as_str(),
        "the export's code objects are for another CPython: {regenerate} on the workers' version"
    );
    assert_eq!(
        export["arrow_default"], arrow_default,
        "the export's arrow default: {regenerate}"
    );
    match export["cases"].take() {
        Value::Object(m) => m,
        _ => panic!("cases is not an object"),
    }
}

fn python_version(python: &Path) -> String {
    let out = Command::new(python)
        .args([
            "-c",
            "import sys; print('%d.%d' % sys.version_info[:2], end='')",
        ])
        .output()
        .expect("version");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `s` as Python's `repr` of a `str`.
fn py_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `b` as Python's `repr` of `bytes`.
fn py_bytes(b: &[u8]) -> String {
    let mut out = String::from("b'");
    for &c in b {
        match c {
            b'\\' => out.push_str("\\\\"),
            b'\'' => out.push_str("\\'"),
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            b'\r' => out.push_str("\\r"),
            0x20..=0x7e => out.push(char::from(c)),
            _ => out.push_str(&format!("\\x{c:02x}")),
        }
    }
    out.push('\'');
    out
}

/// Days since 1970-01-01 as (year, month, day) (Howard Hinnant's `civil_from_days`).
fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// An unscaled decimal as Python's `Decimal` shows it.
fn decimal(v: i128, scale: i8) -> String {
    let digits = v.unsigned_abs().to_string();
    let scale = usize::try_from(scale).unwrap_or(0);
    let sign = if v < 0 { "-" } else { "" };
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    let digits = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = digits.split_at(digits.len() - scale);
    format!("{sign}{int}.{frac}")
}

/// A float as `cases.cell` keeps it: NaN and infinities as text.
fn float(v: f64) -> Value {
    if v.is_nan() {
        Value::from("nan")
    } else if v.is_infinite() {
        Value::from(if v > 0.0 { "inf" } else { "-inf" })
    } else {
        Value::from(v)
    }
}

/// The value at `r` of `c` as Python's `repr` (inside a `Row`).
fn repr(c: &dyn Array, r: usize) -> String {
    if c.is_null(r) {
        return "None".to_owned();
    }
    match cell(c, r) {
        Value::String(s)
            if matches!(
                c.data_type(),
                ArrowType::Utf8 | ArrowType::Utf8View | ArrowType::LargeUtf8
            ) =>
        {
            py_str(&s)
        }
        Value::String(s) => s,
        Value::Bool(b) => if b { "True" } else { "False" }.to_owned(),
        other => other.to_string(),
    }
}

/// The value at `r` of `c` as `cases.cell` gives it.
fn cell(c: &dyn Array, r: usize) -> Value {
    if c.is_null(r) {
        return Value::Null;
    }
    match c.data_type() {
        ArrowType::Boolean => Value::from(c.as_boolean().value(r)),
        ArrowType::Int8 => Value::from(c.as_primitive::<Int8Type>().value(r)),
        ArrowType::Int16 => Value::from(c.as_primitive::<Int16Type>().value(r)),
        ArrowType::Int32 => Value::from(c.as_primitive::<Int32Type>().value(r)),
        ArrowType::Int64 => Value::from(c.as_primitive::<Int64Type>().value(r)),
        ArrowType::Float32 => float(f64::from(c.as_primitive::<Float32Type>().value(r))),
        ArrowType::Float64 => float(c.as_primitive::<Float64Type>().value(r)),
        ArrowType::Utf8 => Value::from(c.as_string::<i32>().value(r)),
        ArrowType::LargeUtf8 => Value::from(c.as_string::<i64>().value(r)),
        ArrowType::Utf8View => Value::from(c.as_string_view().value(r)),
        ArrowType::Binary => Value::from(py_bytes(c.as_binary::<i32>().value(r))),
        ArrowType::LargeBinary => Value::from(py_bytes(c.as_binary::<i64>().value(r))),
        ArrowType::BinaryView => Value::from(py_bytes(c.as_binary_view().value(r))),
        ArrowType::Date32 => {
            let (y, m, d) = civil(i64::from(c.as_primitive::<Date32Type>().value(r)));
            Value::from(format!("datetime.date({y}, {m}, {d})"))
        }
        ArrowType::Decimal128(_, scale) => {
            let v = c
                .as_primitive::<arrow_array::types::Decimal128Type>()
                .value(r);
            Value::from(format!("Decimal('{}')", decimal(v, *scale)))
        }
        ArrowType::Decimal64(_, scale) => {
            let v = c
                .as_primitive::<arrow_array::types::Decimal64Type>()
                .value(r);
            Value::from(format!("Decimal('{}')", decimal(i128::from(v), *scale)))
        }
        ArrowType::Struct(fields) => {
            let s = c.as_struct();
            let parts: Vec<String> = fields
                .iter()
                .zip(s.columns())
                .map(|(f, col)| format!("{}={}", f.name(), repr(col.as_ref(), r)))
                .collect();
            Value::from(format!("Row({})", parts.join(", ")))
        }
        ArrowType::List(_)
        | ArrowType::LargeList(_)
        | ArrowType::ListView(_)
        | ArrowType::FixedSizeList(..) => {
            let v: ArrayRef = match c.data_type() {
                ArrowType::List(_) => c.as_list::<i32>().value(r),
                ArrowType::LargeList(_) => c.as_list::<i64>().value(r),
                _ => return Value::from(format!("{:?}", c.data_type())),
            };
            Value::Array((0..v.len()).map(|i| cell(v.as_ref(), i)).collect())
        }
        ArrowType::Map(_, _) => {
            let m = c.as_map().value(r);
            let (k, v) = (m.column(0), m.column(1));
            let mut out = serde_json::Map::new();
            for i in 0..m.len() {
                let key = match cell(k.as_ref(), i) {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                out.insert(key, cell(v.as_ref(), i));
            }
            Value::Object(out)
        }
        other => Value::from(format!("{other:?}")),
    }
}

/// The schema as Spark's `simpleString` and the rows, sorted.
fn answer(schema: &arrow_schema::Schema, batches: &[RecordBatch]) -> (String, Vec<String>) {
    let fields: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| {
            let t = DataType::from_arrow(f.data_type())
                .map_or_else(|e| format!("{e}"), |t| t.simple_string());
            format!("{}:{t}", f.name())
        })
        .collect();
    let mut rows = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let row: Vec<Value> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
            rows.push(Value::Array(row).to_string());
        }
    }
    rows.sort();
    (format!("struct<{}>", fields.join(",")), rows)
}

/// What differs between Krishiv's answer and Spark's `want`, if anything.
fn error_difference(e: &krishiv_common::error::KrishivError, want: &Value) -> Option<String> {
    let text = e.to_string();
    let Some(err) = want.get("error") else {
        return Some(format!("raised {text}"));
    };
    let condition = err["condition"].as_str().unwrap_or_default();
    if condition.is_empty() {
        return Some(format!("raised {text}, Spark {}", err["message"]));
    }
    // A legacy class has no `[CLASS]` prefix in its message.
    let legacy = condition.starts_with("_LEGACY_ERROR_TEMP_") && e.message() == err["message"];
    if !legacy && !text.starts_with(&format!("[{condition}]")) {
        return Some(format!(
            "raised {text}, Spark [{condition}] {}",
            err["message"]
        ));
    }
    let worker = &err["worker"];
    if worker.is_object() {
        let line = format!(
            "[{}] {}",
            worker["error_class"].as_str().unwrap_or_default(),
            worker["message"].as_str().unwrap_or_default()
        );
        return (!text.contains(&line)).then(|| format!("no {line:?} in: {text}"));
    }
    if let Some(p) = err["python_error"].as_str() {
        return (!text.contains(p)).then(|| format!("no {p:?} in: {text}"));
    }
    None
}

const VIEWS: [&str; 2] = [
    "CREATE OR REPLACE TEMP VIEW t AS SELECT * FROM VALUES (2, 'a', 20), (1, 'a', 10), (4, 'b', CAST(NULL AS INT)), \
     (3, 'b', 30), (5, 'c', 50) AS t(id, g, v)",
    "CREATE OR REPLACE TEMP VIEW e AS SELECT * FROM t WHERE id < 0",
];

/// One case run: `None` when it answers as Spark did.
async fn run_case(
    session: &Session,
    udfs: &SessionUdfs,
    worker: &str,
    record: &Value,
    want: &Value,
) -> Option<String> {
    for u in record["udtfs"].as_array().expect("udtfs") {
        let code = i32::try_from(u["eval_type"].as_i64().expect("eval type")).expect("code");
        let eval_type = PythonEvalType::from_code(code).expect("a UDTF eval type");
        let return_type = u["return_type"]
            .as_str()
            .map(|t| match t.strip_prefix("json:") {
                Some(json) => DataType::from_json(json).expect("return type"),
                None => krishiv_types::parse_data_type(t).expect("return type"),
            });
        let name = u["name"].as_str().expect("name");
        let def = PythonUdtfDef {
            name: name.to_owned(),
            eval_type,
            return_type,
            deterministic: u["deterministic"].as_bool().unwrap_or(false),
        };
        let entry = match session.register_python_udtf(def) {
            Ok(entry) => entry,
            // A return type that is no struct: the client refuses it (`UDTF_RETURN_TYPE_MISMATCH`) before it
            // registers anything.
            Err(e) => {
                let condition = want["error"]["condition"].as_str().unwrap_or_default();
                return (condition != "UDTF_RETURN_TYPE_MISMATCH")
                    .then(|| format!("registration: {e}"));
            }
        };
        let hex = u["command"].as_str().expect("command");
        let command: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
            .collect();
        udfs.register(
            name.to_lowercase(),
            entry.version,
            PythonPayload {
                command: Arc::from(command),
                eval_type: code,
                python_ver: worker.to_owned(),
            },
        );
    }
    let sql = record["sql"].as_str().expect("sql");
    let run = async {
        let df = session.sql(sql).await?;
        let schema = df.schema().await?;
        let batches = df.collect().await?;
        Ok::<_, krishiv_common::error::KrishivError>((schema, batches))
    };
    match run.await {
        Ok((schema, batches)) => {
            let (got_schema, got_rows) = answer(&schema, &batches);
            if want.get("error").is_some() {
                return Some(format!(
                    "returned {got_schema} {got_rows:?}, Spark raised {}",
                    want["error"]
                ));
            }
            let mut want_rows: Vec<String> = want["rows"]
                .as_array()
                .expect("rows")
                .iter()
                .map(Value::to_string)
                .collect();
            want_rows.sort();
            (got_schema != want["schema"] || got_rows != want_rows).then(|| {
                format!(
                    "returned {got_schema} {got_rows:?}, Spark {} {want_rows:?}",
                    want["schema"]
                )
            })
        }
        Err(e) => error_difference(&e, want),
    }
}

/// Every SQL case of `goldens.json` answers as PySpark 4.2.0 did, on the pool.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn udtfs_match_pyspark_4_2_0() {
    let Some(worker_py) = interpreter() else {
        return;
    };
    let goldens: Value = serde_json::from_str(
        &std::fs::read_to_string(repo().join("python/tests/pyspark_udf/udtf/goldens.json"))
            .expect("goldens"),
    )
    .expect("json");
    assert_eq!(goldens["spark"], "4.2.0");
    let session = Session::new().expect("session");
    for (k, v) in goldens["conf"].as_object().expect("conf") {
        session
            .conf()
            .set(k, v.as_str().expect("conf value"))
            .expect("set conf");
    }
    // The client asks the server for the setting that decides a bare `@udtf`'s eval type.
    let arrow_default = session
        .conf()
        .get("spark.sql.execution.pythonUDTF.arrow.enabled")
        .expect("the setting is registered");
    assert_eq!(
        Value::from(arrow_default.clone()),
        goldens["defaults"]["spark.sql.execution.pythonUDTF.arrow.enabled"]
    );
    let records = export(&worker_py, &arrow_default);
    let udfs = Arc::new(SessionUdfs::new(
        pool(&worker_py),
        Owner {
            principal: "alice".to_owned(),
            session: "s1".to_owned(),
        },
    ));
    session.set_python_udf_runtime(Arc::clone(&udfs) as Arc<_>);
    for v in VIEWS {
        session
            .sql(v)
            .await
            .expect("view")
            .collect()
            .await
            .expect("view");
    }
    let worker = python_version(&worker_py);
    let cases = goldens["cases"].as_object().expect("cases");
    assert_eq!(cases.len(), 354);
    let mut differ: BTreeMap<String, String> = BTreeMap::new();
    let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
    for (case, want) in cases {
        let record = &records[case];
        let diff = match record["kind"].as_str() {
            Some("sql") => {
                *tally.entry("sql").or_default() += 1;
                run_case(&session, &udfs, &worker, record, want).await
            }
            Some("client_error") => {
                *tally.entry("client").or_default() += 1;
                (record["condition"] != want["error"]["condition"])
                    .then(|| format!("client raised {record}"))
            }
            _ => {
                *tally.entry("dataframe").or_default() += 1;
                continue;
            }
        };
        let gap = KNOWN_GAPS
            .iter()
            .chain(INCOMPATIBLE)
            .find(|(c, _)| c == case);
        match (diff, gap) {
            (None, None) => *tally.entry("matched").or_default() += 1,
            (Some(_), Some(_)) => *tally.entry("listed").or_default() += 1,
            (None, Some((_, reason))) => {
                differ.insert(
                    case.clone(),
                    format!("matches now; remove it from the lists ({reason})"),
                );
            }
            (Some(d), None) => {
                differ.insert(case.clone(), d);
            }
        }
    }
    udfs.release();
    eprintln!("UDTF goldens: {tally:?}");
    assert!(
        differ.is_empty(),
        "{} cases differ: {differ:#?}",
        differ.len()
    );
}

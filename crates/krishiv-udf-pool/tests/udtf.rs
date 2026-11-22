// SPDX-License-Identifier: Apache-2.0
//! Python table functions (UDTFs, eval types 300, 301, 302) through the pool with real workers (P3-T014 PR-26c-3
//! part B; design `P3-T085-embedded-python-udf-kinds.md` §2, §3.3): a task per Spark task fed the call's
//! argument columns, `eval` per row (per batch for 302), `terminate` at each `PARTITION BY` boundary and at the
//! end; the `UDTF_*` errors of pyspark 4.2.0's `read_udtf`; and `analyze` on a worker
//! (`UdfRuntime::analyze_udtf`), whose pickled result reaches the UDTF's `__init__`.
//!
//! As `tests/relations.rs`: the interpreter comes from `KRISHIV_TEST_UDF_PYTHON`, and without it every test
//! passes after a note unless `KRISHIV_UDF_POOL_REQUIRED` is set. Workers start sandboxed, through this test
//! binary ([`worker_child`]).
#![cfg(target_os = "linux")]
#![expect(
    clippy::print_stderr,
    clippy::expect_used,
    reason = "the skip note when no interpreter is configured; test helpers sit outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Arc;

use arrow_array::types::Date32Type;
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, Date32Array, Decimal64Array, Int32Array, Int64Array,
    ListArray, RecordBatch, StringViewArray, StructArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::python_eval::UdfRuntime;
use krishiv_exec::python_relation::{Frame, PreparedRelationUdf, RelationCallSpec, RunnerConf};
use krishiv_exec::python_udtf::{UdtfAnalyzeArg, UdtfAnalyzeSpec, UdtfArg, UdtfCall};
use krishiv_plan::function::PythonEvalType;
use krishiv_types::{DataType, StructField, StructType};
use krishiv_udf_pool::{Launcher, Owner, Pool, PoolConfig, PythonPayload, SessionUdfs};

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

fn python() -> Option<PathBuf> {
    match std::env::var_os("KRISHIV_TEST_UDF_PYTHON") {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            assert!(
                std::env::var_os("KRISHIV_UDF_POOL_REQUIRED").is_none(),
                "KRISHIV_UDF_POOL_REQUIRED is set but KRISHIV_TEST_UDF_PYTHON is not"
            );
            eprintln!("note: KRISHIV_TEST_UDF_PYTHON is not set; the UDTF tests did not run");
            None
        }
    }
}

fn udfs(python: &Path) -> (Pool, SessionUdfs) {
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
    let udfs = SessionUdfs::new(
        pool.clone(),
        Owner {
            principal: "alice".to_owned(),
            session: "s1".to_owned(),
            tenant: None,
            capped: true,
        },
    );
    (pool, udfs)
}

/// Spark Connect's UDTF `command` for `body` (a Python class `F`): the class alone, cloudpickled.
fn udtf_command(python: &Path, body: &str, eval_type: i32) -> PythonPayload {
    let code = format!(
        "import sys\nfrom pyspark import cloudpickle\nfrom pyspark.sql.types import *\n\
         from pyspark.sql.udtf import AnalyzeArgument, AnalyzeResult\n{body}\n\
         sys.stdout.buffer.write(cloudpickle.dumps(F))"
    );
    let out = Command::new(python)
        .args(["-c", &code])
        .output()
        .expect("run the interpreter");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let version = Command::new(python)
        .args([
            "-c",
            "import sys; print('%d.%d' % sys.version_info[:2], end='')",
        ])
        .output()
        .expect("version");
    PythonPayload {
        command: Arc::from(out.stdout),
        eval_type,
        python_ver: String::from_utf8_lossy(&version.stdout).into_owned(),
    }
}

static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Registers `payload` under a fresh name; the name.
fn register(udfs: &SessionUdfs, payload: PythonPayload) -> String {
    let name = format!(
        "t{}",
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    udfs.register(name.clone(), 1, payload);
    name
}

/// The call site of UDTF `name` of `eval_type` over frame columns `inputs`, declared to return the `BIGINT`
/// columns `output`.
fn prepare(
    udfs: &SessionUdfs,
    name: &str,
    eval_type: PythonEvalType,
    inputs: &[(&str, DataType)],
    output: &[&str],
    call: UdtfCall,
) -> Result<Arc<dyn PreparedRelationUdf>, KrishivError> {
    let output_arrow: SchemaRef = Arc::new(Schema::new(
        output
            .iter()
            .map(|n| Field::new(*n, ArrowType::Int64, true))
            .collect::<Vec<_>>(),
    ));
    let spec = RelationCallSpec {
        name: name.to_owned(),
        version: 1,
        eval_type,
        thread_safe: true,
        input_types: vec![inputs.iter().map(|(_, t)| t.clone()).collect()],
        input_names: vec![inputs.iter().map(|(n, _)| (*n).to_owned()).collect()],
        key_count: 0,
        arity: inputs.len(),
        output_names: output.iter().map(|n| (*n).to_owned()).collect(),
        output_types: output.iter().map(|_| DataType::Long).collect(),
        output_arrow,
        conf: RunnerConf {
            max_records_per_batch: 10_000,
            settings: BTreeMap::new(),
            ..RunnerConf::default()
        },
        udtf: Some(call),
    };
    udfs.prepare_relation(&spec)
}

fn positional(n: usize) -> UdtfCall {
    UdtfCall {
        args: (0..n)
            .map(|offset| UdtfArg { offset, name: None })
            .collect(),
        ..UdtfCall::default()
    }
}

/// Sends `frames` to a new task of `prepared` while it reads the results.
fn run(
    prepared: &dyn PreparedRelationUdf,
    frames: Vec<Frame>,
) -> Result<Vec<RecordBatch>, KrishivError> {
    let task = prepared.open(None)?;
    let send = async {
        for frame in frames {
            task.send(frame).await?;
        }
        Ok::<(), KrishivError>(())
    };
    let receive = async {
        let mut out = Vec::new();
        while let Some(batch) = task.next().await? {
            out.push(batch);
        }
        Ok::<_, KrishivError>(out)
    };
    let (sent, received) = block_on(futures::future::join(send, receive));
    let out = received?;
    sent?;
    Ok(out)
}

fn longs(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

fn frame(columns: Vec<(&str, ArrayRef)>) -> Frame {
    Frame::Batch {
        side: 0,
        batch: RecordBatch::try_from_iter(columns).expect("batch"),
    }
}

/// The rows of `batches` as text, `|` between cells.
fn rows(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| match c.as_any().downcast_ref::<Int64Array>() {
                    Some(_) if c.is_null(r) => "null".to_owned(),
                    Some(a) => a.value(r).to_string(),
                    None => format!("{:?}", c.data_type()),
                })
                .collect();
            out.push(cells.join("|"));
        }
    }
    out
}

/// Two batches of `x` (1, 2, then 3), then the end.
fn x_frames() -> Vec<Frame> {
    vec![
        frame(vec![("_0", longs(&[1, 2]))]),
        frame(vec![("_0", longs(&[3]))]),
        Frame::End,
    ]
}

/// `eval` runs per row and `terminate` once at the end of the task, `cleanup` after it, for `@udtf` (300)
/// and `@udtf(useArrow=True)` (301); `@arrow_udtf` (302) gets each batch's column as one Arrow array.
#[test]
fn udtfs_run_eval_per_row_and_terminate_at_the_end() {
    let Some(py) = python() else { return };
    let (_pool, udfs) = udfs(&py);
    let body = "class F:\n    def __init__(self):\n        self.n = 0\n    def eval(self, x):\n        \
                self.n += 1\n        yield x, x * 10\n    def terminate(self):\n        yield -1, self.n";
    for (code, eval_type) in [
        (300, PythonEvalType::TableUdf),
        (301, PythonEvalType::ArrowTableUdf),
    ] {
        let name = register(&udfs, udtf_command(&py, body, code));
        let prepared = prepare(
            &udfs,
            &name,
            eval_type,
            &[("_0", DataType::Long)],
            &["a", "b"],
            positional(1),
        )
        .expect("prepare");
        let out = run(prepared.as_ref(), x_frames()).unwrap_or_else(|e| panic!("{code}: {e}"));
        assert_eq!(rows(&out), ["1|10", "2|20", "3|30", "-1|3"], "{code}");
    }
    let arrow = "import pyarrow as pa\nimport pyarrow.compute as pc\nclass F:\n    \
                 def __init__(self):\n        self.batches = 0\n    def eval(self, x):\n        \
                 self.batches += 1\n        yield pa.table({'a': x, 'b': pc.call_function('multiply', [x, 10])})\n    \
                 def terminate(self):\n        yield pa.table({'a': [-1], 'b': [self.batches]})";
    let name = register(&udfs, udtf_command(&py, arrow, 302));
    let prepared = prepare(
        &udfs,
        &name,
        PythonEvalType::ArrowUdtf,
        &[("_0", DataType::Long)],
        &["a", "b"],
        positional(1),
    )
    .expect("prepare");
    let out = run(prepared.as_ref(), x_frames()).expect("302");
    assert_eq!(rows(&out), ["1|10", "2|20", "3|30", "-1|2"]);
}

/// A `LATERAL` call (300, 301): the input columns after the arguments are carried, not passed; each result
/// row leads with the values of the row whose `eval` gave it, `terminate`'s rows with the last row's
/// (Spark's `EvalPythonUDTFExec` join), and a row giving no result shows nowhere.
#[test]
fn a_lateral_call_carries_the_input_row_of_each_result() {
    let Some(py) = python() else { return };
    let (_pool, udfs) = udfs(&py);
    let body = "class F:\n    def __init__(self):\n        self.n = 0\n    def eval(self, x):\n        \
                self.n += 1\n        for i in range(x - 1):\n            yield x * 10 + i,\n    \
                def terminate(self):\n        yield -self.n,";
    for (code, eval_type) in [
        (300, PythonEvalType::TableUdf),
        (301, PythonEvalType::ArrowTableUdf),
    ] {
        let name = register(&udfs, udtf_command(&py, body, code));
        let call = UdtfCall {
            passthrough: vec![1, 2],
            ..positional(1)
        };
        let prepared = prepare(
            &udfs,
            &name,
            eval_type,
            &[
                ("_0", DataType::Long),
                ("id", DataType::Long),
                ("w", DataType::Long),
            ],
            &["id", "w", "r"],
            call,
        )
        .expect("prepare");
        let frames = vec![
            frame(vec![
                ("_0", longs(&[1, 2])),
                ("id", longs(&[100, 200])),
                ("w", longs(&[5, 6])),
            ]),
            frame(vec![
                ("_0", longs(&[3])),
                ("id", longs(&[300])),
                ("w", longs(&[7])),
            ]),
            Frame::End,
        ];
        let out = run(prepared.as_ref(), frames).unwrap_or_else(|e| panic!("{code}: {e}"));
        assert_eq!(
            rows(&out),
            ["200|6|20", "300|7|30", "300|7|31", "300|7|-3"],
            "{code}"
        );
    }
}

/// A `TABLE` argument with `PARTITION BY`: the row is a struct whose projected partition values are removed
/// before `eval` sees it, and `terminate` plus a fresh instance mark each change of those values
/// (`UDTFWithPartitions`, `ArrowUDTFWithPartition`).
#[test]
fn a_table_argument_partitioned_terminates_each_partition() {
    let Some(py) = python() else { return };
    let (_pool, udfs) = udfs(&py);
    let row_type = DataType::Struct(StructType::new([
        StructField::new("v".to_owned(), DataType::Long, true),
        StructField::new("partition_by_0".to_owned(), DataType::Long, true),
    ]));
    let table = |v: &[i64], k: &[i64]| -> ArrayRef {
        Arc::new(StructArray::from(vec![
            (Arc::new(Field::new("v", ArrowType::Int64, true)), longs(v)),
            (
                Arc::new(Field::new("partition_by_0", ArrowType::Int64, true)),
                longs(k),
            ),
        ]))
    };
    let frames = || {
        vec![
            frame(vec![("_0", table(&[1, 2, 3], &[7, 7, 8]))]),
            frame(vec![("_0", table(&[4], &[8]))]),
            Frame::End,
        ]
    };
    let call = UdtfCall {
        args: vec![UdtfArg {
            offset: 0,
            name: None,
        }],
        partition_child_indexes: vec![1],
        table_arg_offsets: vec![0],
        analyze_result: None,
        passthrough: Vec::new(),
    };
    // The row has `v` only; the sum of a partition comes out at its end.
    let rows_body = "class F:\n    def __init__(self):\n        self.s = 0\n    def eval(self, row):\n        \
                     assert list(row.asDict()) == ['v'], row\n        self.s += row['v']\n        \
                     return iter(())\n    def terminate(self):\n        yield self.s, 0";
    let name = register(&udfs, udtf_command(&py, rows_body, 300));
    let prepared = prepare(
        &udfs,
        &name,
        PythonEvalType::TableUdf,
        &[("_0", row_type.clone())],
        &["sum", "z"],
        call.clone(),
    )
    .expect("prepare");
    let out = run(prepared.as_ref(), frames()).expect("300");
    assert_eq!(rows(&out), ["3|0", "7|0"]);
    let arrow_body = "import pyarrow as pa\nclass F:\n    def __init__(self):\n        self.s = 0\n    \
                      def eval(self, batch):\n        assert batch.schema.names == ['v'], batch.schema\n        \
                      self.s += sum(batch.column(0).to_pylist())\n        return iter(())\n    \
                      def terminate(self):\n        yield pa.table({'sum': [self.s], 'z': [0]})";
    let name = register(&udfs, udtf_command(&py, arrow_body, 302));
    let prepared = prepare(
        &udfs,
        &name,
        PythonEvalType::ArrowUdtf,
        &[("_0", row_type)],
        &["sum", "z"],
        call,
    )
    .expect("prepare");
    let out = run(prepared.as_ref(), frames()).expect("302");
    assert_eq!(rows(&out), ["3|0", "7|0"]);
}

/// `read_udtf`'s errors reach the user as `PYTHON_EXCEPTION` carrying the PySpark error class.
#[test]
fn udtf_errors_carry_spark_s_classes() {
    let Some(py) = python() else { return };
    let (_pool, udfs) = udfs(&py);
    let cases = [
        (
            300,
            "class F:\n    def eval(self, x):\n        raise ValueError('boom')",
            "[UDTF_EXEC_ERROR] User defined table function encountered an error in the 'eval' method: boom",
        ),
        (
            300,
            "class F:\n    def eval(self, x):\n        return 1",
            "UDTF_RETURN_NOT_ITERABLE",
        ),
        (
            300,
            "class F:\n    def eval(self, x):\n        yield (x,)",
            "UDTF_RETURN_SCHEMA_MISMATCH",
        ),
        (
            300,
            "class F:\n    def eval(self, x, y):\n        yield x, y",
            "UDTF_EVAL_METHOD_ARGUMENTS_DO_NOT_MATCH_SIGNATURE",
        ),
        (
            301,
            "class F:\n    def eval(self, x):\n        yield 'a', 'b'",
            "UDTF_ARROW_DATA_CONVERSION_ERROR",
        ),
        (
            302,
            "class F:\n    def eval(self, x):\n        yield 1",
            "UDTF_ARROW_TYPE_CONVERSION_ERROR",
        ),
        (
            300,
            "class F:\n    def __init__(self, a):\n        pass\n    def eval(self, x):\n        yield x, x",
            "UDTF_CONSTRUCTOR_INVALID_NO_ANALYZE_METHOD",
        ),
    ];
    for (code, body, class) in cases {
        let eval_type = PythonEvalType::from_code(code).expect("code");
        let name = register(&udfs, udtf_command(&py, body, code));
        let prepared = prepare(
            &udfs,
            &name,
            eval_type,
            &[("_0", DataType::Long)],
            &["a", "b"],
            positional(1),
        )
        .expect("prepare");
        let err = run(prepared.as_ref(), x_frames()).expect_err(body);
        assert_eq!(
            err.error_class(),
            ErrorClass::PythonException,
            "{body}: {err}"
        );
        assert!(err.to_string().contains(class), "{body}: {err}");
    }
}

/// A UDTF call site needs its call data, and a UDTF payload runs only as its own kind.
#[test]
fn a_udtf_call_needs_its_call_data_and_its_own_kind() {
    let Some(py) = python() else { return };
    let (_pool, udfs) = udfs(&py);
    let name = register(
        &udfs,
        udtf_command(
            &py,
            "class F:\n    def eval(self, x):\n        yield x, x",
            300,
        ),
    );
    let mut call = positional(1);
    let refused = prepare(
        &udfs,
        &name,
        PythonEvalType::ArrowTableUdf,
        &[("_0", DataType::Long)],
        &["a", "b"],
        call.clone(),
    );
    let Err(e) = refused else {
        panic!("a 300 payload ran as 301")
    };
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivPythonUdfEvalTypeMismatch
    );
    call.args.clear();
    let output_arrow: SchemaRef =
        Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int64, true)]));
    let spec = RelationCallSpec {
        name,
        version: 1,
        eval_type: PythonEvalType::TableUdf,
        thread_safe: true,
        input_types: vec![vec![DataType::Long]],
        input_names: vec![vec!["_0".to_owned()]],
        key_count: 0,
        arity: 1,
        output_names: vec!["a".to_owned()],
        output_types: vec![DataType::Long],
        output_arrow,
        conf: RunnerConf::default(),
        udtf: None,
    };
    let Err(e) = udfs.prepare_relation(&spec) else {
        panic!("a UDTF call without its call data was prepared")
    };
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
}

const ANALYZED: &str = "class F:\n    def __init__(self, analyze_result):\n        \
    self.k = analyze_result.schema.fields[0].name\n    \
    @staticmethod\n    def analyze(x, n):\n        \
    assert n.isConstantExpression and n.value == 3 and n.dataType == IntegerType(), n\n        \
    return AnalyzeResult(schema=StructType().add('c' + str(n.value), LongType()).add('k', LongType()),\n            \
    partitionBy=[], orderBy=[])\n    \
    def eval(self, x, n):\n        yield x, len(self.k)";

fn analyze_spec(name: &str) -> UdtfAnalyzeSpec {
    UdtfAnalyzeSpec {
        name: name.to_owned(),
        version: 1,
        args: vec![
            UdtfAnalyzeArg {
                data_type: DataType::Long,
                value: None,
                is_table: false,
                name: None,
            },
            UdtfAnalyzeArg {
                data_type: DataType::Integer,
                value: Some(Arc::new(Int32Array::from(vec![3]))),
                is_table: false,
                name: Some("n".to_owned()),
            },
        ],
    }
}

/// `analyze` runs on a worker at analysis time; its schema is the call's result and its pickled
/// `AnalyzeResult` reaches `__init__` of every instance the call's tasks make.
#[test]
fn analyze_gives_the_schema_and_its_result_reaches_init() {
    let Some(py) = python() else { return };
    let (_pool, udfs) = udfs(&py);
    let name = register(&udfs, udtf_command(&py, ANALYZED, 300));
    let analysis = block_on(udfs.analyze_udtf(&analyze_spec(&name))).expect("analyze");
    let DataType::Struct(schema) = &analysis.schema else {
        panic!("{:?}", analysis.schema)
    };
    let names: Vec<&str> = schema.fields().iter().map(StructField::name).collect();
    assert_eq!(names, ["c3", "k"]);
    assert!(!analysis.with_single_partition);
    let call = UdtfCall {
        args: vec![
            UdtfArg {
                offset: 0,
                name: None,
            },
            UdtfArg {
                offset: 1,
                name: Some("n".to_owned()),
            },
        ],
        analyze_result: Some(Arc::clone(&analysis.pickled)),
        ..UdtfCall::default()
    };
    let prepared = prepare(
        &udfs,
        &name,
        PythonEvalType::TableUdf,
        &[("_0", DataType::Long), ("_1", DataType::Long)],
        &["c3", "k"],
        call,
    )
    .expect("prepare");
    let frames = vec![
        frame(vec![("_0", longs(&[5, 6])), ("_1", longs(&[3, 3]))]),
        Frame::End,
    ];
    let out = run(prepared.as_ref(), frames).expect("run");
    // `self.k` is the analyzed schema's first field name, `c3`.
    assert_eq!(rows(&out), ["5|2", "6|2"]);
}

/// A failing `analyze` is `TABLE_VALUED_FUNCTION_FAILED_TO_ANALYZE_IN_PYTHON` with Spark's message (the text
/// after the traceback's last `PySparkValueError:`); a UDTF without `analyze`, or an `@arrow_udtf`, too.
#[test]
fn a_failing_analyze_is_spark_s_class() {
    let Some(py) = python() else { return };
    let (_pool, udfs) = udfs(&py);
    let cases = [
        (
            "class F:\n    @staticmethod\n    def analyze(x, n):\n        return 1\n    \
             def eval(self, x, n):\n        yield x,",
            300,
            "Failed to analyze the Python user defined table function: Failed to evaluate the user-defined table \
             function 't",
        ),
        (
            "class F:\n    @staticmethod\n    def analyze(x):\n        return None\n    \
             def eval(self, x):\n        yield x,",
            300,
            "because the function arguments did not match the expected signature of the static 'analyze' method",
        ),
        (
            "class F:\n    @staticmethod\n    def analyze(x, n):\n        raise ValueError('nope')\n    \
             def eval(self, x, n):\n        yield x,",
            301,
            "ValueError: nope",
        ),
        (
            "class F:\n    def eval(self, x, n):\n        yield x,",
            300,
            "has not implemented the 'analyze' static method",
        ),
        // Spark's `worker_run` catches every exception, `SystemExit` too (review R1-7).
        (
            "class F:\n    @staticmethod\n    def analyze(x, n):\n        raise SystemExit(3)\n    \
             def eval(self, x, n):\n        yield x,",
            300,
            "SystemExit: 3",
        ),
    ];
    for (body, code, want) in cases {
        let name = register(&udfs, udtf_command(&py, body, code));
        let err = block_on(udfs.analyze_udtf(&analyze_spec(&name))).expect_err(body);
        assert_eq!(
            err.error_class(),
            ErrorClass::TableValuedFunctionFailedToAnalyzeInPython,
            "{body}: {err}"
        );
        assert!(err.to_string().contains(want), "{body}: {err}");
    }
    // A payload that does not unpickle fails the analysis with its class too: Spark reads the UDTF inside
    // `analyze_udtf.py`'s `main` (review R1-7).
    let mut broken = udtf_command(&py, "class F:\n    pass", 300);
    broken.command = Arc::from(&b"not a pickle"[..]);
    let name = register(&udfs, broken);
    let err = block_on(udfs.analyze_udtf(&analyze_spec(&name))).expect_err("unpickle");
    assert_eq!(
        err.error_class(),
        ErrorClass::TableValuedFunctionFailedToAnalyzeInPython,
        "{err}"
    );
    let name = register(
        &udfs,
        udtf_command(
            &py,
            "class F:\n    def eval(self, x):\n        yield x,",
            302,
        ),
    );
    let err = block_on(udfs.analyze_udtf(&analyze_spec(&name))).expect_err("302");
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivPythonUdfEvalTypeMismatch
    );
}

/// One `analyze` argument of `data_type`, constant with `value` (one row) when given.
fn analyze_arg(data_type: DataType, value: Option<ArrayRef>) -> UdtfAnalyzeArg {
    UdtfAnalyzeArg {
        data_type,
        value,
        is_table: false,
        name: None,
    }
}

/// A large constant argument and a large `AnalyzeResult` (Spark's pattern of a subclass that carries state) do
/// not ride in the size-capped control messages (review R1-2): both reach `analyze` and `__init__` whole.
#[test]
fn large_analyze_arguments_and_results_reach_init() {
    let Some(py) = python() else { return };
    let (_pool, udfs) = udfs(&py);
    let body = "import dataclasses\n@dataclasses.dataclass\nclass Big(AnalyzeResult):\n    \
                state: str = ''\nclass F:\n    def __init__(self, analyze_result):\n        \
                self.n = len(analyze_result.state)\n    @staticmethod\n    def analyze(x, s):\n        \
                return Big(schema=StructType().add('a', LongType()).add('n', LongType()), state=s.value * 2)\n    \
                def eval(self, x, s):\n        yield x, self.n";
    let name = register(&udfs, udtf_command(&py, body, 300));
    let big = "x".repeat(300 * 1024);
    let spec = UdtfAnalyzeSpec {
        name: name.clone(),
        version: 1,
        args: vec![
            analyze_arg(DataType::Long, None),
            analyze_arg(
                DataType::String,
                Some(Arc::new(StringViewArray::from(vec![big.as_str()]))),
            ),
        ],
    };
    let analysis = block_on(udfs.analyze_udtf(&spec)).expect("analyze");
    assert!(analysis.pickled.len() > 600 * 1024);
    let call = UdtfCall {
        analyze_result: Some(Arc::clone(&analysis.pickled)),
        ..positional(2)
    };
    let prepared = prepare(
        &udfs,
        &name,
        PythonEvalType::TableUdf,
        &[("_0", DataType::Long), ("_1", DataType::Long)],
        &["a", "n"],
        call,
    )
    .expect("prepare");
    let frames = vec![
        frame(vec![("_0", longs(&[5])), ("_1", longs(&[0]))]),
        Frame::End,
    ];
    let out = run(prepared.as_ref(), frames).expect("run");
    assert_eq!(rows(&out), ["5|614400"]);
}

/// An `analyze` result over the limit (64 MiB, as the server reads it) fails the analysis with Spark's class,
/// and the worker stays usable (review R1-2): the next `analyze` answers.
#[test]
fn an_oversize_analyze_result_is_spark_s_class() {
    let Some(py) = python() else { return };
    let (pool, udfs) = udfs(&py);
    let body = "import dataclasses\n@dataclasses.dataclass\nclass Big(AnalyzeResult):\n    \
                state: bytes = b''\nclass F:\n    @staticmethod\n    def analyze(x, n):\n        \
                return Big(schema=StructType().add('a', LongType()), state=bytes(n.value << 20))\n    \
                def eval(self, x, n):\n        yield x,";
    let name = register(&udfs, udtf_command(&py, body, 300));
    let spec = |mib: i64| UdtfAnalyzeSpec {
        name: name.clone(),
        version: 1,
        args: vec![
            analyze_arg(DataType::Long, None),
            analyze_arg(DataType::Long, Some(longs(&[mib]))),
        ],
    };
    let err = block_on(udfs.analyze_udtf(&spec(40))).expect_err("over the limit");
    assert_eq!(
        err.error_class(),
        ErrorClass::TableValuedFunctionFailedToAnalyzeInPython,
        "{err}"
    );
    assert!(err.to_string().contains("over the limit"), "{err}");
    let live = pool.live_workers();
    block_on(udfs.analyze_udtf(&spec(1))).expect("the worker still answers");
    assert_eq!(pool.live_workers(), live, "no worker was replaced");
}

/// `analyze` sees each constant argument as the Python value of its type, as Spark converts it
/// (`analyze_udtf.py` L76-82 at v4.2.0: the pickled internal value through `dataType.fromInternal`): a `date`,
/// a naive local `datetime` for `timestamp`, a naive `datetime` for `timestamp_ntz`, a `Decimal` keeping its
/// scale, `bytes` for `binary` (`spark.sql.execution.pyspark.binaryAsBytes`, true by default), a list and a
/// `Row` of converted values, and `None` for a `NULL` constant (review R1-3).
#[test]
fn analyze_gets_constants_as_python_values_of_their_type() {
    let Some(py) = python() else { return };
    let (_pool, udfs) = udfs(&py);
    let body = "import datetime, decimal\nfrom pyspark.sql import Row\nclass F:\n    @staticmethod\n    \
                def analyze(d, ts, ntz, dec, b, arr, st, nul):\n        \
                want = [\n            \
                (d, datetime.date(2024, 1, 2)),\n            \
                (ts, datetime.datetime.fromtimestamp(1704164645).replace(microsecond=123456)),\n            \
                (ntz, datetime.datetime(2024, 1, 2, 3, 4, 5, 123456)),\n            \
                (dec, decimal.Decimal('12.50')),\n            \
                (b, b'\\x00\\xff'),\n            \
                (arr, [datetime.date(1970, 1, 2), None]),\n            \
                (st, Row(x=datetime.date(1970, 1, 1), y=7)),\n            \
                (nul, None),\n        ]\n        \
                for i, (arg, value) in enumerate(want):\n            \
                assert arg.isConstantExpression, i\n            \
                assert type(arg.value) is type(value) and arg.value == value, (i, arg.value, value)\n        \
                assert str(dec.value) == '12.50' and ts.value.tzinfo is None and ntz.value.tzinfo is None\n        \
                return AnalyzeResult(schema=StructType().add('a', LongType()))\n    \
                def eval(self, *args):\n        yield 1,";
    let name = register(&udfs, udtf_command(&py, body, 300));
    let st = DataType::Struct(StructType::new([
        StructField::new("x", DataType::Date, true),
        StructField::new("y", DataType::Integer, true),
    ]));
    let spec = UdtfAnalyzeSpec {
        name,
        version: 1,
        args: vec![
            analyze_arg(
                DataType::Date,
                Some(Arc::new(Date32Array::from(vec![19724]))),
            ),
            analyze_arg(
                DataType::Timestamp,
                Some(Arc::new(
                    TimestampMicrosecondArray::from(vec![1_704_164_645_123_456_i64])
                        .with_timezone("UTC"),
                )),
            ),
            analyze_arg(
                DataType::TimestampNtz,
                Some(Arc::new(TimestampMicrosecondArray::from(vec![
                    1_704_164_645_123_456_i64,
                ]))),
            ),
            analyze_arg(
                DataType::Decimal(krishiv_types::DecimalType::try_new(5, 2).expect("decimal")),
                Some(Arc::new(
                    Decimal64Array::from(vec![1250])
                        .with_precision_and_scale(5, 2)
                        .expect("decimal"),
                )),
            ),
            analyze_arg(
                DataType::Binary,
                Some(Arc::new(BinaryViewArray::from(vec![&[0_u8, 255][..]]))),
            ),
            analyze_arg(
                DataType::Array(Arc::new(krishiv_types::ArrayType::new(
                    DataType::Date,
                    true,
                ))),
                Some(Arc::new(
                    ListArray::from_iter_primitive::<Date32Type, _, _>(vec![Some(vec![
                        Some(1),
                        None,
                    ])]),
                )),
            ),
            analyze_arg(
                st,
                Some(Arc::new(StructArray::from(vec![
                    (
                        Arc::new(Field::new("x", ArrowType::Date32, true)),
                        Arc::new(Date32Array::from(vec![0])) as ArrayRef,
                    ),
                    (
                        Arc::new(Field::new("y", ArrowType::Int32, true)),
                        Arc::new(Int32Array::from(vec![7])) as ArrayRef,
                    ),
                ]))),
            ),
            analyze_arg(
                DataType::Integer,
                Some(Arc::new(Int32Array::from(vec![None::<i32>]))),
            ),
        ],
    };
    block_on(udfs.analyze_udtf(&spec)).expect("analyze");
}

/// SQL that spells the encoded call name of an inline (unregistered) UDTF, forged: nothing was granted to the
/// runtime under that name and version, so the call is refused with `KRISHIV_UDF_NOT_AVAILABLE` and no worker
/// starts, whether the name declares a result (the relation task is refused) or not (`analyze` is refused)
/// (review R2-3 of PR-26c-4).
#[test]
fn a_forged_inline_udtf_name_runs_nothing() {
    let Some(python) = python() else {
        return;
    };
    let (pool, udfs) = udfs(&python);
    let session = krishiv_session::Session::new().expect("session");
    session.set_python_udf_runtime(Arc::new(udfs));
    let version = krishiv_session::ANONYMOUS_VERSION_BIT | 7;
    let result = DataType::Struct(StructType::new([StructField::new(
        "a",
        DataType::Integer,
        true,
    )]))
    .json();
    let m = '\u{0}';
    for declared in [result.as_str(), ""] {
        let name = format!("{m}krishiv.python_udtf{m}{version}{m}300{m}1{m}{declared}{m}forged");
        let sql = format!("SELECT * FROM session.`{name}`(1)");
        let err = block_on(async { session.sql(&sql).await?.collect().await })
            .expect_err("a forged name is refused");
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivUdfNotAvailable,
            "{declared:?}: {err}"
        );
    }
    assert_eq!(pool.live_workers(), 0, "a worker started");
}

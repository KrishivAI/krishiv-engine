// SPDX-License-Identifier: Apache-2.0
//! The relation kinds through the pool with real worker processes (P3-T014 PR-26c-2; design
//! `P3-T085-embedded-python-udf-kinds.md` §3.2): the map, grouped-map, cogrouped-map and grouped-aggregate eval
//! types, each fed [`Frame`]s by a [`UdfTask`] as the relation operators feed it, with Spark 4.2.0's argument
//! and result shapes; a UDF error, a result of the wrong schema and a cancel mid-group.
//!
//! As `tests/pool.rs`: the interpreter comes from `KRISHIV_TEST_UDF_PYTHON`, and without it every test passes
//! after a note unless `KRISHIV_UDF_POOL_REQUIRED` is set. Workers start sandboxed, through this test binary
//! ([`worker_child`]).
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
use std::time::{Duration, Instant};

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::python_eval::UdfRuntime;
use krishiv_exec::python_relation::{Frame, RelationCallSpec, RunnerConf, UdfTask};
use krishiv_plan::function::PythonEvalType;
use krishiv_types::DataType;
use krishiv_udf_pool::{
    Launcher, Owner, Pool, PoolConfig, PythonArchive, PythonPayload, SessionUdfs,
};

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
            eprintln!("note: KRISHIV_TEST_UDF_PYTHON is not set; the relation tests did not run");
            None
        }
    }
}

fn pool(python: &Path) -> Pool {
    pool_of(python, 8)
}

/// A pool whose sessions run at most `max_per_session` workers.
fn pool_of(python: &Path, max_per_session: usize) -> Pool {
    let mut config = PoolConfig::default();
    config.max_per_session = max_per_session;
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

fn udfs(pool: &Pool) -> SessionUdfs {
    SessionUdfs::new(
        pool.clone(),
        Owner {
            principal: "alice".to_owned(),
            session: "s1".to_owned(),
            tenant: None,
            capped: true,
        },
    )
}

/// Spark's `command` for `body` (a Python function `f`) and `return_type` (a pyspark type expression), made
/// by the interpreter's own cloudpickle, as a client makes it.
fn command(python: &Path, body: &str, return_type: &str, eval_type: i32) -> PythonPayload {
    let code = format!(
        "import sys\nfrom pyspark import cloudpickle\nfrom pyspark.sql.types import *\n{body}\n\
         sys.stdout.buffer.write(cloudpickle.dumps((f, {return_type})))"
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

/// A column of a frame or a result: its name and Spark type.
fn col(name: &str, t: DataType) -> (String, DataType) {
    (name.to_owned(), t)
}

/// The client form the engine declares for `t` (views for strings), as the relation operators do.
fn client(t: &DataType) -> ArrowType {
    krishiv_types::arrow::client_arrow_type(&t.to_arrow())
}

/// What a test calls: the UDF's frames (one list per side) and its declared result.
struct Call {
    eval_type: PythonEvalType,
    sides: Vec<Vec<(String, DataType)>>,
    key_count: usize,
    output: Vec<(String, DataType)>,
}

/// Registers `payload` under a fresh name and prepares `call` over it.
fn prepare(
    udfs: &SessionUdfs,
    payload: PythonPayload,
    call: &Call,
) -> Result<Arc<dyn krishiv_exec::python_relation::PreparedRelationUdf>, KrishivError> {
    prepare_with(udfs, payload, call, BTreeMap::new(), &[])
}

/// As [`prepare`], the plan's runner conf holding `settings`, and the result columns named in `non_null`
/// declared non-nullable.
fn prepare_with(
    udfs: &SessionUdfs,
    payload: PythonPayload,
    call: &Call,
    settings: BTreeMap<String, String>,
    non_null: &[&str],
) -> Result<Arc<dyn krishiv_exec::python_relation::PreparedRelationUdf>, KrishivError> {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let name = format!(
        "r{}",
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    udfs.register(name.clone(), 1, payload);
    let output_arrow: SchemaRef = Arc::new(Schema::new(
        call.output
            .iter()
            .map(|(n, t)| Field::new(n, client(t), !non_null.contains(&n.as_str())))
            .collect::<Vec<_>>(),
    ));
    let spec = RelationCallSpec {
        name,
        version: 1,
        eval_type: call.eval_type,
        thread_safe: true,
        input_types: call
            .sides
            .iter()
            .map(|s| s.iter().map(|(_, t)| t.clone()).collect())
            .collect(),
        input_names: call
            .sides
            .iter()
            .map(|s| s.iter().map(|(n, _)| n.clone()).collect())
            .collect(),
        key_count: call.key_count,
        arity: call.sides[0].len() - call.key_count,
        output_names: call.output.iter().map(|(n, _)| n.clone()).collect(),
        output_types: call.output.iter().map(|(_, t)| t.clone()).collect(),
        output_arrow,
        conf: RunnerConf {
            max_records_per_batch: 10_000,
            settings,
            ..RunnerConf::default()
        },
        udtf: None,
    };
    udfs.prepare_relation(&spec)
}

/// Sends `frames` to a new task of `prepared` while it reads the results, as the relation sink does.
fn run(
    prepared: &dyn krishiv_exec::python_relation::PreparedRelationUdf,
    frames: Vec<Frame>,
) -> Result<Vec<RecordBatch>, KrishivError> {
    let task = prepared.open(None)?;
    drive(task.as_ref(), frames)
}

fn drive(task: &dyn UdfTask, frames: Vec<Frame>) -> Result<Vec<RecordBatch>, KrishivError> {
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

/// A frame of `side` whose columns are `columns` (names and arrays).
fn batch(side: u8, columns: Vec<(&str, ArrayRef)>) -> Frame {
    let batch = RecordBatch::try_from_iter(columns).expect("batch");
    Frame::Batch { side, batch }
}

fn longs(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

fn strings(values: &[&str]) -> ArrayRef {
    Arc::new(StringViewArray::from(values.to_vec()))
}

/// The rows of `batches` as text, `|` between cells, in result order.
fn rows(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(r) {
                        return "null".to_owned();
                    }
                    if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
                        a.value(r).to_string()
                    } else if let Some(a) = c.as_any().downcast_ref::<Float64Array>() {
                        a.value(r).to_string()
                    } else if let Some(a) = c.as_any().downcast_ref::<StringViewArray>() {
                        a.value(r).to_owned()
                    } else {
                        format!("{:?}", c.data_type())
                    }
                })
                .collect();
            out.push(cells.join("|"));
        }
    }
    out
}

/// The schema every batch of `batches` must have.
fn assert_schema(batches: &[RecordBatch], names: &[&str], types: &[ArrowType]) {
    for b in batches {
        let got: Vec<(&str, &ArrowType)> = b
            .schema_ref()
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type()))
            .collect();
        let want: Vec<(&str, &ArrowType)> = names.iter().copied().zip(types).collect();
        assert_eq!(got, want);
    }
}

/// A map task's frames: two batches of `id` and `s`, then the end.
fn map_frames() -> Vec<Frame> {
    vec![
        batch(0, vec![("id", longs(&[1, 2])), ("s", strings(&["a", "b"]))]),
        batch(0, vec![("id", longs(&[3])), ("s", strings(&["c"]))]),
        Frame::End,
    ]
}

fn map_call(eval_type: PythonEvalType) -> Call {
    Call {
        eval_type,
        sides: vec![vec![col("id", DataType::Long), col("s", DataType::String)]],
        key_count: 0,
        output: vec![col("s", DataType::String), col("v", DataType::Long)],
    }
}

const MAP_RESULT: &str =
    "StructType([StructField('s', StringType()), StructField('v', LongType())])";

#[test]
fn map_in_pandas_and_map_in_arrow_see_each_batch_and_yield_any_batches() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    // mapInPandas (205): an iterator of DataFrames (the frame's columns by name) in, DataFrames out; the
    // result's columns are bound by name, so their order in the DataFrame does not matter.
    let pandas = command(
        &py,
        "def f(it):\n    for pdf in it:\n        yield pdf.assign(v=pdf.id * 10)[['v', 's']]\n        \
         yield pdf.iloc[:0].assign(v=pdf.id.iloc[:0])[['s', 'v']]",
        MAP_RESULT,
        205,
    );
    let prepared = prepare(&udfs, pandas, &map_call(PythonEvalType::MapPandasIter)).expect("205");
    let out = run(prepared.as_ref(), map_frames()).expect("mapInPandas");
    assert_schema(&out, &["s", "v"], &[ArrowType::Utf8View, ArrowType::Int64]);
    assert_eq!(rows(&out), ["a|10", "b|20", "c|30"]);
    // mapInArrow (207): an iterator of pyarrow RecordBatches (Spark's string type) in, batches out.
    let arrow = command(
        &py,
        "import pyarrow as pa\nimport pyarrow.compute as pc\n\
         def f(it):\n    for b in it:\n        assert b.schema.field('s').type == pa.string(), b.schema\n        \
         yield pa.RecordBatch.from_arrays([b.column('s'), pc.multiply(b.column('id'), 2)], ['s', 'v'])",
        MAP_RESULT,
        207,
    );
    let prepared = prepare(&udfs, arrow, &map_call(PythonEvalType::MapArrowIter)).expect("207");
    let out = run(prepared.as_ref(), map_frames()).expect("mapInArrow");
    assert_schema(&out, &["s", "v"], &[ArrowType::Utf8View, ArrowType::Int64]);
    assert_eq!(rows(&out), ["a|2", "b|4", "c|6"]);
    assert_eq!(pool.live_workers(), 1, "one worker, reused");
}

/// A grouped task's frames: keys `k` lead (the key is also a data column), two groups, the first over two
/// frames.
fn grouped_frames() -> Vec<Frame> {
    let frame =
        |k: &[i64], v: &[i64]| batch(0, vec![("k", longs(k)), ("k", longs(k)), ("v", longs(v))]);
    vec![
        frame(&[1, 1], &[10, 20]),
        frame(&[1], &[30]),
        Frame::EndGroup,
        frame(&[2], &[5]),
        Frame::EndGroup,
        Frame::End,
    ]
}

fn grouped_call(eval_type: PythonEvalType) -> Call {
    Call {
        eval_type,
        sides: vec![vec![
            col("k", DataType::Long),
            col("k", DataType::Long),
            col("v", DataType::Long),
        ]],
        key_count: 1,
        output: vec![col("k", DataType::Long), col("total", DataType::Long)],
    }
}

const GROUPED_RESULT: &str =
    "StructType([StructField('k', LongType()), StructField('total', LongType())])";

#[test]
fn grouped_maps_see_whole_groups_and_their_keys() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let cases = [
        // applyInPandas (201) with `f(key, pdf)`: `key` is a tuple of the group's key values.
        (
            201,
            PythonEvalType::GroupedMapPandas,
            "import pandas as pd\ndef f(key, pdf):\n    \
             return pd.DataFrame({'total': [int(pdf.v.sum())], 'k': [key[0]]})",
        ),
        // applyInPandas (201) with `f(pdf)`: the DataFrame holds the data columns, keys included.
        (
            201,
            PythonEvalType::GroupedMapPandas,
            "import pandas as pd\ndef f(pdf):\n    \
             return pd.DataFrame({'k': [pdf.k.iloc[0]], 'total': [int(pdf.v.sum())]})",
        ),
        // applyInArrow (209) with `f(key, table)`: the key values are pyarrow scalars.
        (
            209,
            PythonEvalType::GroupedMapArrow,
            "import pyarrow as pa\nimport pyarrow.compute as pc\ndef f(key, t):\n    \
             assert isinstance(key[0], pa.Scalar)\n    \
             return pa.table({'k': [key[0].as_py()], 'total': [pc.sum(t.column('v')).as_py()]})",
        ),
        // The iterator forms (216, 215): the group's frames as they come.
        (
            216,
            PythonEvalType::GroupedMapPandasIter,
            "import pandas as pd\ndef f(key, it):\n    total = sum(int(p.v.sum()) for p in it)\n    \
             yield pd.DataFrame({'k': [key[0]], 'total': [total]})",
        ),
        (
            215,
            PythonEvalType::GroupedMapArrowIter,
            "import pyarrow as pa\nimport pyarrow.compute as pc\ndef f(it):\n    batches = list(it)\n    \
             total = sum(pc.sum(b.column('v')).as_py() for b in batches)\n    \
             yield pa.RecordBatch.from_pydict({'k': [batches[0].column('k')[0].as_py()], 'total': [total]})",
        ),
    ];
    for (code, eval_type, body) in cases {
        let payload = command(&py, body, GROUPED_RESULT, code);
        let prepared = prepare(&udfs, payload, &grouped_call(eval_type)).expect("prepare");
        let out =
            run(prepared.as_ref(), grouped_frames()).unwrap_or_else(|e| panic!("{code}: {e}"));
        assert_schema(&out, &["k", "total"], &[ArrowType::Int64, ArrowType::Int64]);
        assert_eq!(rows(&out), ["1|60", "2|5"], "{code}: {body}");
    }
}

/// A cogroup's frames: key 1 on both sides (the left over two frames), key 2 on the right only (a zero-row
/// left frame first).
fn cogroup_frames() -> Vec<Frame> {
    let left =
        |k: &[i64], v: &[i64]| batch(0, vec![("k", longs(k)), ("k", longs(k)), ("v", longs(v))]);
    let right =
        |k: &[i64], w: &[&str]| batch(1, vec![("k", longs(k)), ("k", longs(k)), ("w", strings(w))]);
    vec![
        left(&[1], &[1]),
        left(&[1], &[2]),
        right(&[1, 1, 1], &["x", "y", "z"]),
        Frame::EndGroup,
        left(&[], &[]),
        right(&[2], &["q"]),
        Frame::EndGroup,
        Frame::End,
    ]
}

fn cogroup_call(eval_type: PythonEvalType) -> Call {
    Call {
        eval_type,
        sides: vec![
            vec![
                col("k", DataType::Long),
                col("k", DataType::Long),
                col("v", DataType::Long),
            ],
            vec![
                col("k", DataType::Long),
                col("k", DataType::Long),
                col("w", DataType::String),
            ],
        ],
        key_count: 1,
        output: vec![
            col("k", DataType::Long),
            col("left", DataType::Long),
            col("right", DataType::String),
        ],
    }
}

const COGROUP_RESULT: &str = "StructType([StructField('k', LongType()), \
     StructField('left', LongType()), StructField('right', StringType())])";

#[test]
fn cogrouped_maps_see_the_left_then_the_right_rows_of_each_key() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let cases = [
        // cogroup().applyInPandas (206): `f(key, left, right)`; the key comes from the right when the left is
        // empty.
        (
            206,
            PythonEvalType::CogroupedMapPandas,
            "import pandas as pd\ndef f(key, l, r):\n    \
             return pd.DataFrame({'k': [key[0]], 'left': [len(l)], 'right': [''.join(r.w)]})",
        ),
        // cogroup().applyInArrow (210): `f(left, right)` over two tables.
        (
            210,
            PythonEvalType::CogroupedMapArrow,
            "import pyarrow as pa\ndef f(l, r):\n    \
             k = (l if l.num_rows else r).column('k')[0].as_py()\n    \
             return pa.table({'k': [k], 'left': [l.num_rows], 'right': [''.join(r.column('w').to_pylist())]})",
        ),
    ];
    for (code, eval_type, body) in cases {
        let payload = command(&py, body, COGROUP_RESULT, code);
        let prepared = prepare(&udfs, payload, &cogroup_call(eval_type)).expect("prepare");
        let out =
            run(prepared.as_ref(), cogroup_frames()).unwrap_or_else(|e| panic!("{code}: {e}"));
        assert_schema(
            &out,
            &["k", "left", "right"],
            &[ArrowType::Int64, ArrowType::Int64, ArrowType::Utf8View],
        );
        assert_eq!(rows(&out), ["1|2|xyz", "2|0|q"], "{code}: {body}");
    }
}

/// An aggregate's frames: the argument columns of three groups, the first over two frames.
fn aggregate_frames() -> Vec<Frame> {
    let frame = |v: &[i64], w: &[i64]| batch(0, vec![("_0", longs(v)), ("_1", longs(w))]);
    vec![
        frame(&[1, 2], &[1, 1]),
        frame(&[3], &[2]),
        Frame::EndGroup,
        frame(&[10], &[0]),
        Frame::EndGroup,
        frame(&[7, 7], &[1, 1]),
        Frame::EndGroup,
        Frame::End,
    ]
}

fn aggregate_call(eval_type: PythonEvalType) -> Call {
    Call {
        eval_type,
        sides: vec![vec![col("_0", DataType::Long), col("_1", DataType::Long)]],
        key_count: 0,
        output: vec![col("pythonUDAF0", DataType::Double)],
    }
}

#[test]
fn grouped_aggregates_yield_one_value_per_group() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let cases = [
        // A grouped-aggregate pandas_udf (202): a Series per argument, the whole group; a scalar back.
        (
            202,
            PythonEvalType::GroupedAggPandas,
            "def f(v, w):\n    return float((v * w).sum())",
        ),
        // Its iterator form (217): tuples of Series, one per frame.
        (
            217,
            PythonEvalType::GroupedAggPandasIter,
            "def f(it):\n    return float(sum((v * w).sum() for v, w in it))",
        ),
        // A grouped-aggregate arrow_udf (252): pyarrow arrays; the result is coerced to the declared type.
        (
            252,
            PythonEvalType::GroupedAggArrow,
            "import pyarrow.compute as pc\ndef f(v, w):\n    return pc.sum(pc.multiply(v, w))",
        ),
        (
            254,
            PythonEvalType::GroupedAggArrowIter,
            "import pyarrow.compute as pc\ndef f(it):\n    \
             return float(sum(pc.sum(pc.multiply(v, w)).as_py() for v, w in it))",
        ),
    ];
    for (code, eval_type, body) in cases {
        let payload = command(&py, body, "DoubleType()", code);
        let prepared = prepare(&udfs, payload, &aggregate_call(eval_type)).expect("prepare");
        let out =
            run(prepared.as_ref(), aggregate_frames()).unwrap_or_else(|e| panic!("{code}: {e}"));
        assert_schema(&out, &["pythonUDAF0"], &[ArrowType::Float64]);
        assert_eq!(rows(&out), ["9", "0", "14"], "{code}: {body}");
    }
}

#[test]
fn a_udf_error_is_python_exception_and_the_worker_stays() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let payload = command(
        &py,
        "def f(key, pdf):\n    if key[0] == 2:\n        raise ValueError('bad group')\n    return pdf[['k', 'v']]",
        "StructType([StructField('k', LongType()), StructField('v', LongType())])",
        201,
    );
    let mut call = grouped_call(PythonEvalType::GroupedMapPandas);
    call.output = vec![col("k", DataType::Long), col("v", DataType::Long)];
    let prepared = prepare(&udfs, payload, &call).expect("prepare");
    let err = run(prepared.as_ref(), grouped_frames()).expect_err("raises");
    assert_eq!(err.error_class(), ErrorClass::PythonException, "{err}");
    assert!(err.to_string().contains("ValueError: bad group"), "{err}");
    // The worker answered the error and stays in step: the next task runs on it.
    let ok = run(
        prepared.as_ref(),
        vec![
            batch(
                0,
                vec![("k", longs(&[1])), ("k", longs(&[1])), ("v", longs(&[4]))],
            ),
            Frame::EndGroup,
            Frame::End,
        ],
    )
    .expect("the worker is reused");
    assert_eq!(rows(&ok), ["1|4"]);
    assert_eq!(pool.live_workers(), 1);
}

#[test]
fn a_result_of_the_wrong_schema_is_refused() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    // applyInPandas: a column the declared schema does not have (Spark's `verify_pandas_result`).
    let payload = command(
        &py,
        "import pandas as pd\ndef f(pdf):\n    return pd.DataFrame({'k': [1], 'sum': [2]})",
        GROUPED_RESULT,
        201,
    );
    let prepared = prepare(
        &udfs,
        payload,
        &grouped_call(PythonEvalType::GroupedMapPandas),
    )
    .expect("prepare");
    let err = run(prepared.as_ref(), grouped_frames()).expect_err("wrong names");
    assert_eq!(err.error_class(), ErrorClass::PythonException, "{err}");
    assert!(
        err.to_string().contains("RESULT_COLUMN_NAMES_MISMATCH"),
        "{err}"
    );
    // applyInArrow: a column of another type (Spark's `verify_arrow_table`).
    let payload = command(
        &py,
        "import pyarrow as pa\ndef f(t):\n    return pa.table({'k': [1], 'total': ['x']})",
        GROUPED_RESULT,
        209,
    );
    let prepared = prepare(
        &udfs,
        payload,
        &grouped_call(PythonEvalType::GroupedMapArrow),
    )
    .expect("prepare");
    let err = run(prepared.as_ref(), grouped_frames()).expect_err("wrong type");
    assert_eq!(err.error_class(), ErrorClass::PythonException, "{err}");
    assert!(
        err.to_string().contains("RESULT_COLUMN_TYPES_MISMATCH"),
        "{err}"
    );
    // mapInArrow checks nothing in Python: the pool's own check of the result refuses it.
    let payload = command(
        &py,
        "import pyarrow as pa\ndef f(it):\n    for b in it:\n        \
         yield pa.RecordBatch.from_pydict({'s': [1], 'v': [2]})",
        MAP_RESULT,
        207,
    );
    let prepared =
        prepare(&udfs, payload, &map_call(PythonEvalType::MapArrowIter)).expect("prepare");
    let err = run(prepared.as_ref(), map_frames()).expect_err("wrong type");
    assert_eq!(err.error_class(), ErrorClass::PythonException, "{err}");
    assert!(err.to_string().contains("expected Utf8View"), "{err}");
}

#[test]
fn a_cancel_mid_group_kills_the_worker_within_1s() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let payload = command(
        &py,
        "import time\ndef f(pdf):\n    time.sleep(60)\n    return pdf",
        "StructType([StructField('k', LongType()), StructField('v', LongType())])",
        201,
    );
    let mut call = grouped_call(PythonEvalType::GroupedMapPandas);
    call.output = vec![col("k", DataType::Long), col("v", DataType::Long)];
    let prepared = prepare(&udfs, payload, &call).expect("prepare");
    let task = prepared.open(None).expect("open");
    let mut frames = grouped_frames().into_iter();
    // The first group, whole: the UDF then sleeps in it.
    for frame in frames.by_ref().take(3) {
        block_on(task.send(frame)).expect("send");
    }
    let started = Instant::now();
    while pool.live_workers() == 0 && started.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(500));
    let mut next = task.next();
    assert!(
        futures::FutureExt::now_or_never(&mut next).is_none(),
        "still in the UDF"
    );
    drop(next);
    let cancelled = Instant::now();
    drop(task);
    while pool.live_workers() > 0 && cancelled.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(pool.live_workers(), 0, "the worker was not stopped");
    assert!(
        cancelled.elapsed() < Duration::from_secs(1),
        "{:?}",
        cancelled.elapsed()
    );
}

#[test]
fn a_scalar_payload_is_not_a_relation() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    // The plan names a relation kind the registered payload does not have (the store finds payloads by name
    // and version only).
    let payload = command(&py, "def f(x):\n    return x", "LongType()", 101);
    let err = prepare(&udfs, payload, &map_call(PythonEvalType::MapPandasIter))
        .err()
        .expect("refused");
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivPythonUdfEvalTypeMismatch,
        "{err}"
    );
    assert_eq!(pool.live_workers(), 0, "refused before any worker starts");
}

/// Runs `f` on a thread of its own and gives its result, or fails the test after `deadline`: a hang fails
/// rather than stalls the run.
fn within<T: Send + 'static>(deadline: Duration, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(deadline).expect("hung past the deadline")
}

/// With `spark.sql.execution.arrow.pyspark.validateSchema.enabled` a `mapInArrow` result of another type is
/// Spark 4.1+'s `ARROW_TYPE_MISMATCH` (`MapInBatchEvaluatorFactory.scala` L81-91 at v4.2.0), with Spark's
/// parameters; a non-nullable result column fits a nullable declared one.
#[test]
fn map_in_arrow_under_validate_schema_raises_arrow_type_mismatch() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let on = BTreeMap::from([(
        "krishiv.sql.execution.arrow.pyspark.validateSchema.enabled".to_owned(),
        "true".to_owned(),
    )]);
    let wrong = command(
        &py,
        "import pyarrow as pa\ndef f(it):\n    for b in it:\n        \
         yield pa.RecordBatch.from_pydict({'s': [1], 'v': [2]})",
        MAP_RESULT,
        207,
    );
    let prepared = prepare_with(
        &udfs,
        wrong,
        &map_call(PythonEvalType::MapArrowIter),
        on.clone(),
        &[],
    )
    .expect("prepare");
    let err = run(prepared.as_ref(), map_frames()).expect_err("wrong type");
    assert_eq!(err.error_class(), ErrorClass::ArrowTypeMismatch, "{err}");
    assert_eq!(
        err.message(),
        "Invalid schema from SQL_MAP_ARROW_ITER_UDF: expected \
         StructType(StructField(s,StringType,true),StructField(v,LongType,true)), got \
         StructType(StructField(s,LongType,true),StructField(v,LongType,true))."
    );
    let strict = command(
        &py,
        "import pyarrow as pa\ndef f(it):\n    for b in it:\n        \
         s = pa.schema([pa.field('s', pa.string(), nullable=False), pa.field('v', pa.int64(), nullable=False)])\n        \
         yield pa.RecordBatch.from_arrays([b.column('s'), b.column('id')], schema=s)",
        MAP_RESULT,
        207,
    );
    let prepared = prepare_with(
        &udfs,
        strict,
        &map_call(PythonEvalType::MapArrowIter),
        on,
        &[],
    )
    .expect("prepare");
    let out = run(prepared.as_ref(), map_frames()).expect("compatible nullability");
    assert_eq!(rows(&out), ["a|1", "b|2", "c|3"]);
}

/// A UDF that stops reading before the last frame ends the task; the frames still sent are dropped, and
/// `send` does not fail for them.
#[test]
fn map_in_arrow_done_before_the_end_drops_the_frames_left() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let first = command(
        &py,
        "import pyarrow as pa\ndef f(it):\n    b = next(it)\n    \
         yield pa.RecordBatch.from_arrays([b.column('s'), b.column('id')], ['s', 'v'])",
        MAP_RESULT,
        207,
    );
    let prepared = prepare(&udfs, first, &map_call(PythonEvalType::MapArrowIter)).expect("prepare");
    let mut frames = map_frames();
    frames.insert(
        1,
        batch(0, vec![("id", longs(&[9])), ("s", strings(&["z"]))]),
    );
    let out = within(Duration::from_secs(60), move || {
        run(prepared.as_ref(), frames)
    })
    .expect("done before the end");
    assert_eq!(rows(&out), ["a|1", "b|2"]);
    assert_eq!(pool.live_workers(), 1, "the worker ended well and is kept");
}

/// A task dropped before its first frame starts no worker, or stops it at once.
#[test]
fn a_task_dropped_before_its_first_frame_holds_no_worker() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let payload = command(&py, "def f(it):\n    yield from it", MAP_RESULT, 205);
    let prepared =
        prepare(&udfs, payload, &map_call(PythonEvalType::MapPandasIter)).expect("prepare");
    drop(prepared.open(None).expect("open"));
    let started = Instant::now();
    while pool.live_workers() > 0 && started.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(pool.live_workers(), 0);
    // The slot is free again: a second task opens and runs.
    let out = within(Duration::from_secs(60), move || {
        run(
            prepared.as_ref(),
            vec![
                batch(0, vec![("s", strings(&["x"])), ("v", longs(&[1]))]),
                Frame::End,
            ],
        )
    })
    .expect("a new task runs");
    assert_eq!(rows(&out), ["x|1"]);
}

/// `maxPerSession` 2: a relation task holds one worker while it waits for frames, and a scalar UDF call of the
/// same session (a `PythonEval` below the relation operator) still runs on the other.
#[test]
fn a_scalar_call_beside_an_open_task_runs_on_the_reserved_worker() {
    let Some(py) = python() else { return };
    let pool = pool_of(&py, 2);
    let udfs = Arc::new(udfs(&pool));
    let payload = command(&py, "def f(it):\n    yield from it", MAP_RESULT, 205);
    let prepared =
        prepare(&udfs, payload, &map_call(PythonEvalType::MapPandasIter)).expect("prepare");
    let task = prepared.open(None).expect("open");
    block_on(task.send(batch(0, vec![("s", strings(&["x"])), ("v", longs(&[1]))]))).expect("send");
    let scalar = command(&py, "def f(x):\n    return x + 1", "LongType()", 101);
    udfs.register("plus", 1, scalar);
    let calls = Arc::clone(&udfs);
    let out = within(Duration::from_secs(60), move || {
        let prepared = calls.prepare(&krishiv_exec::python_eval::UdfCallSpec {
            name: "plus".to_owned(),
            version: 1,
            eval_type: PythonEvalType::ArrowBatched,
            thread_safe: true,
            arg_types: vec![DataType::Long],
            return_type: DataType::Long,
            return_arrow: ArrowType::Int64,
        })?;
        let args = RecordBatch::try_from_iter([("a", longs(&[1, 2]))]).expect("args");
        block_on(prepared.call(args))
    })
    .expect("the scalar call ran");
    assert_eq!(out.len(), 2);
    let rest = drive(task.as_ref(), vec![Frame::End]).expect("the task ends");
    assert_eq!(rows(&rest), ["x|1"]);
}

/// `maxPerSession` 1: the one worker is the task's, so a scalar call beside it fails at once, naming the
/// setting, instead of waiting for a task that may wait for it.
#[test]
fn with_one_worker_a_scalar_call_beside_an_open_task_fails_fast() {
    let Some(py) = python() else { return };
    let pool = pool_of(&py, 1);
    let udfs = Arc::new(udfs(&pool));
    let payload = command(&py, "def f(it):\n    yield from it", MAP_RESULT, 205);
    let prepared =
        prepare(&udfs, payload, &map_call(PythonEvalType::MapPandasIter)).expect("prepare");
    let task = prepared.open(None).expect("open");
    let scalar = command(&py, "def f(x):\n    return x + 1", "LongType()", 101);
    udfs.register("plus", 1, scalar);
    let calls = Arc::clone(&udfs);
    let err = within(Duration::from_secs(30), move || {
        let prepared = calls.prepare(&krishiv_exec::python_eval::UdfCallSpec {
            name: "plus".to_owned(),
            version: 1,
            eval_type: PythonEvalType::ArrowBatched,
            thread_safe: true,
            arg_types: vec![DataType::Long],
            return_type: DataType::Long,
            return_arrow: ArrowType::Int64,
        })?;
        let args = RecordBatch::try_from_iter([("a", longs(&[1]))]).expect("args");
        block_on(prepared.call(args))
    })
    .expect_err("refused");
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivPythonWorkerLimit,
        "{err}"
    );
    assert!(
        err.to_string()
            .contains("krishiv.python.worker.maxPerSession"),
        "{err}"
    );
    // A second task of the session would block: it is not queued behind the first.
    assert!(prepared.open(None).is_err(), "no second slot");
    drop(task);
}

/// `mapInArrow` returning more columns than declared: Spark's JVM sees every column, so under
/// `validateSchema` it is `ARROW_TYPE_MISMATCH`; with the setting off Krishiv refuses it (`PYTHON_EXCEPTION`,
/// the recorded divergence: Spark reads the declared columns only). Never dropped silently.
#[test]
fn map_in_arrow_extra_columns_are_refused_not_dropped() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let extra = || {
        command(
            &py,
            "import pyarrow as pa\ndef f(it):\n    for b in it:\n        \
             yield pa.RecordBatch.from_arrays([b.column('s'), b.column('id'), b.column('id')], ['s', 'v', 'w'])",
            MAP_RESULT,
            207,
        )
    };
    let on = BTreeMap::from([(
        "krishiv.sql.execution.arrow.pyspark.validateSchema.enabled".to_owned(),
        "true".to_owned(),
    )]);
    let prepared = prepare_with(
        &udfs,
        extra(),
        &map_call(PythonEvalType::MapArrowIter),
        on,
        &[],
    )
    .expect("prepare");
    let err = run(prepared.as_ref(), map_frames()).expect_err("three columns");
    assert_eq!(err.error_class(), ErrorClass::ArrowTypeMismatch, "{err}");
    assert!(
        err.message().contains("StructField(w,LongType,true)"),
        "{err}"
    );
    let prepared =
        prepare(&udfs, extra(), &map_call(PythonEvalType::MapArrowIter)).expect("prepare");
    let err = run(prepared.as_ref(), map_frames()).expect_err("three columns");
    assert_eq!(err.error_class(), ErrorClass::PythonException, "{err}");
    assert!(err.to_string().contains("3 columns, expected 2"), "{err}");
}

/// A relation's runner conf is read through the session's `ConfSource` at each `prepare_relation`, as a
/// scalar call's is: a change the source shows reaches the next call site's tasks. Here the source turns
/// `validateSchema` off between two call sites of one `mapInArrow` UDF returning extra columns, which changes
/// the refusal's class (`ARROW_TYPE_MISMATCH`, then `PYTHON_EXCEPTION`).
#[test]
fn a_relation_reads_its_runner_conf_through_the_conf_source() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let live = Arc::new(std::sync::Mutex::new(BTreeMap::from([(
        "krishiv.sql.execution.arrow.pyspark.validateSchema.enabled".to_owned(),
        "true".to_owned(),
    )])));
    let source = Arc::clone(&live);
    let udfs = udfs(&pool).with_conf_source(Arc::new(move || {
        source
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }));
    // Ignored while a source is set.
    udfs.set_runner_conf(BTreeMap::new());
    let extra = || {
        command(
            &py,
            "import pyarrow as pa\ndef f(it):\n    for b in it:\n        \
             yield pa.RecordBatch.from_arrays([b.column('s'), b.column('id'), b.column('id')], ['s', 'v', 'w'])",
            MAP_RESULT,
            207,
        )
    };
    let prepared =
        prepare(&udfs, extra(), &map_call(PythonEvalType::MapArrowIter)).expect("prepare");
    let err = run(prepared.as_ref(), map_frames()).expect_err("three columns");
    assert_eq!(err.error_class(), ErrorClass::ArrowTypeMismatch, "{err}");
    live.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    let prepared =
        prepare(&udfs, extra(), &map_call(PythonEvalType::MapArrowIter)).expect("prepare");
    let err = run(prepared.as_ref(), map_frames()).expect_err("three columns");
    assert_eq!(err.error_class(), ErrorClass::PythonException, "{err}");
}

/// `mapInPandas` under `validateSchema` with a column declared non-nullable: Spark 4.2.0's pandas serializer
/// builds the result with every field nullable (`conversion.py` L443 `RecordBatch.from_arrays(arrays,
/// schema.names)`, wrapped by `wrap_struct` L108 with the batch's own fields), and the JVM's check
/// (`MapInBatchEvaluatorFactory.scala` L84-90) refuses a nullable result field for a non-nullable declared one:
/// `ARROW_TYPE_MISMATCH`, as here. Read from the v4.2.0 source; a Spark golden is still owed (no Spark on the
/// build host). Nullable declared columns pass.
#[test]
fn map_in_pandas_under_validate_schema_compares_nullability_as_spark_4_2_0() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let on = BTreeMap::from([(
        "krishiv.sql.execution.arrow.pyspark.validateSchema.enabled".to_owned(),
        "true".to_owned(),
    )]);
    let body = "def f(it):\n    for pdf in it:\n        yield pdf.assign(v=pdf.id)[['s', 'v']]";
    let call = map_call(PythonEvalType::MapPandasIter);
    let prepared = prepare_with(
        &udfs,
        command(&py, body, MAP_RESULT, 205),
        &call,
        on.clone(),
        &[],
    )
    .expect("prepare");
    let out = run(prepared.as_ref(), map_frames()).expect("nullable declared columns");
    assert_eq!(rows(&out), ["a|1", "b|2", "c|3"]);
    let prepared = prepare_with(
        &udfs,
        command(&py, body, MAP_RESULT, 205),
        &call,
        on,
        &["v"],
    )
    .expect("prepare");
    let err = run(prepared.as_ref(), map_frames()).expect_err("v declared non-nullable");
    assert_eq!(err.error_class(), ErrorClass::ArrowTypeMismatch, "{err}");
    assert!(err.message().contains("SQL_MAP_PANDAS_ITER_UDF"), "{err}");
}

/// R2-5 regression: `maxPerSession` 1. Once a task's results ended, its slot is free, so a scalar call the plan
/// makes next (`mapInPandas(...).select(py_udf(c))`) runs instead of being refused as beside an open task.
/// Several rounds, since the old order (slot freed after the results ended) lost only a race.
#[test]
fn a_scalar_call_right_after_a_task_s_results_end_runs() {
    let Some(py) = python() else { return };
    let pool = pool_of(&py, 1);
    let udfs = Arc::new(udfs(&pool));
    let payload = command(&py, "def f(it):\n    yield from it", MAP_RESULT, 205);
    let prepared =
        prepare(&udfs, payload, &map_call(PythonEvalType::MapPandasIter)).expect("prepare");
    let scalar = command(&py, "def f(x):\n    return x + 1", "LongType()", 101);
    udfs.register("plus", 1, scalar);
    let calls = Arc::clone(&udfs);
    within(Duration::from_secs(120), move || {
        let plus = calls
            .prepare(&krishiv_exec::python_eval::UdfCallSpec {
                name: "plus".to_owned(),
                version: 1,
                eval_type: PythonEvalType::ArrowBatched,
                thread_safe: true,
                arg_types: vec![DataType::Long],
                return_type: DataType::Long,
                return_arrow: ArrowType::Int64,
            })
            .expect("prepare the scalar call");
        for round in 0..5 {
            let out = run(
                prepared.as_ref(),
                vec![
                    batch(0, vec![("s", strings(&["x"])), ("v", longs(&[1]))]),
                    Frame::End,
                ],
            )
            .expect("the task");
            assert_eq!(rows(&out), ["x|1"]);
            let args = RecordBatch::try_from_iter([("a", longs(&[1]))]).expect("args");
            let column = block_on(plus.call(args))
                .unwrap_or_else(|e| panic!("round {round}: the scalar call was refused: {e}"));
            assert_eq!(column.len(), 1);
        }
    });
}

// --- PR-26c-3: the window aggregate (203, 253), the scalar iterator (204, 251) and artifacts ---

fn ints(values: &[i32]) -> ArrayRef {
    Arc::new(arrow_array::Int32Array::from(values.to_vec()))
}

/// A window's frames: two partitions of `v`, each row with its frame's start and end (`ROWS BETWEEN 1
/// PRECEDING AND CURRENT ROW`) when `bounded`.
fn window_frames(bounded: bool) -> Vec<Frame> {
    let frame = |begin: &[i32], end: &[i32], v: &[i64]| {
        let mut columns = Vec::new();
        if bounded {
            columns.push(("_b", ints(begin)));
            columns.push(("_e", ints(end)));
        }
        columns.push(("_0", longs(v)));
        batch(0, columns)
    };
    vec![
        frame(&[0, 0], &[1, 2], &[1, 2]),
        frame(&[1, 2], &[3, 4], &[3, 4]),
        Frame::EndGroup,
        frame(&[0], &[1], &[10]),
        Frame::EndGroup,
        Frame::End,
    ]
}

fn window_call(eval_type: PythonEvalType, bounded: bool) -> Call {
    let mut side = Vec::new();
    if bounded {
        side.push(col("_b", DataType::Integer));
        side.push(col("_e", DataType::Integer));
    }
    side.push(col("_0", DataType::Long));
    Call {
        eval_type,
        sides: vec![side],
        key_count: 0,
        output: vec![col("pythonUDAF0", DataType::Double)],
    }
}

/// The runner conf entry the window operator sets per call (Spark's `window_bound_types`).
fn bound_conf(bounded: bool) -> BTreeMap<String, String> {
    BTreeMap::from([(
        "window_bound_types".to_owned(),
        if bounded { "bounded" } else { "unbounded" }.to_owned(),
    )])
}

/// A grouped-aggregate UDF over a window runs as Spark's window kind (202 as 203, 252 as 253): one value per
/// row, the whole partition's for an unbounded frame, each row's frame's for a bounded one (the rows between
/// its start and end indexes, `wrap_bounded_window_agg_pandas_udf`).
#[test]
fn window_aggregates_yield_one_value_per_row() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let cases = [
        (
            202,
            PythonEvalType::WindowAggPandas,
            "def f(v):\n    return float(v.sum())",
        ),
        (
            252,
            PythonEvalType::WindowAggArrow,
            "import pyarrow.compute as pc\ndef f(v):\n    return float(pc.sum(v).as_py() or 0)",
        ),
    ];
    for (code, eval_type, body) in cases {
        for bounded in [false, true] {
            let payload = command(&py, body, "DoubleType()", code);
            let prepared = prepare_with(
                &udfs,
                payload,
                &window_call(eval_type, bounded),
                bound_conf(bounded),
                &[],
            )
            .expect("prepare");
            let out = run(prepared.as_ref(), window_frames(bounded))
                .unwrap_or_else(|e| panic!("{code} bounded={bounded}: {e}"));
            assert_schema(&out, &["pythonUDAF0"], &[ArrowType::Float64]);
            let want: &[&str] = if bounded {
                &["1", "3", "5", "7", "10"]
            } else {
                &["10", "10", "10", "10", "10"]
            };
            assert_eq!(rows(&out), want, "{code} bounded={bounded}");
        }
    }
}

/// A window UDF's error is `PYTHON_EXCEPTION`; a task of the window kind over a payload of another kind is
/// refused when it is prepared.
#[test]
fn a_window_udf_error_is_python_exception() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let payload = command(
        &py,
        "def f(v):\n    raise ValueError('bad window')",
        "DoubleType()",
        202,
    );
    let prepared = prepare_with(
        &udfs,
        payload,
        &window_call(PythonEvalType::WindowAggPandas, true),
        bound_conf(true),
        &[],
    )
    .expect("prepare");
    let err = run(prepared.as_ref(), window_frames(true)).expect_err("raises");
    assert_eq!(err.error_class(), ErrorClass::PythonException, "{err}");
    assert!(err.to_string().contains("ValueError: bad window"), "{err}");
    let map = command(&py, "def f(it):\n    yield from it", "LongType()", 205);
    let refused = prepare_with(
        &udfs,
        map,
        &window_call(PythonEvalType::WindowAggPandas, false),
        bound_conf(false),
        &[],
    );
    assert!(refused.is_err(), "a map payload is no window UDF");
}

/// A scalar iterator's frames: its argument batches, then the end.
fn iter_frames() -> Vec<Frame> {
    vec![
        batch(0, vec![("_0", longs(&[1, 2, 3]))]),
        batch(0, vec![("_0", longs(&[4, 5]))]),
        Frame::End,
    ]
}

fn iter_call(eval_type: PythonEvalType) -> Call {
    Call {
        eval_type,
        sides: vec![vec![col("_0", DataType::Long)]],
        key_count: 0,
        output: vec![col("pythonUDF0", DataType::Long)],
    }
}

/// A scalar iterator UDF is called once with an iterator over the task's batches, and may yield its values
/// in batches of any size (the operator aligns them with the rows).
#[test]
fn scalar_iterators_yield_one_value_per_row_in_any_batches() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let cases = [
        (
            204,
            PythonEvalType::ScalarPandasIter,
            "def f(it):\n    for s in it:\n        yield s + 1",
            vec![3, 2],
        ),
        (
            204,
            PythonEvalType::ScalarPandasIter,
            "import pandas as pd\ndef f(it):\n    yield pd.concat(list(it), ignore_index=True) + 1",
            vec![5],
        ),
        (
            251,
            PythonEvalType::ScalarArrowIter,
            "import pyarrow.compute as pc\ndef f(it):\n    for a in it:\n        yield pc.add(a, 1)",
            vec![3, 2],
        ),
    ];
    for (code, eval_type, body, sizes) in cases {
        let payload = command(&py, body, "LongType()", code);
        let prepared = prepare(&udfs, payload, &iter_call(eval_type)).expect("prepare");
        let out = run(prepared.as_ref(), iter_frames()).unwrap_or_else(|e| panic!("{code}: {e}"));
        assert_schema(&out, &["pythonUDF0"], &[ArrowType::Int64]);
        assert_eq!(rows(&out), ["2", "3", "4", "5", "6"], "{code}: {body}");
        let got: Vec<usize> = out.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(got, sizes, "{code}: {body}");
    }
}

/// Spark's row checks of the scalar iterators (`worker.py` L2586-2646, L3183-3246 at v4.2.0), raised in the
/// worker: `PYTHON_EXCEPTION` carrying the PySpark error class.
#[test]
fn scalar_iterator_row_checks_raise_spark_s_classes() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let cases = [
        (
            204,
            PythonEvalType::ScalarPandasIter,
            "def f(it):\n    for s in it:\n        yield s.iloc[:1]",
            "RESULT_ROWS_MISMATCH",
        ),
        (
            204,
            PythonEvalType::ScalarPandasIter,
            "import pandas as pd\ndef f(it):\n    for s in it:\n        yield pd.concat([s, s])",
            "OUTPUT_EXCEEDS_INPUT_ROWS",
        ),
        (
            204,
            PythonEvalType::ScalarPandasIter,
            "def f(it):\n    yield next(it)",
            "INPUT_NOT_FULLY_CONSUMED",
        ),
        (
            251,
            PythonEvalType::ScalarArrowIter,
            "def f(it):\n    for a in it:\n        yield a.slice(0, 1)",
            "RESULT_ROWS_MISMATCH",
        ),
        (
            251,
            PythonEvalType::ScalarArrowIter,
            "def f(it):\n    yield next(it)",
            "INPUT_NOT_FULLY_CONSUMED",
        ),
        (
            251,
            PythonEvalType::ScalarArrowIter,
            "def f(it):\n    for a in it:\n        yield [1]",
            "UDF_RETURN_TYPE",
        ),
    ];
    for (code, eval_type, body, class) in cases {
        let payload = command(&py, body, "LongType()", code);
        let prepared = prepare(&udfs, payload, &iter_call(eval_type)).expect("prepare");
        let err = run(prepared.as_ref(), iter_frames()).expect_err(body);
        assert_eq!(
            err.error_class(),
            ErrorClass::PythonException,
            "{body}: {err}"
        );
        assert!(err.to_string().contains(class), "{body}: {err}");
    }
}

/// A canonical temporary directory (the artifact paths must be canonical).
fn artifact_dir() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = std::fs::canonicalize(dir.path()).expect("canonical");
    (dir, path)
}

/// Writes an archive with `python` running `script` (given `out`, the archive's path).
fn make_archive(python: &Path, out: &Path, script: &str) {
    let code = format!("import sys\nout = sys.argv[1]\n{script}");
    let status = Command::new(python)
        .args(["-c", &code])
        .arg(out)
        .status()
        .expect("python");
    assert!(status.success());
}

/// A scalar iterator importing `module` and adding its `V` to each value.
fn importing(py: &Path, module: &str) -> PythonPayload {
    command(
        py,
        &format!("def f(it):\n    import {module}\n    for s in it:\n        yield s + {module}.V"),
        "LongType()",
        204,
    )
}

/// A scalar iterator adding to each value the `V = n` its working directory's file `path` holds.
fn reading(py: &Path, path: &str) -> PythonPayload {
    command(
        py,
        &format!(
            "def f(it):\n    v = int(open({path:?}).read().split('=')[1])\n    for s in it:\n        yield s + v"
        ),
        "LongType()",
        204,
    )
}

/// `set_artifacts`, as Spark 4.2.0 places artifacts: a pyfile on the workers' `sys.path`; an archive unpacked
/// (read only, in the sandbox) into the files directory under its alias or its file name, reachable from the
/// worker's working directory and importable as a package of the files directory, but not itself on
/// `sys.path`; a later set gets workers of its own.
#[test]
fn pyfiles_and_archives_reach_the_workers_as_in_spark() {
    let Some(py) = python() else { return };
    let pool = pool(&py);
    let udfs = udfs(&pool);
    let (_keep, dir) = artifact_dir();
    std::fs::write(dir.join("helper.py"), "V = 100\n").expect("pyfile");
    let archive = dir.join("deps.zip");
    make_archive(
        &py,
        &archive,
        "import zipfile\nwith zipfile.ZipFile(out, 'w') as z:\n    z.writestr('pkg/__init__.py', 'V = 7\\n')",
    );
    let tar = dir.join("more.tar.gz");
    make_archive(
        &py,
        &tar,
        "import io, tarfile\nwith tarfile.open(out, 'w:gz') as t:\n    data = b'V = 1000\\n'\n    \
         i = tarfile.TarInfo('tpkg/__init__.py')\n    i.size = len(data)\n    t.addfile(i, io.BytesIO(data))",
    );
    udfs.set_artifacts(
        vec![dir.join("helper.py")],
        vec![
            PythonArchive {
                path: archive,
                alias: Some("deps".to_owned()),
            },
            PythonArchive {
                path: tar,
                alias: None,
            },
        ],
    );
    let cases = [
        ("helper", importing(&py, "helper"), 100),
        ("deps.pkg", importing(&py, "deps.pkg"), 7),
        ("./deps", reading(&py, "deps/pkg/__init__.py"), 7),
        (
            "./more.tar.gz",
            reading(&py, "more.tar.gz/tpkg/__init__.py"),
            1000,
        ),
    ];
    for (what, payload, add) in cases {
        let prepared =
            prepare(&udfs, payload, &iter_call(PythonEvalType::ScalarPandasIter)).expect("prepare");
        let out = run(prepared.as_ref(), iter_frames()).unwrap_or_else(|e| panic!("{what}: {e}"));
        let want: Vec<String> = (1..=5).map(|v| (v + add).to_string()).collect();
        assert_eq!(rows(&out), want, "{what}");
    }
    // An archive's own directory is not on `sys.path` (Spark's `setup_spark_files` does not add it).
    let prepared = prepare(
        &udfs,
        importing(&py, "pkg"),
        &iter_call(PythonEvalType::ScalarPandasIter),
    )
    .expect("prepare");
    let err = run(prepared.as_ref(), iter_frames()).expect_err("pkg is not on sys.path");
    assert!(err.to_string().contains("ModuleNotFoundError"), "{err}");
    // A worker of the old set never serves the new one: without the pyfile the import fails.
    udfs.set_artifacts(Vec::new(), Vec::new());
    let prepared = prepare(
        &udfs,
        importing(&py, "helper"),
        &iter_call(PythonEvalType::ScalarPandasIter),
    )
    .expect("prepare");
    let err = run(prepared.as_ref(), iter_frames()).expect_err("no helper");
    assert!(err.to_string().contains("ModuleNotFoundError"), "{err}");
}

/// A symbolic link is refused (`INVALID_ARTIFACT_PATH`); an archive whose member links out of its directory, or
/// over the pool's artifact limit, is refused (`KRISHIV_PYTHON_ARTIFACT_REFUSED`); the calls fail, no worker
/// sees them.
#[test]
fn unsafe_or_oversized_artifacts_are_refused() {
    let Some(py) = python() else { return };
    let (_keep, dir) = artifact_dir();
    std::fs::write(dir.join("real.py"), "V = 1\n").expect("pyfile");
    std::os::unix::fs::symlink("/etc/passwd", dir.join("link.py")).expect("link");
    let evil = dir.join("evil.tar");
    make_archive(
        &py,
        &evil,
        "import tarfile\nwith tarfile.open(out, 'w') as t:\n    \
         i = tarfile.TarInfo('out')\n    i.type = tarfile.SYMTYPE\n    i.linkname = '/etc/passwd'\n    t.addfile(i)",
    );
    let big = dir.join("big.zip");
    make_archive(
        &py,
        &big,
        "import zipfile\nwith zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:\n    \
         z.writestr('zeros.bin', bytes(4 << 20))",
    );
    let mut config = PoolConfig::default();
    config.executables = vec![py.display().to_string()];
    config.artifact_limit = 1 << 20;
    config.launcher = Launcher {
        program: std::env::current_exe().expect("test binary"),
        args: ["worker_child", "--exact", "--nocapture", "--"]
            .map(OsString::from)
            .to_vec(),
    };
    let pool = Pool::new(config).expect("pool");
    let udfs = udfs(&pool);
    let cases = [
        (
            vec![dir.join("link.py")],
            Vec::new(),
            ErrorClass::InvalidArtifactPath,
            "link.py",
        ),
        (
            Vec::new(),
            vec![PythonArchive {
                path: evil,
                alias: None,
            }],
            ErrorClass::KrishivPythonArtifactRefused,
            "leaves the archive's directory",
        ),
        (
            Vec::new(),
            vec![PythonArchive {
                path: big,
                alias: None,
            }],
            ErrorClass::KrishivPythonArtifactRefused,
            "over the 1048576 bytes left",
        ),
    ];
    for (pyfiles, archives, class, why) in cases {
        udfs.set_artifacts(pyfiles, archives);
        let prepared = prepare(
            &udfs,
            importing(&py, "real"),
            &iter_call(PythonEvalType::ScalarPandasIter),
        )
        .expect("prepare");
        let err = run(prepared.as_ref(), iter_frames()).expect_err("refused");
        assert_eq!(err.error_class(), class, "{err}");
        assert!(err.to_string().contains(why), "{err}");
    }
    assert_eq!(
        pool.live_workers(),
        0,
        "no worker started for a refused set"
    );
}

// SPDX-License-Identifier: Apache-2.0
//! Native Rust scalar UDFs end to end: registration, SQL and DataFrame
//! calls, search-path precedence, replacement, null handling, contract
//! errors and constant folding.
#![cfg(test)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StringArray, StringViewArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::functions::{call_function, col};
use krishiv_session::{
    ArgField, InvokeContext, NullHandling, ReturnField, ScalarUdf, Session, Signature, UdfArg,
    UdfError, UdfInputType, UdfProperties,
};
use krishiv_types::DataType;

fn people() -> (SchemaRef, Vec<RecordBatch>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int32, false),
        Field::new("name", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![3, 1, 2])),
            Arc::new(StringArray::from(vec![Some("c"), None, Some("b")])),
        ],
    )
    .unwrap();
    (schema, vec![batch])
}

fn session() -> Session {
    let s = Session::new().unwrap();
    let (schema, batches) = people();
    s.register_batches("people", schema, batches).unwrap();
    s
}

fn ints(batches: &[RecordBatch]) -> Vec<Option<i32>> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_primitive::<Int32Type>()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn strings(batches: &[RecordBatch]) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|b| {
            let column = b.column(0);
            if let Some(view) = column.as_any().downcast_ref::<StringViewArray>() {
                view.iter()
                    .map(|v| v.map(str::to_owned))
                    .collect::<Vec<_>>()
            } else {
                column
                    .as_string::<i32>()
                    .iter()
                    .map(|v| v.map(str::to_owned))
                    .collect::<Vec<_>>()
            }
        })
        .collect()
}

fn collect(s: &Session, query: &str) -> Vec<RecordBatch> {
    let df = block_on(s.sql(query)).unwrap();
    block_on(df.collect()).unwrap()
}

fn int_sig() -> Signature {
    Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)])
}

/// `add_one(x) = x + 1`.
#[derive(Debug)]
struct AddOne;

impl ScalarUdf for AddOne {
    fn name(&self) -> &str {
        "add_one"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(int_sig)
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: true,
        })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        let input = args[0].array();
        let input = input
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| UdfError::new("add_one needs INT"))?;
        assert_eq!(
            input.len(),
            if matches!(args[0], UdfArg::Scalar(_)) {
                1
            } else {
                rows
            }
        );
        Ok(Arc::new(
            input
                .iter()
                .map(|v| v.map(|v| v.wrapping_add(1)))
                .collect::<Int32Array>(),
        ))
    }
}

#[test]
fn a_registered_udf_runs_in_sql() {
    let s = session();
    s.register_udf(AddOne);
    let rows = collect(&s, "SELECT add_one(id) FROM people ORDER BY id");
    assert_eq!(ints(&rows), [Some(2), Some(3), Some(4)]);
}

#[test]
fn a_udf_runs_through_the_dataframe_api_identically() {
    let s = session();
    s.register_udf(AddOne);
    let via_sql = collect(&s, "SELECT add_one(id) AS x FROM people ORDER BY id");
    let df = s
        .table("people")
        .select([call_function("add_one", [col("id")]).alias("x")]);
    let via_df = block_on(df.order_by([col("x").asc()]).collect()).unwrap();
    assert_eq!(ints(&via_sql), ints(&via_df));
    assert_eq!(ints(&via_sql), [Some(2), Some(3), Some(4)]);
}

/// `upper` implemented as a UDF, shadowing the built-in by name only.
#[derive(Debug)]
struct UpperUdf;

impl ScalarUdf for UpperUdf {
    fn name(&self) -> &str {
        "upper"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Coercible(vec![UdfInputType::Exact(DataType::String)]))
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::String,
            nullable: true,
        })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        let input = args[0].array();
        assert_eq!(
            input.len(),
            if matches!(args[0], UdfArg::Scalar(_)) {
                1
            } else {
                rows
            }
        );
        // String columns arrive in any UTF-8 carrier.
        let upper = |v: Option<&str>| v.map(|v| format!("udf:{v}"));
        if let Some(view) = input.as_any().downcast_ref::<StringViewArray>() {
            return Ok(Arc::new(view.iter().map(upper).collect::<StringArray>()));
        }
        let plain = input
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| UdfError::new("upper needs STRING"))?;
        Ok(Arc::new(plain.iter().map(upper).collect::<StringArray>()))
    }
}

fn one_string(s: &Session, query: &str) -> String {
    let rows = collect(s, query);
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    strings(&rows).into_iter().next().unwrap().unwrap()
}

#[test]
fn builtins_win_over_a_same_name_udf() {
    let s = Session::new().unwrap();
    s.register_udf(UpperUdf);
    // The bare name is the built-in, as in Spark 4.2.
    assert_eq!(one_string(&s, "SELECT upper('a')"), "A");
    // `system.builtin` reaches built-ins only.
    assert_eq!(one_string(&s, "SELECT system.builtin.upper('a')"), "A");
    // `system.session` and `session` reach the UDF only.
    assert_eq!(one_string(&s, "SELECT system.session.upper('a')"), "udf:a");
    assert_eq!(one_string(&s, "SELECT session.upper('a')"), "udf:a");
    // `system.builtin` of a UDF-only name is unknown.
    let e = block_on(s.sql("SELECT system.builtin.add_one(1)")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnresolvedRoutine);
}

#[test]
fn unknown_functions_keep_todays_unresolved_routine() {
    let s = Session::new().unwrap();
    let e = block_on(s.sql("SELECT nosuchfn(1)")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnresolvedRoutine);
    assert_eq!(e.sql_state(), Some("42883"));
    assert_eq!(
        e.message_parameters(),
        [
            ("routineName", "`nosuchfn`".to_owned()),
            (
                "searchPath",
                "[`system`.`builtin`, `system`.`session`, `spark_catalog`.`default`]".to_owned()
            ),
        ]
    );
}

/// `v1` then `v2`: re-registering replaces the implementation.
#[derive(Debug)]
struct Versioned {
    version: u32,
    calls: Arc<AtomicUsize>,
}

impl ScalarUdf for Versioned {
    fn name(&self) -> &str {
        "versioned"
    }

    fn version(&self) -> u32 {
        self.version
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Exact(Vec::new()))
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: false,
        })
    }

    fn invoke(
        &self,
        _: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let version =
            i32::try_from(self.version).map_err(|_| UdfError::new("version out of range"))?;
        Ok(Arc::new(Int32Array::from(vec![version; rows])))
    }
}

#[test]
fn re_registering_a_name_replaces_it() {
    let s = Session::new().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    s.register_udf(Versioned {
        version: 1,
        calls: Arc::clone(&calls),
    });
    let rows = collect(&s, "SELECT versioned()");
    assert_eq!(ints(&rows), [Some(1)]);
    // Re-registering replaces the implementation (and its version).
    s.register_udf(Versioned {
        version: 2,
        calls: Arc::clone(&calls),
    });
    let rows = collect(&s, "SELECT versioned()");
    assert_eq!(ints(&rows), [Some(2)]);
}

/// A strict UDF counts its invocations; a custom one counts its nulls.
#[derive(Debug)]
struct NullProbe {
    strict: bool,
    invocations: Arc<AtomicUsize>,
    nulls_seen: Arc<AtomicUsize>,
}

impl ScalarUdf for NullProbe {
    fn name(&self) -> &str {
        if self.strict {
            "strict_probe"
        } else {
            "custom_probe"
        }
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(int_sig)
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: true,
        })
    }

    fn properties(&self) -> UdfProperties {
        UdfProperties {
            deterministic: true,
            null_handling: if self.strict {
                NullHandling::Strict
            } else {
                NullHandling::Custom
            },
        }
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        _rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        self.invocations.fetch_add(1, Ordering::Relaxed);
        let input = args[0]
            .array()
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| UdfError::new("probe needs INT"))?
            .clone();
        self.nulls_seen
            .fetch_add(input.null_count(), Ordering::Relaxed);
        if self.strict {
            assert_eq!(input.null_count(), 0, "a strict UDF never sees a null");
        }
        Ok(Arc::new(input))
    }
}

#[test]
fn strict_never_sees_nulls_custom_does() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))],
    )
    .unwrap();
    s.register_batches("nums", schema, vec![batch]).unwrap();
    let strict_calls = Arc::new(AtomicUsize::new(0));
    let strict_nulls = Arc::new(AtomicUsize::new(0));
    let custom_calls = Arc::new(AtomicUsize::new(0));
    let custom_nulls = Arc::new(AtomicUsize::new(0));
    s.register_udf(NullProbe {
        strict: true,
        invocations: Arc::clone(&strict_calls),
        nulls_seen: Arc::clone(&strict_nulls),
    });
    s.register_udf(NullProbe {
        strict: false,
        invocations: Arc::clone(&custom_calls),
        nulls_seen: Arc::clone(&custom_nulls),
    });
    // `a` is `[1, NULL, 3]`.
    let rows = collect(
        &s,
        "SELECT strict_probe(a), custom_probe(a) FROM nums ORDER BY a",
    );
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    // The null row is null without invoking under Strict; Custom sees it.
    assert_eq!(strict_nulls.load(Ordering::Relaxed), 0);
    assert_eq!(custom_nulls.load(Ordering::Relaxed), 1);
    let first: Vec<Option<i32>> = rows
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_primitive::<Int32Type>()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect();
    let second: Vec<Option<i32>> = rows
        .iter()
        .flat_map(|b| {
            b.column(1)
                .as_primitive::<Int32Type>()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(first, [None, Some(1), Some(3)]);
    assert_eq!(second, [None, Some(1), Some(3)]);
}

/// Records the row counts it is invoked with.
#[derive(Debug)]
struct RowRecorder {
    rows: Arc<std::sync::Mutex<Vec<usize>>>,
}

impl ScalarUdf for RowRecorder {
    fn name(&self) -> &str {
        "recorder"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| {
            Signature::Exact(vec![
                UdfInputType::Exact(DataType::Integer),
                UdfInputType::Exact(DataType::Integer),
            ])
        })
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: false,
        })
    }

    fn invoke(
        &self,
        _: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        self.rows
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(rows);
        Ok(Arc::new(Int32Array::from(vec![1; rows])))
    }
}

#[test]
fn all_literal_calls_invoke_once_and_broadcast() {
    let s = session();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    s.register_udf(RowRecorder {
        rows: Arc::clone(&seen),
    });
    // Two literal arguments over three rows: computed once, broadcast.
    let rows = collect(&s, "SELECT recorder(1, 2) FROM people");
    assert_eq!(ints(&rows), [Some(1), Some(1), Some(1)]);
    assert_eq!(seen.lock().unwrap().as_slice(), [1]);
}

/// A UDF echoing its string argument, for implicit-cast parity.
#[derive(Debug)]
struct EchoString;

impl ScalarUdf for EchoString {
    fn name(&self) -> &str {
        "echo_str"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::String)]))
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::String,
            nullable: true,
        })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        let input = args[0].array();
        assert_eq!(
            input.len(),
            if matches!(args[0], UdfArg::Scalar(_)) {
                1
            } else {
                rows
            }
        );
        if let Some(view) = input.as_any().downcast_ref::<StringViewArray>() {
            return Ok(Arc::new(view.iter().collect::<StringArray>()));
        }
        let plain = input
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| UdfError::new("echo_str needs STRING"))?;
        Ok(Arc::new(plain.clone()))
    }
}

#[test]
fn implicit_casts_match_a_builtin_with_the_same_inputs() {
    let s = Session::new().unwrap();
    s.register_udf(EchoString);
    // `INT` implicitly casts to `STRING`, exactly as for the built-in
    // `upper` with the same declared input.
    assert_eq!(one_string(&s, "SELECT echo_str(1)"), "1");
    assert_eq!(one_string(&s, "SELECT upper(1)"), "1");
}

/// A UDF with coercible (strict) integer input.
#[derive(Debug)]
struct StrictInt;

impl ScalarUdf for StrictInt {
    fn name(&self) -> &str {
        "strict_int"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Coercible(vec![UdfInputType::Exact(DataType::Integer)]))
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: true,
        })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        let input = args[0].array();
        let input = input
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| UdfError::new("strict_int needs INT"))?;
        assert_eq!(
            input.len(),
            if matches!(args[0], UdfArg::Scalar(_)) {
                1
            } else {
                rows
            }
        );
        Ok(Arc::new(input.clone()))
    }
}

#[test]
fn coercible_positions_match_strictly() {
    let s = Session::new().unwrap();
    s.register_udf(AddOne);
    s.register_udf(StrictInt);
    // `Exact` coerces a `BIGINT` down to `INT`.
    let rows = collect(&s, "SELECT add_one(CAST(5 AS BIGINT))");
    assert_eq!(ints(&rows), [Some(6)]);
    // `Coercible` does not: a `BIGINT` is the wrong input type.
    let e = block_on(s.sql("SELECT strict_int(CAST(5 AS BIGINT))")).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
}

/// A failing, panicking or contract-breaking UDF.
#[derive(Debug)]
struct Explosive {
    what: &'static str,
}

impl ScalarUdf for Explosive {
    fn name(&self) -> &str {
        match self.what {
            "short" => "short_result",
            "type" => "explosive_type",
            "nullable" => "explosive_nullable",
            "error" => "explosive_error",
            _ => "explosive_panic",
        }
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(int_sig)
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            // Never nullable: a null result is a contract break.
            nullable: false,
        })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        match self.what {
            "type" => Ok(Arc::new(StringArray::from(vec!["x"; rows]))),
            "short" => Ok(Arc::new(Int32Array::from(vec![Some(1)]))),
            "nullable" => Ok(Arc::new(Int32Array::from(vec![None; rows]))),
            "error" => Err(UdfError::new("cannot compute this")),
            "panic" => panic!("udf went bang"),
            _ => {
                let input = args[0].array();
                Ok(input)
            }
        }
    }
}

fn explosive_fails(s: &Session, query: &str, reason: &str) {
    let df = block_on(s.sql(query)).unwrap();
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedExecuteUdf, "{query}");
    assert_eq!(e.sql_state(), Some("39000"), "{query}");
    let text = e.to_string();
    assert!(text.contains(reason), "{query}: {text}");
}

#[test]
fn contract_breaks_user_errors_and_panics_fail_the_query() {
    let s = session();
    for what in ["type", "nullable", "error", "panic"] {
        s.register_udf(Explosive { what });
    }
    s.register_udf(Explosive { what: "short" });
    // Wrong result type, nullability, user error and panic.
    explosive_fails(&s, "SELECT explosive_type(1)", "explosive_type");
    explosive_fails(&s, "SELECT explosive_nullable(1)", "explosive_nullable");
    explosive_fails(&s, "SELECT explosive_error(1)", "cannot compute this");
    explosive_fails(&s, "SELECT explosive_panic(1)", "udf went bang");
    // Wrong result length: one row for three (a column argument, so no
    // all-literal broadcast hides it).
    let df = block_on(s.sql("SELECT short_result(id) FROM people")).unwrap();
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedExecuteUdf);
    assert!(e.to_string().contains("short_result"), "{e}");
    // The wrong arity never reaches the UDF.
    let e = block_on(s.sql("SELECT short_result(1, 2)")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
}

/// A counting echo: deterministic or not.
#[derive(Debug)]
struct CountingEcho {
    deterministic: bool,
    calls: Arc<AtomicUsize>,
}

impl ScalarUdf for CountingEcho {
    fn name(&self) -> &str {
        if self.deterministic {
            "det_echo"
        } else {
            "nondet_echo"
        }
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::String)]))
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::String,
            nullable: true,
        })
    }

    fn properties(&self) -> UdfProperties {
        UdfProperties {
            deterministic: self.deterministic,
            null_handling: NullHandling::Strict,
        }
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        use arrow_array::{StringArray, StringViewArray};
        self.calls.fetch_add(1, Ordering::Relaxed);
        let input = args[0].array();
        assert_eq!(
            input.len(),
            if matches!(args[0], UdfArg::Scalar(_)) {
                1
            } else {
                rows
            }
        );
        if let Some(view) = input.as_any().downcast_ref::<StringViewArray>() {
            return Ok(Arc::new(view.iter().collect::<StringArray>()));
        }
        let plain = input
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| UdfError::new("echo needs STRING"))?;
        Ok(Arc::new(plain.clone()))
    }
}

#[test]
fn only_deterministic_calls_fold() {
    let s = Session::new().unwrap();
    let nondet_calls = Arc::new(AtomicUsize::new(0));
    s.register_udf(CountingEcho {
        deterministic: true,
        calls: Arc::new(AtomicUsize::new(0)),
    });
    s.register_udf(CountingEcho {
        deterministic: false,
        calls: Arc::clone(&nondet_calls),
    });
    // The deterministic call folds to a literal: the plan holds `'a'`,
    // not the call (the output name still cites it, as for built-ins).
    let plan = explain(&s, "SELECT det_echo('a')");
    assert!(plan.contains("'a' AS"), "{plan}");
    assert!(!plan.contains("det_echo('a')"), "{plan}");
    assert_eq!(one_string(&s, "SELECT det_echo('a')"), "a");
    // The nondeterministic call stays in the plan and runs.
    let plan = explain(&s, "SELECT nondet_echo('a')");
    assert!(plan.contains("nondet_echo('a')"), "{plan}");
    assert_eq!(one_string(&s, "SELECT nondet_echo('a')"), "a");
    assert!(nondet_calls.load(Ordering::Relaxed) >= 1);
}

fn explain(s: &Session, query: &str) -> String {
    let df = block_on(s.sql(&format!("EXPLAIN {query}"))).unwrap();
    let rows = block_on(df.collect()).unwrap();
    let mut out = String::new();
    for b in &rows {
        for c in b.columns() {
            if let Some(view) = c.as_any().downcast_ref::<StringViewArray>() {
                for v in view.iter().flatten() {
                    out.push_str(v);
                    out.push('\n');
                }
            }
            if let Some(plain) = c.as_any().downcast_ref::<StringArray>() {
                for v in plain.iter().flatten() {
                    out.push_str(v);
                    out.push('\n');
                }
            }
        }
    }
    out
}

/// A variadic string join.
#[derive(Debug)]
struct JoinStrings;

impl ScalarUdf for JoinStrings {
    fn name(&self) -> &str {
        "join_strings"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Variadic {
            element: UdfInputType::Exact(DataType::String),
            min_arity: 1,
        })
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::String,
            nullable: true,
        })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        use arrow_array::{StringArray, StringViewArray};
        fn text(arg: &ArrayRef) -> Result<Vec<Option<String>>, UdfError> {
            if let Some(view) = arg.as_any().downcast_ref::<StringViewArray>() {
                return Ok(view.iter().map(|v| v.map(str::to_owned)).collect());
            }
            arg.as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| UdfError::new("join_strings needs STRING"))
                .map(|a| a.iter().map(|v| v.map(str::to_owned)).collect())
        }
        let columns: Vec<Vec<Option<String>>> = args
            .iter()
            .map(|a| text(&a.array()))
            .collect::<Result<_, _>>()?;
        let mut out = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut parts = Vec::new();
            for (j, column) in columns.iter().enumerate() {
                let at = if column.len() == 1 { 0 } else { i };
                match column.get(at) {
                    Some(Some(v)) => parts.push(v.clone()),
                    _ => parts.push(format!("<null{j}>")),
                }
            }
            out.push(Some(parts.join(",")));
        }
        Ok(Arc::new(StringArray::from(out)))
    }
}

#[test]
fn variadic_udfs_take_any_arity_above_the_minimum() {
    let s = Session::new().unwrap();
    s.register_udf(JoinStrings);
    assert_eq!(one_string(&s, "SELECT join_strings('a')"), "a");
    assert_eq!(
        one_string(&s, "SELECT join_strings('a', 'b', 'c')"),
        "a,b,c"
    );
    let e = block_on(s.sql("SELECT join_strings()")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
}

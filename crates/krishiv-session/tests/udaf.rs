// SPDX-License-Identifier: Apache-2.0
//! Native Rust aggregate UDFs end to end (ADR 0045): grouped and global
//! results against the built-in they mirror, over many drivers, through the
//! plan codec and under a spill; FILTER, NULL skipping and DISTINCT; names
//! shared with built-ins; contract breaks and panics; windows.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::{
    AggregateProperties, AggregateUdf, ArgField, GroupsAccumulator, InvokeContext, ReturnField,
    Session, Signature, StateField, UdfArg, UdfError, UdfInputType,
};
use krishiv_types::DataType;

/// How [`UdSum`] breaks its contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    PanicInUpdate,
    WrongResultType,
    ShortState,
}

/// `name(x BIGINT)`: the sum of the non-NULL `x`, NULL for none (as
/// `sum`), with the state `(sum BIGINT NULL)`.
#[derive(Debug)]
struct UdSum {
    name: &'static str,
    fault: Fault,
    signature: Signature,
}

impl UdSum {
    fn new(name: &'static str, fault: Fault) -> Self {
        Self {
            name,
            fault,
            signature: Signature::Exact(vec![UdfInputType::Exact(DataType::Long)]),
        }
    }
}

impl AggregateUdf for UdSum {
    fn name(&self) -> &str {
        self.name
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_field(&self, _args: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Long,
            nullable: true,
        })
    }

    fn state_fields(&self, _args: &[ArgField]) -> Result<Vec<StateField>, UdfError> {
        Ok(vec![StateField {
            name: "sum".to_owned(),
            data_type: DataType::Long,
            nullable: true,
        }])
    }

    fn properties(&self) -> AggregateProperties {
        AggregateProperties::DEFAULT
    }

    fn accumulator(
        &self,
        _args: &[ArgField],
        _ctx: &InvokeContext,
    ) -> Result<Box<dyn GroupsAccumulator>, UdfError> {
        Ok(Box::new(Sums {
            sums: Vec::new(),
            fault: self.fault,
        }))
    }
}

#[derive(Debug)]
struct Sums {
    sums: Vec<Option<i64>>,
    fault: Fault,
}

impl Sums {
    fn add(&mut self, g: u32, v: i64) -> Result<(), UdfError> {
        let s = &mut self.sums[g as usize];
        *s = Some(
            s.unwrap_or(0)
                .checked_add(v)
                .ok_or_else(|| UdfError::new("overflow"))?,
        );
        Ok(())
    }
}

impl GroupsAccumulator for Sums {
    fn resize(&mut self, groups: usize) {
        self.sums.resize(groups, None);
    }

    fn update(&mut self, args: &[UdfArg<'_>], groups: &[u32]) -> Result<(), UdfError> {
        assert!(self.fault != Fault::PanicInUpdate, "boom");
        let array = args[0].array();
        let values = array.as_primitive::<Int64Type>();
        // The engine skips NULL rows of a strict UDF.
        if values.null_count() > 0 {
            return Err(UdfError::new("got a NULL row"));
        }
        for (i, &g) in groups.iter().enumerate() {
            self.add(g, values.value(i))?;
        }
        Ok(())
    }

    fn merge(&mut self, states: &[ArrayRef], groups: &[u32]) -> Result<(), UdfError> {
        let sums = states[0].as_primitive::<Int64Type>();
        for (i, &g) in groups.iter().enumerate() {
            if sums.is_valid(i) {
                self.add(g, sums.value(i))?;
            }
        }
        Ok(())
    }

    fn state(&self, groups: &[u32]) -> Result<Vec<ArrayRef>, UdfError> {
        if self.fault == Fault::ShortState {
            return Ok(vec![Arc::new(Int64Array::from(Vec::<i64>::new()))]);
        }
        Ok(vec![self.values(groups)])
    }

    fn evaluate(&self, groups: &[u32]) -> Result<ArrayRef, UdfError> {
        if self.fault == Fault::WrongResultType {
            return Ok(Arc::new(arrow_array::Int32Array::from(vec![
                0;
                groups.len()
            ])));
        }
        Ok(self.values(groups))
    }

    fn size(&self) -> usize {
        self.sums.capacity() * size_of::<Option<i64>>()
    }
}

impl Sums {
    fn values(&self, groups: &[u32]) -> ArrayRef {
        Arc::new(Int64Array::from_iter(
            groups.iter().map(|&g| self.sums[g as usize]),
        ))
    }
}

const ROWS: i64 = 200_000;

/// `t(k BIGINT, v BIGINT NULL)`: `keys` groups, a NULL `v` every seventh
/// row, in batches of 4096.
fn register_t(s: &Session, keys: i64) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", ArrowType::Int64, false),
        Field::new("v", ArrowType::Int64, true),
    ]));
    let rows: Vec<i64> = (0..ROWS).collect();
    let batches = rows
        .chunks(4096)
        .map(|c| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(c.iter().map(|i| i % keys))),
                    Arc::new(Int64Array::from_iter(
                        c.iter().map(|i| (i % 7 != 0).then_some(i % 1000)),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("t", schema, batches).unwrap();
}

fn session_with(config: &[(&str, &str)], fault: Fault) -> Session {
    let mut b = Session::builder();
    for (k, v) in config {
        b = b.config(*k, *v);
    }
    let s = b.build().unwrap();
    s.register_udaf(UdSum::new("udsum", fault)).unwrap();
    s
}

/// The rows of a two-column `BIGINT` result.
fn pairs(batches: &[RecordBatch]) -> Vec<(Option<i64>, Option<i64>)> {
    let mut out = Vec::new();
    for b in batches {
        let (a, c) = (
            b.column(0).as_primitive::<Int64Type>(),
            b.column(1).as_primitive::<Int64Type>(),
        );
        for i in 0..b.num_rows() {
            out.push((
                (!a.is_null(i)).then(|| a.value(i)),
                (!c.is_null(i)).then(|| c.value(i)),
            ));
        }
    }
    out
}

fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap()
}

fn error_of(s: &Session, sql: &str) -> krishiv_common::error::KrishivError {
    block_on(async {
        match s.sql(sql).await {
            Ok(df) => df.collect().await.map(|_| ()),
            Err(e) => Err(e),
        }
    })
    .unwrap_err()
}

/// The UDF and the built-in over the same groups.
fn check_against_sum(s: &Session) {
    let got = run(s, "SELECT k, udsum(v) FROM t GROUP BY k ORDER BY k");
    let want = run(s, "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k");
    assert_eq!(pairs(&got), pairs(&want));
    let got = run(s, "SELECT count(*), udsum(v) FROM t");
    let want = run(s, "SELECT count(*), sum(v) FROM t");
    assert_eq!(pairs(&got), pairs(&want));
}

#[test]
fn grouped_and_global_results_match_the_builtin_over_many_drivers() {
    for keys in [1, 4, 16, 50_000] {
        let s = session_with(&[], Fault::None);
        register_t(&s, keys);
        check_against_sum(&s);
    }
}

#[test]
fn plans_round_trip_through_the_codec() {
    let s = session_with(&[("krishiv.exec.debug.planRoundTrip", "true")], Fault::None);
    register_t(&s, 1000);
    check_against_sum(&s);
}

#[test]
fn a_spilling_aggregate_merges_the_udf_states() {
    let dir = tempfile::tempdir().unwrap();
    let spill = dir.path().join("spill");
    let s = session_with(
        &[
            ("krishiv.spill.dir", spill.to_str().unwrap()),
            ("krishiv.memory.limit", "6m"),
        ],
        Fault::None,
    );
    register_t(&s, 60_000);
    let df = block_on(s.sql("SELECT k, udsum(v) FROM t GROUP BY k ORDER BY k")).unwrap();
    let (got, profile) = block_on(df.collect_with_profile()).unwrap();
    let files: u64 = profile
        .metrics
        .operators
        .values()
        .map(|e| e.metrics.spill_files)
        .sum();
    assert!(files > 0, "the aggregate did not spill");
    let want = run(&s, "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k");
    assert_eq!(pairs(&got), pairs(&want));
}

#[test]
fn filter_nulls_and_distinct_are_the_engines() {
    let s = session_with(&[], Fault::None);
    register_t(&s, 10);
    let got = run(
        &s,
        "SELECT k, udsum(v) FILTER (WHERE v > 500) FROM t GROUP BY k ORDER BY k",
    );
    let want = run(
        &s,
        "SELECT k, sum(v) FILTER (WHERE v > 500) FROM t GROUP BY k ORDER BY k",
    );
    assert_eq!(pairs(&got), pairs(&want));
    let got = run(
        &s,
        "SELECT k, udsum(DISTINCT v) FROM t GROUP BY k ORDER BY k",
    );
    let want = run(&s, "SELECT k, sum(DISTINCT v) FROM t GROUP BY k ORDER BY k");
    assert_eq!(pairs(&got), pairs(&want));
    // Only NULLs, and no rows: NULL, as the built-in.
    let got = run(&s, "SELECT count(*), udsum(v) FROM t WHERE v IS NULL");
    assert_eq!(pairs(&got), [(Some(ROWS / 7 + 1), None)]);
    let got = run(&s, "SELECT count(*), udsum(v) FROM t WHERE k < 0");
    assert_eq!(pairs(&got), [(Some(0), None)]);
}

#[test]
fn a_udaf_named_like_a_builtin_runs_only_under_its_session_name() {
    let s = session_with(&[], Fault::None);
    s.register_udaf(UdSum::new("max", Fault::None)).unwrap();
    register_t(&s, 3);
    // `max` is the built-in, `session.max` the UDF (a sum here).
    let got = run(
        &s,
        "SELECT max(v), session.max(v) FROM t WHERE k = 0 AND v IS NOT NULL",
    );
    let want = run(
        &s,
        "SELECT max(v), sum(v) FROM t WHERE k = 0 AND v IS NOT NULL",
    );
    assert_eq!(pairs(&got), pairs(&want));
}

#[test]
fn having_and_distinct_rewrites_keep_the_udaf() {
    let s = session_with(&[], Fault::None);
    s.register_udaf(UdSum::new("max", Fault::None)).unwrap();
    register_t(&s, 5);
    // HAVING reuses the SELECT's calls by meaning: the UDF and the
    // built-in of its name are different calls, either way round.
    for (got, want) in [
        (
            "SELECT k, session.max(v) FROM t GROUP BY k HAVING max(v) > 990 ORDER BY k",
            "SELECT k, sum(v) FROM t GROUP BY k HAVING max(v) > 990 ORDER BY k",
        ),
        (
            "SELECT k, max(v) FROM t GROUP BY k HAVING session.max(v) > 0 ORDER BY k",
            "SELECT k, max(v) FROM t GROUP BY k HAVING sum(v) > 0 ORDER BY k",
        ),
    ] {
        assert_eq!(pairs(&run(&s, got)), pairs(&run(&s, want)), "{got}");
    }
    // A filtered DISTINCT call is rewritten and bound again.
    let got = run(
        &s,
        "SELECT count(*), session.max(DISTINCT v) FILTER (WHERE v > 3) FROM t",
    );
    let want = run(
        &s,
        "SELECT count(*), sum(DISTINCT v) FILTER (WHERE v > 3) FROM t",
    );
    assert_eq!(pairs(&got), pairs(&want));
}

#[test]
fn a_distinct_rewrite_never_merges_or_loses_a_udaf() {
    let s = session_with(&[], Fault::None);
    s.register_udaf(UdSum::new("max", Fault::None)).unwrap();
    register_t(&s, 5);
    // The built-in and the UDF, both rewritten: still two calls.
    let got = run(
        &s,
        "SELECT max(DISTINCT v) FILTER (WHERE v > 0), \
         session.max(DISTINCT v) FILTER (WHERE v > 0) FROM t",
    );
    let want = run(
        &s,
        "SELECT max(DISTINCT v) FILTER (WHERE v > 0), sum(DISTINCT v) FILTER (WHERE v > 0) FROM t",
    );
    assert_eq!(pairs(&got), pairs(&want));
    // A regular UDF call next to the rewrite: grouped, as its built-in twin...
    let got = run(
        &s,
        "SELECT session.max(v), sum(DISTINCT v) FILTER (WHERE v > 0) FROM t GROUP BY k ORDER BY k",
    );
    let want = run(
        &s,
        "SELECT sum(v), sum(DISTINCT v) FILTER (WHERE v > 0) FROM t GROUP BY k ORDER BY k",
    );
    assert_eq!(pairs(&got), pairs(&want));
    // ... but refused without GROUP BY, where its empty-group value would be lost.
    let e = error_of(
        &s,
        "SELECT session.max(v), sum(DISTINCT v) FILTER (WHERE v > 0) FROM t",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
}

/// `name()`: the number of rows, with no argument.
#[derive(Debug)]
struct Rows(Signature);

impl AggregateUdf for Rows {
    fn name(&self) -> &str {
        "count"
    }
    fn version(&self) -> u32 {
        1
    }
    fn signature(&self) -> &Signature {
        &self.0
    }
    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Long,
            nullable: true,
        })
    }
    fn state_fields(&self, _: &[ArgField]) -> Result<Vec<StateField>, UdfError> {
        Ok(vec![StateField {
            name: "sum".to_owned(),
            data_type: DataType::Long,
            nullable: true,
        }])
    }
    fn accumulator(
        &self,
        _: &[ArgField],
        _: &InvokeContext,
    ) -> Result<Box<dyn GroupsAccumulator>, UdfError> {
        Ok(Box::new(RowCount(Vec::new())))
    }
}

#[derive(Debug)]
struct RowCount(Vec<i64>);

impl GroupsAccumulator for RowCount {
    fn resize(&mut self, groups: usize) {
        self.0.resize(groups, 0);
    }
    fn update(&mut self, _: &[UdfArg<'_>], groups: &[u32]) -> Result<(), UdfError> {
        for &g in groups {
            self.0[g as usize] += 1;
        }
        Ok(())
    }
    fn merge(&mut self, states: &[ArrayRef], groups: &[u32]) -> Result<(), UdfError> {
        let s = states[0].as_primitive::<Int64Type>();
        for (i, &g) in groups.iter().enumerate() {
            self.0[g as usize] += s.value(i);
        }
        Ok(())
    }
    fn state(&self, groups: &[u32]) -> Result<Vec<ArrayRef>, UdfError> {
        Ok(vec![self.evaluate(groups)?])
    }
    fn evaluate(&self, groups: &[u32]) -> Result<ArrayRef, UdfError> {
        Ok(Arc::new(Int64Array::from_iter_values(
            groups.iter().map(|&g| self.0[g as usize]),
        )))
    }
    fn size(&self) -> usize {
        self.0.capacity() * 8
    }
}

#[test]
fn built_in_special_cases_skip_a_same_name_udaf() {
    let s = session_with(&[], Fault::None);
    s.register_udaf(Rows(Signature::Exact(Vec::new()))).unwrap();
    s.register_udaf(UdSum::new("max", Fault::None)).unwrap();
    register_t(&s, 5);
    // `count()` with no argument is the built-in's error; the UDF's
    // signature takes none.
    let got = run(&s, "SELECT k, session.count() FROM t GROUP BY k ORDER BY k");
    let want = run(&s, "SELECT k, count(*) FROM t GROUP BY k ORDER BY k");
    assert_eq!(pairs(&got), pairs(&want));
    // A HAVING aggregate next to a DISTINCT rewrite: only the built-in
    // extrema may run over the rewrite's deduplicated rows; the UDF is
    // refused, as the built-in `sum` is there, never computed over them.
    for sql in [
        "SELECT k, sum(DISTINCT v) FILTER (WHERE v > 0) FROM t GROUP BY k HAVING session.max(k) > 10",
        "SELECT k, sum(DISTINCT v) FILTER (WHERE v > 0) FROM t GROUP BY k HAVING sum(k) > 10",
    ] {
        let e = error_of(&s, sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql}: {e}"
        );
    }
}

#[test]
fn a_correlated_subquery_never_invents_a_udafs_empty_result() {
    let s = session_with(&[], Fault::None);
    // A sum under `count`'s name: NULL over no rows, where `count` is 0.
    s.register_udaf(UdSum::new("count", Fault::None)).unwrap();
    register_t(&s, 5);
    let sql = "SELECT CAST(o.k AS BIGINT), (SELECT session.count(v) FROM t WHERE t.k = o.k) \
               FROM VALUES (99) AS o(k)";
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(batches) => assert_eq!(pairs(&batches), [(Some(99), None)]),
        Err(e) => assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{e}"
        ),
    }
}

#[test]
fn contract_breaks_and_panics_fail_the_query_naming_the_udf() {
    for fault in [Fault::PanicInUpdate, Fault::WrongResultType] {
        let s = session_with(&[], fault);
        register_t(&s, 4);
        let e = error_of(&s, "SELECT k, udsum(v) FROM t GROUP BY k");
        assert_eq!(
            e.error_class(),
            ErrorClass::FailedExecuteUdf,
            "{fault:?}: {e}"
        );
        assert!(e.message().contains("udsum"), "{e}");
    }
    // A broken state shows when the state is read: under a spill.
    let dir = tempfile::tempdir().unwrap();
    let spill = dir.path().join("spill");
    let s = session_with(
        &[
            ("krishiv.spill.dir", spill.to_str().unwrap()),
            ("krishiv.memory.limit", "6m"),
        ],
        Fault::ShortState,
    );
    register_t(&s, 60_000);
    let e = error_of(&s, "SELECT k, udsum(v) FROM t GROUP BY k");
    assert_eq!(e.error_class(), ErrorClass::FailedExecuteUdf, "{e}");
}

#[test]
fn a_window_over_a_udaf_is_not_supported_yet() {
    let s = session_with(&[], Fault::None);
    register_t(&s, 4);
    let e = error_of(&s, "SELECT udsum(v) OVER (PARTITION BY k) FROM t");
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
}

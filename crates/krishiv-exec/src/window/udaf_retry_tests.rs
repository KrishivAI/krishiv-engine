// SPDX-License-Identifier: Apache-2.0
//! An aggregate UDF's window state the pool refuses (design
//! `P2-window-udaf-charge-retry`): the window asks the driver for the whole
//! state and evaluates again, in `execute`, at the end of input (`flush`) and
//! in the flush phase after a spill; a request the driver cannot meet ends
//! in the memory error, not a loop; no row is lost or taken twice.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing; never runs on a compute worker"
)]

use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::{DataType, Field};
use krishiv_common::error::ErrorClass;
use krishiv_memory::Reservation;

use krishiv_types::batch::Batch;
use krishiv_udf_sdk::{
    AggregateProperties, AggregateUdf, ArgField, GroupsAccumulator, InvokeContext, ReturnField,
    Signature, StateField, UdfArg, UdfError, UdfLibrary,
};

use super::tests::{Row, batch, outputs, plan, schema, spill_scope, task_with};
use super::*;
use crate::aggregate::{AggInput, AggregateExpr, AggregateFunction, UdafCall, UdafEnv};
use crate::operator::{FlushOutput, Operator, OperatorOutput};
use crate::{ExecConfig, TaskContext};

/// The bytes of state [`Grow`] reports per row it was given.
const PER_ROW: usize = 256;

/// `grow(x)`: the number of rows, with a state that reports `.1` bytes
/// for every row it was given (a state that keeps its rows).
#[derive(Debug)]
struct Grow(Signature, usize);

impl AggregateUdf for Grow {
    fn name(&self) -> &str {
        "grow"
    }
    fn version(&self) -> u32 {
        1
    }
    fn signature(&self) -> &Signature {
        &self.0
    }
    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: krishiv_types::DataType::Long,
            nullable: false,
        })
    }
    fn state_fields(&self, _: &[ArgField]) -> Result<Vec<StateField>, UdfError> {
        Ok(vec![StateField {
            name: "rows".to_owned(),
            data_type: krishiv_types::DataType::Long,
            nullable: false,
        }])
    }
    fn properties(&self) -> AggregateProperties {
        AggregateProperties::DEFAULT
    }
    fn accumulator(
        &self,
        _: &[ArgField],
        _: &InvokeContext,
    ) -> Result<Box<dyn GroupsAccumulator>, UdfError> {
        Ok(Box::new(Rows {
            rows: Vec::new(),
            seen: 0,
            per_row: self.1,
        }))
    }
}

#[derive(Debug)]
struct Rows {
    rows: Vec<i64>,
    seen: usize,
    per_row: usize,
}

impl GroupsAccumulator for Rows {
    fn resize(&mut self, groups: usize) {
        self.rows.resize(groups, 0);
    }
    fn update(&mut self, _: &[UdfArg<'_>], groups: &[u32]) -> Result<(), UdfError> {
        for &g in groups {
            self.rows[g as usize] += 1;
        }
        self.seen += groups.len();
        Ok(())
    }
    fn merge(&mut self, states: &[ArrayRef], groups: &[u32]) -> Result<(), UdfError> {
        let s = states[0].as_primitive::<Int64Type>();
        for (i, &g) in groups.iter().enumerate() {
            self.rows[g as usize] += s.value(i);
        }
        Ok(())
    }
    fn state(&self, groups: &[u32]) -> Result<Vec<ArrayRef>, UdfError> {
        self.evaluate(groups).map(|a| vec![a])
    }
    fn evaluate(&self, groups: &[u32]) -> Result<ArrayRef, UdfError> {
        Ok(Arc::new(Int64Array::from_iter_values(
            groups.iter().map(|&g| self.rows[g as usize]),
        )))
    }
    fn size(&self) -> usize {
        self.seen * self.per_row
    }
}

fn grow() -> WindowFunction {
    WindowFunction::Aggregate(AggregateExpr {
        function: AggregateFunction::Udf(Box::new(UdafCall {
            name: "grow".to_owned(),
            version: 1,
            args: vec![Field::new("arg0", DataType::Int64, true)],
            arg_types: vec![krishiv_types::DataType::Long],
            result: krishiv_types::DataType::Long,
            nullable: false,
            state: vec![Field::new("rows", DataType::Int64, false)],
        })),
        args: vec![3],
        input: AggInput::Raw,
        mask: None,
        arg_type: DataType::Int64,
        name: "a".to_owned(),
        element: None,
    })
}

/// A window of `grow(v)` over `frame`, partitioned by `p`, ordered by `o`.
fn window(frame: WindowFrame, spill: Option<krishiv_memory::SpillScope>) -> Window {
    window_of(vec![(grow(), frame)], spill, PER_ROW)
}

/// A window of `functions` whose `grow` reports `per_row` bytes a row.
fn window_of(
    functions: Vec<(WindowFunction, WindowFrame)>,
    spill: Option<krishiv_memory::SpillScope>,
    per_row: usize,
) -> Window {
    let mut w = Window::new(schema(), plan(functions, true), true, spill).unwrap();
    let library = UdfLibrary::new();
    library
        .register_aggregate(Arc::new(Grow(
            Signature::Exact(vec![krishiv_udf_sdk::UdfInputType::Exact(
                krishiv_types::DataType::Long,
            )]),
            per_row,
        )))
        .unwrap();
    Arc::get_mut(&mut w.env).unwrap().udafs = Some(UdafEnv {
        library: Arc::new(library),
        time_zone: None,
    });
    w
}

/// 1,200 rows in partitions of 300, in batches of 100, sorted.
fn input() -> Vec<Batch> {
    let rows: Vec<Row> = (0..1_200)
        .map(|i| (Some(i / 300), Some(i), i, Some(i % 7)))
        .collect();
    rows.chunks(100).map(batch).collect()
}

/// What the test's driver does with a `NeedMemory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OnNeed {
    /// Frees the hog and parks the bytes as the consumer's credit.
    Grant,
    /// Gives `bytes` back from the hog and parks exactly that as credit.
    GrantExact,
    /// Asks the window to spill (its buffered rows); others free 1 MiB.
    Spill,
    /// Grants nothing (the driver found no memory).
    Nothing,
}

/// Where a `NeedMemory` came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Execute,
    Flush,
}

struct Run {
    out: Vec<Batch>,
    needs: Vec<(Phase, usize)>,
}

/// Drives `op` over `input` as the driver does: the same batch again after
/// `NeedMemory`, then `finish`, `end_input` and `flush` (the window with an
/// aggregate UDF needs it). `hog` holds the pool's memory; `policy` says what
/// a `NeedMemory` gets. At most `budget` calls, so a loop fails the test.
async fn drive(
    op: &Window,
    ctx: &TaskContext,
    hog: &mut Reservation,
    policy: &mut dyn FnMut(Phase) -> OnNeed,
) -> crate::Result<Run> {
    let budget = 10_000;
    let mut calls = 0;
    let mut local = op.local_state(ctx)?;
    let mut run = Run {
        out: Vec::new(),
        needs: Vec::new(),
    };
    let mut answer =
        |phase: Phase, bytes: usize, local: &mut LocalState, run: &mut Run| -> Option<OnNeed> {
            run.needs.push((phase, bytes));
            let how = policy(phase);
            match how {
                OnNeed::Grant | OnNeed::GrantExact => {
                    let give = if how == OnNeed::Grant {
                        hog.size()
                    } else {
                        // The bytes asked for, rounded up past the pool's quantum.
                        (bytes + (32 << 10)).min(hog.size())
                    };
                    hog.shrink(give);
                    let mut r = ctx.consumer().reservation();
                    r.try_grow(bytes).unwrap();
                    r.shrink_to_credit(bytes);
                    None
                }
                OnNeed::Spill => {
                    // Other consumers give some memory back too, so the flush
                    // phase can read the spilled rows.
                    hog.shrink((1 << 20).min(hog.size()));
                    let _ = local;
                    Some(how)
                }
                OnNeed::Nothing => Some(how),
            }
        };
    for b in input_of(ctx) {
        loop {
            calls += 1;
            assert!(calls < budget, "the window loops");
            match op.execute(&b, &mut local)? {
                OperatorOutput::NeedMoreInput(o) => {
                    run.out.extend(o);
                    break;
                }
                OperatorOutput::HaveMoreOutput(o) => run.out.push(o),
                OperatorOutput::NeedMemory { bytes } => {
                    match answer(Phase::Execute, bytes, &mut local, &mut run) {
                        Some(OnNeed::Spill) => {
                            // A refused batch with no rows buffered frees
                            // nothing; the memory others gave back covers it.
                            op.spill(&mut local, bytes, ctx).await?;
                        }
                        Some(_) => {
                            return Err(krishiv_memory::MemoryError::Exhausted {
                                requested: bytes,
                                available: 0,
                            }
                            .into());
                        }
                        None => {}
                    }
                }
                OperatorOutput::Finished(_) => unreachable!("a window never finishes early"),
            }
        }
    }
    while let Some(b) = op.finish(&mut local)? {
        run.out.push(b);
    }
    assert!(op.needs_flush(), "a window with an aggregate UDF flushes");
    op.end_input(&mut local, ctx).await?;
    loop {
        calls += 1;
        assert!(calls < budget, "the window loops");
        match op.flush(&mut local, ctx).await? {
            FlushOutput::Batch(b) => run.out.push(b),
            FlushOutput::Continue => {}
            FlushOutput::Done => break,
            FlushOutput::NeedMemory { bytes } => {
                if answer(Phase::Flush, bytes, &mut local, &mut run) == Some(OnNeed::Spill) {
                    op.spill(&mut local, bytes, ctx).await?;
                }
            }
        }
    }
    drop(local);
    ctx.consumer().consumer().release_credit();
    Ok(run)
}

fn input_of(_: &TaskContext) -> Vec<Batch> {
    input()
}

/// The output of `frame` on an unconstrained pool.
async fn reference(frame: WindowFrame) -> Vec<String> {
    reference_of(&window(frame, None), 1).await
}

/// The output of `op` (with `functions` window columns) on an unconstrained pool.
async fn reference_of(op: &Window, functions: usize) -> Vec<String> {
    let (_root, ctx) = task_with(1 << 30, config());
    let pool = Arc::clone(ctx.query().pool());
    let mut hog = Reservation::new(Arc::clone(&pool), pool.register("hog", None));
    let run = drive(op, &ctx, &mut hog, &mut |_| unreachable!("no refusal"))
        .await
        .unwrap();
    assert!(run.needs.is_empty());
    outputs(&run.out, functions)
}

/// Complete rows flush in `execute` once 64 of them wait.
fn config() -> ExecConfig {
    ExecConfig {
        batch_size: 64,
        ..ExecConfig::default()
    }
}

/// A 64 MiB pool whose `hog` holds all but `free` bytes.
fn tight(free: usize) -> (Arc<krishiv_memory::TreePool>, TaskContext, Reservation) {
    tight_with(free, config())
}

/// [`tight`] under `config`.
fn tight_with(
    free: usize,
    config: ExecConfig,
) -> (Arc<krishiv_memory::TreePool>, TaskContext, Reservation) {
    let limit = 64 << 20;
    let (root, ctx) = task_with(limit, config);
    let pool = Arc::clone(ctx.query().pool());
    let mut hog = Reservation::new(Arc::clone(&pool), pool.register("hog", None));
    let mut step = limit;
    while step > 0 {
        if hog.try_grow(step).is_err() {
            step /= 2;
        }
    }
    hog.shrink(free);
    (root, ctx, hog)
}

/// Batches of 10,000 rows: nothing flushes before the input ends.
fn at_the_end() -> ExecConfig {
    ExecConfig {
        batch_size: 10_000,
        ..ExecConfig::default()
    }
}

fn frames() -> [WindowFrame; 3] {
    [
        WindowFrame::whole(FrameUnits::Rows),
        WindowFrame::running(FrameUnits::Rows),
        WindowFrame {
            units: FrameUnits::Rows,
            lower: FrameBound::Offset(-2),
            upper: FrameBound::Offset(1),
        },
    ]
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn a_refused_state_in_execute_asks_for_memory_and_takes_the_batch_once() {
    runtime().block_on(async {
        for frame in frames() {
            let want = reference(frame).await;
            let op = window(frame, None);
            let (_root, ctx, mut hog) = tight(48 << 10);
            let run = drive(&op, &ctx, &mut hog, &mut |_| OnNeed::Grant)
                .await
                .unwrap();
            assert!(
                run.needs.iter().any(|(p, _)| *p == Phase::Execute),
                "{frame:?}: the state was refused in execute: {:?}",
                run.needs
            );
            assert_eq!(outputs(&run.out, 1), want, "{frame:?}");
            // The batch passed again was not appended again.
            assert_eq!(op.rows.load(Ordering::Relaxed), 1_200, "{frame:?}");
        }
    });
}

#[test]
fn each_request_names_the_whole_state_so_requests_only_grow() {
    runtime().block_on(async {
        let frame = WindowFrame::running(FrameUnits::Rows);
        let want = reference(frame).await;
        let op = window(frame, None);
        let (_root, ctx, mut hog) = tight(48 << 10);
        let run = drive(&op, &ctx, &mut hog, &mut |_| OnNeed::GrantExact)
            .await
            .unwrap();
        assert_eq!(outputs(&run.out, 1), want);
        assert!(!run.needs.is_empty());
        // A request covers what the evaluation held at the refusal plus the
        // refused growth, never one fold's delta (one row: `PER_ROW`), so a
        // flush is retried a few times, not once per row.
        for (_, bytes) in &run.needs {
            assert!(*bytes >= 16 * PER_ROW, "{:?}", run.needs);
        }
        assert!(run.needs.len() <= 4, "{:?}", run.needs);
    });
}

#[test]
fn the_end_of_input_is_evaluated_in_flush_where_a_refusal_asks_for_memory() {
    runtime().block_on(async {
        for frame in frames() {
            let want = reference(frame).await;
            let op = window(frame, None);
            // Batches of 10,000: nothing flushes before the input ends.
            let (_root, ctx, mut hog) = tight_with(200 << 10, at_the_end());
            // The input fits; the state of 1,200 rows (300 KiB) does not.
            let run = drive(&op, &ctx, &mut hog, &mut |_| OnNeed::GrantExact)
                .await
                .unwrap();
            // A sliding frame's up-front cost may also be refused while
            // its batches are taken; the state is refused at the end.
            assert!(
                run.needs.iter().any(|(p, _)| *p == Phase::Flush),
                "{frame:?}: {:?}",
                run.needs
            );
            assert_eq!(outputs(&run.out, 1), want, "{frame:?}");
        }
    });
}

#[test]
fn a_request_the_driver_cannot_meet_is_the_memory_error_not_a_loop() {
    runtime().block_on(async {
        let op = window(WindowFrame::whole(FrameUnits::Rows), None);
        let (_root, ctx, mut hog) = tight_with(200 << 10, at_the_end());
        let mut needs = 0;
        let e = drive(&op, &ctx, &mut hog, &mut |_| {
            needs += 1;
            OnNeed::Nothing
        })
        .await
        .err()
        .unwrap();
        assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory, "{e}");
        // Asked once; the same refusal again with nothing granted fails.
        assert_eq!(needs, 1);
    });
}

#[test]
fn a_spill_while_a_refused_state_waits_loses_no_row() {
    runtime().block_on(async {
        // 16 KiB of state a row: a partition's state (4.8 MB) is refused in
        // the flush phase too, after the spill, where the flush is retried
        // before the phase checks that its input ended.
        let big = 16 << 10;
        for frame in [
            WindowFrame::whole(FrameUnits::Rows),
            WindowFrame::running(FrameUnits::Rows),
        ] {
            let want = reference_of(&window_of(vec![(grow(), frame)], None, big), 1).await;
            // In execute: the buffered rows spill instead of a grant, and
            // wait for the flush phase.
            let (_dir, scope) = spill_scope();
            let op = window_of(vec![(grow(), frame)], Some(scope), big);
            let (_root, ctx, mut hog) = tight(48 << 10);
            let run = drive(&op, &ctx, &mut hog, &mut |p| match p {
                Phase::Execute => OnNeed::Spill,
                Phase::Flush => OnNeed::GrantExact,
            })
            .await
            .unwrap();
            assert!(op.counters.spills.load(Ordering::Relaxed) > 0, "{frame:?}");
            assert!(
                run.needs.iter().any(|(p, _)| *p == Phase::Flush),
                "{frame:?}: refused in the flush phase: {:?}",
                run.needs
            );
            assert_eq!(outputs(&run.out, 1), want, "{frame:?}");
            assert_eq!(op.rows.load(Ordering::Relaxed), 1_200, "{frame:?}");

            // At the end of input: the rows spill while the state waits,
            // and are read back from disk.
            let (_dir, scope) = spill_scope();
            let op = window_of(vec![(grow(), frame)], Some(scope), big);
            let (_root, ctx, mut hog) = tight_with(200 << 10, at_the_end());
            let mut first = true;
            let run = drive(&op, &ctx, &mut hog, &mut |_| {
                if std::mem::take(&mut first) {
                    OnNeed::Spill
                } else {
                    OnNeed::GrantExact
                }
            })
            .await
            .unwrap();
            assert!(op.counters.spills.load(Ordering::Relaxed) > 0, "{frame:?}");
            assert!(run.needs.len() > 1, "{frame:?}: {:?}", run.needs);
            assert_eq!(outputs(&run.out, 1), want, "{frame:?}");
        }
    });
}

#[test]
fn two_udafs_of_one_evaluation_ask_for_their_states_together() {
    runtime().block_on(async {
        let functions = || {
            vec![
                (grow(), WindowFrame::whole(FrameUnits::Rows)),
                (grow(), WindowFrame::running(FrameUnits::Rows)),
            ]
        };
        let want = reference_of(&window_of(functions(), None, PER_ROW), 2).await;
        let op = window_of(functions(), None, PER_ROW);
        let (_root, ctx, mut hog) = tight(48 << 10);
        let run = drive(&op, &ctx, &mut hog, &mut |_| OnNeed::GrantExact)
            .await
            .unwrap();
        assert!(!run.needs.is_empty());
        assert_eq!(outputs(&run.out, 2), want);
    });
}

// SPDX-License-Identifier: Apache-2.0
//! `RANGE` frames with a year-month interval bound (calendar months) over `DATE` and `TIMESTAMP` keys, from disk as
//! in memory, in a zone with daylight saving time: the zone is the session's (a `TIMESTAMP` column carries only a
//! marker), so the same plan over the same rows frames differently in `America/Los_Angeles` and in UTC.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing and temp-dir cleanup; never runs on a compute worker"
)]

use std::sync::Arc;

use arrow_array::{ArrayRef, Date32Array, Int64Array, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use krishiv_kernels::datetime::{FixedOffset, ZoneRef, resolve_zone};
use krishiv_types::batch::Batch;

use super::tests::{Spills, agg, between, drive, drive_spilling, spill_scope, task};
use super::*;
use crate::aggregate::AggregateFunction;

const HOUR_US: i64 = 3_600_000_000;
/// 2024-01-01T00:00:00Z.
const START_US: i64 = 1_704_067_200_000_000;

/// 600 rows ordered by `ts` (ties, then NULLs first), 14 hours apart: the year takes in both of 2024's transitions
/// in Los Angeles (March 10, November 3). `d` is the same instant's day; `v` is a small value to sum.
fn rows(schema: &SchemaRef) -> Vec<Batch> {
    let ids: Vec<i64> = (0..600).collect();
    ids.chunks(100)
        .map(|c| {
            let ts = TimestampMicrosecondArray::from_iter(c.iter().map(|&i| {
                (i >= 10).then_some(START_US + (i / 2) * 28 * HOUR_US + (i % 2) * 14 * HOUR_US)
            }))
            .with_timezone("UTC");
            let d = Date32Array::from_iter(c.iter().map(|&i| {
                (i >= 10).then(|| {
                    i32::try_from(
                        (START_US + (i / 2) * 28 * HOUR_US + (i % 2) * 14 * HOUR_US)
                            / (24 * HOUR_US),
                    )
                    .unwrap_or(0)
                })
            }));
            let v = Int64Array::from_iter(c.iter().map(|&i| Some(i % 17 + 1)));
            let columns: Vec<ArrayRef> = vec![Arc::new(ts), Arc::new(d), Arc::new(v)];
            Batch::new(RecordBatch::try_new(Arc::clone(schema), columns).unwrap())
        })
        .collect()
}

fn concat(batches: &[Batch]) -> RecordBatch {
    let data: Vec<RecordBatch> = batches.iter().map(|b| b.data().clone()).collect();
    arrow_select::concat::concat_batches(&data[0].schema(), &data).unwrap()
}

fn window(column: usize, frame: WindowFrame) -> WindowPlan {
    WindowPlan {
        partition_by: Vec::new(),
        order_by: vec![SortKey::asc(column)],
        functions: vec![WindowFunctionPlan {
            function: agg(AggregateFunction::Sum, Some(2)),
            frame,
            filter: None,
            name: "w".to_owned(),
        }],
    }
}

fn in_memory(schema: &SchemaRef, plan: &WindowPlan, zone: ZoneRef, input: &[Batch]) -> RecordBatch {
    let op = Window::with_zone(Arc::clone(schema), plan.clone(), true, zone, None).unwrap();
    let (_root, ctx) = task(1 << 30);
    concat(&drive(&op, &ctx, input).unwrap())
}

#[tokio::test]
async fn month_bounds_from_disk_match_memory_in_a_dst_zone() {
    use FrameBound::{CurrentRow, RangeMonths};
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new("d", DataType::Date32, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let input = rows(&schema);
    let la = || resolve_zone("America/Los_Angeles").unwrap();
    let utc = || ZoneRef::Fixed(FixedOffset(0));
    let mut zone_mattered = false;
    for column in [0, 1] {
        for frame in [
            between(FrameUnits::Range, RangeMonths(-1), CurrentRow),
            between(FrameUnits::Range, RangeMonths(-1), RangeMonths(1)),
            between(FrameUnits::Range, CurrentRow, RangeMonths(2)),
        ] {
            let plan = window(column, frame);
            let want = in_memory(&schema, &plan, la(), &input);
            // From disk: the same answer.
            let (_manager, scope) = spill_scope();
            let op = Window::with_zone(Arc::clone(&schema), plan.clone(), true, la(), Some(scope))
                .unwrap();
            let limit = 1 << 30;
            let (root, ctx) = task(limit);
            let spills = Spills {
                every: 1,
                slack: Some(256 << 10),
            };
            let got = drive_spilling(&op, &ctx, limit, &root, &input, spills)
                .await
                .unwrap();
            assert_eq!(concat(&got), want, "column {column} {frame:?}");
            assert!(
                op.counters
                    .external
                    .load(std::sync::atomic::Ordering::Relaxed)
                    >= 1,
                "column {column} {frame:?}: nothing was evaluated from disk"
            );
            drop((got, ctx));
            // A TIMESTAMP key frames by the session zone; a DATE key has none.
            let in_utc = in_memory(&schema, &plan, utc(), &input);
            if column == 0 {
                zone_mattered |= in_utc != want;
            } else {
                assert_eq!(in_utc, want, "a DATE key does not depend on the zone");
            }
        }
    }
    assert!(
        zone_mattered,
        "no month bound over the TIMESTAMP key differed between Los Angeles and UTC"
    );
}

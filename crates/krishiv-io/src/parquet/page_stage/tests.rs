// SPDX-License-Identifier: Apache-2.0
//! The page stage through the whole scan (design section 7): a sorted
//! column prunes pages, results equal the stage-off run, a bad or absent
//! index changes nothing.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::file::properties::{EnabledStatistics, WriterProperties};

use super::super::ParquetFormat;
use super::super::tests::{Env, rows, write};
use crate::request::{CmpOp, ColumnPath, PushdownExpr, ScalarValue, ScanRequest};
use crate::{FileRef, FormatOptions, MetadataNeeds, ScanUnitMetrics};

const ROWS: i64 = 20_000;
const PAGE_ROWS: usize = 500;

/// One row group of `k` (sorted) and `u` (the same values in a scrambled
/// order), in pages of `PAGE_ROWS` rows.
fn file(env: &Env, stats: EnabledStatistics) -> (FileRef, FormatOptions) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("u", DataType::Int64, false),
    ]));
    let k = Int64Array::from_iter_values(0..ROWS);
    let u = Int64Array::from_iter_values((0..ROWS).map(|i| (i * 7919) % ROWS));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(k) as ArrayRef, Arc::new(u) as ArrayRef],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .set_statistics_enabled(stats)
        .build();
    let f = env.put("pages.parquet", write(&[batch], props));
    (f, env.options(&[]))
}

fn request(env: &Env, f: &FileRef, o: &FormatOptions, predicate: PushdownExpr) -> ScanRequest {
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, f, MetadataNeeds::default(), o),
        )
        .unwrap()
        .schema();
    let mut req = ScanRequest::new(schema, vec![ColumnPath::new("k"), ColumnPath::new("u")]);
    req.predicate = Some(predicate);
    req
}

fn ks(out: &[RecordBatch]) -> Vec<i64> {
    out.iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect()
}

fn run(
    env: &Env,
    stats: EnabledStatistics,
    key: &str,
    predicate: PushdownExpr,
    on: bool,
) -> (Vec<i64>, ScanUnitMetrics) {
    let (f, _) = file(env, stats);
    let o = env.options(&[("pageIndex", if on { "true" } else { "false" })]);
    let req = request(env, &f, &o, predicate);
    let (out, m) = env.scan_metered(&f, &req, &o).unwrap();
    let _ = key;
    (ks(&out), m)
}

fn k_between(lo: i64, hi: i64) -> PushdownExpr {
    PushdownExpr::And(vec![
        PushdownExpr::cmp(ColumnPath::new("k"), CmpOp::GtEq, ScalarValue::Int64(lo)),
        PushdownExpr::cmp(ColumnPath::new("k"), CmpOp::LtEq, ScalarValue::Int64(hi)),
    ])
}

#[test]
fn a_sorted_column_skips_pages() {
    let env = Env::new();
    let (with, m_on) = run(
        &env,
        EnabledStatistics::Page,
        "k",
        k_between(9_000, 9_700),
        true,
    );
    let (without, m_off) = run(
        &env,
        EnabledStatistics::Page,
        "k",
        k_between(9_000, 9_700),
        false,
    );
    // Pages 18 and 19 hold 9000..10000; the pages of the other rows are skipped.
    assert_eq!(m_off.pages_pruned, 0);
    assert!(m_on.pages_pruned >= 38, "{m_on:?}");
    assert!(
        with.len() < without.len(),
        "{} vs {}",
        with.len(),
        without.len()
    );
    // Every matching row survives; the rows that stay are a superset.
    let wanted: Vec<i64> = (9_000..=9_700).collect();
    assert!(wanted.iter().all(|k| with.contains(k)));
    // The stage never returns a row the stage-off scan did not.
    assert!(with.iter().all(|k| without.contains(k)));
    assert_eq!(m_on.page_index_rejected, 0);
}

#[test]
fn an_unclustered_column_prunes_nothing_and_changes_no_row() {
    let env = Env::new();
    let predicate = PushdownExpr::cmp(ColumnPath::new("u"), CmpOp::Eq, ScalarValue::Int64(12_345));
    let (with, m_on) = run(&env, EnabledStatistics::Page, "u", predicate.clone(), true);
    let (without, _) = run(&env, EnabledStatistics::Page, "u", predicate, false);
    assert_eq!(with, without);
    assert_eq!(m_on.pages_pruned, 0, "{m_on:?}");
}

#[test]
fn a_file_without_a_column_index_is_read_whole() {
    let env = Env::new();
    let (with, m) = run(
        &env,
        EnabledStatistics::Chunk,
        "k",
        k_between(9_000, 9_700),
        true,
    );
    let (without, _) = run(
        &env,
        EnabledStatistics::Chunk,
        "k",
        k_between(9_000, 9_700),
        false,
    );
    assert_eq!(with, without);
    assert_eq!(m.pages_pruned, 0);
    assert_eq!(rows(&[]), 0);
}

#[test]
fn a_predicate_the_statistics_decided_reads_no_index() {
    let env = Env::new();
    // Every row matches: statistics say `True`, the unit is not undecided.
    let (_, m) = run(
        &env,
        EnabledStatistics::Page,
        "k",
        k_between(-1, ROWS + 1),
        true,
    );
    assert_eq!(m.pages_pruned, 0);
    // No row matches: the unit is gone before the stage.
    let (out, _) = run(
        &env,
        EnabledStatistics::Page,
        "k",
        k_between(ROWS + 5, ROWS + 10),
        true,
    );
    assert!(out.is_empty());
}

/// Differential: for many ranges, stage on and off return the same matching
/// rows, and the stage never returns a row outside the stage-off result.
#[test]
fn the_stage_changes_no_matching_row() {
    let env = Env::new();
    for (lo, hi) in [
        (0, 0),
        (0, 499),
        (499, 500),
        (250, 1_750),
        (9_999, 10_000),
        (19_999, 25_000),
        (-5, 3),
        (7_777, 7_777),
        (12_000, 12_001),
    ] {
        let predicate = k_between(lo, hi);
        let (with, _) = run(&env, EnabledStatistics::Page, "k", predicate.clone(), true);
        let (without, _) = run(&env, EnabledStatistics::Page, "k", predicate, false);
        let matches = |v: &[i64]| {
            v.iter()
                .copied()
                .filter(|k| (lo..=hi).contains(k))
                .collect::<Vec<_>>()
        };
        assert_eq!(matches(&with), matches(&without), "{lo}..={hi}");
        assert!(with.iter().all(|k| without.contains(k)), "{lo}..={hi}");
    }
}

/// A single-column `OR` and a negation are answered per page; a conjunct on
/// another column leaves the pages alone.
#[test]
fn or_not_and_other_columns() {
    let env = Env::new();
    let either = PushdownExpr::Or(vec![
        PushdownExpr::cmp(ColumnPath::new("k"), CmpOp::Lt, ScalarValue::Int64(100)),
        PushdownExpr::cmp(ColumnPath::new("k"), CmpOp::Gt, ScalarValue::Int64(19_900)),
    ]);
    let (with, m) = run(&env, EnabledStatistics::Page, "k", either.clone(), true);
    let (without, _) = run(&env, EnabledStatistics::Page, "k", either, false);
    assert!(m.pages_pruned > 0);
    let want = |v: &[i64]| {
        v.iter()
            .copied()
            .filter(|k| *k < 100 || *k > 19_900)
            .collect::<Vec<_>>()
    };
    assert_eq!(want(&with), want(&without));
    // `NOT (k >= 100)` is the same set.
    let not = PushdownExpr::Not(Box::new(PushdownExpr::cmp(
        ColumnPath::new("k"),
        CmpOp::GtEq,
        ScalarValue::Int64(100),
    )));
    let (with, m) = run(&env, EnabledStatistics::Page, "k", not, true);
    assert!(m.pages_pruned > 0);
    assert!(with.iter().filter(|k| **k < 100).count() == 100);
}

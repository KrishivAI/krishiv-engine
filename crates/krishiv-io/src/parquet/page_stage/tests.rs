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
        // The dictionary page of a column is always fetched whole: keep it
        // out of the byte comparison.
        .set_dictionary_enabled(false)
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
    predicate: PushdownExpr,
    on: bool,
) -> (Vec<i64>, ScanUnitMetrics) {
    let (f, _) = file(env, stats);
    let o = env.options(&[("pageIndex", if on { "true" } else { "false" })]);
    let req = request(env, &f, &o, predicate);
    let (out, m) = env.scan_metered(&f, &req, &o).unwrap();
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
    let (with, m_on) = run(&env, EnabledStatistics::Page, k_between(9_000, 9_700), true);
    let (without, m_off) = run(
        &env,
        EnabledStatistics::Page,
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
    // The offset indexes are attached: only the selected pages are fetched.
    assert!(
        m_on.bytes_pushed < m_off.bytes_pushed / 2,
        "{} vs {}",
        m_on.bytes_pushed,
        m_off.bytes_pushed
    );
    assert!(m_on.index_bytes > 0);
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
    let (with, m_on) = run(&env, EnabledStatistics::Page, predicate.clone(), true);
    let (without, _) = run(&env, EnabledStatistics::Page, predicate, false);
    assert_eq!(with, without);
    assert_eq!(m_on.pages_pruned, 0, "{m_on:?}");
}

#[test]
fn a_file_without_a_column_index_is_read_whole() {
    let env = Env::new();
    let (with, m) = run(
        &env,
        EnabledStatistics::Chunk,
        k_between(9_000, 9_700),
        true,
    );
    let (without, _) = run(
        &env,
        EnabledStatistics::Chunk,
        k_between(9_000, 9_700),
        false,
    );
    assert_eq!(with, without);
    assert_eq!(m.pages_pruned, 0);
}

#[test]
fn a_predicate_the_statistics_decided_reads_no_index() {
    let env = Env::new();
    // Every row matches: statistics say `True`, the unit is not undecided.
    let (_, m) = run(&env, EnabledStatistics::Page, k_between(-1, ROWS + 1), true);
    assert_eq!(m.pages_pruned, 0);
    // No row matches: the unit is gone before the stage.
    let (out, _) = run(
        &env,
        EnabledStatistics::Page,
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
        let (with, _) = run(&env, EnabledStatistics::Page, predicate.clone(), true);
        let (without, _) = run(&env, EnabledStatistics::Page, predicate, false);
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
    let (with, m) = run(&env, EnabledStatistics::Page, either.clone(), true);
    let (without, _) = run(&env, EnabledStatistics::Page, either, false);
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
    let (with, m) = run(&env, EnabledStatistics::Page, not, true);
    assert!(m.pages_pruned > 0);
    assert!(with.iter().filter(|k| **k < 100).count() == 100);
}

/// A predicate on two indexed columns: each column's pages are checked
/// against its own chunk, and both prune.
#[test]
fn two_indexed_columns_both_prune() {
    let env = Env::new();
    let predicate = PushdownExpr::And(vec![
        k_between(9_000, 9_700),
        PushdownExpr::cmp(ColumnPath::new("u"), CmpOp::GtEq, ScalarValue::Int64(0)),
        PushdownExpr::cmp(
            ColumnPath::new("u"),
            CmpOp::Lt,
            ScalarValue::Int64(ROWS - 10_000),
        ),
    ]);
    let (with, m) = run(&env, EnabledStatistics::Page, predicate.clone(), true);
    let (without, _) = run(&env, EnabledStatistics::Page, predicate, false);
    assert_eq!(m.page_index_rejected, 0, "{m:?}");
    assert!(m.pages_pruned > 0);
    let wanted: Vec<i64> = (9_000..=9_700).collect();
    assert!(
        wanted
            .iter()
            .filter(|k| (0..ROWS - 10_000).contains(&((*k * 7919) % ROWS)))
            .all(|k| with.contains(k))
    );
    assert!(with.iter().all(|k| without.contains(k)));
}

/// A row group whose every page is ruled out is dropped from the scan: no
/// row, no chunk fetched for it, and it counts as a pruned unit.
#[test]
fn a_unit_with_every_page_pruned_is_dropped() {
    let env = Env::new();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    // 0..10000 then 20000..30000: a hole statistics cannot see.
    let k =
        Int64Array::from_iter_values((0..20_000).map(|i| if i < 10_000 { i } else { i + 10_000 }));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k) as ArrayRef]).unwrap();
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .build();
    let f = env.put("hole.parquet", write(&[batch], props));
    let o = env.options(&[]);
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &f, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    let mut req = ScanRequest::new(schema, vec![ColumnPath::new("k")]);
    req.predicate = Some(k_between(12_000, 13_000));
    let (out, m) = env.scan_metered(&f, &req, &o).unwrap();
    assert_eq!(rows(&out), 0);
    assert_eq!(m.units_pruned, 1, "{m:?}");
    assert!(m.pages_pruned >= 40, "{m:?}");
    // The data chunk was never fetched: only the index was read.
    assert_eq!(m.bytes_pushed, 0, "{m:?}");
}

/// `krishiv.io.parquet.pageIndex.enabled` is the session's default for the
/// `pageIndex` read option, and the option beats it.
#[test]
fn the_session_setting_switches_the_stage() {
    let env = Env::new();
    let key = "krishiv.io.parquet.pageIndex.enabled";
    let pruned = |session: &[(&str, &str)], user: &[(&str, &str)]| {
        let (f, _) = file(&env, EnabledStatistics::Page);
        let o = env.options_over(session, user);
        let req = request(&env, &f, &o, k_between(9_000, 9_700));
        env.scan_metered(&f, &req, &o).unwrap().1.pages_pruned
    };
    assert!(pruned(&[], &[]) > 0);
    assert!(pruned(&[(key, "true")], &[]) > 0);
    assert_eq!(pruned(&[(key, "false")], &[]), 0);
    assert!(pruned(&[(key, "false")], &[("pageIndex", "true")]) > 0);
    assert_eq!(pruned(&[], &[("pageIndex", "false")]), 0);
}

// ---- stored Bloom filters (stage 2a) ----

/// The string key of the number `x`: 20 zero-padded digits.
fn key(x: i64) -> String {
    format!("{x:020}")
}

/// Two row groups of one string column: the keys of the even numbers below
/// 20,000, then of the odd ones. Their min/max overlap, so only the Bloom
/// filter can tell a row group does not hold a key. (Strings: a filter
/// costs about a byte per value, which must be small next to the column
/// chunk, a rule that 8-byte integers rarely meet.)
fn parity_file(env: &Env, bloom: bool) -> FileRef {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, false)]));
    let evens = (0..10_000_i64).map(|i| key(2 * i));
    let odds = (0..10_000_i64).map(|i| key(2 * i + 1));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::StringArray::from_iter_values(
            evens.chain(odds),
        )) as ArrayRef],
    )
    .unwrap();
    let mut props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(10_000))
        .set_dictionary_enabled(false)
        // Uncompressed: the filter is judged against the chunk's size.
        .set_compression(parquet::basic::Compression::UNCOMPRESSED);
    if bloom {
        props = props
            .set_bloom_filter_enabled(true)
            .set_bloom_filter_max_ndv(10_000);
    }
    env.put("parity.parquet", write(&[batch], props.build()))
}

fn parity_scan(
    env: &Env,
    f: &FileRef,
    o: &FormatOptions,
    predicate: PushdownExpr,
) -> (Vec<String>, ScanUnitMetrics) {
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, f, MetadataNeeds::default(), o),
        )
        .unwrap()
        .schema();
    let mut req = ScanRequest::new(schema, vec![ColumnPath::new("v")]);
    req.predicate = Some(predicate);
    let (out, m) = env.scan_metered(f, &req, o).unwrap();
    let values = out
        .iter()
        .flat_map(|b| {
            let a = b.column(0).as_string_view();
            (0..b.num_rows())
                .map(|i| a.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    (values, m)
}

fn v_eq(x: i64) -> PushdownExpr {
    PushdownExpr::cmp(ColumnPath::new("v"), CmpOp::Eq, ScalarValue::Utf8(key(x)))
}

fn parity(values: &[String]) -> Vec<i64> {
    values
        .iter()
        .filter_map(|v| v.parse::<i64>().ok())
        .collect()
}

#[test]
fn a_point_lookup_skips_the_row_group_whose_bloom_filter_lacks_it() {
    let env = Env::new();
    let f = parity_file(&env, true);
    let o = env.options(&[]);
    let (odd, m) = parity_scan(&env, &f, &o, v_eq(4_001));
    // The even row group is ruled out by its filter; the odd one is read
    // whole (no row filter applies in this request) and holds the key.
    assert_eq!(m.units_pruned_bloom, 1, "{m:?}");
    assert_eq!(m.bloom_rejected, 0);
    let odd = parity(&odd);
    assert!(odd.contains(&4_001));
    assert!(
        odd.iter().all(|v| v % 2 == 1),
        "the even row group was not read"
    );
    // An even key prunes the other row group.
    let (even, m) = parity_scan(&env, &f, &o, v_eq(4_000));
    assert_eq!(m.units_pruned_bloom, 1);
    let even = parity(&even);
    assert!(even.contains(&4_000) && even.iter().all(|v| v % 2 == 0));
    // IN: a row group is read if it may hold any literal.
    let both = PushdownExpr::In {
        column: ColumnPath::new("v"),
        values: vec![ScalarValue::Utf8(key(4_000)), ScalarValue::Utf8(key(4_001))],
        collation: crate::request::Collation::Utf8Binary,
    };
    let (_, m) = parity_scan(&env, &f, &o, both);
    assert_eq!(m.units_pruned_bloom, 0);
}

#[test]
fn bloom_pruning_changes_no_matching_row() {
    let env = Env::new();
    let f = parity_file(&env, true);
    let on = env.options(&[]);
    let off = env.options(&[("bloomFilter", "false")]);
    for x in [0, 1, 4_000, 4_001, 19_998, 19_999, 7_777] {
        let (with, _) = parity_scan(&env, &f, &on, v_eq(x));
        let (without, m) = parity_scan(&env, &f, &off, v_eq(x));
        assert_eq!(m.units_pruned_bloom, 0);
        let hits = |v: &[String]| v.iter().filter(|k| **k == key(x)).count();
        assert_eq!(hits(&with), 1, "{x}");
        assert_eq!(hits(&without), 1, "{x}");
    }
}

#[test]
fn a_file_without_a_bloom_filter_reads_nothing_extra() {
    let env = Env::new();
    let f = parity_file(&env, false);
    let o = env.options(&[]);
    let (rows_read, m) = parity_scan(&env, &f, &o, v_eq(4_001));
    assert_eq!((m.units_pruned_bloom, m.bloom_rejected), (0, 0), "{m:?}");
    assert_eq!(rows_read.len(), 20_000);
}

/// `krishiv.io.parquet.bloomFilter.enabled` is the session's default for
/// the `bloomFilter` read option, and the option beats it.
#[test]
fn the_session_setting_switches_the_bloom_stage() {
    let env = Env::new();
    let key = "krishiv.io.parquet.bloomFilter.enabled";
    let f = parity_file(&env, true);
    let pruned = |session: &[(&str, &str)], user: &[(&str, &str)]| {
        let o = env.options_over(session, user);
        parity_scan(&env, &f, &o, v_eq(4_001)).1.units_pruned_bloom
    };
    assert_eq!(pruned(&[], &[]), 1);
    assert_eq!(pruned(&[(key, "false")], &[]), 0);
    assert_eq!(pruned(&[(key, "false")], &[("bloomFilter", "true")]), 1);
    assert_eq!(pruned(&[], &[("bloomFilter", "false")]), 0);
}

// ---- review of #450 ----

/// The Bloom stage does not depend on the page-index switch.
#[test]
fn bloom_works_with_the_page_index_off() {
    let env = Env::new();
    let f = parity_file(&env, true);
    let o = env.options(&[("pageIndex", "false")]);
    let (_, m) = parity_scan(&env, &f, &o, v_eq(4_001));
    assert_eq!(m.units_pruned_bloom, 1, "{m:?}");
    assert_eq!(m.pages_pruned, 0);
    // And the page stage works with Bloom off.
    let (f2, _) = file(&env, EnabledStatistics::Page);
    let o = env.options(&[("bloomFilter", "false")]);
    let req = request(&env, &f2, &o, k_between(9_000, 9_700));
    let (_, m) = env.scan_metered(&f2, &req, &o).unwrap();
    assert!(m.pages_pruned > 0 && m.units_pruned_bloom == 0, "{m:?}");
}

/// A predicate column outside the output projection is part of the
/// attach coverage: the row filter fetches it under a selection too, so a
/// column without an offset index there means nothing is attached and the
/// scan is still correct.
#[test]
fn a_predicate_only_column_does_not_break_the_sparse_read() {
    let env = Env::new();
    let (f, _) = file(&env, EnabledStatistics::Page);
    let o = env.options(&[]);
    let schema = env
        .rt
        .block_on(
            env.io
                .read_metadata(&ParquetFormat, &f, MetadataNeeds::default(), &o),
        )
        .unwrap()
        .schema();
    // Project `k` only, filter on `k` and the unprojected `u`.
    let mut req = ScanRequest::new(schema, vec![ColumnPath::new("k")]);
    req.predicate = Some(PushdownExpr::And(vec![
        k_between(9_000, 9_700),
        PushdownExpr::cmp(ColumnPath::new("u"), CmpOp::GtEq, ScalarValue::Int64(0)),
    ]));
    for policy in [
        crate::request::RowFilterPolicy::On,
        crate::request::RowFilterPolicy::Off,
    ] {
        req.row_filter = policy;
        let (out, m) = env.scan_metered(&f, &req, &o).unwrap();
        let got = ks(&out);
        assert!(
            (9_000..=9_700).all(|k| got.contains(&k)),
            "{policy:?} {m:?}"
        );
        assert!(got.len() < 20_000);
    }
}

/// A runtime filter on an unprojected column, still pending when the scan
/// opens and published while it reads: the row filter then reads its
/// column in the next row group, and the scan stays correct. The filter's
/// column counts toward the offset-index coverage from the start (Codex on
/// #450); this file did not fail without that, so the test guards the
/// result, not the bookkeeping failure.
#[test]
#[cfg(not(loom))] // Runtime filters use loom's primitives under `--cfg loom`.
fn a_runtime_filter_published_after_open_does_not_break_the_sparse_read() {
    use crate::ByteRange;
    use crate::{
        FilterId, FilterSnapshot, KeyFilterBuilder, KeyFilterOptions, RuntimeFilter, ScanFilter,
        ScanOptions, ScanStream,
    };

    let env = Env::new();
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
        .set_max_row_group_row_count(Some(10_000))
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();
    let f = env.put("two_groups.parquet", write(&[batch], props));
    let o = env.options(&[]);
    let mut req = ScanRequest::new(Arc::clone(&schema), vec![ColumnPath::new("k")]);
    req.predicate = Some(k_between(9_000, 11_000));
    req.row_filter = crate::request::RowFilterPolicy::On;
    let filter = Arc::new(RuntimeFilter::new(
        FilterId(1),
        ColumnPath::new("u"),
        DataType::Int64,
    ));
    req.dynamic
        .push(ScanFilter::new(Arc::clone(&filter), &schema).unwrap());
    let mut b = KeyFilterBuilder::new(DataType::Int64, KeyFilterOptions::default()).unwrap();
    let keep: Vec<i64> = (0..ROWS).step_by(3).collect();
    b.add(&Int64Array::from(keep), None).unwrap();
    let keys = Arc::new(b.finish());
    let got = env.rt.block_on(async {
        let pool: Arc<dyn krishiv_memory::MemoryPool> =
            krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
        let id = pool.register("scan", None);
        let mut s = ScanStream::open(
            &env.io,
            &ParquetFormat,
            f.clone(),
            ByteRange::whole(f.size),
            &req,
            &o,
            krishiv_memory::Reservation::new(pool, id),
            ScanOptions::default(),
        )
        .await
        .unwrap();
        let mut out = vec![s.next_batch().await.unwrap().unwrap()];
        filter.publish(FilterSnapshot::Final(Arc::clone(&keys)));
        while let Some(b) = s.next_batch().await.unwrap() {
            out.push(b);
        }
        ks(&out)
    });
    // Every row in range whose `u` passes survives; nothing outside the
    // pages the predicate kept is read.
    let passes = |k: i64| ((k * 7919) % ROWS) % 3 == 0;
    assert!(
        (9_000..=11_000)
            .filter(|&k| k < 10_000 || passes(k))
            .all(|k| got.contains(&k))
    );
    assert!(got.len() < 20_000);
}

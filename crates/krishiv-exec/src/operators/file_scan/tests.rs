// SPDX-License-Identifier: Apache-2.0
//! Unit tests of `FileScan`: metrics, metadata columns and Spark file paths.

use super::*;

fn value(extra: &crate::metrics::ExtraMetrics, name: &str) -> Option<u64> {
    extra
        .iter()
        .find(|(n, _)| n == name)
        .and_then(|(_, v)| v.value())
}

/// The decode over-commit shows in `EXPLAIN ANALYZE` only when there was
/// any, summed over the scan's splits.
#[test]
fn decode_over_commit_is_reported_when_there_was_any() {
    let mut m = FileScanMetrics::default();
    assert!(m.extra().is_empty());
    m.add(&ScanUnitMetrics::default(), 0, 0);
    assert_eq!(value(&m.extra(), "decode_overcommit_bytes"), None);
    m.add(&ScanUnitMetrics::default(), 3, 0);
    m.add(&ScanUnitMetrics::default(), 4, 0);
    assert_eq!(value(&m.extra(), "decode_overcommit_bytes"), Some(7));
}

/// The time streams waited for decode shares shows only when there was
/// a wait, summed over the scan's splits.
#[test]
fn decode_gate_waits_are_reported_when_there_were_any() {
    let mut m = FileScanMetrics::default();
    m.add(&ScanUnitMetrics::default(), 0, 0);
    assert_eq!(value(&m.extra(), "decode_gate_wait_ns"), None);
    m.add(&ScanUnitMetrics::default(), 0, 5);
    m.add(&ScanUnitMetrics::default(), 0, 6);
    assert_eq!(value(&m.extra(), "decode_gate_wait_ns"), Some(11));
}

/// The scan counters are summed over splits; the row filter's appear
/// only where it ran.
#[test]
fn read_costs_are_reported_once_a_read_was_made_and_sum_over_splits() {
    let mut m = FileScanMetrics::default();
    assert_eq!(value(&m.extra(), "scan_reads"), None);
    let mut a = krishiv_io::ReadStats::default();
    a.reads = 3;
    a.bytes = 3_000;
    a.wait_ns = 500;
    a.inflight_ns = 1_500;
    m.add_reads(&a);
    m.add_reads(&a);
    // Reads without a decoded split still show, so the I/O is never hidden.
    let extra = m.extra();
    for (name, want) in [
        ("scan_reads", 6),
        ("scan_read_bytes", 6_000),
        ("scan_read_wait_ns", 1_000),
        ("scan_read_inflight_ns", 3_000),
    ] {
        assert_eq!(value(&extra, name), Some(want), "{name}");
    }
}

#[test]
fn scan_counters_sum_over_splits() {
    let mut m = FileScanMetrics::default();
    let mut unit = ScanUnitMetrics::default();
    unit.units = 4;
    unit.units_pruned = 1;
    unit.rows = 100;
    unit.bytes_pushed = 1000;
    m.add(&unit, 0, 0);
    let extra = m.extra();
    assert_eq!(value(&extra, "scan_splits"), Some(1));
    assert_eq!(value(&extra, "row_filter_rows"), None);
    assert_eq!(value(&extra, "scan_pages_pruned"), None);
    assert_eq!(value(&extra, "scan_units_pruned_bloom"), None);
    let mut filtered = unit;
    filtered.units_filtered = 2;
    filtered.units_unfiltered = 1;
    filtered.row_filter_rows = 50;
    filtered.row_filter_kept = 5;
    filtered.pages_pruned = 3;
    filtered.page_index_rejected = 1;
    filtered.units_pruned_bloom = 1;
    filtered.bloom_rejected = 2;
    m.add(&filtered, 0, 0);
    let extra = m.extra();
    for (name, want) in [
        ("scan_splits", 2),
        ("scan_units", 8),
        ("scan_units_pruned", 2),
        ("scan_rows_decoded", 200),
        ("scan_bytes", 2000),
        ("row_filter_units", 2),
        ("row_filter_units_skipped", 1),
        ("row_filter_rows", 50),
        ("row_filter_kept", 5),
        ("scan_pages_pruned", 3),
        ("page_index_rejected", 1),
        ("scan_units_pruned_bloom", 1),
        ("bloom_rejected", 2),
    ] {
        assert_eq!(value(&extra, name), Some(want), "{name}");
    }
}

#[test]
fn spark_paths_and_constants() {
    let file = FileRef {
        store: krishiv_io::StoreId::new("file://"),
        path: object_store::path::Path::from("tmp/pq/a b.parquet"),
        size: 10,
        version: krishiv_io::VersionToken::Immutable,
        modified_micros: Some(5),
    };
    let none = PathRoots::default();
    assert_eq!(spark_file_path(&file, &none), "file:/tmp/pq/a%20b.parquet");
    assert_eq!(spark_file_name(&file), "a%20b.parquet");
    // Spark 4.2 keeps non-ASCII text and quotes spaces and `%`.
    let odd = FileRef {
        path: object_store::path::Path::from("tmp/données x/é%41/f.parquet"),
        ..file.clone()
    };
    assert_eq!(
        spark_file_path(&odd, &none),
        "file:/tmp/données%20x/é%2541/f.parquet"
    );
    let s3 = FileRef {
        store: krishiv_io::StoreId::new("s3://bucket"),
        ..file
    };
    assert_eq!(
        spark_file_path(&s3, &none),
        "s3://bucket/tmp/pq/a%20b.parquet"
    );
    // Under the root of the read path nearest above the file (Spark 4.2.0 run: an `s3a://`
    // read prints `s3a://`); of two paths with one key the first; no root keeps the store id.
    let bucket = krishiv_io::StoreId::new("s3://bucket");
    let key = |k: &str| object_store::path::Path::from(k);
    let roots = PathRoots::new([
        (
            krishiv_io::StoreId::new("s3://other"),
            key(""),
            "s3a://other".to_owned(),
        ),
        (bucket.clone(), key("tmp"), "s3a://bucket".to_owned()),
        (bucket.clone(), key("tmp"), "s3://bucket".to_owned()),
        (bucket.clone(), key("dir"), "s3://bucket".to_owned()),
        (bucket, key(""), "s3n://bucket".to_owned()),
    ]);
    assert_eq!(
        spark_file_path(&s3, &roots),
        "s3a://bucket/tmp/pq/a%20b.parquet"
    );
    assert_eq!(
        spark_file_path(&odd, &roots),
        "file:/tmp/données%20x/é%2541/f.parquet"
    );
    let elsewhere = FileRef {
        path: key("tmpx/f.parquet"),
        ..s3.clone()
    };
    assert_eq!(
        spark_file_path(&elsewhere, &roots),
        "s3n://bucket/tmpx/f.parquet"
    );
    // The name is the last segment of the quoted path (Spark 4.2.0 run:
    // `with%20space%25.parquet`, `we+ird%23.parquet`).
    let named = |name: &str| FileRef {
        path: object_store::path::Path::from_iter(["dir", name]),
        ..s3.clone()
    };
    assert_eq!(
        spark_file_name(&named("with space%.parquet")),
        "with%20space%25.parquet"
    );
    assert_eq!(
        spark_file_name(&named("we+ird#.parquet")),
        "we+ird%23.parquet"
    );
    assert_eq!(
        spark_file_path(&named("we+ird#.parquet"), &roots),
        "s3://bucket/dir/we+ird%23.parquet"
    );
    for s in ["short", "a value longer than twelve bytes"] {
        let a = constant_string(s, 3).unwrap_or_else(|e| unreachable!("{e}"));
        let a = a
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap_or_else(|| unreachable!());
        assert_eq!(a.iter().collect::<Vec<_>>(), [Some(s); 3]);
    }
}

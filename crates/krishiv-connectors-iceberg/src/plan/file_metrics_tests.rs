// SPDX-License-Identifier: Apache-2.0
//! Tests of a written file's Iceberg metrics (`file_metrics`, PR-36a): the
//! modes, truncation as Java's `UnicodeUtil`/`BinaryUtil`, NaN, and columns
//! without statistics.

use krishiv_io::{ColumnPath, ColumnStatistics, FileStatistics, Precision, ScalarValue as S};

use super::file_metrics::{
    file_metrics, truncate_binary_max, truncate_string, truncate_string_max,
};
use super::metrics::MetricsMode;
use super::types::{IcebergType, SchemaField};

fn field(id: i32, name: &str, ty: IcebergType) -> SchemaField {
    SchemaField {
        id,
        path: ColumnPath::new(name),
        ty,
    }
}

fn col(name: &str, nulls: u64, min: S, max: S) -> ColumnStatistics {
    let mut c = ColumnStatistics::new(ColumnPath::new(name));
    c.null_count = Precision::Exact(nulls);
    c.min = Precision::Exact(min);
    c.max = Precision::Exact(max);
    c
}

fn stats(rows: u64, columns: Vec<ColumnStatistics>) -> FileStatistics {
    let mut s = FileStatistics::with_rows(Precision::Exact(rows));
    s.columns = columns;
    s
}

#[test]
fn strings_truncate_as_java() {
    assert_eq!(truncate_string("abcdef", 3), "abc");
    assert_eq!(truncate_string("ab", 3), "ab");
    // Code points, not bytes or UTF-16 units.
    assert_eq!(truncate_string("😀😁😂", 2), "😀😁");
    assert_eq!(truncate_string_max("abcdef", 3).as_deref(), Some("abd"));
    assert_eq!(truncate_string_max("ab", 3).as_deref(), Some("ab"));
    assert_eq!(truncate_string_max("a\u{10FFFF}z", 2).as_deref(), Some("b"));
    assert_eq!(truncate_string_max("\u{10FFFF}\u{10FFFF}z", 2), None);
    // U+D7FF skips the surrogates.
    assert_eq!(
        truncate_string_max("\u{D7FF}z", 1).as_deref(),
        Some("\u{E000}")
    );
    assert_eq!(truncate_binary_max(&[1, 2, 3], 2), Some(vec![1, 3]));
    assert_eq!(truncate_binary_max(&[1, 0xFF, 3], 2), Some(vec![2]));
    assert_eq!(truncate_binary_max(&[0xFF, 0xFF, 3], 2), None);
    assert_eq!(truncate_binary_max(&[1, 2], 2), Some(vec![1, 2]));
}

#[test]
fn modes_choose_what_is_recorded() {
    let fields = [
        field(1, "id", IcebergType::Long),
        field(2, "s", IcebergType::String),
        field(3, "f", IcebergType::Double),
        field(4, "b", IcebergType::Binary),
    ];
    let mut f = col("f", 0, S::Float64(-1.5), S::Float64(2.0));
    f.nan_count = Precision::Exact(2);
    let st = stats(
        10,
        vec![
            col("id", 1, S::Int64(-5), S::Int64(7)),
            col("s", 0, S::Utf8("apple".into()), S::Utf8("banana".into())),
            f,
            col(
                "b",
                0,
                S::Binary(vec![1, 2, 3]),
                S::Binary(vec![9, 0xFF, 1]),
            ),
        ],
    );
    let m = file_metrics(&st, &fields, |id| match id {
        1 => MetricsMode::Counts,
        2 => MetricsMode::Truncate(3),
        4 => MetricsMode::Truncate(1),
        _ => MetricsMode::Full,
    });
    assert_eq!(m.value_counts[&1], 10);
    assert_eq!(m.null_counts[&1], 1);
    assert!(!m.lower_bounds.contains_key(&1));
    assert_eq!(m.lower_bounds[&2], b"app");
    assert_eq!(m.upper_bounds[&2], b"bao");
    assert_eq!(m.nan_counts[&3], 2);
    assert_eq!(m.lower_bounds[&3], (-1.5f64).to_le_bytes());
    assert_eq!(m.lower_bounds[&4], [1]);
    assert_eq!(m.upper_bounds[&4], [10]);
    assert!(!m.nan_counts.contains_key(&1));

    let none = file_metrics(&st, &fields, |_| MetricsMode::None);
    assert!(none.value_counts.is_empty());
}

#[test]
fn nan_unknown_and_missing_statistics() {
    let fields = [
        field(1, "f", IcebergType::Float),
        field(2, "g", IcebergType::Long),
        field(3, "u", IcebergType::Uuid),
        field(4, "x", IcebergType::Long),
    ];
    let mut g = col("g", 0, S::Int64(1), S::Int64(2));
    g.null_count = Precision::Absent;
    g.max = Precision::Inexact(S::Int64(2));
    let st = stats(
        4,
        vec![
            col("f", 0, S::Float32(f32::NAN), S::Float32(f32::NAN)),
            g,
            col(
                "u",
                0,
                S::Utf8("00000000-0000-0000-0000-000000000000".into()),
                S::Utf8("ffffffff-ffff-ffff-ffff-ffffffffffff".into()),
            ),
        ],
    );
    let m = file_metrics(&st, &fields, |_| MetricsMode::Full);
    // A NaN bound leaves both bounds out; counts stay.
    assert_eq!(m.value_counts[&1], 4);
    assert!(!m.lower_bounds.contains_key(&1) && !m.upper_bounds.contains_key(&1));
    // An unknown null count is left out; an inexact bound leaves both out
    // (Java reads min and max of one footer statistic together).
    assert!(!m.null_counts.contains_key(&2));
    assert!(!m.lower_bounds.contains_key(&2) && !m.upper_bounds.contains_key(&2));
    assert!(!m.lower_bounds.contains_key(&3));
    // A column without statistics has no metrics.
    assert!(!m.value_counts.contains_key(&4));
}

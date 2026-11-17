// SPDX-License-Identifier: Apache-2.0
//! Static partition pruning for the file scan (P3-T025): the pushed filters on partition columns rule a file
//! out by its partition values, so it is never opened.

use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, FieldRef};
use krishiv_io::runtime_filter::ScanFilter;
use krishiv_io::{
    ColumnPath, ColumnStatistics, FileStatistics, Precision, PushdownExpr, ScalarValue, TriState,
    prune,
};

use super::scan::PushedFilter;

/// The one value of a partition column in a file, for the types partitions are made of.
fn partition_scalar(array: &dyn Array) -> Option<ScalarValue> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Date32Type, Int8Type, Int16Type, Int32Type, Int64Type};
    if array.is_empty() || array.is_null(0) {
        return None;
    }
    Some(match array.data_type() {
        DataType::Boolean => ScalarValue::Boolean(array.as_boolean().value(0)),
        DataType::Int8 => ScalarValue::Int8(array.as_primitive::<Int8Type>().value(0)),
        DataType::Int16 => ScalarValue::Int16(array.as_primitive::<Int16Type>().value(0)),
        DataType::Int32 => ScalarValue::Int32(array.as_primitive::<Int32Type>().value(0)),
        DataType::Int64 => ScalarValue::Int64(array.as_primitive::<Int64Type>().value(0)),
        DataType::Date32 => ScalarValue::Date32(array.as_primitive::<Date32Type>().value(0)),
        DataType::Utf8 => ScalarValue::Utf8(array.as_string::<i32>().value(0).to_owned()),
        DataType::Utf8View => ScalarValue::Utf8(array.as_string_view().value(0).to_owned()),
        _ => return None,
    })
}

/// Whether the pushed filters on partition columns (`filters`) cannot match any row of a file with these
/// partition `values` (one per field of `fields`; each is the one value of its column in the file, so `prune`
/// decides exactly; a null or an unhandled type is unknown and keeps the file).
pub(super) fn statically_dropped(
    filters: &[PushdownExpr],
    fields: &[FieldRef],
    values: &[ArrayRef],
) -> bool {
    if filters.is_empty() {
        return false;
    }
    let mut stats = FileStatistics::unknown();
    stats.rows = Precision::Exact(1);
    for (field, array) in fields.iter().zip(values) {
        let Some(value) = partition_scalar(array.as_ref()) else {
            continue;
        };
        let mut column = ColumnStatistics::new(ColumnPath::new(field.name()));
        column.null_count = Precision::Exact(0);
        column.min = Precision::Exact(value.clone());
        column.max = Precision::Exact(value);
        stats.columns.push(column);
    }
    filters.iter().any(|e| prune(e, &stats) == TriState::False)
}

/// The typed trees static partition pruning may drop a file by: of each pushed filter that reads a partition
/// column, its conjuncts that cannot raise ([`PushedFilter::partitions`]). A conjunct that may raise under ANSI
/// never prunes a file, so the filter above evaluates it on the file's rows and raises as Spark's listing does
/// (`(CAST(p AS INT) > 0 AND p LIKE '2%') OR p = 'zzz'` keeps `p = 'bad'`); the others prune in any order, as
/// Spark splits the partition filters off and prunes by them before any other conjunct runs.
pub(super) fn static_partition_filters(
    partition_columns: &[FieldRef],
    filters: &[PushedFilter],
) -> Vec<PushdownExpr> {
    filters
        .iter()
        .filter(|f| reads_partition_column(partition_columns, f))
        .filter_map(|f| f.partitions.clone())
        .collect()
}

/// Whether the typed tree of `filter` reads a partition column of the scan
/// (a filter without one prunes nothing).
pub(super) fn reads_partition_column(
    partition_columns: &[FieldRef],
    filter: &PushedFilter,
) -> bool {
    if partition_columns.is_empty() {
        return false;
    }
    filter.expr.as_ref().is_some_and(|e| {
        e.columns()
            .iter()
            .any(|c| partition_columns.iter().any(|p| p.name() == c.root()))
    })
}

/// The runtime filter that no row of a file with these partition values
/// passes, if any.
pub(super) fn dropping_filter<'a>(
    filters: &'a [(usize, ScanFilter)],
    values: &[ArrayRef],
) -> Option<&'a ScanFilter> {
    filters.iter().find_map(|(k, filter)| {
        let mask = filter.evaluate(values.get(*k)?.as_ref())?;
        (mask.is_valid(0) && !mask.value(0)).then_some(filter)
    })
}

#[cfg(test)]
mod tests {
    use super::super::file_scan::FileScanMetrics;

    /// A scan whose every file was pruned unopened (no split read) still reports its counters.
    #[test]
    fn a_scan_that_pruned_every_file_reports_its_counters() {
        let m = FileScanMetrics {
            units_pruned: 3,
            ..FileScanMetrics::default()
        };
        let extra: Vec<(String, Option<u64>)> = m
            .extra()
            .iter()
            .map(|(n, v)| (n.to_string(), v.value()))
            .collect();
        assert_eq!(
            extra,
            [
                ("scan_splits".to_owned(), Some(0)),
                ("scan_files".to_owned(), Some(0)),
                ("scan_units".to_owned(), Some(0)),
                ("scan_units_pruned".to_owned(), Some(3)),
                ("scan_rows_decoded".to_owned(), Some(0)),
                ("scan_bytes".to_owned(), Some(0)),
            ]
        );
        assert!(FileScanMetrics::default().extra().is_empty());
    }
}

// SPDX-License-Identifier: Apache-2.0
//! Partition advisor core (P3-T025, ADR 0020 layer 1): given what queries filtered a table on and the
//! table's column statistics, rank the columns worth partitioning by. Pure and read-only: it runs off the
//! planning path and never touches a table.

use std::collections::HashMap;

/// A column with more distinct values than this makes too many partition directories.
pub const MAX_PARTITIONS: u64 = 10_000;

/// A column whose distinct count is at least this fraction of the rows is a key, not a partition column.
const KEY_FRACTION: f64 = 0.5;

/// One column of the table: its name and distinct-value count (`None` when unknown, which is never advised).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisedColumn {
    /// The column name.
    pub name: String,
    /// Distinct non-null values.
    pub distinct: Option<u64>,
}

/// The table the advice is for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisedTable {
    /// Row count.
    pub rows: u64,
    /// Its columns.
    pub columns: Vec<AdvisedColumn>,
}

/// One partition-column suggestion.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionAdvice {
    /// The column to partition by.
    pub column: String,
    /// Queries that filtered the table on it by equality or `IN`.
    pub queries: u64,
    /// Its distinct values: the partitions the table would have.
    pub distinct: u64,
    /// The expected fraction of scanned bytes those queries would skip, averaged over all `total` queries.
    pub pruned_fraction: f64,
}

impl PartitionAdvice {
    /// `partition advice: <column> (<n> queries, <d> partitions, ~<p>% of bytes pruned)`, the `EXPLAIN` line.
    #[must_use]
    pub fn explain_line(&self) -> String {
        format!(
            "partition advice: {} ({} queries, {} partitions, ~{:.0}% of bytes pruned)",
            self.column,
            self.queries,
            self.distinct,
            self.pruned_fraction * 100.0
        )
    }
}

/// Ranks the partition-column candidates of `table`. `filters` counts, per column name, the queries that
/// filtered on it by equality; `total` is the number of queries seen on the table. An equality on a column of
/// `d` distinct values reads about `1/d` of the bytes, so a column scores `queries / total * (1 - 1/d)`.
/// Columns with unknown statistics, fewer than 2 or more than [`MAX_PARTITIONS`] values, or that look like a
/// key (distinct count at least half the rows) are not suggested. Ties go to the column with fewer partitions,
/// then to the name, so the order is deterministic.
#[must_use]
pub fn advise(
    table: &AdvisedTable,
    filters: &HashMap<String, u64>,
    total: u64,
) -> Vec<PartitionAdvice> {
    if total == 0 {
        return Vec::new();
    }
    let mut out: Vec<PartitionAdvice> = table
        .columns
        .iter()
        .filter_map(|c| {
            let queries = *filters.get(&c.name)?;
            let distinct = c.distinct.filter(|d| (2..=MAX_PARTITIONS).contains(d))?;
            if queries == 0 || (distinct as f64) >= KEY_FRACTION * table.rows as f64 {
                return None;
            }
            let pruned = (queries.min(total) as f64 / total as f64) * (1.0 - 1.0 / distinct as f64);
            Some(PartitionAdvice {
                column: c.name.clone(),
                queries,
                distinct,
                pruned_fraction: pruned,
            })
        })
        .collect();
    out.sort_by(|a, b| {
        b.pruned_fraction
            .total_cmp(&a.pruned_fraction)
            .then(a.distinct.cmp(&b.distinct))
            .then_with(|| a.column.cmp(&b.column))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(cols: &[(&str, Option<u64>)], rows: u64) -> AdvisedTable {
        AdvisedTable {
            rows,
            columns: cols
                .iter()
                .map(|(n, d)| AdvisedColumn {
                    name: (*n).to_owned(),
                    distinct: *d,
                })
                .collect(),
        }
    }

    fn uses(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(n, c)| ((*n).to_owned(), *c)).collect()
    }

    #[test]
    fn advisor_finds_planted_partition_key() {
        let t = table(&[("d", Some(365)), ("c", Some(3))], 1_000_000);
        let a = advise(&t, &uses(&[("d", 100), ("c", 20)]), 120);
        assert_eq!(a[0].column, "d");
        assert_eq!(a[1].column, "c");
        assert!(a[0].pruned_fraction > 0.8);
    }

    #[test]
    fn advisor_rejects_unique_key() {
        let t = table(&[("id", Some(1_000_000)), ("k", Some(900_000))], 1_000_000);
        assert!(advise(&t, &uses(&[("id", 500), ("k", 500)]), 1000).is_empty());
    }

    #[test]
    fn unknown_constant_or_unfiltered_columns_are_skipped() {
        let t = table(&[("a", None), ("b", Some(1)), ("c", Some(10))], 1000);
        assert!(advise(&t, &uses(&[("a", 5), ("b", 5)]), 10).is_empty());
        assert!(advise(&t, &HashMap::new(), 0).is_empty());
    }

    #[test]
    fn ties_are_ordered_by_name() {
        let t = table(&[("y", Some(10)), ("x", Some(10))], 1000);
        let a = advise(&t, &uses(&[("x", 4), ("y", 4)]), 4);
        assert_eq!(a[0].column, "x");
        assert_eq!(
            a[0].explain_line(),
            "partition advice: x (4 queries, 10 partitions, ~90% of bytes pruned)"
        );
    }
}

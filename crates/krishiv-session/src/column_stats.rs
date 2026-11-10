// SPDX-License-Identifier: Apache-2.0
//! Column statistics of `ANALYZE TABLE .. FOR COLUMNS` (Spark's `AnalyzeColumnCommand` and
//! `CommandUtils.statExprs`): one aggregate over the table, kept in the catalog and printed by
//! `DESCRIBE EXTENDED t col`. A write or a change of the table's partitions drops them, as
//! `CommandUtils.updateTableStats` does.

use std::future::Future;
use std::pin::Pin;

use arrow_array::{Array, ArrayRef, RecordBatch, cast::AsArray};
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_types::DataType;

use crate::Session;
use crate::store::{CatalogStore as _, CatalogTable, ColumnStats, StoreColumn};
use crate::table_commands::TableName;

/// `spark.sql.statistics.ndv.maxError`'s default.
const DEFAULT_NDV_MAX_ERROR: &str = "0.05";

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

fn quoted(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Spark's `DataType.defaultSize`, for the types `FOR COLUMNS` accepts.
fn default_size(t: &DataType) -> Result<u32, KrishivError> {
    Ok(match t {
        DataType::Boolean | DataType::Byte => 1,
        DataType::Short => 2,
        DataType::Integer | DataType::Float | DataType::Date => 4,
        DataType::Long | DataType::Double | DataType::Timestamp | DataType::TimestampNtz => 8,
        DataType::Decimal(d) => {
            if d.precision() <= 18 {
                8
            } else {
                16
            }
        }
        DataType::String => 20,
        DataType::Binary => 100,
        _ => {
            return Err(unsupported(
                "ANALYZE TABLE ... FOR COLUMNS of this column type",
            ));
        }
    })
}

/// The select items for one column, in [`ColumnStats`] order, each a string.
fn stat_items(c: &StoreColumn, max_error: &str) -> Result<[String; 6], KrishivError> {
    let q = quoted(&c.name);
    let size = default_size(&c.data_type)?;
    let text = |e: String| format!("CAST({e} AS STRING)");
    let varying = matches!(c.data_type, DataType::String | DataType::Binary);
    let bound = |f: &str| match c.data_type {
        DataType::String | DataType::Binary => "CAST(NULL AS STRING)".to_owned(),
        DataType::Timestamp => text(format!("unix_micros({f}({q}))")),
        DataType::TimestampNtz => {
            format!("date_format({f}({q}), 'yyyy-MM-dd HH:mm:ss.SSSSSS')")
        }
        _ => text(format!("{f}({q})")),
    };
    let (avg, max) = if varying {
        (
            text(format!("coalesce(ceil(avg(length({q}))), {size})")),
            text(format!("coalesce(max(length({q})), {size})")),
        )
    } else {
        (format!("'{size}'"), format!("'{size}'"))
    };
    Ok([
        bound("min"),
        bound("max"),
        text(format!("count(1) - count({q})")),
        text(format!(
            "least(approx_count_distinct({q}, {max_error}), count({q}))"
        )),
        avg,
        max,
    ])
}

fn cell(a: &ArrayRef) -> Option<String> {
    if a.is_null(0) {
        return None;
    }
    a.as_string_opt::<i32>().map_or_else(
        || a.as_string_view_opt().map(|s| s.value(0).to_owned()),
        |s| Some(s.value(0).to_owned()),
    )
}

/// The first row of `batches`, every column a string (`None` for a NULL).
pub(crate) fn first_row_text(batches: &[RecordBatch]) -> Result<Vec<Option<String>>, KrishivError> {
    let batch = batches
        .iter()
        .find(|b| b.num_rows() > 0)
        .ok_or_else(|| unsupported("a statistics query with no row"))?;
    Ok(batch.columns().iter().map(cell).collect())
}

impl Session {
    /// The rows of `query`, run from inside a statement `sql` is already running. The future is boxed as `dyn`
    /// because `sql` awaits statements that await this.
    pub(crate) fn nested_query<'a>(
        &'a self,
        query: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RecordBatch>, KrishivError>> + Send + 'a>> {
        Box::pin(async move { self.sql(query).await?.collect().await })
    }

    /// Computes the statistics of `columns` and merges them into the table's: newly computed ones replace old
    /// ones, the others stay (`AnalyzeColumnCommand`).
    pub(crate) async fn store_column_stats(
        &self,
        name: &TableName,
        table: &CatalogTable,
        columns: &[&StoreColumn],
    ) -> Result<(), KrishivError> {
        let conf = self.conf();
        let histogram = conf
            .get("spark.sql.statistics.histogram.enabled")
            .is_some_and(|v| v.eq_ignore_ascii_case("true"));
        if histogram
            && columns
                .iter()
                .any(|c| !matches!(c.data_type, DataType::String | DataType::Binary))
        {
            return Err(unsupported(
                "ANALYZE TABLE ... FOR COLUMNS with spark.sql.statistics.histogram.enabled",
            ));
        }
        let max_error = conf
            .get("spark.sql.statistics.ndv.maxError")
            .unwrap_or_else(|| DEFAULT_NDV_MAX_ERROR.to_owned());
        max_error
            .parse::<f64>()
            .map_err(|_| unsupported("this spark.sql.statistics.ndv.maxError"))?;
        let mut items = Vec::with_capacity(columns.len() * 6);
        for c in columns {
            items.extend(stat_items(c, &max_error)?);
        }
        let query = format!(
            "SELECT {} FROM {}.{}",
            items.join(", "),
            quoted(&table.namespace),
            quoted(&table.name)
        );
        let batches = self.nested_query(&query).await?;
        let row = first_row_text(&batches)?;
        let required = |v: &Option<String>| {
            v.clone()
                .ok_or_else(|| unsupported("a NULL column statistic"))
        };
        let mut computed = Vec::with_capacity(columns.len());
        for (c, [min, max, nulls, distinct, avg, max_len]) in
            columns.iter().zip(row.as_chunks::<6>().0)
        {
            computed.push((
                c.name.clone(),
                ColumnStats {
                    min: min.clone(),
                    max: max.clone(),
                    num_nulls: required(nulls)?,
                    distinct_count: required(distinct)?,
                    avg_col_len: required(avg)?,
                    max_col_len: required(max_len)?,
                },
            ));
        }
        let store = &self.state.catalog.store;
        loop {
            let Some(current) = store.table(&name.namespace, &name.name) else {
                return Ok(());
            };
            let mut changed = (*current).clone();
            changed.column_stats.extend(computed.iter().cloned());
            if store.replace_table(&current, changed) {
                return Ok(());
            }
        }
    }

    /// Drops the table's column statistics: its data changed (`updateTableStats`).
    pub(crate) fn forget_column_stats(&self, name: &TableName) {
        let store = &self.state.catalog.store;
        loop {
            let Some(current) = store.table(&name.namespace, &name.name) else {
                return;
            };
            if current.column_stats.is_empty() {
                return;
            }
            let mut changed = (*current).clone();
            changed.column_stats.clear();
            if store.replace_table(&current, changed) {
                return;
            }
        }
    }
}

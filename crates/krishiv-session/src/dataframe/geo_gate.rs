// SPDX-License-Identifier: Apache-2.0
//! The write gate for spatial columns (P2-T181): a child of `dataframe` so `dataframe.rs` does not
//! grow (`cargo xtask file-size`).

use krishiv_common::error::{
    KrishivError, KrishivUnsupportedFeature, UnsupportedDataTypeForDatasource,
};
use krishiv_types::{DataType, Srid};

/// Parquet holds a top-level geometry or geography as a WKB column with a logical type
/// (`krishiv-io` `parquet/geo.rs`). Any other format, and a spatial value nested in another type,
/// is refused rather than written as the `{srid, wkb}` struct Spark cannot read. A column of
/// `GEOMETRY(ANY)` / `GEOGRAPHY(ANY)` has no CRS to write: refused here, before a dry check passes
/// or an overwrite drops the existing table.
///
/// # Errors
/// `UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE` for an ANY-SRID column in Parquet,
/// `KRISHIV_UNSUPPORTED_FEATURE` for any other spatial column that cannot be stored.
pub(super) fn check(
    output: &[krishiv_plan::BoundColumn],
    format: &str,
) -> Result<(), KrishivError> {
    let parquet = format.eq_ignore_ascii_case("parquet");
    if parquet
        && let Some(c) = output.iter().find(|c| {
            matches!(
                c.data_type,
                DataType::Geometry { srid: Srid::Any } | DataType::Geography { srid: Srid::Any }
            )
        })
    {
        return Err(UnsupportedDataTypeForDatasource {
            format: "Parquet".to_owned(),
            column_name: format!("`{}`", c.name),
            column_type: format!("\"{}\"", c.data_type.sql()),
        }
        .into());
    }
    if output.iter().any(|c| {
        krishiv_types::geo::holds_spatial(&c.data_type)
            && !(parquet
                && matches!(
                    c.data_type,
                    DataType::Geometry { .. } | DataType::Geography { .. }
                ))
    }) {
        return Err(KrishivUnsupportedFeature {
            feature: "writing GEOMETRY or GEOGRAPHY columns to files".to_owned(),
        }
        .into());
    }
    Ok(())
}

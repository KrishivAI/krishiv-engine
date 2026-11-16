// SPDX-License-Identifier: Apache-2.0
//! Delta's write options (P3-T021 PR-37e, design section 6.3) as delta-spark 4.4.1's
//! `DeltaOptions` reads them (`DeltaOptions.scala`; probe of Spark 4.2.0 + delta-spark 4.4.1 in
//! `tests/fixtures/delta-4.4.1/OPTIONS.json`), for `df.write.format("delta").option(..)`,
//! `saveAsTable` and `INSERT .. WITH (..)`:
//!
//! - names are case insensitive; a value delta-spark parses and cannot is
//!   `DELTA_ILLEGAL_OPTION` (`toBoolean` on `mergeSchema`, `overwriteSchema`, `dataChange`,
//!   `optimizeWrite`, `useNullIntolerantEqualityWithDPO`; `txnVersion` a non-negative long;
//!   `partitionOverwriteMode` `STATIC` or `DYNAMIC` on every save mode of a partitioned table,
//!   never read for an unpartitioned one), and one of
//!   `txnAppId`/`txnVersion` without the other is `DELTA_INVALID_IDEMPOTENT_WRITES_OPTIONS`;
//! - an unknown name is accepted and has no effect: delta-spark only logs it
//!   (`DeltaOptions.verifyOptions`), and the probe's `unknown_option` commits;
//! - honoured: `replaceWhere`, `partitionOverwriteMode`, `maxRecordsPerFile`, `userMetadata`
//!   (`commitInfo.userMetadata`), `mergeSchema` (`canMergeSchema`; other columns stay refused),
//!   `overwriteSchema` (refused as a replace on an existing table); accepted with no effect on
//!   the table's rows or log: `optimizeWrite`, `compression`, `parquet.writer.version`, and
//!   `writePartitionColumns` (delta-spark keeps only `maxRecordsPerFile`, `compression`,
//!   `parquet.writer.version` and the output timestamp type of the write options and decides
//!   from the table whether partition columns go into the files, `TransactionalWrite.scala:
//!   477-492`; the probe's `write_partition_columns` commits as a plain append);
//! - refused by name (`KRISHIV_UNSUPPORTED_FEATURE`), where delta-spark changes what it commits
//!   and Krishiv cannot: idempotent writes (`txnAppId` + `txnVersion`, a `txn` action),
//!   `dataChange = false`, `replaceOn`, `replaceUsing`, `targetAlias` and
//!   `useNullIntolerantEqualityWithDPO = true` (the last two from the source only, no probe).
//!
//! [`merge_schema_check`] is `mergeSchema`'s type rule.
//!
//! Options whose names start with `__krishiv` are the session's own and never pass from a user.

use std::collections::BTreeMap;

use krishiv_common::error::{
    DeltaFailedToMergeFields, DeltaIllegalOption, DeltaInvalidIdempotentWritesOptions,
    KrishivError, KrishivUnsupportedFeature,
};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;

/// The prefix of the session's own write options.
const INTERNAL_PREFIX: &str = "__krishiv";

/// The options of a Delta write to a table that is `partitioned` or not, checked as delta-spark
/// checks them: the ones the connector sees (the user's, without the session's own names).
///
/// # Errors
/// `DELTA_ILLEGAL_OPTION`, `DELTA_INVALID_IDEMPOTENT_WRITES_OPTIONS`, and
/// `KRISHIV_UNSUPPORTED_FEATURE` for the options Krishiv does not implement.
pub(crate) fn delta_write_options(
    options: &[(String, String)],
    partitioned: bool,
) -> Result<BTreeMap<String, String>, KrishivError> {
    let get = |name: &str| {
        options
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    };
    // `DeltaWriteOptionsImpl`'s constructor first: `txnVersion`, then the pair.
    let txn_version = get("txnVersion");
    if let Some(v) = txn_version
        && !v.parse::<i64>().is_ok_and(|n| n >= 0)
    {
        return Err(illegal("txnVersion", v, "must be a non-negative integer"));
    }
    let txn_app_id = get("txnAppId");
    if txn_version.is_some() != txn_app_id.is_some() {
        return Err(DeltaInvalidIdempotentWritesOptions {
            reason: "Both txnVersion and txnAppId must be specified for idempotent data frame \
                     writes"
                .to_owned(),
        }
        .into());
    }
    // Then `DeltaWriteOptions`' own values, then what `WriteIntoDelta` reads.
    let mut flags = BTreeMap::new();
    for name in [
        "useNullIntolerantEqualityWithDPO",
        "optimizeWrite",
        "mergeSchema",
        "overwriteSchema",
    ] {
        if let Some(v) = get(name) {
            flags.insert(name, to_boolean(name, v)?);
        }
    }
    if let Some(v) = get("partitionOverwriteMode").filter(|_| partitioned)
        && !["STATIC", "DYNAMIC"]
            .iter()
            .any(|m| v.eq_ignore_ascii_case(m))
    {
        return Err(illegal(
            "partitionOverwriteMode",
            v,
            "must be 'STATIC' or 'DYNAMIC'",
        ));
    }
    if let Some(v) = get("dataChange") {
        flags.insert("dataChange", to_boolean("dataChange", v)?);
    }
    let refused = if txn_app_id.is_some() {
        Some("idempotent Delta writes (the txnAppId and txnVersion options)")
    } else if flags.get("dataChange") == Some(&false) {
        Some("a Delta write with the option dataChange = false")
    } else if get("replaceOn").is_some() || get("replaceUsing").is_some() {
        Some("a Delta write with the option replaceOn or replaceUsing")
    } else if get("targetAlias").is_some() {
        Some("a Delta write with the option targetAlias")
    } else if flags.get("useNullIntolerantEqualityWithDPO") == Some(&true) {
        Some("a Delta write with the option useNullIntolerantEqualityWithDPO = true")
    } else {
        None
    };
    if let Some(feature) = refused {
        return Err(KrishivUnsupportedFeature {
            feature: feature.to_owned(),
        }
        .into());
    }
    Ok(options
        .iter()
        .filter(|(k, _)| !k.starts_with(INTERNAL_PREFIX))
        .cloned()
        .collect())
}

/// What merging a table type with a written type gives (delta-spark 4.4.1
/// `SchemaMergingUtils.mergeDataTypes`, `schema/SchemaMergingUtils.scala:193-305`).
#[derive(Debug, PartialEq, Eq)]
enum Merged {
    /// The table's type stays: the rows are stored in it.
    Kept,
    /// delta-spark changes the table's schema; the kernel cannot (refused by name).
    Changed(&'static str),
    /// `DELTA_MERGE_INCOMPATIBLE_*`, which the enclosing column reports as
    /// `DELTA_FAILED_TO_MERGE_FIELDS`.
    Incompatible,
}

/// `mergeDataTypes(current, update)` as `ImplicitMetadataOperation.mergeSchema` calls it
/// (`schema/ImplicitMetadataOperation.scala:222-236`: no implicit conversions, the existing
/// type not kept), with type widening as the table has it (`TypeWideningMode.TypeEvolution`,
/// same-family widenings by default).
fn merge_type(current: &DataType, update: &DataType, widening: bool) -> Merged {
    use DataType as T;
    let both = |a: Merged, b: Merged| match (a, b) {
        (Merged::Incompatible, _) | (_, Merged::Incompatible) => Merged::Incompatible,
        (Merged::Changed(w), _) | (_, Merged::Changed(w)) => Merged::Changed(w),
        _ => Merged::Kept,
    };
    match (current, update) {
        // `:203-238`: fields by name (Delta is case insensitive), the current nullability
        // kept; a field only the update has is added to the schema.
        (T::Struct(c), T::Struct(u)) => {
            let mut out = Merged::Kept;
            for cf in c.fields() {
                if let Some(uf) = u
                    .fields()
                    .iter()
                    .find(|f| f.name().eq_ignore_ascii_case(cf.name()))
                {
                    out = both(out, merge_type(cf.data_type(), uf.data_type(), widening));
                }
            }
            let added = u.fields().iter().any(|uf| {
                !c.fields()
                    .iter()
                    .any(|cf| cf.name().eq_ignore_ascii_case(uf.name()))
            });
            if added {
                out = both(
                    out,
                    Merged::Changed("a new nested field (schema evolution)"),
                );
            }
            out
        }
        // `:239-249`: the current `containsNull` kept.
        (T::Array(c), T::Array(u)) => merge_type(c.element_type(), u.element_type(), widening),
        (T::Map(c), T::Map(u)) => both(
            merge_type(c.key_type(), u.key_type(), widening),
            merge_type(c.value_type(), u.value_type(), widening),
        ),
        (c, u) if c.equals_ignore_nullability(u) => Merged::Kept,
        // `:251-255`: a widening the table allows changes the column's type.
        (c, u) if widening && widens(c, u) => Merged::Changed("type widening on write"),
        // `:288-299`: Parquet stores BYTE, SHORT and INT alike; the wider of the two is the
        // merged type.
        // `:301-304`: a NULL-typed update keeps the current type.
        (T::Integer, T::Short | T::Byte) | (T::Short, T::Byte) | (_, T::Null) => Merged::Kept,
        (T::Byte, T::Short | T::Integer) | (T::Short, T::Integer) => {
            Merged::Changed("a wider column type (schema evolution)")
        }
        (T::Null, _) => Merged::Changed("a NULL-typed column given a type (schema evolution)"),
        // `:262-287` (decimals of another precision or scale) and `:305`.
        _ => Merged::Incompatible,
    }
}

/// delta-spark's widenings for schema evolution (`TypeWidening.scala:111-123`): an integral
/// type to one at least as wide, `FLOAT` to `DOUBLE`, a decimal to a wider one, `DATE` to
/// `TIMESTAMP_NTZ`.
fn widens(from: &DataType, to: &DataType) -> bool {
    use DataType as T;
    let size = |t: &DataType| match t {
        T::Byte => Some(1),
        T::Short => Some(2),
        T::Integer => Some(4),
        T::Long => Some(8),
        _ => None,
    };
    match (from, to) {
        (T::Float, T::Double) | (T::Date, T::TimestampNtz) => true,
        (T::Decimal(f), T::Decimal(t)) => {
            t.scale() >= f.scale()
                && i32::from(t.precision()) - i32::from(t.scale())
                    >= i32::from(f.precision()) - i32::from(f.scale())
        }
        _ => matches!((size(from), size(to)), (Some(f), Some(t)) if f <= t),
    }
}

/// The schema merge of a DataFrame `save(path)` of `data` into a Delta table of `table` columns
/// (`widening`: `delta.enableTypeWidening`). delta-spark runs it on every such write, with or
/// without `mergeSchema` (`ImplicitMetadataOperation.scala:87, 216-236`; the frame reaches
/// `WriteIntoDelta` uncast): the types it keeps pass (the rows are then stored in the table's
/// types); a column it cannot merge is `DELTA_FAILED_TO_MERGE_FIELDS` with the table's and the
/// data's names (the first in table order, `SchemaMergingUtils.scala:206-226`); a merge that
/// changes the table's schema (type widening, a wider integral) is, without `merge_schema`,
/// delta-spark's `DELTA_METADATA_MISMATCH` (`ImplicitMetadataOperation.scala:152-165`), and with
/// it refused by name, since the kernel commits no schema change on a write (design section
/// 4.1). Columns only one side has are left to the caller. SQL `INSERT`, `insertInto` and
/// `saveAsTable` append cast to the table's types first and do not call this.
///
/// # Errors
/// `DELTA_FAILED_TO_MERGE_FIELDS`; `DELTA_METADATA_MISMATCH`; `KRISHIV_UNSUPPORTED_FEATURE`.
pub(crate) fn merge_schema_check(
    table: &[TableColumn],
    data: &arrow_schema::Schema,
    widening: bool,
    merge_schema: bool,
    quoted: &str,
) -> Result<(), KrishivError> {
    for c in table {
        let Some(f) = data
            .fields()
            .iter()
            .find(|f| f.name().eq_ignore_ascii_case(&c.name))
        else {
            continue;
        };
        let update = DataType::from_arrow_field(f)?;
        match merge_type(&c.data_type, &update, widening) {
            Merged::Kept => {}
            Merged::Incompatible => {
                return Err(DeltaFailedToMergeFields {
                    current_field: c.name.clone(),
                    update_field: f.name().clone(),
                }
                .into());
            }
            Merged::Changed(_) if !merge_schema => {
                // delta-spark raises the parent class bare (`OPTIONS.json` `merge_schema`
                // `int_into_short_no_merge`).
                return Err(krishiv_common::error::DeltaMetadataMismatch {}.into());
            }
            Merged::Changed(what) => {
                return Err(KrishivUnsupportedFeature {
                    feature: format!(
                        "a Delta write to {quoted} with mergeSchema that changes column {}: {what}",
                        c.name
                    ),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// Scala's `String.toBoolean` (`true`/`false`, any case, nothing else), as `DeltaOptionParser`
/// reads a flag.
fn to_boolean(name: &str, value: &str) -> Result<bool, KrishivError> {
    if value.eq_ignore_ascii_case("true") {
        Ok(true)
    } else if value.eq_ignore_ascii_case("false") {
        Ok(false)
    } else {
        Err(illegal(name, value, "must be 'true' or 'false'"))
    }
}

fn illegal(name: &str, input: &str, explain: &str) -> KrishivError {
    DeltaIllegalOption {
        input: input.to_owned(),
        name: name.to_owned(),
        explain: explain.to_owned(),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    fn class(pairs: &[(&str, &str)]) -> String {
        delta_write_options(&opts(pairs), true)
            .map(|_| "ok".to_owned())
            .unwrap_or_else(|e| e.error_class().name().to_owned())
    }

    #[test]
    fn values_are_checked_as_delta_spark_checks_them() {
        // `tests/fixtures/delta-4.4.1/OPTIONS.json`.
        for (pairs, want) in [
            (&[("mergeSchema", "maybe")][..], "DELTA_ILLEGAL_OPTION"),
            (&[("overwriteSchema", "maybe")], "DELTA_ILLEGAL_OPTION"),
            (&[("dataChange", "maybe")], "DELTA_ILLEGAL_OPTION"),
            (&[("optimizeWrite", "maybe")], "DELTA_ILLEGAL_OPTION"),
            (
                &[("partitionOverwriteMode", "bogus")],
                "DELTA_ILLEGAL_OPTION",
            ),
            (
                &[("txnVersion", "-1"), ("txnAppId", "a")],
                "DELTA_ILLEGAL_OPTION",
            ),
            (
                &[("txnVersion", "1")],
                "DELTA_INVALID_IDEMPOTENT_WRITES_OPTIONS",
            ),
            (
                &[("txnAppId", "a")],
                "DELTA_INVALID_IDEMPOTENT_WRITES_OPTIONS",
            ),
            (&[("someUnknownOption", "x")], "ok"),
            (&[("MERGESCHEMA", "TRUE")], "ok"),
            (&[("optimizeWrite", "true")], "ok"),
            (&[("userMetadata", "hello")], "ok"),
            (&[("partitionOverwriteMode", "dynamic")], "ok"),
            (&[("PARTITIONOVERWRITEMODE", "Static")], "ok"),
            (
                &[("txnVersion", "1"), ("txnAppId", "a")],
                "KRISHIV_UNSUPPORTED_FEATURE",
            ),
            (&[("dataChange", "false")], "KRISHIV_UNSUPPORTED_FEATURE"),
            (&[("replaceUsing", "id")], "KRISHIV_UNSUPPORTED_FEATURE"),
            (
                &[("replaceOn", "t.id = s.id")],
                "KRISHIV_UNSUPPORTED_FEATURE",
            ),
            (&[("writePartitionColumns", "true")], "ok"),
        ] {
            assert_eq!(class(pairs), want, "{pairs:?}");
        }
    }

    #[test]
    fn merge_types_as_schema_merging_utils() {
        use krishiv_types::{ArrayType, StructField, StructType};
        use std::sync::Arc;
        let st = |nullable: bool, t: DataType| {
            DataType::Struct(StructType::new([StructField::new("a", t, nullable)]))
        };
        for (current, update, widening, want) in [
            (DataType::Integer, DataType::Integer, false, Merged::Kept),
            (DataType::Integer, DataType::Short, false, Merged::Kept),
            (DataType::Integer, DataType::Byte, false, Merged::Kept),
            (DataType::Integer, DataType::Null, false, Merged::Kept),
            (
                DataType::Integer,
                DataType::Long,
                false,
                Merged::Incompatible,
            ),
            (
                DataType::Long,
                DataType::Integer,
                false,
                Merged::Incompatible,
            ),
            (
                DataType::Integer,
                DataType::String,
                false,
                Merged::Incompatible,
            ),
            (
                DataType::Integer,
                DataType::Long,
                true,
                Merged::Changed("type widening on write"),
            ),
            (
                DataType::Integer,
                DataType::String,
                true,
                Merged::Incompatible,
            ),
            (DataType::Integer, DataType::Short, true, Merged::Kept),
            (
                st(true, DataType::Integer),
                st(false, DataType::Integer),
                false,
                Merged::Kept,
            ),
            (
                st(true, DataType::Integer),
                st(true, DataType::Long),
                false,
                Merged::Incompatible,
            ),
            (
                DataType::Array(Arc::new(ArrayType::new(DataType::Integer, true))),
                DataType::Array(Arc::new(ArrayType::new(DataType::Short, false))),
                false,
                Merged::Kept,
            ),
        ] {
            assert_eq!(
                merge_type(&current, &update, widening),
                want,
                "{current:?} {update:?} {widening}"
            );
        }
    }

    #[test]
    fn an_unpartitioned_table_never_reads_the_overwrite_mode() {
        // `OPTIONS.json` `pom_bad_unpartitioned`, `insert_overwrite_with_pom_bad`.
        delta_write_options(&opts(&[("partitionOverwriteMode", "bogus")]), false).unwrap();
    }

    #[test]
    fn the_message_is_delta_sparks() {
        let e = delta_write_options(&opts(&[("mergeSchema", "maybe")]), true).unwrap_err();
        assert_eq!(
            e.to_string().lines().next().unwrap(),
            "[DELTA_ILLEGAL_OPTION] Invalid value 'maybe' for option 'mergeSchema', must be \
             'true' or 'false' SQLSTATE: 42616"
        );
    }

    #[test]
    fn the_sessions_own_options_never_pass() {
        let passed = delta_write_options(
            &opts(&[
                ("__krishiv_delta_reads_target", "true"),
                ("userMetadata", "m"),
            ]),
            true,
        )
        .unwrap();
        assert_eq!(
            passed.into_iter().collect::<Vec<_>>(),
            vec![("userMetadata".to_owned(), "m".to_owned())]
        );
    }
}

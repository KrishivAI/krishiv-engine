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
//!   the table's rows or log: `optimizeWrite`, `compression`, `parquet.writer.version`;
//! - refused by name (`KRISHIV_UNSUPPORTED_FEATURE`), where delta-spark changes what it commits
//!   and Krishiv cannot: idempotent writes (`txnAppId` + `txnVersion`, a `txn` action),
//!   `dataChange = false`, `replaceOn`, `replaceUsing`, `targetAlias`,
//!   `writePartitionColumns = true` and `useNullIntolerantEqualityWithDPO = true`.
//!
//! Options whose names start with `__krishiv` are the session's own and never pass from a user.

use std::collections::BTreeMap;

use krishiv_common::error::{
    DeltaIllegalOption, DeltaInvalidIdempotentWritesOptions, KrishivError,
    KrishivUnsupportedFeature,
};

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
    } else if get("writePartitionColumns").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
        Some("a Delta write with the option writePartitionColumns = true")
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
            (
                &[("writePartitionColumns", "true")],
                "KRISHIV_UNSUPPORTED_FEATURE",
            ),
        ] {
            assert_eq!(class(pairs), want, "{pairs:?}");
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

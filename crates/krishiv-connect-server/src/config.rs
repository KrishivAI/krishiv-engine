// SPDX-License-Identifier: Apache-2.0
//! `Config` (design P3-T011a §3.6; Spark's `SparkConnectConfigHandler`,
//! v4.3.0 lines 49-251) over the session's [`RuntimeConf`]:
//!
//! - `set(pairs, silent)`: per pair, in order; with `silent` a failure is
//!   the warning `Failed to set <key> to Some(<value>) due to <message>`
//!   (Spark prints the Scala `Option`; Krishiv shows a redacted value, see
//!   below, as `Some(*********(redacted))`) and the next pair is tried. Who may
//!   set or unset a key (ADR 0046 §3, §8, threat model T-Z3): a `Startup`
//!   key is `CANNOT_MODIFY_STATIC_CONFIG` for everyone (`check_static`);
//!   any other key is the session's policy, asked as SQL `SET` asks it
//!   (`Session::authorize_set_key`: an `Admin` key or a server-scoped store
//!   key needs `Manage` on the cluster, else `KRISHIV_INSUFFICIENT_PERMISSIONS`
//!   (42501), audited and counted). The listener keeps no role check of its own.
//! - `get(keys)`: an unset key without a default is `SQL_CONF_NOT_FOUND`
//!   (Spark's `SQLConf.getConfString`, `SQLConf.scala:9801-9808`).
//! - `get_with_default`: a set value, else the caller's default (a
//!   registered key's own default is not used, `SQLConf.scala:9842-9856`).
//! - `get_option`, `unset`, `is_modifiable` (`"true"`/`"false"`).
//! - `get_all(prefix)`: the explicitly set keys (`RuntimeConf::entries`),
//!   with the prefix stripped.
//!
//! **Redaction** (Spark 4.3.0, `ConfigH:50-53, 109-167, 227-233`): a key or
//! value matching `spark.redaction.regex`'s default
//! (`(?i)secret|password|token|access[.]?key|credential`) is hidden: `get`
//! fails as for an unset key, `get_with_default` answers the caller's
//! default, `get_option` nothing, `get_all` leaves it out (matched before
//! the prefix is stripped). Krishiv redacts for 4.2 sessions too (design
//! §7 question 3), and never resolves a secret reference here (ADR 0038).
//!
//! Every key read or written gets a warning when it is one of the two SQL
//! configs Spark Connect does not support, else Spark's deprecation text.

use std::borrow::Cow;

use krishiv_common::config::{SettableBy, lookup, resolve_alternative};
use krishiv_common::error::{
    CannotModifyStaticConfig, ConnectInvalidPlanInvalidOneOfFieldNotSet,
    InvalidConfValueRequirement, KrishivError, SqlConfNotFound,
};
use krishiv_common::redact::{REDACTED, is_redacted_setting};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::config_request::operation::OpType;
use krishiv_session::RuntimeConf;

use crate::manager::SessionEntry;

/// A `Startup` key cannot change in a running session, for anyone. The key is looked up as
/// `Session::authorize_set_key` looks it up (its Spark alias or legacy alternative too). Public
/// so Flight SQL's `SetSessionOptions` refuses the same keys as `Config` (P3-T101 §2 H4, §3.2).
///
/// # Errors
/// `CANNOT_MODIFY_STATIC_CONFIG` for a `Startup` key.
pub fn check_static(key: &str) -> Result<(), KrishivError> {
    let entry = lookup(key).or_else(|| lookup(resolve_alternative(key)));
    if entry.is_some_and(|e| e.settable_by == SettableBy::Startup) {
        return Err(CannotModifyStaticConfig {
            key: format!("\"{key}\""),
        }
        .into());
    }
    Ok(())
}

// TODO(P3-T002 follow-up): hide `Startup` and `Admin` keys from `get_all` for callers the
// session's policy would not let set them.

/// The two SQL configs Spark Connect warns about
/// (`SparkConnectConfigHandler.unsupportedConfigurations`, line 217-218).
const UNSUPPORTED: [&str; 2] = [
    "spark.sql.execution.arrow.enabled",
    "spark.sql.execution.arrow.pyspark.fallback.enabled",
];

/// `SQLConf.deprecatedSQLConfigs` (v4.3.0, `SQLConf.scala:8379-8412`):
/// key, version, comment.
const DEPRECATED: [(&str, &str, &str); 13] = [
    (
        "spark.sql.legacy.execution.pandas.groupedMap.assignColumnsByName",
        "2.4",
        "The config allows to switch to the behaviour before Spark 2.4 and will be removed in \
         the future releases.",
    ),
    (
        "spark.sql.execution.arrow.enabled",
        "3.0",
        "Use 'spark.sql.execution.arrow.pyspark.enabled' instead of it.",
    ),
    (
        "spark.sql.execution.arrow.fallback.enabled",
        "3.0",
        "Use 'spark.sql.execution.arrow.pyspark.fallback.enabled' instead of it.",
    ),
    (
        "spark.sql.adaptive.shuffle.targetPostShuffleInputSize",
        "3.0",
        "Use 'spark.sql.adaptive.advisoryPartitionSizeInBytes' instead of it.",
    ),
    (
        "spark.sql.optimizer.metadataOnly",
        "3.0",
        "Avoid to depend on this optimization to prevent a potential correctness issue. If you \
         must use, use 'SparkSessionExtensions' instead to inject it as a custom rule.",
    ),
    (
        "spark.sql.sources.schemaStringLengthThreshold",
        "3.2",
        "Use 'spark.sql.hive.tablePropertyLengthThreshold' instead.",
    ),
    (
        "spark.sql.legacy.replaceDatabricksSparkAvro.enabled",
        "3.2",
        "Use `.format(\"avro\")` in `DataFrameWriter` or `DataFrameReader` instead.",
    ),
    (
        "spark.sql.adaptive.coalescePartitions.minPartitionNum",
        "3.2",
        "Use 'spark.sql.adaptive.coalescePartitions.minPartitionSize' instead.",
    ),
    (
        "spark.sql.legacy.parquet.datetimeRebaseModeInRead",
        "3.2",
        "Use 'spark.sql.parquet.datetimeRebaseModeInRead' instead.",
    ),
    (
        "spark.sql.parser.escapedStringLiterals",
        "4.0",
        "Use raw string literals with the `r` prefix instead. ",
    ),
    (
        "spark.connect.copyFromLocalToFs.allowDestLocal",
        "4.0",
        "Use 'spark.sql.artifact.copyFromLocalToFs.allowDestLocal' instead.",
    ),
    (
        "spark.sql.legacy.allowZeroIndexInFormatString",
        "4.0",
        "Increase indexes by 1 in `strfmt` of the `format_string` function. Refer to the first \
         argument by \"1$\".",
    ),
    (
        "spark.sql.shuffleDependency.fileCleanup.enabled",
        "4.1",
        "Use 'spark.sql.connect.shuffleDependency.fileCleanup.enabled' instead.",
    ),
];

/// `getWarning` (`ConfigH:206-212`).
fn warning(key: &str) -> Option<String> {
    if UNSUPPORTED.contains(&key) {
        return Some(format!(
            "The SQL config '{key}' is NOT supported in Spark Connect"
        ));
    }
    DEPRECATED
        .iter()
        .find(|(k, _, _)| *k == key)
        .map(|(k, version, comment)| {
            format!(
                "The SQL config '{k}' has been deprecated in Spark v{version} and may be removed \
                 in the future. {comment}"
            )
        })
}

/// `SparkConnectConfigHandler.isRedacted` (`ConfigH:227-233`): the key or
/// the value matches Spark's pattern, or Krishiv's secret classifier (a
/// difference toward safety); see [`is_redacted_setting`].
#[must_use]
pub fn is_redacted(key: &str, value: Option<&str>) -> bool {
    is_redacted_setting(key, value)
}

/// A silent `set`'s warning for a pair that failed: Spark's
/// `Failed to set <key> to <Option> due to <message>` (`ConfigH:92-93`).
/// A difference toward safety: Spark prints the value as given, Krishiv
/// replaces it by [`REDACTED`] when [`is_redacted`] holds for the pair, also
/// where the error message repeats it, so a token sent through a client
/// builder (`.config("spark.sql.catalog.x.token", ...)`) never comes back in
/// the response.
fn failed_to_set(key: &str, value: Option<&str>, message: &str) -> String {
    match value {
        None => format!("Failed to set {key} to None due to {message}"),
        Some(v) if is_redacted(key, Some(v)) => {
            // Every occurrence goes, even inside other words or the SQLSTATE when the value is
            // very short (`1`): garbling a warning is the safe side. An empty value is skipped,
            // since `replace("")` would insert the marker between every character.
            let message = if v.is_empty() {
                Cow::Borrowed(message)
            } else {
                Cow::Owned(message.replace(v, REDACTED))
            };
            format!("Failed to set {key} to Some({REDACTED}) due to {message}")
        }
        Some(v) => format!("Failed to set {key} to Some({v}) due to {message}"),
    }
}

/// The value `key` was set to, if it was (not a registered default).
fn explicit(conf: &RuntimeConf, key: &str) -> Option<String> {
    let canonical = lookup(resolve_alternative(key)).map_or(key, |e| e.key);
    conf.entries()
        .into_iter()
        .find(|(k, _)| k == canonical)
        .map(|(_, v)| v)
}

/// The read-only key a Krishiv client reads to learn the `krishiv.v1` capabilities (P3-T015): the
/// comma-separated set, never listed by `get_all` and never settable, so a stock client never sees it.
fn capabilities_key(key: &str) -> Option<String> {
    (key == krishiv_connect::extension::CAPABILITIES_KEY)
        .then(krishiv_connect::extension::capabilities_value)
}

fn pair(key: &str, value: Option<String>) -> pb::KeyValue {
    pb::KeyValue {
        key: key.to_owned(),
        value,
    }
}

/// The message Spark's `e.getMessage` gives: the client form.
fn message(e: &KrishivError) -> String {
    e.for_client().error().to_string()
}

/// Answers `request` on the session `entry` for a principal of `role`.
///
/// # Errors
/// The first failure of a non-silent `set`, `unset` or `get` (see the
/// module docs); `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET` for a
/// request without an operation (Spark throws a classless
/// `UnsupportedOperationException`: a recorded difference).
pub(crate) async fn config(
    entry: &SessionEntry,
    request: &pb::ConfigRequest,
) -> Result<pb::ConfigResponse, KrishivError> {
    let conf = entry.session().conf();
    let mut out = pb::ConfigResponse {
        session_id: request.session_id.clone(),
        server_side_session_id: entry.server_side_id().to_owned(),
        pairs: Vec::new(),
        warnings: Vec::new(),
    };
    let op = request.operation.as_ref().and_then(|o| o.op_type.as_ref());
    match op {
        None => {
            return Err(ConnectInvalidPlanInvalidOneOfFieldNotSet {
                full_name: "spark.connect.ConfigRequest.Operation".to_owned(),
                name: "OPTYPE_NOT_SET".to_owned(),
            }
            .into());
        }
        Some(OpType::Set(set)) => {
            let silent = set.silent.unwrap_or(false);
            for kv in &set.pairs {
                match set_one(entry, &kv.key, kv.value.as_deref()).await {
                    Ok(()) => out.warnings.extend(warning(&kv.key)),
                    Err(e) if silent => {
                        out.warnings.push(failed_to_set(
                            &kv.key,
                            kv.value.as_deref(),
                            &message(&e),
                        ));
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        Some(OpType::Get(get)) => {
            for key in &get.keys {
                let value = capabilities_key(key).or_else(|| conf.get(key));
                if value.is_none() || is_redacted(key, value.as_deref()) {
                    return Err(SqlConfNotFound {
                        sql_conf: format!("\"{key}\""),
                    }
                    .into());
                }
                out.pairs.push(pair(key, value));
                out.warnings.extend(warning(key));
            }
        }
        Some(OpType::GetWithDefault(g)) => {
            for kv in &g.pairs {
                let stored = match explicit(&conf, &kv.key) {
                    Some(v) => Some(v),
                    None => {
                        // Spark checks a registered key's default against
                        // its type (`entry.valueConverter(defaultValue)`).
                        if let (Some(e), Some(d)) = (lookup(&kv.key), kv.value.as_deref()) {
                            e.parse(&kv.key, d)?;
                        }
                        kv.value.clone()
                    }
                };
                let value = if is_redacted(&kv.key, stored.as_deref()) {
                    kv.value.clone()
                } else {
                    stored
                };
                out.pairs.push(pair(&kv.key, value));
                out.warnings.extend(warning(&kv.key));
            }
        }
        Some(OpType::GetOption(g)) => {
            for key in &g.keys {
                let value = capabilities_key(key)
                    .or_else(|| conf.get(key).filter(|v| !is_redacted(key, Some(v))));
                out.pairs.push(pair(key, value));
                out.warnings.extend(warning(key));
            }
        }
        Some(OpType::GetAll(g)) => {
            for (key, value) in conf.entries() {
                // A registered key is listed under the Spark name a client
                // sets it by, when it has one.
                let key = lookup(&key)
                    .and_then(|e| e.spark_alias)
                    .map_or(key, str::to_owned);
                if is_redacted(&key, Some(&value)) {
                    continue;
                }
                let shown = match &g.prefix {
                    Some(prefix) => match key.strip_prefix(prefix.as_str()) {
                        Some(rest) => rest.to_owned(),
                        None => continue,
                    },
                    None => key,
                };
                out.warnings.extend(warning(&shown));
                out.pairs.push(pair(&shown, Some(value)));
            }
        }
        Some(OpType::Unset(u)) => {
            for key in &u.keys {
                authorize(entry, key).await?;
                conf.unset(key)?;
                out.warnings.extend(warning(key));
            }
        }
        Some(OpType::IsModifiable(m)) => {
            for key in &m.keys {
                out.pairs
                    .push(pair(key, Some(conf.is_modifiable(key).to_string())));
                out.warnings.extend(warning(key));
            }
        }
    }
    Ok(out)
}

/// The checks of a `set` or `unset` of `key`: a `Startup` key is refused for everyone first
/// (no policy call, nothing counted: not an auth denial, ADR 0046 §3), then the session asks
/// its policy as SQL `SET` does (`Session::authorize_set_key`: an `Admin` key or a
/// server-scoped store key needs `Manage` on the cluster; a denial is audited and counted).
async fn authorize(entry: &SessionEntry, key: &str) -> Result<(), KrishivError> {
    check_static(key)?;
    entry.session().authorize_set_key(key).await
}

async fn set_one(entry: &SessionEntry, key: &str, value: Option<&str>) -> Result<(), KrishivError> {
    authorize(entry, key).await?;
    let conf = entry.session().conf();
    // Spark's `require(value != null)` is a classless
    // `IllegalArgumentException`: a recorded difference.
    let Some(value) = value else {
        return Err(InvalidConfValueRequirement {
            conf_name: format!("\"{key}\""),
            conf_value: "null".to_owned(),
            conf_requirement: format!("value cannot be null for key: {key}"),
        }
        .into());
    };
    conf.set(key, value)
}

#[cfg(test)]
mod tests;

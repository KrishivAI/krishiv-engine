// SPDX-License-Identifier: Apache-2.0
//! `Config` (design P3-T011a §3.6; Spark's `SparkConnectConfigHandler`,
//! v4.3.0 lines 49-251) over the session's [`RuntimeConf`]:
//!
//! - `set(pairs, silent)`: per pair, in order; with `silent` a failure is
//!   the warning `Failed to set <key> to Some(<value>) due to <message>`
//!   (Spark prints the Scala `Option`) and the next pair is tried. Who may
//!   set a key is the registry's `SettableBy` (ADR 0046 §8, threat model
//!   T-Z3): a `Startup` key is `CANNOT_MODIFY_STATIC_CONFIG` for every role,
//!   an `Admin` key `KRISHIV_INSUFFICIENT_PERMISSIONS` (42501) for a viewer or
//!   operator ([`check_settable`]).
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

use krishiv_common::config::{SettableBy, lookup};
use krishiv_common::error::{
    CannotModifyStaticConfig, ConnectInvalidPlanInvalidOneOfFieldNotSet,
    InvalidConfValueRequirement, KrishivError, KrishivInsufficientPermissions, SqlConfNotFound,
};
use krishiv_common::principal::Principal;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::config_request::operation::OpType;
use krishiv_session::RuntimeConf;

use crate::manager::SessionEntry;

/// What a principal may do with configuration (ADR 0046 §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigRole {
    /// Reads and sets `User` keys only.
    Viewer,
    /// Reads and sets `User` keys only.
    Operator,
    /// Sets `Admin` keys too.
    Admin,
}

/// The configuration role of `principal`.
///
/// TODO(P3-T002): roles arrive with authentication; until then every
/// principal is the local user on a loopback listener (design P3-T001
/// §3.6) and is treated as the admin.
#[must_use]
pub fn role_of(_principal: &Principal) -> ConfigRole {
    ConfigRole::Admin
}

/// Whether `role` may set or unset `key` (the registry's `SettableBy`).
///
/// # Errors
/// `CANNOT_MODIFY_STATIC_CONFIG` for a `Startup` key (every role);
/// `KRISHIV_INSUFFICIENT_PERMISSIONS` for an `Admin` key and a role other
/// than [`ConfigRole::Admin`].
pub fn check_settable(
    role: ConfigRole,
    principal: &Principal,
    key: &str,
) -> Result<(), KrishivError> {
    match lookup(key).map(|e| e.settable_by) {
        Some(SettableBy::Startup) => Err(CannotModifyStaticConfig {
            key: format!("\"{key}\""),
        }
        .into()),
        Some(SettableBy::Admin) if role != ConfigRole::Admin => {
            Err(KrishivInsufficientPermissions {
                principal: principal.name.clone(),
                privilege: "SET".to_owned(),
                securable: format!("configuration `{key}`"),
            }
            .into())
        }
        _ => Ok(()),
    }
}

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

/// The default of `spark.redaction.regex`
/// (`core/.../internal/config/package.scala:1325-1333`),
/// `(?i)secret|password|token|access[.]?key|credential`, as plain
/// substrings: Java's `(?i)` folds ASCII only, and `[.]` is a literal dot.
const REDACTION_WORDS: [&str; 6] = [
    "secret",
    "password",
    "token",
    "accesskey",
    "access.key",
    "credential",
];

fn matches_redaction(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    REDACTION_WORDS.iter().any(|w| lower.contains(w))
}

/// `SparkConnectConfigHandler.isRedacted` (`ConfigH:227-233`): the key or
/// the value matches.
#[must_use]
pub fn is_redacted(key: &str, value: Option<&str>) -> bool {
    matches_redaction(key) || value.is_some_and(matches_redaction)
}

/// The value `key` was set to, if it was (not a registered default).
fn explicit(conf: &RuntimeConf, key: &str) -> Option<String> {
    let canonical = lookup(key).map_or(key, |e| e.key);
    conf.entries()
        .into_iter()
        .find(|(k, _)| k == canonical)
        .map(|(_, v)| v)
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
pub fn config(
    entry: &SessionEntry,
    request: &pb::ConfigRequest,
    principal: &Principal,
    role: ConfigRole,
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
                match set_one(&conf, principal, role, &kv.key, kv.value.as_deref()) {
                    Ok(()) => out.warnings.extend(warning(&kv.key)),
                    Err(e) if silent => {
                        let shown = kv
                            .value
                            .as_ref()
                            .map_or_else(|| "None".to_owned(), |v| format!("Some({v})"));
                        out.warnings.push(format!(
                            "Failed to set {} to {shown} due to {}",
                            kv.key,
                            message(&e)
                        ));
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        Some(OpType::Get(get)) => {
            for key in &get.keys {
                let value = conf.get(key);
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
                let value = conf.get(key).filter(|v| !is_redacted(key, Some(v)));
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
                check_settable(role, principal, key)?;
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

fn set_one(
    conf: &RuntimeConf,
    principal: &Principal,
    role: ConfigRole,
    key: &str,
    value: Option<&str>,
) -> Result<(), KrishivError> {
    check_settable(role, principal, key)?;
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

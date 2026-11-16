// SPDX-License-Identifier: Apache-2.0
//! Unit tests of `SettableBy`, redaction and the per-key warnings.
use super::{check_static, is_redacted, warning};

const STARTUP: &str = "krishiv.resultCache.maxBytes";
const ADMIN: &str = "krishiv.resultCache.ttl";
const USER: &str = "spark.sql.ansi.enabled";

/// A `Startup` key is refused here for everyone; an `Admin` key is left to the session's
/// policy (`tests/auth.rs` `config_refusals_are_counted_like_sql_set`).
#[test]
fn config_settable_by() {
    let e = check_static(STARTUP).unwrap_err();
    assert_eq!(e.error_class().name(), "CANNOT_MODIFY_STATIC_CONFIG");
    assert_eq!(e.error_class().sql_state(), Some("46110"));
    check_static(ADMIN).unwrap();
    check_static(USER).unwrap();
    check_static("my.app.flag").unwrap();
}

#[test]
fn redaction_matches_the_key_or_the_value() {
    assert!(is_redacted("spark.my.secret.value", Some("x")));
    assert!(is_redacted("fs.s3a.ACCESS.KEY", None));
    assert!(is_redacted("fs.s3a.accessKey", None));
    assert!(is_redacted("a.b", Some("jdbc:x?Password=p")));
    assert!(is_redacted("my.Credentials", None));
    assert!(is_redacted("auth.token", None));
    assert!(
        !super::matches_redaction("access_key"),
        "Spark's [.]? is a literal dot"
    );
    // Krishiv's own classifier (ADR 0038) on top of Spark's pattern.
    assert!(is_redacted("access_key", None));
    assert!(is_redacted(
        "fs.azure.account.key.acct.dfs.core.windows.net",
        Some("abc")
    ));
    assert!(is_redacted("kafka.sasl.jaas.config", Some("x")));
    assert!(is_redacted("my.apiKey", None));
    // A value holding URL credentials, whatever its key.
    assert!(is_redacted("my.db", Some("postgres://u:pw@h/db")));
    assert!(!is_redacted("my.db", Some("postgres://h/db")));
    assert!(!is_redacted("spark.sql.ansi.enabled", Some("true")));
    assert!(!is_redacted("spark.sql.ansi.enabled", None));
}

#[test]
fn warnings_name_unsupported_then_deprecated_keys() {
    assert_eq!(
        warning("spark.sql.execution.arrow.enabled").as_deref(),
        Some(
            "The SQL config 'spark.sql.execution.arrow.enabled' is NOT supported in Spark Connect"
        )
    );
    assert_eq!(
        warning("spark.sql.execution.arrow.pyspark.fallback.enabled").as_deref(),
        Some(
            "The SQL config 'spark.sql.execution.arrow.pyspark.fallback.enabled' is NOT \
             supported in Spark Connect"
        )
    );
    assert_eq!(
        warning("spark.sql.optimizer.metadataOnly").as_deref(),
        Some(
            "The SQL config 'spark.sql.optimizer.metadataOnly' has been deprecated in Spark v3.0 \
             and may be removed in the future. Avoid to depend on this optimization to prevent a \
             potential correctness issue. If you must use, use 'SparkSessionExtensions' instead \
             to inject it as a custom rule."
        )
    );
    assert_eq!(warning("spark.sql.ansi.enabled"), None);
}

#[test]
fn explicit_resolves_a_legacy_alternative_to_its_primary_entry() {
    let alt = "spark.sql.legacy.avro.datetimeRebaseModeInRead";
    let session = krishiv_session::Session::new().unwrap();
    let conf = session.conf();
    conf.set(alt, "LEGACY").unwrap();
    assert_eq!(super::explicit(&conf, alt).as_deref(), Some("LEGACY"));
}

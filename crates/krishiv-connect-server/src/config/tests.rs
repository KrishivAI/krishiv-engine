// SPDX-License-Identifier: Apache-2.0
//! Unit tests of `SettableBy`, redaction and the per-key warnings.
use krishiv_common::principal::{Principal, PrincipalKind};

use super::{ConfigRole, check_settable, is_redacted, role_of, warning};

const STARTUP: &str = "krishiv.resultCache.maxBytes";
const ADMIN: &str = "krishiv.resultCache.ttl";
const USER: &str = "spark.sql.ansi.enabled";

#[test]
fn config_settable_by() {
    let p = Principal::new("ann", PrincipalKind::User);
    for role in [ConfigRole::Viewer, ConfigRole::Operator, ConfigRole::Admin] {
        let e = check_settable(role, &p, STARTUP).unwrap_err();
        assert_eq!(e.error_class().name(), "CANNOT_MODIFY_STATIC_CONFIG");
        assert_eq!(e.error_class().sql_state(), Some("46110"));
        check_settable(role, &p, USER).unwrap();
        check_settable(role, &p, "my.app.flag").unwrap();
    }
    for role in [ConfigRole::Viewer, ConfigRole::Operator] {
        let e = check_settable(role, &p, ADMIN).unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
        assert_eq!(e.error_class().sql_state(), Some("42501"));
    }
    check_settable(ConfigRole::Admin, &p, ADMIN).unwrap();
    // Until T002 the local principal is the admin.
    assert_eq!(role_of(&Principal::local_user()), ConfigRole::Admin);
}

#[test]
fn redaction_matches_the_key_or_the_value() {
    assert!(is_redacted("spark.my.secret.value", Some("x")));
    assert!(is_redacted("fs.s3a.ACCESS.KEY", None));
    assert!(is_redacted("fs.s3a.accessKey", None));
    assert!(is_redacted("a.b", Some("jdbc:x?Password=p")));
    assert!(is_redacted("my.Credentials", None));
    assert!(is_redacted("auth.token", None));
    assert!(!is_redacted("access_key", None), "[.]? is a literal dot");
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

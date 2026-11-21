// SPDX-License-Identifier: Apache-2.0
//! A refused or invalid setting hides its value in the error when `Config.get` would hide the pair
//! (`krishiv_common::redact::is_redacted_setting`): the value is a secret-like word or a
//! credential, or the key names a secret. Spark repeats the value; Krishiv shows
//! `*********(redacted)`, a difference toward safety (threat model T-C8). Every other pair keeps
//! Spark's text. This holds for `RuntimeConf::set` and SQL `SET`, which goes through it.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

const REDACTED: &str = "*********(redacted)";

/// Asserts that `e`, raised for `key` = `canary`, keeps its class and names the key, but does not
/// show the value anywhere: message, parameters or the client form.
fn hides(e: &KrishivError, class: ErrorClass, key: &str, canary: &str) {
    assert_eq!(e.error_class(), class, "{e}");
    let shown = [
        e.to_string(),
        e.for_client().error().to_string(),
        format!("{:?}", e.message_parameters()),
    ];
    for text in &shown {
        assert!(!text.contains(canary), "{text}");
    }
    assert!(shown[0].contains(REDACTED), "{}", shown[0]);
    assert!(shown[0].contains(key), "{}", shown[0]);
}

#[test]
fn a_credential_in_a_value_is_hidden_in_the_error() {
    // A URL with credentials as a time zone, as a boolean, and as a `SET PATH` element list.
    let canary = "canary-pw-7f3";
    let url = format!("postgres://u:{canary}@db.local/x");
    let s = Session::new().unwrap();
    for (key, class) in [
        (
            "spark.sql.session.timeZone",
            ErrorClass::InvalidConfValueTimeZone,
        ),
        (
            "spark.sql.ansi.enabled",
            ErrorClass::KrishivInvalidConfigValue,
        ),
        (
            "krishiv.sql.defaultPath",
            ErrorClass::InvalidConfValueRequirement,
        ),
    ] {
        let e = s.conf().set(key, &url).unwrap_err();
        hides(&e, class, key, canary);
    }
}

#[test]
fn a_secret_like_word_in_a_value_is_hidden_in_the_error() {
    // Spark's `spark.redaction.regex` words, in any case.
    let s = Session::new().unwrap();
    for value in ["mySecret-1", "TOKEN-abc", "aws.access.key=1"] {
        let e = s.conf().set("spark.sql.ansi.enabled", value).unwrap_err();
        hides(
            &e,
            ErrorClass::KrishivInvalidConfigValue,
            "spark.sql.ansi.enabled",
            value,
        );
    }
}

#[test]
fn a_secret_like_key_never_shows_its_value() {
    // The only registered key whose name matches (`credential`) is static: its refusal names the
    // key and never the value, redacted or not.
    let key = "krishiv.io.credentials.refreshSkew";
    let s = Session::new().unwrap();
    let e = s.conf().set(key, "canary-skew").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::CannotModifyStaticConfig, "{e}");
    assert!(e.to_string().contains(key), "{e}");
    assert!(!e.to_string().contains("canary-skew"), "{e}");
}

#[test]
fn sql_set_hides_the_value_too() {
    let s = Session::new().unwrap();
    let e = block_on(s.sql("SET spark.sql.session.timeZone = postgres://u:canary-sql-9@h/db"))
        .map(drop)
        .unwrap_err();
    hides(
        &e,
        ErrorClass::InvalidConfValueTimeZone,
        "spark.sql.session.timeZone",
        "canary-sql-9",
    );
}

#[test]
fn a_plain_pair_keeps_sparks_text() {
    let s = Session::new().unwrap();
    let e = s
        .conf()
        .set("spark.sql.session.timeZone", "Mars/Base")
        .unwrap_err();
    assert_eq!(
        e.message_parameters(),
        [
            ("confValue", "Mars/Base".to_owned()),
            ("confName", "spark.sql.session.timeZone".to_owned())
        ]
    );
    assert!(!e.to_string().contains(REDACTED), "{e}");
}

#[test]
fn an_unhonoured_legacy_setting_hides_the_value() {
    // A `spark.sql.legacy.*` key Spark registers but Krishiv does not honour is refused for a
    // non-default value; the refusal names the pair, so a credential in the value must go.
    let key = "spark.sql.legacy.allowNegativeScaleOfDecimal";
    let canary = "canary-legacy";
    let s = Session::new().unwrap();
    let e = s
        .conf()
        .set(key, &format!("postgres://u:{canary}@h/x"))
        .unwrap_err();
    hides(&e, ErrorClass::KrishivUnsupportedFeature, key, canary);
    // SQL `SET` takes the same path.
    let e = block_on(s.sql(&format!("SET {key} = postgres://u:{canary}@h/x")))
        .map(drop)
        .unwrap_err();
    hides(&e, ErrorClass::KrishivUnsupportedFeature, key, canary);
}

#[test]
fn a_value_inside_a_secret_like_key_leaves_the_key_readable() {
    // `krishiv.foo.token` is unknown and secret-like; the value `o` occurs in the key itself.
    let key = "krishiv.foo.token";
    let s = Session::new().unwrap();
    let e = s.conf().set(key, "o").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnknownConfig, "{e}");
    assert!(e.to_string().contains(key), "{e}");
}

// SPDX-License-Identifier: Apache-2.0
//! A refused or invalid setting hides its value in the error when `Config.get` would hide the pair
//! (`krishiv_common::redact::is_redacted_setting`): the value is a secret-like word or a
//! credential, or the key names a secret. Spark repeats the value; Krishiv shows
//! `*********(redacted)`, a difference toward safety (threat model T-C8). Every other pair keeps
//! Spark's text. This holds for `RuntimeConf::set` and SQL `SET`, which goes through it.
//!
//! The output of SQL `SET` and `SET key` hides what Spark's hides (`SQLConf.redactOptions`), and
//! `SET -v` a registered secret.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StringArray, StringViewArray};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::{EngineRuntime, Session};

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

/// The `key` and `value` cells of each row `query` returns.
fn pairs(s: &Session, query: &str) -> Vec<(String, String)> {
    let batches: Vec<RecordBatch> = block_on(block_on(s.sql(query)).unwrap().collect()).unwrap();
    let cell = |b: &RecordBatch, col: usize, row: usize| {
        let c = b.column(col);
        c.as_any().downcast_ref::<StringArray>().map_or_else(
            || {
                c.as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap()
                    .value(row)
                    .to_owned()
            },
            |a| a.value(row).to_owned(),
        )
    };
    batches
        .iter()
        .flat_map(|b| (0..b.num_rows()).map(move |i| (cell(b, 0, i), cell(b, 1, i))))
        .collect()
}

/// Spark 4.3.0 (`SetCommand` through `SQLConf.redactOptions`; run on the 4.3.0 oracle,
/// acceptance R1-D-18): `SET key` and `SET` print `*********(redacted)` for a pair whose key or
/// value matches `spark.redaction.regex` or `spark.sql.redaction.options.regex` (`(?i)url`).
/// Only the echo of `SET key = value` shows the value.
#[test]
fn set_and_set_key_print_no_secret() {
    let s = Session::new().unwrap();
    let hidden = [
        ("spark.my.password", "canary-1"),
        ("spark.x.token", "canary-2"),
        ("spark.hadoop.fs.s3a.secret.key", "canary-3"),
        // The value names a secret-like word; Krishiv's classifier knows the key too.
        ("fs.azure.account.key.acct", "canary-4-secret"),
        // `url` in the key: hidden in `SET`, though `Config.get` shows a URL without credentials.
        ("spark.my.jdbc.URL", "postgres://canary-5/db"),
        // `url` in the value only.
        ("spark.my.plain", "canary-6-curl"),
    ];
    for (key, value) in hidden {
        let echo = pairs(&s, &format!("SET {key}={value}"));
        assert_eq!(
            echo,
            [(key.to_owned(), value.to_owned())],
            "the echo is Spark's"
        );
        assert_eq!(
            pairs(&s, &format!("SET {key}")),
            [(key.to_owned(), REDACTED.to_owned())]
        );
    }
    block_on(s.sql("SET spark.my.shown=plain-value"))
        .map(drop)
        .unwrap();
    let all = pairs(&s, "SET");
    for (key, _) in hidden {
        assert!(
            all.contains(&(key.to_owned(), REDACTED.to_owned())),
            "{key}: {all:?}"
        );
    }
    assert!(all.iter().all(|(_, v)| !v.contains("canary")), "{all:?}");
    // A pair nothing marks is shown, by both statements.
    assert!(
        all.contains(&("spark.my.shown".to_owned(), "plain-value".to_owned())),
        "{all:?}"
    );
    assert_eq!(
        pairs(&s, "SET spark.my.shown"),
        [("spark.my.shown".to_owned(), "plain-value".to_owned())]
    );
    assert_eq!(
        pairs(&s, "SET spark.my.unset"),
        [("spark.my.unset".to_owned(), "<undefined>".to_owned())]
    );
    // A secret-like key with no value: Spark asks the pattern about the key first, so it prints
    // the replacement text, not `<undefined>`.
    assert_eq!(
        pairs(&s, "SET spark.never.set.password"),
        [("spark.never.set.password".to_owned(), REDACTED.to_owned())]
    );
}

/// `SET -v` lists registered keys with their values. Spark hides nothing there; Krishiv hides the
/// value of a registered key that holds a secret (a difference toward safety).
#[test]
fn set_verbose_hides_a_registered_secret() {
    let key = "krishiv.access.external.token";
    let rt = EngineRuntime::builder()
        .config(key, "env:CANARY_VERBOSE")
        .build()
        .unwrap();
    let s = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let all = pairs(&s, "SET -v");
    assert!(
        all.contains(&(key.to_owned(), REDACTED.to_owned())),
        "{key} is not hidden"
    );
    assert!(all.iter().all(|(_, v)| !v.contains("CANARY")));
    // Other registered keys keep their values.
    assert!(all.iter().any(|(_, v)| v == "true"));
}

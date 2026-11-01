// SPDX-License-Identifier: Apache-2.0
//! Query-text redaction: a canary secret written into query text in every
//! way never survives redaction, and never reaches a lowered plan's
//! `Display`, `Debug` or canonical encoding.

use krishiv_plan::{Plan, SourceId};
use krishiv_sql::{ParserOptions, parse_and_lower, redact_sql};

fn lower(sql: &str) -> Result<Plan, krishiv_sql::ParseError> {
    parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
}

const CANARY: &str = "Canary7f3a9c";

/// Every way a secret is written into query text; `CANARY` (or the digits
/// `70319753`) is the secret.
const SECRETS: [&str; 28] = [
    "SELECT a FROM t WITH ('password' = 'Canary7f3a9c', 'path' = 's3a://b/k')",
    "SELECT a FROM t WITH (fs.s3a.secret.key = 'Canary7f3a9c', other = 1)",
    "SELECT a FROM t WITH (token 'Canary7f3a9c')",
    "SELECT a FROM t WITH (`api.Key` = R'Canary7f3a9c')",
    "SELECT a FROM t WITH ('url' = 'jdbc:postgresql://u:Canary7f3a9c@h/db?ssl=true')",
    "SELECT * FROM parquet.`s3a://AKIA:Canary7f3a9c@bucket/k`",
    "SELECT 'https://u:Canary7f3a9c@h/x?sig=Canary7f3a9c' AS s FROM t",
    "SELECT a FROM t -- password=Canary7f3a9c",
    "SELECT a /* token: Canary7f3a9c */ FROM t",
    "SELECT a FROM t WITH ('password' = 'Canary7f3a9c",
    "SELECT a FROM t WITH ('password' = $s$Canary7f3a9c$s$)",
    "SELECT 'https://u:pa\\\\Canary7f3a9c@h/' AS s FROM t",
    "SELECT * FROM parquet.`s3a://AKIA:Canary7f3a9c@b/k",
    // Adjacent literals concatenate: every part is the value.
    "SELECT a FROM t WITH ('password' = 'x' 'Canary7f3a9c')",
    "SELECT a FROM t WITH ('password' 'x' 'Canary7f3a9c', 'k' = 'v')",
    // Signs and numbers, unquoted words, booleans.
    "SELECT a FROM t WITH (password = -70319753)",
    "SELECT a FROM t WITH (password = +70319753)",
    "SELECT a FROM t WITH (password = Canary7f3a9c)",
    "SELECT a FROM t WITH (password Canary7f3a9c)",
    // Escaped and quoted keys.
    "SELECT a FROM t WITH ('pass\\u0077ord' = 'Canary7f3a9c')",
    "SELECT a FROM t WITH (\"password\" = 'Canary7f3a9c')",
    "SELECT a FROM t WITH (`pass``word` = 'Canary7f3a9c', `secret` 'Canary7f3a9c')",
    // SET statements carry raw values.
    "SET spark.hadoop.fs.s3a.secret.key=Canary7f3a9c",
    "SET spark.hadoop.fs.s3a.secret.key = 'Canary7f3a9c' ;",
    "CREATE TABLE x USING parquet OPTIONS (password 'Canary7f3a9c') TBLPROPERTIES ('token' = 'Canary7f3a9c')",
    "SELECT a FROM t WITH (password = 'x' 'Canary7f3a9c'",
    // An escape that only decodes into the URL delimiters.
    "SELECT 'https://u:Canary7f3a9c\\u0040h/' AS s FROM t",
    "SELECT \"https://u:Canary7f3a9c\"\"@h/\" AS s FROM t",
];

fn leaks(text: &str) -> bool {
    text.contains(CANARY) || text.contains("70319753")
}

/// A doubled quote inside a key part reads as `pwd` / `password` when the
/// legacy consecutive-literal mode drops it; the text redactor has no parser
/// options, so each reading is classified on its own and any sensitive one
/// hides the value (the lowered plan uses the real options, where the key is
/// `p'wd`).
#[test]
fn a_doubled_quote_in_a_key_part_cannot_hide_a_secret_in_text() {
    for sql in [
        "SELECT a FROM t WITH ('p''wd' = 'Canary7f3a9c')",
        "SELECT a FROM t WITH ('pass''word' = 'Canary7f3a9c')",
    ] {
        let r = redact_sql(sql);
        assert!(!leaks(r.as_str()), "{sql} -> {}", r.as_str());
        assert!(r.as_str().contains("*********(redacted)"), "{sql}");
    }
}

#[test]
fn query_text_never_holds_the_canary() {
    for sql in SECRETS {
        let r = redact_sql(sql);
        for text in [r.as_str().to_owned(), r.to_string(), format!("{r:?}")] {
            assert!(!leaks(&text), "{sql} -> {text}");
        }
        assert!(r.as_str().contains("*********(redacted)"), "{sql}");
        for at in 0..sql.len() {
            assert!(!leaks(r.fragment(at..sql.len())), "{sql} @ {at}");
        }
    }
    // Only secrets are replaced; references and other options stay.
    let reference = "SELECT a FROM t WITH ('password' = secret('scope', 'Canary-ref'), 'x' = 'y')";
    assert_eq!(redact_sql(reference).as_str(), reference);
    let plain = "SELECT 'Canary7f3a9c' AS s FROM t WITH ('path' = 's3a://b/k', n = -1)";
    assert_eq!(redact_sql(plain).as_str(), plain);
    let reset = "RESET spark.hadoop.fs.s3a.secret.key";
    assert_eq!(redact_sql(reset).as_str(), reset);
    assert_eq!(
        redact_sql("SELECT a FROM t WITH ('password' = 'x' 'y', 'k' = 'v')").as_str(),
        "SELECT a FROM t WITH ('password' = *********(redacted), 'k' = 'v')"
    );
}

/// Parsing, lowering and their errors (message, parameters, `Debug`) never
/// hold a secret, wherever it is written; nor does a lowered plan.
#[test]
fn errors_and_lowered_plans_never_hold_the_canary() {
    let broken = [
        "SELECT a FROM t WITH (password = Canary7f3a9c)",
        "SELECT a FROM t WITH (password = Canary7f3a9c Canary7f3a9c)",
        "SELECT a FROM t WITH (password = 'x' Canary7f3a9c)",
        "SELECT a FROM t WITH (token = 70319753 70319753)",
        "SELECT a FROM t WITH (password = -Canary7f3a9c)",
    ];
    for sql in SECRETS.iter().chain(broken.iter()) {
        match lower(sql) {
            Ok(plan) => {
                for out in [
                    format!("{plan:?}"),
                    plan.root().to_string(),
                    String::from_utf8_lossy(&plan.canonical_encoding()).into_owned(),
                ] {
                    assert!(!leaks(&out), "{sql}: {out}");
                }
            }
            Err(e) => {
                let params = format!("{:?}", e.error.message_parameters());
                for out in [e.to_string(), format!("{e:?}"), params] {
                    assert!(!leaks(&out), "{sql}: {out}");
                }
                let fragment = redact_sql(sql).fragment(e.span.range()).to_owned();
                assert!(!leaks(&fragment), "{sql}: {fragment}");
            }
        }
    }
}

/// The key argument of the AES functions is a secret in query text, however
/// the call is written; the other arguments stay.
#[test]
fn an_aes_key_argument_is_redacted_in_query_text() {
    for sql in [
        "SELECT aes_encrypt(1, 'Canary7f3a9c')",
        "SELECT AES_DECRYPT(unhex('00'), 'Canary7f3a9c', 'GCM')",
        "SELECT try_aes_decrypt(f(a, b), 'Canary7f3a9c' || x, 'GCM')",
        "SELECT builtin.aes_encrypt(1, Canary7f3a9c)",
        "SELECT `aes_encrypt`(1, 'Canary7f3a9c'",
        "SELECT aes_encrypt(DISTINCT 1, 'Canary7f3a9c')",
        "SELECT aes_encrypt(map('a', 'S')::MAP<STRING,STRING>['a'], 'Canary7f3a9c', 'XYZ')",
        "SELECT aes_encrypt(x::STRUCT<a: ARRAY<INT>, b: INT>.b, 'Canary7f3a9c')",
    ] {
        let r = redact_sql(sql);
        for at in 0..sql.len() {
            assert!(!leaks(r.fragment(at..sql.len())), "{sql} @ {at}");
        }
        assert!(!leaks(r.as_str()), "{sql} -> {}", r.as_str());
    }
    assert_eq!(
        redact_sql("SELECT aes_encrypt(f(1, 2), 'k', 'GCM')").as_str(),
        "SELECT aes_encrypt(f(1, 2), *********(redacted), 'GCM')"
    );
    let plain = "SELECT aes_encrypt(1) , other(1, 'Canary7f3a9c')";
    assert_eq!(redact_sql(plain).as_str(), plain);
}

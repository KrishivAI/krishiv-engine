// SPDX-License-Identifier: Apache-2.0
//! Query-text redaction: a canary secret written into query text in every
//! way never survives redaction, and never reaches a lowered plan's
//! `Display`, `Debug` or canonical encoding.

use krishiv_plan::{Plan, SourceId};
use krishiv_sql::{ParserOptions, parse_and_lower, redact_sql};

fn lower(sql: &str) -> Result<Plan, krishiv_sql::ParseError> {
    parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
}

const CANARY: &str = "Canary-7f3a9c";

/// Every way a secret is written into query text.
const SECRETS: [&str; 13] = [
    "SELECT a FROM t WITH ('password' = 'Canary-7f3a9c', 'path' = 's3a://b/k')",
    "SELECT a FROM t WITH (fs.s3a.secret.key = 'Canary-7f3a9c', other = 1)",
    "SELECT a FROM t WITH (token 'Canary-7f3a9c')",
    "SELECT a FROM t WITH (`api.Key` = R'Canary-7f3a9c')",
    "SELECT a FROM t WITH ('url' = 'jdbc:postgresql://u:Canary-7f3a9c@h/db?ssl=true')",
    "SELECT * FROM parquet.`s3a://AKIA:Canary-7f3a9c@bucket/k`",
    "SELECT 'https://u:Canary-7f3a9c@h/x?sig=Canary-7f3a9c' AS s FROM t",
    "SELECT a FROM t -- password=Canary-7f3a9c",
    "SELECT a /* token: Canary-7f3a9c */ FROM t",
    "SELECT a FROM t WITH ('password' = 'Canary-7f3a9c",
    "SELECT a FROM t WITH ('password' = $s$Canary-7f3a9c$s$)",
    "SELECT 'https://u:pa\\\\Canary-7f3a9c@h/' AS s FROM t",
    "SELECT * FROM parquet.`s3a://AKIA:Canary-7f3a9c@b/k",
];

#[test]
fn query_text_never_holds_the_canary() {
    for sql in SECRETS {
        let r = redact_sql(sql);
        for text in [r.as_str().to_owned(), r.to_string(), format!("{r:?}")] {
            assert!(!text.contains(CANARY), "{sql} -> {text}");
        }
        assert!(r.as_str().contains("*********(redacted)"), "{sql}");
        for at in 0..sql.len() {
            assert!(!r.fragment(at..sql.len()).contains(CANARY), "{sql} @ {at}");
        }
    }
    // Only secrets are replaced; references stay.
    let reference = "SELECT a FROM t WITH ('password' = secret('scope', 'Canary-ref'), 'x' = 'y')";
    assert_eq!(redact_sql(reference).as_str(), reference);
    let plain = "SELECT 'Canary-7f3a9c' AS s FROM t WITH ('path' = 's3a://b/k')";
    assert_eq!(redact_sql(plain).as_str(), plain);
}

#[test]
fn lowered_plans_never_hold_the_canary() {
    for sql in [SECRETS[0], SECRETS[1], SECRETS[4], SECRETS[5]] {
        let plan = lower(sql).unwrap();
        for out in [
            format!("{plan:?}"),
            plan.root().to_string(),
            String::from_utf8_lossy(&plan.canonical_encoding()).into_owned(),
        ] {
            assert!(!out.contains(CANARY), "{sql}: {out}");
        }
    }
    // An error context over a secret shows it redacted.
    let sql = "SELECT a FROM t WITH ('password' = 'Canary-7f3a9c'";
    let e = lower(sql).unwrap_err();
    let fragment = redact_sql(sql).fragment(e.span.range()).to_owned();
    assert!(!fragment.contains(CANARY));
}

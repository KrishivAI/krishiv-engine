// SPDX-License-Identifier: Apache-2.0
//! ADR 0038 canary tests: a known secret, injected through every path an
//! option can take, never appears in `Debug`, `Display` (EXPLAIN text), the
//! canonical encoding or an error the IR raises.

use std::collections::BTreeMap;

use krishiv_common::secret::OptionValue;

use crate::{Expr, LogicalPlan, MAX_DEPTH, Plan, PlanNode, ReadSource, Scan};

const CANARY: &str = "Canary-7f3a9c";

/// Options carrying the canary as a key-classified secret, in a URL's
/// userinfo and parameters, in signed-URL parameters, and a reference.
fn options(secret: &str) -> BTreeMap<String, OptionValue> {
    [
        ("Password", secret.to_owned()),
        ("fs.s3a.secret.key", secret.to_owned()),
        (
            "url",
            format!("jdbc:postgresql://user:{secret}@h/db?password={secret}&ssl=true"),
        ),
        (
            "path",
            format!("https://b.s3.amazonaws.com/k?X-Amz-Signature={secret}&X-Amz-Expires=60"),
        ),
        (
            "location",
            format!("https://a.blob.core.windows.net/c?sv=1&sig={secret}"),
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), OptionValue::from_user(k, v)))
    .chain([("token".to_owned(), OptionValue::reference("prod", "pg"))])
    .collect()
}

fn read(secret: &str) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "parquet".to_owned(),
        // A presigned URL as a path: paths are typed like options.
        paths: vec![OptionValue::from_user(
            "path",
            format!("https://b.s3.amazonaws.com/k?X-Amz-Signature={secret}&X-Amz-Expires=60"),
        )],
        options: options(secret),
    }))
}

fn scan(secret: &str) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Scan(Scan {
        source: ReadSource::Table {
            name: vec!["t".to_owned()],
            options: options(secret),
        },
        output: Vec::new(),
        version: None,
    }))
}

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

#[test]
fn the_canary_never_leaks() {
    let plan = Plan::try_new(read(CANARY).join(
        scan(CANARY),
        crate::JoinKind::Inner,
        crate::JoinCondition::None,
    ))
    .unwrap();
    let explain = plan.root().to_string();
    assert!(!explain.contains(CANARY), "{explain}");
    assert!(explain.contains(
        "Read parquet [https://b.s3.amazonaws.com/k?X-Amz-Signature=*********(redacted)&X-Amz-Expires=60]"
    ));
    assert!(explain.contains(
        "url=jdbc:postgresql://user:*********(redacted)@h/db?password=*********(redacted)&ssl=true"
    ));
    let debug = format!("{plan:?} {plan:#?}");
    assert!(!debug.contains(CANARY), "{debug}");
    assert!(debug.contains("ssl=true") && debug.contains("Reference"));
    let shown: Vec<String> = options(CANARY).values().map(ToString::to_string).collect();
    assert_eq!(
        shown,
        [
            "*********(redacted)",
            "*********(redacted)",
            "https://a.blob.core.windows.net/c?sv=1&sig=*********(redacted)",
            "https://b.s3.amazonaws.com/k?X-Amz-Signature=*********(redacted)&X-Amz-Expires=60",
            "secret('prod', 'pg')",
            "jdbc:postgresql://user:*********(redacted)@h/db?password=*********(redacted)&ssl=true",
        ]
    );
    assert!(!contains(&plan.canonical_encoding(), CANARY));
    assert!(!plan.structural_hash().to_string().contains(CANARY));
}

#[test]
fn ir_errors_never_hold_the_canary() {
    let deep = (0..MAX_DEPTH).fold(read(CANARY), |p, _| p.filter(Expr::column(["a"])));
    let err = Plan::try_new(deep).unwrap_err();
    let text = format!("{err} {err:?}");
    assert!(!text.contains(CANARY), "{text}");
}

/// Inline secrets are keyed by fingerprint: equal secrets give equal keys
/// (within the process), different ones different keys.
#[test]
fn inline_secrets_are_part_of_the_key() {
    let key = |s: &str| Plan::try_new(read(s)).unwrap().structural_hash();
    assert_eq!(key(CANARY), key(CANARY));
    assert_ne!(key(CANARY), key("Canary-7f3a9d"));
    let plain = |v: &str| {
        let mut options = BTreeMap::new();
        options.insert("k".to_owned(), OptionValue::Plain(v.to_owned()));
        options
    };
    let with = |options| {
        Plan::try_new(LogicalPlan::new(PlanNode::Read(ReadSource::Table {
            name: vec!["t".to_owned()],
            options,
        })))
        .unwrap()
        .canonical_encoding()
    };
    // A plain value is encoded as itself, as before ADR 0038.
    assert!(contains(&with(plain("visible-value")), "visible-value"));
    assert_ne!(with(plain("a")), with(plain("b")));
}

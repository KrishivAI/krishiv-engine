// SPDX-License-Identifier: Apache-2.0

use super::{
    OptionValue, SecretString, SecretUrl, SecretValue, UrlPart, ct_eq, is_sensitive_key, zero,
};

const CANARY: &str = "Canary-7f3a9c";

fn shown(v: &OptionValue) -> String {
    format!("{v} {v:?} {v:#?}")
}

#[test]
fn sensitive_keys() {
    for key in [
        "password",
        "PASSWORD",
        "ssl.keystore.password",
        "fs.s3a.access.key",
        "fs.s3a.secret.key",
        "fs.s3a.session.token",
        "fs.azure.account.key.acct.dfs.core.windows.net",
        "fs.gs.auth.service.account.private.key",
        "sasl.jaas.config",
        "apiKey",
        "api_key",
        "accessKeyId",
        "awsSecretAccessKey",
        "pwd",
        "sas",
        "azure.sas",
        "credentials",
        "ssl.keystore.key",
        "Authorization",
    ] {
        assert!(is_sensitive_key(key), "{key}");
    }
    for key in [
        "url",
        "dbtable",
        "user",
        "header",
        "inferSchema",
        "sasl.mechanism",
        "path",
        "keyspace",
        "primaryKey",
        "passthrough",
    ] {
        assert!(!is_sensitive_key(key), "{key}");
    }
}

#[test]
fn sensitive_keys_are_inline_secrets() {
    let v = OptionValue::from_user("Password", CANARY);
    assert_eq!(
        v,
        OptionValue::Secret(SecretValue::Inline(SecretString::new(CANARY)))
    );
    assert_eq!(
        shown(&v),
        "*********(redacted) Secret(Inline(*********(redacted))) Secret(\n    Inline(\n        *********(redacted),\n    ),\n)"
    );
}

#[test]
fn plain_values_stay_plain() {
    let v = OptionValue::from_user("header", "true");
    assert_eq!(v.as_plain(), Some("true"));
    assert_eq!(v.to_string(), "true");
    let url = OptionValue::from_user("url", "jdbc:postgresql://h:5432/db?ssl=true");
    assert_eq!(url.as_plain(), Some("jdbc:postgresql://h:5432/db?ssl=true"));
}

/// URL credentials, as `(input, printed)`.
const URLS: &[(&str, &str)] = &[
    (
        "jdbc:postgresql://user:Canary-7f3a9c@h/db?password=Canary-7f3a9c&ssl=true",
        "jdbc:postgresql://user:*********(redacted)@h/db?password=*********(redacted)&ssl=true",
    ),
    (
        "https://acct.blob.core.windows.net/c/f?sv=2022-11-02&se=2026-01-01&sp=r&sig=Canary-7f3a9c",
        "https://acct.blob.core.windows.net/c/f?sv=2022-11-02&se=2026-01-01&sp=r&sig=*********(redacted)",
    ),
    (
        "https://b.s3.amazonaws.com/k?X-Amz-Credential=Canary-7f3a9c&X-Amz-Signature=Canary-7f3a9c&X-Amz-Security-Token=Canary-7f3a9c&X-Amz-Expires=60",
        "https://b.s3.amazonaws.com/k?X-Amz-Credential=*********(redacted)&X-Amz-Signature=*********(redacted)&X-Amz-Security-Token=*********(redacted)&X-Amz-Expires=60",
    ),
    (
        "https://storage.googleapis.com/b/o?X-Goog-Credential=Canary-7f3a9c&X-Goog-Signature=Canary-7f3a9c",
        "https://storage.googleapis.com/b/o?X-Goog-Credential=*********(redacted)&X-Goog-Signature=*********(redacted)",
    ),
    (
        "jdbc:sqlserver://h:1433;databaseName=d;user=u;password=Canary-7f3a9c",
        "jdbc:sqlserver://h:1433;databaseName=d;user=u;password=*********(redacted)",
    ),
    (
        "jdbc:oracle:thin:scott/Canary-7f3a9c@//h:1521/svc",
        "jdbc:oracle:thin:scott/*********(redacted)@//h:1521/svc",
    ),
    (
        "https://u:Canary-7f3a9c@h/p#access_token=Canary-7f3a9c&state=1",
        "https://u:*********(redacted)@h/p#access_token=*********(redacted)&state=1",
    ),
    (
        "https://h/p?%74oken=Canary-7f3a9c&pwd=Canary-7f3a9c",
        "https://h/p?%74oken=*********(redacted)&pwd=*********(redacted)",
    ),
    (
        "mongodb://u:p@ss:Canary-7f3a9c@h1,h2/db",
        "mongodb://u:*********(redacted)@h1,h2/db",
    ),
    // An `@` inside a `;` parameter is not a userinfo end.
    (
        "jdbc:sqlserver://h:1433;user=u;password=p@ssCanary-7f3a9c",
        "jdbc:sqlserver://h:1433;user=u;password=*********(redacted)",
    ),
    (
        "jdbc:derby:db;user=u@corp;password=p@s:sCanary-7f3a9c",
        "jdbc:derby:db;user=u@corp;password=*********(redacted)",
    ),
    (
        "jdbc:sqlserver://h;accessToken=Canary-7f3a9c;encrypt=true",
        "jdbc:sqlserver://h;accessToken=*********(redacted);encrypt=true",
    ),
    // SQL Server quotes `;` and `}` in values with braces (`}}` is `}`).
    (
        "jdbc:sqlserver://h;password={p;w}}Canary-7f3a9c=;x};user=u",
        "jdbc:sqlserver://h;password=*********(redacted);user=u",
    ),
    (
        "jdbc:sqlserver://h;password={unterminated;Canary-7f3a9c",
        "jdbc:sqlserver://h;password=*********(redacted)",
    ),
    // A base64 key holds `/` and `:`; the last `@` before parameters ends
    // the userinfo.
    (
        "s3a://AKIA:ab/c:d@Canary-7f3a9c@bucket/k?versionId=1",
        "s3a://AKIA:*********(redacted)@bucket/k?versionId=1",
    ),
    (
        "jdbc:oracle:thin:scott/ti:g@rCanary-7f3a9c@//h:1521/svc;x=1",
        "jdbc:oracle:thin:scott/*********(redacted)@//h:1521/svc;x=1",
    ),
];

#[test]
fn url_credentials_are_lifted_out() {
    for &(input, printed) in URLS {
        let v = OptionValue::from_user("url", input);
        assert_eq!(v.to_string(), printed);
        // Messages name the expected text only, never the value.
        assert!(!shown(&v).contains(CANARY), "{printed}");
        let OptionValue::Url(url) = &v else {
            panic!("{printed}: not a URL");
        };
        assert_eq!(url.expose_secret().expose_secret(), input);
        assert!(url.parts().iter().any(|p| matches!(p, UrlPart::Secret(_))));
    }
}

#[test]
fn non_urls_have_no_url_parts() {
    for s in [
        "/data/x?password=y",
        "a://",
        "1http://u:p@h",
        "jdbc:h2:mem:",
    ] {
        assert!(SecretUrl::parse(s).is_none(), "{s}");
    }
}

#[test]
fn references_print_their_name() {
    let v = OptionValue::reference("prod", "pg'pw");
    assert_eq!(v.to_string(), r"secret('prod', 'pg\'pw')");
    assert_ne!(v, OptionValue::reference("prod", "other"));
}

#[test]
fn equality_and_fingerprints() {
    let (a, b) = (SecretString::new(CANARY), SecretString::new(CANARY));
    let c = SecretString::new("Canary-7f3a9d");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(a, SecretString::new("short"));
    assert_eq!(a.fingerprint(), b.fingerprint());
    assert_ne!(a.fingerprint(), c.fingerprint());
    assert!(ct_eq(b"", b""));
    assert!(!ct_eq(b"a", b"b"));
}

#[test]
fn zeroing_covers_the_whole_allocation() {
    let mut bytes = Vec::with_capacity(64);
    bytes.extend_from_slice(CANARY.as_bytes());
    let capacity = bytes.capacity();
    zero(&mut bytes);
    assert_eq!(bytes.len(), capacity);
    assert!(bytes.iter().all(|&b| b == 0));
}

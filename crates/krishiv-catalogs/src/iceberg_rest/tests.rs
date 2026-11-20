// SPDX-License-Identifier: Apache-2.0
//! End-to-end tests of [`RestClient`] against the fake server: the calls and their wire forms, the token sources,
//! and the hostile-catalog tests of design P3-lakehouse §5.6 for PR-17 (`catalog_location_confined`,
//! `catalog_parse_limits`, `catalog_redirect_not_followed`, `catalog_request_timeout`).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::SecretString;
use url::Url;

use super::test_server::{Reply, Seen, Server, serve};
use super::{RestAuth, RestClient, RestConfig};
use crate::location::RemoteLocationPolicy;

const CONFIG: &str = r#"{"defaults": {}, "overrides": {}}"#;

fn table_json(location: &str) -> String {
    format!(
        r#"{{"metadata-location": "{location}/metadata/v1.metadata.json",
            "metadata": {{"format-version": 2, "table-uuid": "u", "location": "{location}"}},
            "config": {{}}}}"#
    )
}

fn config_for(server: &Server) -> RestConfig {
    let mut c = RestConfig::new(&server.base).unwrap();
    c.allow_http = true;
    c
}

fn client_with(config: RestConfig) -> RestClient {
    RestClient::new("rest", config, Arc::new(RemoteLocationPolicy::default())).unwrap()
}

fn ns(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).to_owned()).collect()
}

fn path(seen: &Seen) -> &str {
    seen.target.split('?').next().unwrap_or("")
}

fn class(e: &KrishivError) -> ErrorClass {
    e.error_class()
}

#[test]
fn config_then_listings_and_load_use_the_prefix_separator_and_token() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(
            200,
            r#"{"defaults": {"prefix": "wrong"}, "overrides": {"prefix": "cat"}}"#.into(),
        ),
        "/v1/cat/namespaces" => Reply::Json(200, r#"{"namespaces": [["a"], ["a b"]]}"#.into()),
        "/v1/cat/namespaces/a%1Fb%20c/tables" => Reply::Json(
            200,
            r#"{"identifiers": [{"namespace": ["a", "b c"], "name": "t x"}]}"#.into(),
        ),
        "/v1/cat/namespaces/a%1Fb%20c/tables/t%20x" => {
            Reply::Json(200, table_json("s3://bucket/wh/t"))
        }
        _ => Reply::Json(
            404,
            r#"{"error": {"message": "no", "type": "NotFoundException", "code": 404}}"#.into(),
        ),
    });
    let mut config = config_for(&server);
    config.warehouse = Some("wh one".into());
    config.auth = RestAuth::Token(SecretString::new("static-token"));
    config.vended_credentials = true;
    let client = client_with(config);

    assert_eq!(
        client.list_namespaces(&[]).unwrap().unwrap(),
        vec![ns(&["a"]), ns(&["a b"])]
    );
    let tables = client.list_tables(&ns(&["a", "b c"])).unwrap().unwrap();
    assert_eq!(
        (tables[0].namespace.clone(), tables[0].name.as_str()),
        (ns(&["a", "b c"]), "t x")
    );
    let t = client
        .load_table(&ns(&["a", "b c"]), "t x")
        .unwrap()
        .unwrap();
    assert_eq!(t.location, "s3://bucket/wh/t");
    assert_eq!(t.format_version, 2);
    assert!(t.metadata.get("table-uuid").is_some());

    let seen = server.seen();
    assert_eq!(seen[0].target, "/v1/config?warehouse=wh+one");
    assert_eq!(
        seen.iter().filter(|s| path(s) == "/v1/config").count(),
        1,
        "config is fetched once"
    );
    for s in &seen {
        assert_eq!(s.method, "GET");
        assert_eq!(s.header("authorization"), Some("Bearer static-token"));
        assert_eq!(s.header("accept-encoding"), Some("identity"));
    }
    let load = seen.last().unwrap();
    assert_eq!(
        load.header("x-iceberg-access-delegation"),
        Some("vended-credentials")
    );
    assert_eq!(seen[1].header("x-iceberg-access-delegation"), None);
}

#[test]
fn parent_namespaces_and_a_custom_separator() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(
            200,
            r#"{"defaults": {}, "overrides": {"namespace-separator": "%2E"}}"#.into(),
        ),
        "/v1/namespaces" => Reply::Json(200, r#"{"namespaces": [["a", "b", "c"]]}"#.into()),
        "/v1/namespaces/a%2Eb/tables" => Reply::Json(200, r#"{"identifiers": []}"#.into()),
        _ => Reply::Json(404, "{}".into()),
    });
    let client = client_with(config_for(&server));
    assert_eq!(
        client.list_namespaces(&ns(&["a", "b"])).unwrap().unwrap(),
        vec![ns(&["a", "b", "c"])]
    );
    assert_eq!(
        client.list_tables(&ns(&["a", "b"])).unwrap().unwrap(),
        vec![]
    );
    let seen = server.seen();
    assert_eq!(seen[1].target, "/v1/namespaces?parent=a.b");
}

#[test]
fn missing_names_are_none() {
    let no_such =
        |kind: &str| format!(r#"{{"error": {{"message": "m", "type": "{kind}", "code": 404}}}}"#);
    let (table, namespace) = (
        no_such("NoSuchTableException"),
        no_such("NoSuchNamespaceException"),
    );
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        "/v1/namespaces/db/tables/t" => Reply::Json(404, table.clone()),
        _ => Reply::Json(404, namespace.clone()),
    });
    let client = client_with(config_for(&server));
    assert!(client.load_table(&ns(&["db"]), "t").unwrap().is_none());
    assert!(client.list_tables(&ns(&["db"])).unwrap().is_none());
    assert!(client.list_namespaces(&ns(&["db"])).unwrap().is_none());

    // The `NoSuch*` answer must fit the call: "no such table" does not prove a namespace absent.
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(
            404,
            r#"{"error": {"message": "m", "type": "NoSuchTableException", "code": 404}}"#.into(),
        ),
    });
    let client = client_with(config_for(&server));
    assert!(client.load_table(&ns(&["db"]), "t").unwrap().is_none());
    let e = client.list_tables(&ns(&["db"])).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);

    // A 404 that is not the catalog's `NoSuch*` answer (a proxy, a wrong prefix) is no proof of absence.
    for body in [
        "{}",
        "<html>not found</html>",
        r#"{"error": {"type": "NotFoundException"}}"#,
    ] {
        let server = serve(move |r| match path(r) {
            "/v1/config" => Reply::Json(200, CONFIG.into()),
            _ => Reply::Json(404, body.into()),
        });
        let client = client_with(config_for(&server));
        let e = client.load_table(&ns(&["db"]), "t").unwrap_err();
        assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure, "{body}");
    }
}

#[test]
fn dot_and_empty_names_are_refused_before_any_request() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, table_json("s3://b/t")),
    });
    let client = client_with(config_for(&server));
    client.catalog_config().unwrap();
    for (namespace, table) in [
        (ns(&["db"]), ".."),
        (ns(&["db"]), "."),
        (ns(&["db"]), ""),
        (ns(&[".."]), "t"),
        (ns(&["."]), "t"),
        (ns(&[""]), "t"),
        (ns(&["a", ".."]), "t"),
    ] {
        let e = client.load_table(&namespace, table).unwrap_err();
        assert_eq!(
            class(&e),
            ErrorClass::KrishivCatalogFailure,
            "{namespace:?} {table}"
        );
        if namespace != ["db"] {
            assert!(client.list_tables(&namespace).is_err(), "{namespace:?}");
        }
    }
    assert!(client.list_namespaces(&ns(&[".."])).is_err());
    assert!(client.list_tables(&[]).is_err());
    assert!(client.load_table(&[], "t").is_err());
    assert_eq!(server.seen().len(), 1, "only the config was requested");
}

#[test]
fn listings_follow_page_tokens() {
    let server = serve(|r| match r.target.as_str() {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        "/v1/namespaces/db/tables" => Reply::Json(
            200,
            r#"{"identifiers": [{"namespace": ["db"], "name": "a"}], "next-page-token": "p 2"}"#
                .into(),
        ),
        "/v1/namespaces/db/tables?pageToken=p+2" => Reply::Json(
            200,
            r#"{"identifiers": [{"namespace": ["db"], "name": "b"}], "next-page-token": null}"#
                .into(),
        ),
        _ => Reply::Json(404, "{}".into()),
    });
    let client = client_with(config_for(&server));
    let names: Vec<_> = client
        .list_tables(&ns(&["db"]))
        .unwrap()
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(names, ["a", "b"]);
}

#[test]
fn a_uri_override_from_the_server_is_ignored() {
    let elsewhere = serve(|_| Reply::Json(200, CONFIG.into()));
    let override_json = format!(
        r#"{{"defaults": {{}}, "overrides": {{"uri": "{}"}}}}"#,
        elsewhere.base
    );
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, override_json.clone()),
        "/v1/namespaces" => Reply::Json(200, r#"{"namespaces": []}"#.into()),
        _ => Reply::Json(404, "{}".into()),
    });
    let mut config = config_for(&server);
    config.auth = RestAuth::Token(SecretString::new("the-callers-token"));
    let client = client_with(config);
    client.list_namespaces(&[]).unwrap();
    assert_eq!(server.seen().len(), 2);
    assert!(
        elsewhere.seen().is_empty(),
        "the token went to the overriding host"
    );
}

#[test]
fn an_unsafe_prefix_or_separator_is_a_catalog_fault() {
    for overrides in [
        r#"{"prefix": "../admin"}"#,
        r#"{"prefix": "a/../../x"}"#,
        r#"{"prefix": "x?y=1"}"#,
        r#"{"namespace-separator": "/"}"#,
    ] {
        let body = format!(r#"{{"defaults": {{}}, "overrides": {overrides}}}"#);
        let server = serve(move |_| Reply::Json(200, body.clone()));
        let client = client_with(config_for(&server));
        let e = client.list_namespaces(&[]).unwrap_err();
        assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure, "{overrides}");
        assert_eq!(server.seen().len(), 1, "{overrides}");
    }
}

#[test]
fn catalog_location_confined() {
    for (location, scope) in [
        ("file:///etc", "s3://b/t"),
        ("http://169.254.169.254/latest", "s3://b/t"),
        ("s3://key:secret@b/t", "s3://b/t"),
        ("s3://b/t", "file:///"),
        ("s3://b/t", "file"),
        ("s3://b/t", "http"),
        ("s3://b/t", "s3:"),
        ("s3://b/t", "b/t"),
        ("s3://b/wh/../../other", "s3://b/t"),
    ] {
        let body = format!(
            r#"{{"metadata": {{"format-version": 2, "location": "{location}"}},
                "storage-credentials": [{{"prefix": "{scope}", "config": {{"s3.secret-access-key": "SK"}}}}]}}"#
        );
        let server = serve(move |r| match path(r) {
            "/v1/config" => Reply::Json(200, CONFIG.into()),
            _ => Reply::Json(200, body.clone()),
        });
        let client = client_with(config_for(&server));
        let e = client.load_table(&ns(&["db"]), "t").unwrap_err();
        assert_eq!(
            class(&e),
            ErrorClass::KrishivCatalogLocationRefused,
            "{location} {scope}"
        );
        let text = e.to_string();
        assert!(
            text.contains("`rest`") && text.contains("`db`.`t`"),
            "{text}"
        );
        assert!(!text.contains(location) && !text.contains("SK"), "{text}");
        assert_eq!(
            server.seen().len(),
            2,
            "nothing is requested after the refusal"
        );
    }
}

fn limit_error(
    reply: impl Fn() -> Reply + Send + 'static,
    limits: impl FnOnce(&mut RestConfig),
) -> KrishivError {
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => reply(),
    });
    let mut config = config_for(&server);
    limits(&mut config);
    client_with(config)
        .load_table(&ns(&["db"]), "t")
        .unwrap_err()
}

#[test]
fn catalog_parse_limits() {
    // Oversize metadata, declared up front.
    let big = table_json("s3://b/t") + &" ".repeat(5000);
    let e = limit_error(
        move || Reply::Json(200, big.clone()),
        |c| c.limits.max_response_bytes = 4096,
    );
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);
    assert!(
        e.to_string().contains("krishiv.catalog.maxMetadataBytes"),
        "{e}"
    );

    // Oversize without a length (read until close).
    let raw = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}{}",
        table_json("s3://b/t"),
        " ".repeat(5000)
    );
    let e = limit_error(
        move || Reply::Raw(raw.clone().into_bytes()),
        |c| c.limits.max_response_bytes = 4096,
    );
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);

    // A gzip bomb: small on the wire, large decompressed. It is never decompressed: a compressed body is refused.
    let gz: &[u8] = &[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
    let mut raw = b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 10\r\nConnection: close\r\n\r\n".to_vec();
    raw.extend_from_slice(gz);
    let e = limit_error(move || Reply::Raw(raw.clone()), |_| {});
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);

    // Depth 65.
    let deep = format!(
        r#"{{"metadata": {{"format-version": 2, "location": "s3://b/t", "x": {}{}}}}}"#,
        "[".repeat(63),
        "]".repeat(63)
    );
    let e = limit_error(move || Reply::Json(200, deep.clone()), |_| {});
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);
    assert!(e.to_string().contains("nesting depth"), "{e}");
    let ok = format!(
        r#"{{"metadata": {{"format-version": 2, "location": "s3://b/t", "x": {}{}}}}}"#,
        "[".repeat(62),
        "]".repeat(62)
    );
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, ok.clone()),
    });
    client_with(config_for(&server))
        .load_table(&ns(&["db"]), "t")
        .unwrap()
        .unwrap();
}

#[test]
fn listing_limits_and_a_repeated_page_token() {
    let page = r#"{"identifiers": [{"namespace": ["db"], "name": "a"}, {"namespace": ["db"], "name": "b"}], "next-page-token": "same"}"#;
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, page.into()),
    });
    let e = client_with(config_for(&server))
        .list_tables(&ns(&["db"]))
        .unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert_eq!(server.seen().len(), 3, "stops at the first repeat");

    let n = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&n);
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => {
            let i = count.fetch_add(1, Ordering::SeqCst);
            Reply::Json(
                200,
                format!(r#"{{"identifiers": [], "next-page-token": "t{i}"}}"#),
            )
        }
    });
    let mut config = config_for(&server);
    config.limits.max_pages = 3;
    let e = client_with(config).list_tables(&ns(&["db"])).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);

    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, page.replace(r#", "next-page-token": "same""#, "")),
    });
    let mut config = config_for(&server);
    config.limits.max_list_entries = 1;
    let e = client_with(config).list_tables(&ns(&["db"])).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);
}

#[test]
fn catalog_redirect_not_followed() {
    let elsewhere = serve(|_| Reply::Json(200, CONFIG.into()));
    let location = format!("{}/v1/config", elsewhere.base);
    let server = serve(move |_| {
        Reply::Raw(
            format!("HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .into_bytes(),
        )
    });
    let mut config = config_for(&server);
    config.auth = RestAuth::Token(SecretString::new("tok"));
    let e = client_with(config).list_namespaces(&[]).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(e.to_string().contains("`rest`"), "{e}");
    assert!(elsewhere.seen().is_empty());
}

/// A connector under the server's rule (`deny`), exempting `exempt` by name.
fn deny_connector(exempt: &[&str]) -> krishiv_store::PolicyConnector {
    use krishiv_store::egress::{EgressPolicy, PrivateAddresses, ProviderEndpoints};
    let policy = EgressPolicy::new(
        PrivateAddresses::Deny,
        Vec::new(),
        ProviderEndpoints::default(),
    );
    krishiv_store::PolicyConnector::new(
        policy.scope(exempt.iter().copied()),
        krishiv_store::Redirects::None,
        None,
    )
    .without_provider_literals()
}

#[test]
fn every_request_goes_through_the_egress_policy_client() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, r#"{"namespaces": [["a"]]}"#.into()),
    });
    // Not exempt: the loopback catalog is refused before anything is sent, the address never named.
    let mut config = config_for(&server);
    config.egress = Some(deny_connector(&[]));
    let e = client_with(config).list_namespaces(&[]).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(!e.to_string().contains("127.0.0.1"), "{e}");
    assert!(server.seen().is_empty());
    // The declared host is exempt by name: the same calls pass, token fetch included.
    let idp = serve(|_| {
        Reply::Json(
            200,
            r#"{"access_token": "t1", "token_type": "bearer"}"#.into(),
        )
    });
    let mut config = config_for(&server);
    config.egress = Some(deny_connector(&["127.0.0.1"]));
    config.auth = RestAuth::ClientCredentials {
        client_id: Some("id".into()),
        client_secret: SecretString::new("s"),
        token_endpoint: Some(Url::parse(&format!("{}/token", idp.base)).unwrap()),
        scope: None,
        extra: Vec::new(),
    };
    let names = client_with(config).list_namespaces(&[]).unwrap().unwrap();
    assert_eq!(names, vec![ns(&["a"])]);
    assert_eq!(idp.seen().len(), 1);
    // A live token endpoint the rule refuses (`localhost`: not the declared host, a loopback
    // address) is never called, so the refusal is the policy's, not a failed connection.
    let refused_idp = serve(|_| {
        Reply::Json(
            200,
            r#"{"access_token": "t2", "token_type": "bearer"}"#.into(),
        )
    });
    let mut config = config_for(&server);
    config.egress = Some(deny_connector(&["127.0.0.1"]));
    config.auth = RestAuth::ClientCredentials {
        client_id: Some("id".into()),
        client_secret: SecretString::new("s"),
        token_endpoint: Some(
            Url::parse(&format!(
                "{}/token",
                refused_idp.base.replace("127.0.0.1", "localhost")
            ))
            .unwrap(),
        ),
        scope: None,
        extra: Vec::new(),
    };
    let before = server.seen().len();
    let e = client_with(config).list_namespaces(&[]).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(
        refused_idp.seen().is_empty(),
        "the refused endpoint is never called"
    );
    assert_eq!(
        server.seen().len(),
        before,
        "nothing reaches the catalog without a token"
    );
}

#[test]
fn credential_prefixes_in_the_three_forms() {
    for prefix in ["s3://b/t", "s3://b", "s3://b/", "s3", "S3", "abfss", "gs"] {
        let body = format!(
            r#"{{"metadata": {{"format-version": 2, "location": "s3://b/t"}},
                "storage-credentials": [{{"prefix": "{prefix}", "config": {{}}}}]}}"#
        );
        let server = serve(move |r| match path(r) {
            "/v1/config" => Reply::Json(200, CONFIG.into()),
            _ => Reply::Json(200, body.clone()),
        });
        let client = client_with(config_for(&server));
        let t = client.load_table(&ns(&["db"]), "t").unwrap().unwrap();
        assert_eq!(t.storage_credentials[0].prefix, prefix);
    }
}

#[test]
fn a_paged_listing_shares_one_deadline() {
    let pages = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&pages);
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => {
            // Each page is fast (~100 ms) and names a new token, forever.
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Reply::Trickle(
                format!(r#"{{"identifiers":[],"next-page-token":"p{n:04}"}}"#),
                Duration::from_millis(2),
            )
        }
    });
    let mut config = config_for(&server);
    config.limits.request_timeout = Duration::from_millis(700);
    let client = client_with(config);
    client.catalog_config().unwrap();
    let started = Instant::now();
    let e = client.list_tables(&ns(&["db"])).unwrap_err();
    let took = started.elapsed();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(took < Duration::from_millis(2500), "{took:?}");
    assert!(
        pages.load(Ordering::SeqCst) > 1,
        "more than one page was read"
    );
}

#[test]
fn a_page_token_past_4_kib_is_refused() {
    let token = "x".repeat(4097);
    let server = serve(move |r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(
            200,
            format!(r#"{{"identifiers":[],"next-page-token":"{token}"}}"#),
        ),
    });
    let client = client_with(config_for(&server));
    let e = client.list_tables(&ns(&["db"])).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
}

#[test]
fn zero_and_absurd_limits_fail_at_startup() {
    let base = "https://h/";
    let cases: [fn(&mut RestConfig); 4] = [
        |c| c.limits.request_timeout = Duration::ZERO,
        |c| c.limits.request_timeout = Duration::from_secs(u64::MAX),
        |c| c.limits.max_pages = 0,
        |c| c.limits.max_depth = 129,
    ];
    for set in cases {
        let mut c = RestConfig::new(base).unwrap();
        set(&mut c);
        let e = RestClient::new("rest", c, Arc::new(RemoteLocationPolicy::default())).unwrap_err();
        assert_eq!(class(&e), ErrorClass::KrishivInvalidConfigValue);
    }
}

#[test]
fn catalog_request_timeout() {
    for reply in [0, 1] {
        let server = serve(move |r| match (path(r), reply) {
            ("/v1/config", _) => Reply::Json(200, CONFIG.into()),
            (_, 0) => Reply::Stall(Duration::from_secs(5)),
            // One byte every 100 ms: no single read waits long, the whole call does.
            _ => Reply::Trickle(table_json("s3://b/t"), Duration::from_millis(100)),
        });
        let mut config = config_for(&server);
        config.limits.request_timeout = Duration::from_millis(700);
        let client = client_with(config);
        client.catalog_config().unwrap();
        let started = Instant::now();
        let e = client.load_table(&ns(&["db"]), "t").unwrap_err();
        let took = started.elapsed();
        assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure, "{reply}");
        assert!(
            took >= Duration::from_millis(600) && took < Duration::from_millis(2500),
            "{reply}: {took:?}"
        );
    }
}

#[test]
fn read_restrictions_and_server_side_planning_are_refused() {
    for extra in [
        r#""read-restrictions": {"required-column-projections": [{"field-id": 1, "action": "mask-alphanum"}]}"#,
        r#""config": {"scan-planning-mode": "server"}"#,
    ] {
        let body =
            format!(r#"{{"metadata": {{"format-version": 2, "location": "s3://b/t"}}, {extra}}}"#);
        let server = serve(move |r| match path(r) {
            "/v1/config" => Reply::Json(200, CONFIG.into()),
            _ => Reply::Json(200, body.clone()),
        });
        let e = client_with(config_for(&server))
            .load_table(&ns(&["db"]), "t")
            .unwrap_err();
        assert_eq!(
            class(&e),
            ErrorClass::KrishivCatalogTableUnsupported,
            "{extra}"
        );
        assert!(e.to_string().contains("`db`.`t`"), "{e}");
    }
}

#[test]
fn errors_never_carry_the_servers_message() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(
            500,
            r#"{"error": {"message": "token=leaked", "type": "X", "code": 500}}"#.into(),
        ),
    });
    let e = client_with(config_for(&server))
        .load_table(&ns(&["db"]), "t")
        .unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(!e.to_string().contains("leaked"), "{e}");
}

#[test]
fn load_credentials_needs_the_advertised_endpoint() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(404, "{}".into()),
    });
    let e = client_with(config_for(&server))
        .load_credentials(&ns(&["db"]), "t")
        .unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);

    let server = serve(|r| {
        match path(r) {
        "/v1/config" => Reply::Json(
            200,
            r#"{"defaults": {}, "overrides": {},
                "endpoints": ["GET /v1/{prefix}/namespaces/{namespace}/tables/{table}/credentials"]}"#
                .into(),
        ),
        "/v1/namespaces/db/tables/t/credentials" => Reply::Json(
            200,
            r#"{"storage-credentials": [{"prefix": "abfss://c@a.dfs.core.windows.net/t", "config": {"adls.sas-token.a": "sv=x"}}]}"#
                .into(),
        ),
        _ => Reply::Json(404, "{}".into()),
    }
    });
    let creds = client_with(config_for(&server))
        .load_credentials(&ns(&["db"]), "t")
        .unwrap()
        .unwrap();
    assert_eq!(creds[0].config["adls.sas-token.a"].expose_secret(), "sv=x");
}

fn client_credentials(server: &Server, token_endpoint: Option<Url>) -> RestConfig {
    let mut config = config_for(server);
    config.auth = RestAuth::ClientCredentials {
        client_id: Some("app id".into()),
        client_secret: SecretString::new("s3cr&t"),
        token_endpoint,
        scope: None,
        extra: vec![("audience".into(), "https://storage.azure.com/".into())],
    };
    config
}

#[test]
fn client_credentials_fetch_reuse_and_renew_after_a_401() {
    let issued = Arc::new(AtomicUsize::new(0));
    let n = Arc::clone(&issued);
    let server = serve(move |r| match path(r) {
        "/v1/oauth/tokens" => {
            let i = n.fetch_add(1, Ordering::SeqCst);
            Reply::Json(
                200,
                format!(
                    r#"{{"access_token": "tok{i}", "token_type": "bearer", "expires_in": 3600}}"#
                ),
            )
        }
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        // The first token is revoked by the time tables are listed.
        _ if r.header("authorization") == Some("Bearer tok0") => Reply::Json(401, "{}".into()),
        _ => Reply::Json(200, r#"{"namespaces": []}"#.into()),
    });
    let client = client_with(client_credentials(&server, None));
    client.list_namespaces(&[]).unwrap();
    client.list_namespaces(&[]).unwrap();
    let seen = server.seen();
    let tokens: Vec<_> = seen
        .iter()
        .filter(|s| path(s) == "/v1/oauth/tokens")
        .collect();
    assert_eq!(tokens.len(), 2, "one fetch, one renewal after the 401");
    assert_eq!(tokens[0].method, "POST");
    assert_eq!(
        tokens[0].header("content-type"),
        Some("application/x-www-form-urlencoded")
    );
    assert_eq!(
        tokens[0].body,
        "grant_type=client_credentials&client_id=app+id&client_secret=s3cr%26t&scope=catalog&audience=https%3A%2F%2Fstorage.azure.com%2F"
    );
    assert_eq!(tokens[0].header("authorization"), None);
    // config (tok0), namespaces (tok0 → 401), token, namespaces (tok1), namespaces (tok1, reused).
    let auth: Vec<_> = seen
        .iter()
        .filter(|s| path(s) != "/v1/oauth/tokens")
        .map(|s| s.header("authorization").unwrap_or(""))
        .collect();
    assert_eq!(
        auth,
        ["Bearer tok0", "Bearer tok0", "Bearer tok1", "Bearer tok1"]
    );
}

#[test]
fn an_expiring_token_is_renewed_before_use() {
    let issued = Arc::new(AtomicUsize::new(0));
    let n = Arc::clone(&issued);
    let server = serve(move |r| match path(r) {
        "/token" => {
            let i = n.fetch_add(1, Ordering::SeqCst);
            Reply::Json(
                200,
                format!(r#"{{"access_token": "tok{i}", "token_type": "Bearer", "expires_in": 0}}"#),
            )
        }
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, r#"{"namespaces": []}"#.into()),
    });
    let endpoint = Url::parse(&format!("{}/token", server.base)).unwrap();
    let client = client_with(client_credentials(&server, Some(endpoint)));
    client.list_namespaces(&[]).unwrap();
    client.list_namespaces(&[]).unwrap();
    assert_eq!(
        issued.load(Ordering::SeqCst),
        3,
        "a zero-lifetime token is fetched for every request"
    );
}

#[test]
fn refused_client_credentials_name_the_catalog_only() {
    let server = serve(|r| match path(r) {
        "/v1/oauth/tokens" => Reply::Json(
            401,
            r#"{"error": "invalid_client", "error_description": "secret s3cr&t is wrong"}"#.into(),
        ),
        _ => Reply::Json(200, CONFIG.into()),
    });
    let e = client_with(client_credentials(&server, None))
        .list_namespaces(&[])
        .unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(!e.to_string().contains("s3cr"), "{e}");
    assert_eq!(server.seen().len(), 1, "no catalog request without a token");
}

#[test]
fn the_onelake_form_against_a_fake_fabric_endpoint() {
    // OneLake's endpoint has a base path (`/iceberg`) and a warehouse of `<workspace>/<item>`; the real host waits
    // for the owner's Azure account, so the form is checked here.
    let server = serve(|r| match path(r) {
        "/iceberg/v1/config" => Reply::Json(
            200,
            r#"{"defaults": {}, "overrides": {"prefix": "ws/sales.Lakehouse"}}"#.into(),
        ),
        "/iceberg/v1/ws/sales.Lakehouse/namespaces/dbo/tables/orders" => Reply::Json(
            200,
            table_json(
                "abfss://ws@onelake.dfs.fabric.microsoft.com/sales.Lakehouse/Tables/dbo/orders",
            ),
        ),
        _ => Reply::Json(404, "{}".into()),
    });
    let mut config =
        RestConfig::onelake("ws", "sales.Lakehouse", SecretString::new("entra-token")).unwrap();
    config.uri = Url::parse(&format!("{}/iceberg", server.base)).unwrap();
    config.allow_http = true;
    let t = client_with(config)
        .load_table(&ns(&["dbo"]), "orders")
        .unwrap()
        .unwrap();
    assert!(
        t.location
            .starts_with("abfss://ws@onelake.dfs.fabric.microsoft.com/")
    );
    let seen = server.seen();
    assert_eq!(
        seen[0].target,
        "/iceberg/v1/config?warehouse=ws%2Fsales.Lakehouse"
    );
    assert!(
        seen.iter()
            .all(|s| s.header("authorization") == Some("Bearer entra-token"))
    );
}

#[test]
fn a_dropped_client_inside_a_runtime_does_not_panic() {
    let server = serve(|_| Reply::Json(200, r#"{"defaults": {}, "overrides": {}}"#.into()));
    let client = client_with(config_for(&server));
    client.catalog_config().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async move { drop(client) });
}

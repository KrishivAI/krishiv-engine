// SPDX-License-Identifier: Apache-2.0
//! Acceptance against a real Iceberg REST server: Apache's `iceberg-rest-fixture` (the Java `RESTCatalogServlet`
//! over an in-memory JDBC catalog). Runs only when `KRISHIV_ICEBERG_REST_URL` names one;
//! `tests/scripts/iceberg_rest_fixture.sh` starts a throw-away container and sets it. The test creates what it reads
//! with plain HTTP (the client is read-only), so the server's own encoding of namespaces and names is what the
//! client's reads must match.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use url::Url;

use super::{RestClient, RestConfig};
use crate::location::{LocationPolicy, LocationRule};

/// The fixture writes metadata to its own local disk (`file:`), which a remote catalog may never name in
/// production; this acceptance is about the wire, so it accepts any location.
struct AnyLocation;

impl LocationPolicy for AnyLocation {
    fn check(&self, _: &str) -> Result<(), LocationRule> {
        Ok(())
    }
}

/// A JSON POST over a fresh connection; returns the status.
fn post(base: &Url, path: &str, body: &str) -> u16 {
    let host = format!(
        "{}:{}",
        base.host_str().unwrap(),
        base.port_or_known_default().unwrap()
    );
    let mut s = TcpStream::connect(&host).unwrap();
    write!(
        s,
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut answer = String::new();
    s.read_to_string(&mut answer).unwrap();
    answer.split_whitespace().nth(1).unwrap().parse().unwrap()
}

#[test]
fn fixture_lists_and_loads_what_the_java_server_wrote() {
    let Ok(base) = std::env::var("KRISHIV_ICEBERG_REST_URL") else {
        return;
    };
    let base = Url::parse(&base).unwrap();
    let ns = ["krishiv_acc".to_owned(), "sub level".to_owned()];
    for (path, body) in [
        (
            "/v1/namespaces",
            r#"{"namespace": ["krishiv_acc"], "properties": {}}"#,
        ),
        (
            "/v1/namespaces",
            r#"{"namespace": ["krishiv_acc", "sub level"], "properties": {}}"#,
        ),
    ] {
        let status = post(&base, path, body);
        assert!(status == 200 || status == 409, "{path}: {status}");
    }
    let table = r#"{"name": "t x+y", "schema": {"type": "struct", "schema-id": 0,
        "fields": [{"id": 1, "name": "a", "required": false, "type": "long"}]}}"#;
    let status = post(
        &base,
        "/v1/namespaces/krishiv_acc%1Fsub%20level/tables",
        table,
    );
    assert!(status == 200 || status == 409, "create table: {status}");

    let mut config = RestConfig::new(base.as_str()).unwrap();
    config.allow_http = true;
    config.vended_credentials = true;
    let client = RestClient::new("fixture", config, Arc::new(AnyLocation)).unwrap();

    let top = client.list_namespaces(&[]).unwrap().unwrap();
    assert!(top.contains(&vec!["krishiv_acc".to_owned()]), "{top:?}");
    let under = client.list_namespaces(&ns[..1]).unwrap().unwrap();
    assert_eq!(under, vec![ns.to_vec()]);
    let tables = client.list_tables(&ns).unwrap().unwrap();
    assert_eq!(tables.len(), 1, "{tables:?}");
    assert_eq!(
        (tables[0].namespace.as_slice(), tables[0].name.as_str()),
        (&ns[..], "t x+y")
    );

    let t = client.load_table(&ns, "t x+y").unwrap().unwrap();
    assert_eq!(t.format_version, 2);
    assert!(
        t.metadata_location
            .as_deref()
            .is_some_and(|m| m.ends_with(".metadata.json")),
        "{t:?}"
    );
    assert!(t.metadata.get("schemas").is_some());
    assert!(!t.restricted);

    assert!(client.load_table(&ns, "absent").unwrap().is_none());
    assert!(
        client
            .list_tables(&["krishiv_absent".to_owned()])
            .unwrap()
            .is_none()
    );
}

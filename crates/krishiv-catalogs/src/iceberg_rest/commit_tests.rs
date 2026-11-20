// SPDX-License-Identifier: Apache-2.0
//! [`RestClient::commit_table_as`] against the fake server (PR-36c part 2): the wire form (POST,
//! JSON body, token, `X-Iceberg-Access-Delegation`), one resend after a 401 only with a renewable
//! token, never a resend after a 5xx, `Retry-After`, a reset after sending told apart from a
//! request that never left (the egress policy), the body cap, and a 200 naming a refused location.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use krishiv_common::secret::SecretString;

use super::test_server::{Reply, Seen, Server, serve};
use super::{RestAuth, RestCaller, RestClient, RestConfig, Snapshots};
use crate::location::RemoteLocationPolicy;

const CONFIG: &str = r#"{"defaults": {}, "overrides": {}}"#;
const TABLE: &str = "/v1/namespaces/db/tables/t";
const BODY: &[u8] =
    br#"{"identifier": {"namespace": ["db"], "name": "t"}, "requirements": [], "updates": []}"#;

fn path(seen: &Seen) -> &str {
    seen.target.split('?').next().unwrap_or("")
}

fn answer(location: &str) -> String {
    format!(
        r#"{{"metadata-location": "{location}/metadata/v2.metadata.json",
            "metadata": {{"format-version": 2, "table-uuid": "u", "location": "{location}"}}}}"#
    )
}

fn client(server: &Server, auth: RestAuth, vended: bool) -> RestClient {
    let mut c = RestConfig::new(&server.base).unwrap();
    c.allow_http = true;
    c.auth = auth;
    c.vended_credentials = vended;
    RestClient::new("rest", c, Arc::new(RemoteLocationPolicy::default())).unwrap()
}

fn commit(client: &RestClient) -> Result<super::CommitReply, super::CommitFailure> {
    client.commit_table_as(&RestCaller::server(), &["db".to_owned()], "t", BODY)
}

fn posts(server: &Server) -> Vec<Seen> {
    server
        .seen()
        .into_iter()
        .filter(|s| s.method == "POST" && path(s) == TABLE)
        .collect()
}

#[test]
fn a_commit_is_one_json_post_with_the_tables_headers() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        TABLE => Reply::Json(200, answer("s3://b/t")),
        _ => Reply::Json(404, "{}".into()),
    });
    let c = client(&server, RestAuth::Token(SecretString::new("tok")), true);
    let reply = commit(&c).unwrap();
    assert_eq!(reply.status, 200);
    let sent = posts(&server);
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].header("content-type"), Some("application/json"));
    assert_eq!(sent[0].header("authorization"), Some("Bearer tok"));
    assert_eq!(
        sent[0].header("x-iceberg-access-delegation"),
        Some("vended-credentials")
    );
    assert_eq!(sent[0].body.as_bytes(), BODY);
}

/// A 401 with a renewable token is resent once with a new token (the catalog applied nothing);
/// a second 401 is the answer. A static token is never resent.
#[test]
fn a_401_is_resent_once_only_with_a_renewable_token() {
    let issued = Arc::new(AtomicUsize::new(0));
    let n = Arc::clone(&issued);
    let server = serve(move |r| match path(r) {
        "/v1/oauth/tokens" => {
            let i = n.fetch_add(1, Ordering::SeqCst);
            Reply::Json(
                200,
                format!(r#"{{"access_token": "tok{i}", "token_type": "bearer"}}"#),
            )
        }
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(401, "{}".into()),
    });
    let c = client(
        &server,
        RestAuth::ClientCredentials {
            client_id: Some("id".into()),
            client_secret: SecretString::new("s"),
            token_endpoint: None,
            scope: None,
            extra: Vec::new(),
        },
        false,
    );
    assert_eq!(commit(&c).unwrap().status, 401);
    let sent = posts(&server);
    let auth: Vec<_> = sent
        .iter()
        .map(|s| s.header("authorization").unwrap_or(""))
        .collect();
    assert_eq!(auth, ["Bearer tok0", "Bearer tok1"]);

    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(401, "{}".into()),
    });
    let c = client(&server, RestAuth::Token(SecretString::new("tok")), false);
    assert_eq!(commit(&c).unwrap().status, 401);
    assert_eq!(posts(&server).len(), 1);
}

/// A 503 is the answer, never resent; its `Retry-After` is kept.
#[test]
fn a_5xx_is_never_resent_and_keeps_retry_after() {
    let server = serve(|r| {
        match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Raw(
            b"HTTP/1.1 503 Unavailable\r\nRetry-After: 7\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
                .to_vec(),
        ),
    }
    });
    let c = client(&server, RestAuth::None, false);
    let reply = commit(&c).unwrap();
    assert_eq!(reply.status, 503);
    assert_eq!(reply.retry_after, Some(Duration::from_secs(7)));
    assert_eq!(posts(&server).len(), 1);
}

/// The connection closed after the request was read: no answer, and it may have been applied.
#[test]
fn a_reset_after_sending_is_sent() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Raw(Vec::new()),
    });
    let c = client(&server, RestAuth::None, false);
    let f = commit(&c).unwrap_err();
    assert!(f.sent, "{f:?}");
    assert_eq!(posts(&server).len(), 1);
}

/// The egress policy refuses the catalog's host: the commit fails before anything is sent.
#[test]
fn a_commit_the_egress_policy_refuses_is_never_sent() {
    use krishiv_store::egress::{EgressPolicy, PrivateAddresses, ProviderEndpoints};
    let server = serve(|_| Reply::Json(200, CONFIG.into()));
    let mut c = RestConfig::new(&server.base).unwrap();
    c.allow_http = true;
    c.egress = Some(
        krishiv_store::PolicyConnector::new(
            EgressPolicy::new(
                PrivateAddresses::Deny,
                Vec::new(),
                ProviderEndpoints::default(),
            )
            .scope(Vec::<String>::new()),
            krishiv_store::Redirects::None,
            None,
        )
        .without_provider_literals(),
    );
    let c = RestClient::new("rest", c, Arc::new(RemoteLocationPolicy::default())).unwrap();
    let f = commit(&c).unwrap_err();
    assert!(!f.sent, "{f:?}");
    assert!(server.seen().is_empty());
}

/// A body over the response limit is refused before sending.
#[test]
fn an_oversized_body_is_never_sent() {
    let server = serve(|_| Reply::Json(200, CONFIG.into()));
    let mut cfg = RestConfig::new(&server.base).unwrap();
    cfg.allow_http = true;
    cfg.limits.max_response_bytes = 16;
    let c = RestClient::new("rest", cfg, Arc::new(RemoteLocationPolicy::default())).unwrap();
    let f = commit(&c).unwrap_err();
    assert!(!f.sent, "{f:?}");
    assert!(server.seen().is_empty());
}

/// A 200 naming a location the policy refuses is never handed on: a failure after sending (the
/// commit may stand).
#[test]
fn a_200_with_a_refused_location_is_a_sent_failure() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, answer("file:///etc")),
    });
    let c = client(&server, RestAuth::None, false);
    let f = commit(&c).unwrap_err();
    assert!(f.sent, "{f:?}");
}

/// `snapshots=all|refs` reaches the query; no parameter by default.
#[test]
fn load_sends_the_snapshots_parameter() {
    let server = serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, CONFIG.into()),
        _ => Reply::Json(200, answer("s3://b/t")),
    });
    let c = client(&server, RestAuth::None, false);
    let ns = ["db".to_owned()];
    let caller = RestCaller::server();
    c.load_table_snapshots_as(&caller, &ns, "t", Some(Snapshots::All))
        .unwrap()
        .unwrap();
    c.load_table_snapshots_as(&caller, &ns, "t", Some(Snapshots::Refs))
        .unwrap()
        .unwrap();
    c.load_table_as(&caller, &ns, "t").unwrap().unwrap();
    let targets: Vec<_> = server
        .seen()
        .into_iter()
        .filter(|s| path(s) == TABLE)
        .map(|s| s.target)
        .collect();
    assert_eq!(
        targets,
        [
            format!("{TABLE}?snapshots=all"),
            format!("{TABLE}?snapshots=refs"),
            TABLE.to_owned()
        ]
    );
}

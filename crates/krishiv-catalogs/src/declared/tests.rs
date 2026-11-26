// SPDX-License-Identifier: Apache-2.0
//! The `iceberg-rest` factory (design P3-T067 §2.3, §3.1, §5.2): declarations to a catalog, the
//! auth modes, every request on the egress policy client.

use std::sync::Arc;

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::{NameForm, TableIdent, TableMeta};
use krishiv_session::catalog_decl::{CatalogDecl, CatalogDeclarations, CatalogType, DeclMode};
use krishiv_session::catalog_registry::{
    CallerCatalog, CatalogCaller, CatalogDeps, CatalogFactory,
};
use krishiv_store::egress::{EgressPolicy, PrivateAddresses, ProviderEndpoints};

use super::IcebergRestFactory;
use crate::iceberg_rest::test_server::{Reply, Seen, Server, serve};

const NO_TABLE: &str =
    r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#;

pub(super) fn path(seen: &Seen) -> &str {
    seen.target.split('?').next().unwrap_or("")
}

/// A valid table metadata document of `s3://b/t (x INT)` without snapshots.
pub(super) const META: &str = r#"{"format-version": 2, "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
    "location": "s3://b/t", "last-sequence-number": 0, "last-updated-ms": 1, "last-column-id": 1,
    "current-schema-id": 0,
    "schemas": [{"type": "struct", "schema-id": 0,
                 "fields": [{"id": 1, "name": "x", "required": false, "type": "int"}]}],
    "default-spec-id": 0, "partition-specs": [{"spec-id": 0, "fields": []}], "last-partition-id": 999,
    "default-sort-order-id": 0, "sort-orders": [{"order-id": 0, "fields": []}], "properties": {},
    "current-snapshot-id": -1, "snapshots": [], "snapshot-log": [], "metadata-log": []}"#;

/// The `loadTable` answer of `db.t` (`META`).
pub(super) fn table_body() -> String {
    format!(
        r#"{{"metadata-location": "s3://b/t/metadata/v1.metadata.json", "metadata": {META}, "config": {{}}}}"#
    )
}

/// A catalog with `db.t` and nothing else.
fn catalog_server() -> Server {
    serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, r#"{"defaults": {}, "overrides": {}}"#.into()),
        "/v1/namespaces/db/tables/t" => Reply::Json(200, table_body()),
        "/token" => Reply::Json(
            200,
            r#"{"access_token": "cc-token", "token_type": "bearer"}"#.into(),
        ),
        _ => Reply::Json(404, NO_TABLE.into()),
    })
}

pub(super) fn decl(pairs: &[(&str, &str)], mode: DeclMode) -> CatalogDecl {
    let mut d = CatalogDeclarations::parse(
        pairs.iter().copied(),
        mode,
        &[CatalogType::IcebergRest],
        None,
    )
    .unwrap();
    d.catalogs.remove(0)
}

pub(super) fn deps(mode: DeclMode, private: PrivateAddresses) -> CatalogDeps {
    CatalogDeps::new(
        mode,
        EgressPolicy::new(private, Vec::new(), ProviderEndpoints::default()),
    )
}

pub(super) fn ident(ns: &[&str], name: &str) -> TableIdent {
    TableIdent {
        catalog: "lake".to_owned(),
        namespace: ns.iter().map(|s| (*s).to_owned()).collect(),
        name: name.to_owned(),
    }
}

pub(super) const FORM: NameForm = NameForm {
    case_sensitive: false,
    qualified: true,
    temporary_only: false,
};

fn load(
    catalog: &Arc<dyn CallerCatalog>,
    id: &TableIdent,
) -> Result<Option<TableMeta>, KrishivError> {
    load_as(catalog, &CatalogCaller::server("lake"), id)
}

#[expect(
    clippy::unwrap_in_result,
    reason = "a test helper: the runtime always builds"
)]
fn load_as(
    catalog: &Arc<dyn CallerCatalog>,
    caller: &CatalogCaller,
    id: &TableIdent,
) -> Result<Option<TableMeta>, KrishivError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(catalog.load_table(caller, id, None, FORM))
}

#[test]
fn declared_rest_catalog_resolves_names_with_a_static_token() {
    let server = catalog_server();
    // Spark's alias in embedded mode: an inline token, an http:// catalog.
    let d = decl(
        &[
            (
                "spark.sql.catalog.lake",
                "org.apache.iceberg.spark.SparkCatalog",
            ),
            ("spark.sql.catalog.lake.type", "rest"),
            ("spark.sql.catalog.lake.uri", server.base.as_str()),
            ("spark.sql.catalog.lake.token", "static-token"),
        ],
        DeclMode::Embedded,
    );
    let deps = deps(DeclMode::Embedded, PrivateAddresses::Allow);
    IcebergRestFactory.validate(&d, &deps).unwrap();
    let catalog = IcebergRestFactory.build(&d, &deps).unwrap();
    assert!(
        load(&catalog, &ident(&["db"], "missing"))
            .unwrap()
            .is_none()
    );
    // No namespace: absent, nothing sent.
    let sent = server.seen().len();
    assert!(load(&catalog, &ident(&[], "t")).unwrap().is_none());
    assert_eq!(server.seen().len(), sent);
    // A table the catalog has: its metadata, under the catalog's own name.
    let meta = load(&catalog, &ident(&["db"], "t")).unwrap().unwrap();
    assert_eq!(meta.ident.parts(), ["lake", "db", "t"]);
    assert_eq!(meta.columns.len(), 1);
    assert_eq!(meta.columns[0].name, "x");
    assert_eq!(meta.version, None);
    for s in server.seen() {
        assert_eq!(s.header("authorization"), Some("Bearer static-token"));
    }
}

#[test]
fn client_credentials_at_the_declared_token_endpoint_under_the_server_rule() {
    let server = catalog_server();
    let token_endpoint = format!("{}/token", server.base);
    // Server mode, `deny`: the loopback catalog and its IdP are the catalog's declared hosts, so
    // they pass by name. The secret is a reference; `PATH` is set in every test environment.
    let pairs = [
        ("krishiv.catalog.lake.type", "iceberg-rest"),
        ("krishiv.catalog.lake.uri", server.base.as_str()),
        ("krishiv.catalog.lake.allowInsecure", "true"),
        ("krishiv.catalog.lake.auth", "client-credentials"),
        ("krishiv.catalog.lake.clientId", "krishiv"),
        ("krishiv.catalog.lake.clientSecret", "env:PATH"),
        (
            "krishiv.catalog.lake.tokenEndpoint",
            token_endpoint.as_str(),
        ),
        ("krishiv.catalog.lake.audience", "catalog-aud"),
    ];
    let d = decl(&pairs, DeclMode::Server);
    let deps = deps(DeclMode::Server, PrivateAddresses::Deny);
    let catalog = IcebergRestFactory.build(&d, &deps).unwrap();
    assert!(
        load(&catalog, &ident(&["db"], "missing"))
            .unwrap()
            .is_none()
    );
    let seen = server.seen();
    let token = seen.iter().find(|s| path(s) == "/token").unwrap();
    assert!(
        token.body.contains("grant_type=client_credentials"),
        "{}",
        token.body
    );
    assert!(
        token.body.contains("audience=catalog-aud"),
        "{}",
        token.body
    );
    let load_call = seen.iter().find(|s| path(s).ends_with("/missing")).unwrap();
    assert_eq!(load_call.header("authorization"), Some("Bearer cc-token"));
}

#[test]
fn an_undeclared_private_address_is_refused_before_sending() {
    let server = catalog_server();
    // The declared host is 10.0.0.1; the client is then pointed at the loopback server, which is
    // no declared host of this catalog: the per-request check refuses it before anything is sent.
    let d = decl(
        &[
            ("krishiv.catalog.lake.type", "iceberg-rest"),
            ("krishiv.catalog.lake.uri", "http://10.0.0.1:1"),
            ("krishiv.catalog.lake.allowInsecure", "true"),
        ],
        DeclMode::Server,
    );
    let deps = deps(DeclMode::Server, PrivateAddresses::Deny);
    let mut config = IcebergRestFactory::config(&d, &deps, true).unwrap();
    config.uri = url::Url::parse(&server.base).unwrap();
    let client = crate::iceberg_rest::RestClient::new(
        "lake",
        config,
        Arc::new(crate::location::RemoteLocationPolicy::default()),
    )
    .unwrap();
    let catalog: Arc<dyn CallerCatalog> = Arc::new(super::RestCatalog::new(
        &d,
        Arc::new(client),
        super::table::METADATA_TTL,
        false,
        super::vended::VendedCache::global(),
    ));
    let e = load(&catalog, &ident(&["db"], "t")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivCatalogFailure);
    assert!(!e.to_string().contains("127.0.0.1"), "{e}");
    assert!(server.seen().is_empty());
}

#[test]
fn startup_validation_refuses_what_the_client_cannot_do() {
    let deps = deps(DeclMode::Server, PrivateAddresses::Deny);
    let d = decl(
        &[
            ("krishiv.catalog.lake.type", "iceberg-rest"),
            (
                "krishiv.catalog.lake.uri",
                "https://catalog.example.com/api",
            ),
            ("krishiv.catalog.lake.option.prefix", "p"),
        ],
        DeclMode::Server,
    );
    let e = IcebergRestFactory.validate(&d, &deps).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("option.prefix"), "{e}");
    // A uri with credentials, refused by the client's own check, never echoed.
    let d = decl(
        &[
            ("krishiv.catalog.lake.type", "iceberg-rest"),
            (
                "krishiv.catalog.lake.uri",
                "https://user:hunter2@catalog.example.com/api",
            ),
        ],
        DeclMode::Server,
    );
    let e = IcebergRestFactory.validate(&d, &deps).unwrap_err();
    assert!(!e.to_string().contains("hunter2"), "{e}");
}

/// Polls `f` to completion on the calling thread, with no tokio runtime anywhere.
fn poll_without_runtime<F: std::future::Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

/// Review 58a2 M1: outside a tokio runtime (an embedded host on `futures::executor`), a table
/// load runs the blocking call on the caller's thread instead of panicking.
#[test]
fn load_table_outside_a_tokio_runtime() {
    let server = catalog_server();
    let d = decl(
        &[
            ("krishiv.catalog.lake.type", "iceberg-rest"),
            ("krishiv.catalog.lake.uri", server.base.as_str()),
        ],
        DeclMode::Embedded,
    );
    let catalog = IcebergRestFactory
        .build(&d, &deps(DeclMode::Embedded, PrivateAddresses::Allow))
        .unwrap();
    assert!(tokio::runtime::Handle::try_current().is_err());
    let server_caller = CatalogCaller::server("lake");
    let found = poll_without_runtime(catalog.load_table(
        &server_caller,
        &ident(&["db"], "missing"),
        None,
        FORM,
    ));
    assert!(found.unwrap().is_none());
    let found =
        poll_without_runtime(catalog.load_table(&server_caller, &ident(&["db"], "t"), None, FORM));
    assert!(found.unwrap().is_some());
}

/// A delegated catalog whose IdP exchanges `<subject>` for `x-<subject>` and whose catalog has
/// `db.t` only for `x-alice-jwt` (catalog-side grants of the end user).
fn delegated_server() -> Server {
    serve(|r| match path(r) {
        "/token" => {
            let subject = url::form_urlencoded::parse(r.body.as_bytes())
                .find(|(k, _)| k == "subject_token")
                .map(|(_, v)| v.into_owned())
                .unwrap_or_default();
            Reply::Json(
                200,
                format!(r#"{{"access_token": "x-{subject}", "token_type": "bearer"}}"#),
            )
        }
        "/v1/config" => Reply::Json(200, r#"{"defaults": {}, "overrides": {}}"#.into()),
        "/v1/namespaces/db/tables/t" if r.header("authorization") == Some("Bearer x-alice-jwt") => {
            Reply::Json(200, table_body())
        }
        _ => Reply::Json(404, NO_TABLE.into()),
    })
}

fn delegated_catalog(server: &Server) -> Arc<dyn CallerCatalog> {
    let token_endpoint = format!("{}/token", server.base);
    let pairs = [
        ("krishiv.catalog.lake.type", "iceberg-rest"),
        ("krishiv.catalog.lake.uri", server.base.as_str()),
        ("krishiv.catalog.lake.allowInsecure", "true"),
        ("krishiv.catalog.lake.auth", "delegated"),
        ("krishiv.catalog.lake.audience", "lake"),
        ("krishiv.catalog.lake.actorToken", "none"),
        (
            "krishiv.catalog.lake.tokenEndpoint",
            token_endpoint.as_str(),
        ),
    ];
    let d = decl(&pairs, DeclMode::Server);
    let deps = deps(DeclMode::Server, PrivateAddresses::Deny);
    IcebergRestFactory.validate(&d, &deps).unwrap();
    IcebergRestFactory.build(&d, &deps).unwrap()
}

fn oidc_caller(name: &str, token: &str) -> CatalogCaller {
    use krishiv_common::principal::{Principal, PrincipalKind};
    use krishiv_session::auth::{CredentialKind, ExchangeClaims, InboundCredential, Method};
    use krishiv_session::catalog_registry::IdentityKey;
    let principal = Principal::new(name, PrincipalKind::User);
    CatalogCaller::new(
        IdentityKey::of("lake", &principal),
        Some(Method::Oidc),
        Some(InboundCredential::new(
            CredentialKind::Oidc,
            krishiv_common::secret::SecretString::new(token),
            // The same claims for both users: only the identity key tells them apart.
            ExchangeClaims::with_audiences(vec!["krishiv".to_owned()]),
            std::time::SystemTime::now() + std::time::Duration::from_secs(3600),
        )),
    )
}

/// P3-T067 §3.5: what a delegated catalog answers user A is never served to user B; each
/// user's call reaches the catalog with that user's exchanged token.
#[test]
fn delegated_metadata_cache_keyed_by_identity() {
    let server = delegated_server();
    let catalog = delegated_catalog(&server);
    let alice = oidc_caller("alice", "alice-jwt");
    let bob = oidc_caller("bob", "bob-jwt");
    let t = ident(&["db"], "t");
    // Alice is granted `db.t` in the catalog: found.
    assert!(load_as(&catalog, &alice, &t).unwrap().is_some());
    // Bob is not: absent for him, whatever Alice loaded (and cached).
    assert!(load_as(&catalog, &bob, &t).unwrap().is_none());
    // Alice's answer is cached for her (one load), and still never Bob's.
    assert!(load_as(&catalog, &alice, &t).unwrap().is_some());
    assert!(load_as(&catalog, &bob, &t).unwrap().is_none());
    let loads: Vec<_> = server
        .seen()
        .iter()
        .filter(|s| path(s) == "/v1/namespaces/db/tables/t")
        .map(|s| s.header("authorization").unwrap_or("").to_owned())
        .collect();
    assert_eq!(
        loads,
        ["Bearer x-alice-jwt", "Bearer x-bob-jwt", "Bearer x-bob-jwt"]
    );
    // A session without a user token gets the delegation class, never the server's identity.
    let e = load_as(&catalog, &CatalogCaller::server("lake"), &t).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogDelegationUnavailable
    );
}

/// Review 58b2 F3 (§3.5): with both users granted, each gets an answer of their own (two loads,
/// each with its own exchanged token, then one cache entry each); once Bob's grant is revoked in
/// the catalog, Alice's cached entry never answers him. A catalog that is not delegated keeps one
/// entry for every caller.
#[test]
fn delegated_metadata_cache_one_entry_per_identity() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let revoked = Arc::new(AtomicBool::new(false));
    let bob_revoked = Arc::clone(&revoked);
    let server = serve(move |r| match path(r) {
        "/token" => {
            let subject = url::form_urlencoded::parse(r.body.as_bytes())
                .find(|(k, _)| k == "subject_token")
                .map(|(_, v)| v.into_owned())
                .unwrap_or_default();
            Reply::Json(
                200,
                format!(r#"{{"access_token": "x-{subject}", "token_type": "bearer"}}"#),
            )
        }
        "/v1/config" => Reply::Json(200, r#"{"defaults": {}, "overrides": {}}"#.into()),
        "/v1/namespaces/db/tables/t"
            if r.header("authorization") == Some("Bearer x-bob-jwt")
                && bob_revoked.load(Ordering::SeqCst) =>
        {
            Reply::Json(
                403,
                r#"{"error": {"message": "no", "type": "ForbiddenException", "code": 403}}"#.into(),
            )
        }
        "/v1/namespaces/db/tables/t" => Reply::Json(200, table_body()),
        _ => Reply::Json(404, NO_TABLE.into()),
    });
    let table_loads = |server: &Server| -> Vec<String> {
        server
            .seen()
            .iter()
            .filter(|s| path(s) == "/v1/namespaces/db/tables/t")
            .map(|s| s.header("authorization").unwrap_or("").to_owned())
            .collect()
    };
    let catalog = delegated_catalog(&server);
    let alice = oidc_caller("alice", "alice-jwt");
    let bob = oidc_caller("bob", "bob-jwt");
    let t = ident(&["db"], "t");
    for _ in 0..2 {
        assert!(load_as(&catalog, &alice, &t).unwrap().is_some());
        assert!(load_as(&catalog, &bob, &t).unwrap().is_some());
    }
    assert_eq!(
        table_loads(&server),
        ["Bearer x-alice-jwt", "Bearer x-bob-jwt"]
    );
    // Bob's grant is revoked; a new catalog (no entry of his) asks the catalog for him, and
    // Alice's entry, loaded first, does not answer him.
    revoked.store(true, Ordering::SeqCst);
    let catalog = delegated_catalog(&server);
    assert!(load_as(&catalog, &alice, &t).unwrap().is_some());
    let r = load_as(&catalog, &bob, &t);
    assert!(!matches!(r, Ok(Some(_))), "{r:?}");
    assert_eq!(
        table_loads(&server)[2..],
        ["Bearer x-alice-jwt", "Bearer x-bob-jwt"]
    );

    // Not delegated: one entry for the catalog, whoever calls.
    let server = catalog_server();
    let catalog = IcebergRestFactory
        .build(
            &decl(
                &[
                    ("krishiv.catalog.lake.type", "iceberg-rest"),
                    ("krishiv.catalog.lake.uri", server.base.as_str()),
                ],
                DeclMode::Embedded,
            ),
            &deps(DeclMode::Embedded, PrivateAddresses::Allow),
        )
        .unwrap();
    load_as(&catalog, &alice, &t).unwrap().unwrap();
    load_as(&catalog, &bob, &t).unwrap().unwrap();
    load_as(&catalog, &CatalogCaller::server("lake"), &t)
        .unwrap()
        .unwrap();
    assert_eq!(table_loads(&server).len(), 1);
}

fn caller_with(groups: &[&str], scope: &str, expires_in: i64) -> CatalogCaller {
    use krishiv_common::principal::{Principal, PrincipalKind};
    use krishiv_session::auth::{CredentialKind, ExchangeClaims, InboundCredential, Method};
    use krishiv_session::catalog_registry::IdentityKey;
    let mut principal = Principal::new("alice", PrincipalKind::User);
    principal.groups = groups.iter().map(|g| (*g).to_owned()).collect();
    let mut claims = ExchangeClaims::with_audiences(vec!["krishiv".to_owned()]);
    claims.scope = Some(scope.to_owned());
    let now = std::time::SystemTime::now();
    let delta = std::time::Duration::from_secs(expires_in.unsigned_abs());
    let exp = if expires_in < 0 {
        now - delta
    } else {
        now + delta
    };
    CatalogCaller::new(
        IdentityKey::of("lake", &principal),
        Some(Method::Oidc),
        Some(InboundCredential::new(
            CredentialKind::Oidc,
            krishiv_common::secret::SecretString::new("alice-jwt"),
            claims,
            exp,
        )),
    )
}

fn limited_catalog(server: &Server, distinct: usize) -> Arc<dyn CallerCatalog> {
    let token_endpoint = format!("{}/token", server.base);
    let pairs = [
        ("krishiv.catalog.lake.type", "iceberg-rest"),
        ("krishiv.catalog.lake.uri", server.base.as_str()),
        ("krishiv.catalog.lake.allowInsecure", "true"),
        ("krishiv.catalog.lake.auth", "delegated"),
        ("krishiv.catalog.lake.audience", "lake"),
        ("krishiv.catalog.lake.actorToken", "none"),
        (
            "krishiv.catalog.lake.tokenEndpoint",
            token_endpoint.as_str(),
        ),
    ];
    let d = decl(&pairs, DeclMode::Server);
    let mut deps = deps(DeclMode::Server, PrivateAddresses::Deny);
    deps.max_distinct_exchanges_per_minute = distinct;
    // No `loadTable` cache: every call reaches the exchange and the catalog.
    IcebergRestFactory::build_with_ttl(&d, &deps, std::time::Duration::ZERO).unwrap()
}

/// Review 58b1-r1 M1: the exchange limits count a principal by kind, tenant, name and
/// `delegated_by`, never by its groups, so ten group sets under `maxDistinct = 1` are all refused.
#[test]
fn exchange_limits_ignore_groups() {
    let server = delegated_server();
    let catalog = limited_catalog(&server, 1);
    let t = ident(&["db"], "t");
    let _ = load_as(&catalog, &caller_with(&["g1"], "s1", 3600), &t);
    let e = load_as(&catalog, &caller_with(&["g1"], "s2", 3600), &t).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogLimitExceeded,
        "{e}"
    );
    for i in 2..12 {
        let g = format!("g{i}");
        let e = load_as(&catalog, &caller_with(&[g.as_str()], "s2", 3600), &t).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivCatalogLimitExceeded,
            "{g}: {e}"
        );
    }
    let exchanges = server.seen().iter().filter(|s| path(s) == "/token").count();
    assert_eq!(exchanges, 1);
}

/// Review 58b1-r1 M2: a token the IdP issued without `expires_in` lives no longer than its
/// subject token and never serves a caller whose subject token has expired.
#[test]
fn exchanged_token_without_expires_in_ends_with_its_subject() {
    let server = delegated_server(); // its `/token` answers without `expires_in`
    let catalog = limited_catalog(&server, 20);
    let t = ident(&["db"], "t");
    let _ = load_as(&catalog, &caller_with(&[], "s", 3600), &t);
    let e = load_as(&catalog, &caller_with(&[], "s", -7 * 86_400), &t).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogDelegationUnavailable,
        "{e}"
    );
    assert!(e.to_string().contains("reconnect"), "{e}");
    let exchanges = server.seen().iter().filter(|s| path(s) == "/token").count();
    assert_eq!(exchanges, 1, "no exchange for an expired subject token");
}

/// Review 58b1-r2 N3: a delegated catalog's `/v1/config` is kept per principal without its
/// groups and per exchange claims, fetched with that caller's own exchanged token.
#[test]
fn delegated_config_is_kept_per_principal_and_claims() {
    let server = delegated_server();
    let catalog = limited_catalog(&server, 20);
    let t = ident(&["db"], "t");
    let _ = load_as(&catalog, &caller_with(&["g1"], "s", 3600), &t);
    let _ = load_as(&catalog, &caller_with(&["g2"], "s", 3600), &t);
    let _ = load_as(&catalog, &caller_with(&["g1"], "narrower", 3600), &t);
    let configs: Vec<_> = server
        .seen()
        .iter()
        .filter(|s| path(s) == "/v1/config")
        .map(|s| s.header("authorization").unwrap_or("").to_owned())
        .collect();
    // Other groups, same claims: the same entry; other claims: a fetch of its own.
    assert_eq!(configs, ["Bearer x-alice-jwt", "Bearer x-alice-jwt"]);
}

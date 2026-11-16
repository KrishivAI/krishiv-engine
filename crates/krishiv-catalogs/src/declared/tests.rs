// SPDX-License-Identifier: Apache-2.0
//! The `iceberg-rest` factory (design P3-T067 §2.3, §3.1, §5.2): declarations to a catalog, the
//! auth modes, every request on the egress policy client.

use std::sync::Arc;

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::{Catalog, NameForm, TableIdent, TableMeta};
use krishiv_session::catalog_decl::{CatalogDecl, CatalogDeclarations, CatalogType, DeclMode};
use krishiv_session::catalog_registry::{CatalogDeps, CatalogFactory};
use krishiv_store::egress::{EgressPolicy, PrivateAddresses, ProviderEndpoints};

use super::IcebergRestFactory;
use crate::iceberg_rest::test_server::{Reply, Seen, Server, serve};

const NO_TABLE: &str =
    r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#;

fn path(seen: &Seen) -> &str {
    seen.target.split('?').next().unwrap_or("")
}

/// A catalog with `db.t` and nothing else.
fn catalog_server() -> Server {
    serve(|r| match path(r) {
        "/v1/config" => Reply::Json(200, r#"{"defaults": {}, "overrides": {}}"#.into()),
        "/v1/namespaces/db/tables/t" => Reply::Json(
            200,
            r#"{"metadata-location": "s3://b/t/metadata/v1.metadata.json",
                "metadata": {"format-version": 2, "table-uuid": "u", "location": "s3://b/t"},
                "config": {}}"#
                .into(),
        ),
        "/token" => Reply::Json(
            200,
            r#"{"access_token": "cc-token", "token_type": "bearer"}"#.into(),
        ),
        _ => Reply::Json(404, NO_TABLE.into()),
    })
}

fn decl(pairs: &[(&str, &str)], mode: DeclMode) -> CatalogDecl {
    let mut d = CatalogDeclarations::parse(
        pairs.iter().copied(),
        mode,
        &[CatalogType::IcebergRest],
        None,
    )
    .unwrap();
    d.catalogs.remove(0)
}

fn deps(mode: DeclMode, private: PrivateAddresses) -> CatalogDeps {
    CatalogDeps::new(
        mode,
        EgressPolicy::new(private, Vec::new(), ProviderEndpoints::default()),
    )
}

fn ident(ns: &[&str], name: &str) -> TableIdent {
    TableIdent {
        catalog: "lake".to_owned(),
        namespace: ns.iter().map(|s| (*s).to_owned()).collect(),
        name: name.to_owned(),
    }
}

const FORM: NameForm = NameForm {
    case_sensitive: false,
    qualified: true,
    temporary_only: false,
};

#[expect(
    clippy::unwrap_in_result,
    reason = "a test helper: the runtime always builds"
)]
fn load(catalog: &Arc<dyn Catalog>, id: &TableIdent) -> Result<Option<TableMeta>, KrishivError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(catalog.load_table(id, None, FORM))
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
    assert_eq!(catalog.name(), "lake");
    assert!(
        load(&catalog, &ident(&["db"], "missing"))
            .unwrap()
            .is_none()
    );
    // No namespace: absent, nothing sent.
    let sent = server.seen().len();
    assert!(load(&catalog, &ident(&[], "t")).unwrap().is_none());
    assert_eq!(server.seen().len(), sent);
    // A table the catalog has is refused until the provider exists, naming it.
    let e = load(&catalog, &ident(&["db"], "t")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("`lake`.`db`.`t`"), "{e}");
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
    let catalog: Arc<dyn Catalog> = Arc::new(super::RestCatalog {
        client: Arc::new(client),
    });
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

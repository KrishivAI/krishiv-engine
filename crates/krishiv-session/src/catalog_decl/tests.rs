// SPDX-License-Identifier: Apache-2.0
//! Declared catalogs (design P3-T067 §8, declarations).

use krishiv_io::egress::{EgressPolicy, PrivateAddresses, ProviderEndpoints};

use super::*;

const ALL: [CatalogType; 6] = [
    CatalogType::IcebergRest,
    CatalogType::Hive,
    CatalogType::Unity,
    CatalogType::Glue,
    CatalogType::Sqlite,
    CatalogType::Memory,
];

fn parse(pairs: &[(&str, &str)], mode: DeclMode) -> Result<CatalogDeclarations, KrishivError> {
    CatalogDeclarations::parse(pairs.iter().copied(), mode, &ALL, None)
}

fn err(pairs: &[(&str, &str)], mode: DeclMode) -> String {
    parse(pairs, mode).unwrap_err().to_string()
}

const REST: [(&str, &str); 6] = [
    ("krishiv.catalog.lake.type", "iceberg-rest"),
    (
        "krishiv.catalog.lake.uri",
        "https://catalog.example.com/api/catalog",
    ),
    ("krishiv.catalog.lake.warehouse", "analytics"),
    ("krishiv.catalog.lake.auth", "client-credentials"),
    ("krishiv.catalog.lake.clientId", "krishiv"),
    ("krishiv.catalog.lake.clientSecret", "env:LAKE_SECRET"),
];

fn rest_with(extra: &[(&'static str, &'static str)]) -> Vec<(&'static str, &'static str)> {
    let mut v = REST.to_vec();
    v.push((
        "krishiv.catalog.lake.tokenEndpoint",
        "https://idp.example.com/oauth2/token",
    ));
    v.extend_from_slice(extra);
    v
}

#[test]
fn catalog_keys_parse_same_in_all_modes() {
    let pairs = rest_with(&[("krishiv.catalog.mem.type", "memory")]);
    let server = parse(&pairs, DeclMode::Server).unwrap();
    let embedded = parse(&pairs, DeclMode::Embedded).unwrap();
    assert_eq!(server, embedded);
    let lake = &server.catalogs[0];
    assert_eq!(lake.name, "lake");
    assert_eq!(lake.kind, CatalogType::IcebergRest);
    assert_eq!(lake.storage, CatalogStorage::Vended);
    assert_eq!(lake.auth.name(), "client-credentials");
    assert_eq!(
        lake.declared_hosts(),
        ["catalog.example.com", "idp.example.com"]
    );
    assert_eq!(server.catalogs[1].storage, CatalogStorage::AllowedPrefixes);
}

#[test]
fn spark_catalog_alias_maps() {
    let pairs = [
        (
            "spark.sql.catalog.lake",
            "org.apache.iceberg.spark.SparkCatalog",
        ),
        ("spark.sql.catalog.lake.type", "rest"),
        (
            "spark.sql.catalog.lake.uri",
            "https://catalog.example.com/api/catalog",
        ),
        ("spark.sql.catalog.lake.warehouse", "analytics"),
        ("spark.sql.catalog.lake.credential", "id:secret"),
        (
            "spark.sql.catalog.lake.oauth2-server-uri",
            "https://idp.example.com/token",
        ),
        ("spark.sql.catalog.lake.scope", "catalog"),
        (
            "spark.sql.catalog.lake.header.X-Iceberg-Access-Delegation",
            "vended-credentials",
        ),
    ];
    let decls = parse(&pairs, DeclMode::Embedded).unwrap();
    let lake = &decls.catalogs[0];
    assert_eq!(lake.kind, CatalogType::IcebergRest);
    assert_eq!(lake.warehouse.as_deref(), Some("analytics"));
    match &lake.auth {
        CatalogAuth::ClientCredentials {
            client_id,
            token_endpoint,
            scope,
            ..
        } => {
            assert_eq!(client_id, "id");
            assert_eq!(token_endpoint, "https://idp.example.com/token");
            assert_eq!(scope.as_deref(), Some("catalog"));
        }
        other => panic!("{other:?}"),
    }
    // A Spark credential inline is refused in server mode (a reference only).
    let text = err(&pairs, DeclMode::Server);
    assert!(text.contains("spark.sql.catalog.lake.credential"), "{text}");
    assert!(!text.contains("id:secret"), "{text}");
    // Unity's class.
    let unity = parse(
        &[
            (
                "spark.sql.catalog.uc",
                "io.unitycatalog.spark.UCSingleCatalog",
            ),
            ("spark.sql.catalog.uc.uri", "https://uc.example.com"),
            ("spark.sql.catalog.uc.token", "env:UC_TOKEN"),
        ],
        DeclMode::Server,
    )
    .unwrap();
    assert_eq!(unity.catalogs[0].kind, CatalogType::Unity);
    assert_eq!(unity.catalogs[0].auth.name(), "token");
}

#[test]
fn unknown_catalog_class_fails_startup() {
    let text = err(
        &[("spark.sql.catalog.x", "com.example.MyCatalog")],
        DeclMode::Embedded,
    );
    assert!(text.contains("spark.sql.catalog.x"), "{text}");
    assert!(text.contains("com.example.MyCatalog"), "{text}");
    let text = err(
        &[("spark.sql.catalog.x.uri", "https://h")],
        DeclMode::Embedded,
    );
    assert!(text.contains("class"), "{text}");
}

#[test]
fn unknown_catalog_option_fails_startup() {
    let text = err(
        &rest_with(&[("krishiv.catalog.lake.uriTypo", "x")]),
        DeclMode::Server,
    );
    assert!(text.contains("krishiv.catalog.lake.uriTypo"), "{text}");
    let text = err(
        &rest_with(&[("krishiv.catalog.lake.option.s3.endpoint", "x")]),
        DeclMode::Server,
    );
    assert!(text.contains("option.s3.endpoint"), "{text}");
    let text = err(
        &[
            (
                "spark.sql.catalog.lake",
                "org.apache.iceberg.spark.SparkCatalog",
            ),
            ("spark.sql.catalog.lake.type", "rest"),
            ("spark.sql.catalog.lake.uri", "https://h"),
            ("spark.sql.catalog.lake.cache-enabled", "false"),
        ],
        DeclMode::Embedded,
    );
    assert!(text.contains("cache-enabled"), "{text}");
}

#[test]
fn java_only_catalog_option_warns() {
    let decls = parse(
        &[
            (
                "spark.sql.catalog.lake",
                "org.apache.iceberg.spark.SparkCatalog",
            ),
            ("spark.sql.catalog.lake.type", "rest"),
            ("spark.sql.catalog.lake.uri", "https://h"),
            (
                "spark.sql.catalog.lake.rest-metrics-reporting-enabled",
                "true",
            ),
            (
                "spark.sql.catalog.lake.io-impl",
                "org.apache.iceberg.aws.s3.S3FileIO",
            ),
        ],
        DeclMode::Server,
    )
    .unwrap();
    assert_eq!(decls.warnings.len(), 2, "{:?}", decls.warnings);
}

#[test]
fn custom_io_impl_fails_startup() {
    let text = err(
        &[
            (
                "spark.sql.catalog.lake",
                "org.apache.iceberg.spark.SparkCatalog",
            ),
            ("spark.sql.catalog.lake.type", "rest"),
            ("spark.sql.catalog.lake.uri", "https://h"),
            (
                "spark.sql.catalog.lake.io-impl",
                "com.example.EncryptingFileIO",
            ),
        ],
        DeclMode::Embedded,
    );
    assert!(text.contains("io-impl"), "{text}");
}

#[test]
fn compiled_out_catalog_type_fails_startup() {
    let e = CatalogDeclarations::parse(
        [("krishiv.catalog.g.type", "glue")],
        DeclMode::Server,
        &[CatalogType::Memory],
        None,
    )
    .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    assert!(e.to_string().contains("glue"), "{e}");
}

#[test]
fn literal_secret_refused_in_server_mode() {
    let mut pairs = rest_with(&[]);
    pairs.retain(|(k, _)| !k.ends_with("clientSecret"));
    pairs.push(("krishiv.catalog.lake.clientSecret", "hunter2-literal"));
    for mode in [DeclMode::Server, DeclMode::Embedded] {
        let text = err(&pairs, mode);
        assert!(text.contains("clientSecret"), "{text}");
        assert!(!text.contains("hunter2"), "{text}");
    }
}

#[test]
fn reserved_and_malformed_names_fail_startup() {
    for name in ["spark_catalog", "system"] {
        let key = format!("krishiv.catalog.{name}.type");
        let text = err(&[(key.as_str(), "memory")], DeclMode::Embedded);
        assert!(text.contains("reserved"), "{text}");
    }
    let text = err(
        &[("krishiv.catalog.Lake.type", "memory")],
        DeclMode::Embedded,
    );
    assert!(text.contains("lowercase"), "{text}");
}

#[test]
fn empty_tenants_fails_startup() {
    let text = err(
        &[
            ("krishiv.catalog.m.type", "memory"),
            ("krishiv.catalog.m.tenants", " , "),
        ],
        DeclMode::Server,
    );
    assert!(text.contains("krishiv.catalog.m.tenants"), "{text}");
}

#[test]
fn catalog_hidden_from_other_tenant() {
    let decls = parse(
        &[
            ("krishiv.catalog.m.type", "memory"),
            ("krishiv.catalog.m.tenants", "acme, globex"),
            ("krishiv.catalog.open.type", "memory"),
        ],
        DeclMode::Server,
    )
    .unwrap();
    let m = &decls.catalogs[0];
    assert!(m.visible_to(Some("acme")));
    assert!(!m.visible_to(Some("initech")));
    let open = &decls.catalogs[1];
    assert!(open.visible_to(Some("initech")));
    assert!(open.visible_to(None));
}

#[test]
fn tenantless_principal_hidden_from_tenant_catalog() {
    let decls = parse(
        &[
            ("krishiv.catalog.m.type", "memory"),
            ("krishiv.catalog.m.tenants", "acme"),
        ],
        DeclMode::Server,
    )
    .unwrap();
    assert!(!decls.catalogs[0].visible_to(None));
}

#[test]
fn http_uri_needs_allow_insecure_in_server_mode() {
    let mut pairs = rest_with(&[]);
    pairs.retain(|(k, _)| !k.ends_with(".uri"));
    pairs.push(("krishiv.catalog.lake.uri", "http://catalog.internal:8181"));
    assert!(err(&pairs, DeclMode::Server).contains("allowInsecure"));
    assert!(parse(&pairs, DeclMode::Embedded).is_ok());
    pairs.push(("krishiv.catalog.lake.allowInsecure", "true"));
    let decls = parse(&pairs, DeclMode::Server).unwrap();
    assert_eq!(decls.warnings.len(), 1);
}

#[test]
fn missing_keys_and_bad_values_fail_startup() {
    assert!(
        err(
            &[("krishiv.catalog.r.type", "iceberg-rest")],
            DeclMode::Embedded
        )
        .contains("uri")
    );
    assert!(err(&[("krishiv.catalog.r.type", "nessie")], DeclMode::Embedded).contains("nessie"));
    assert!(
        err(
            &[("krishiv.catalog.r.uri", "https://h")],
            DeclMode::Embedded
        )
        .contains("type")
    );
    // The token endpoint is required for client credentials (no `/v1/oauth/tokens`).
    let text = err(&REST, DeclMode::Server);
    assert!(text.contains("tokenEndpoint"), "{text}");
    let text = err(
        &[
            ("krishiv.catalog.m.type", "memory"),
            ("krishiv.catalog.default", "other"),
        ],
        DeclMode::Server,
    );
    assert!(text.contains("krishiv.catalog.default"), "{text}");
}

#[test]
fn delegated_and_remote_signing_refused_in_this_build() {
    let pairs = [
        ("krishiv.catalog.r.type", "iceberg-rest"),
        ("krishiv.catalog.r.uri", "https://h"),
        ("krishiv.catalog.r.auth", "delegated"),
    ];
    let e = parse(&pairs, DeclMode::Server).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    let pairs = [
        ("krishiv.catalog.r.type", "iceberg-rest"),
        ("krishiv.catalog.r.uri", "https://h"),
        ("krishiv.catalog.r.storage", "remote-signing"),
    ];
    let e = parse(&pairs, DeclMode::Server).unwrap_err();
    assert!(e.to_string().contains("remote signing"), "{e}");
}

#[test]
fn s3_tables_warehouse_refused() {
    let e = parse(
        &[
            ("krishiv.catalog.r.type", "iceberg-rest"),
            ("krishiv.catalog.r.uri", "https://h"),
            (
                "krishiv.catalog.r.warehouse",
                "arn:aws:s3tables:us-east-1:1:bucket/b",
            ),
        ],
        DeclMode::Server,
    )
    .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn glue_endpoint_checked_when_set() {
    let policy = EgressPolicy::new(
        PrivateAddresses::Deny,
        Vec::new(),
        ProviderEndpoints::default(),
    );
    let pairs = [
        ("krishiv.catalog.g.type", "glue"),
        (
            "krishiv.catalog.g.option.endpoint",
            "http://169.254.169.254",
        ),
    ];
    let decls = parse(&pairs, DeclMode::Server).unwrap();
    let scope = policy.scope(decls.catalogs[0].declared_hosts());
    let e = CatalogDeclarations::parse(pairs, DeclMode::Server, &ALL, Some(&scope)).unwrap_err();
    assert!(e.to_string().contains("option.endpoint"), "{e}");
    assert!(!e.to_string().contains("169.254"), "{e}");
    // A private on-premises endpoint is exempt by name for the Glue client.
    let pairs = [
        ("krishiv.catalog.g.type", "glue"),
        ("krishiv.catalog.g.option.endpoint", "http://10.0.0.7:4566"),
    ];
    let decls = parse(&pairs, DeclMode::Server).unwrap();
    let scope = policy.scope(decls.catalogs[0].declared_hosts());
    assert!(CatalogDeclarations::parse(pairs, DeclMode::Server, &ALL, Some(&scope)).is_ok());
}

#[test]
fn catalog_keys_are_recognised() {
    assert!(is_catalog_key("krishiv.catalog.x.uri"));
    assert!(is_catalog_key("SPARK.SQL.CATALOG.x"));
    assert!(!is_catalog_key("spark.sql.catalogImplementation"));
}

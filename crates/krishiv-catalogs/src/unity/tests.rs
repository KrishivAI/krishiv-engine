// SPDX-License-Identifier: Apache-2.0
//! End-to-end tests of `UnityCatalog` and `UnityClient` against a fake server that replays recorded UC (v0.7.0)
//! responses: the calls and their wire forms, paging, absence, the format and location policies, and the hostile-
//! catalog tests of design P3-lakehouse §5.6 (redirect not followed, request timeout, parse limits, locations).

use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::SecretString;
use krishiv_plan::analyzer::{Catalog, NameForm, TableIdent};
use krishiv_types::DataType;

use super::{UnityCatalog, UnityConfig, UnityFormat};
use crate::http_test_server::{Reply, Seen, Server, serve};
use crate::location::RemoteLocationPolicy;

fn path(seen: &Seen) -> &str {
    seen.target.split('?').next().unwrap_or("")
}

fn config_for(server: &Server) -> UnityConfig {
    let mut c = UnityConfig::new(&server.base, "unity").unwrap();
    c.allow_http = true;
    c
}

fn catalog_with(config: UnityConfig) -> UnityCatalog {
    UnityCatalog::new("uc", config, Arc::new(RemoteLocationPolicy::default())).unwrap()
}

fn catalog_of(server: &Server) -> UnityCatalog {
    catalog_with(config_for(server))
}

fn class(e: &KrishivError) -> ErrorClass {
    e.error_class()
}

/// A table body as `GET /tables/unity.default.marksheet` answers (OSS UC 0.7.0 shape).
fn table(format: &str, location: &str) -> String {
    format!(
        r#"{{"name": "marksheet", "catalog_name": "unity", "schema_name": "default",
            "table_type": "MANAGED", "data_source_format": "{format}", "storage_location": "{location}",
            "columns": [
              {{"name": "mark", "type_text": "int", "type_json": "{{}}", "type_name": "INT", "position": 1,
                "nullable": true, "comment": "the mark"}},
              {{"name": "id", "type_text": "int", "type_name": "INT", "position": 0, "nullable": false}},
              {{"name": "dt", "type_text": "string", "type_name": "STRING", "position": 2, "partition_index": 0}}
            ],
            "properties": {{"delta.minReaderVersion": "1", "n": 3}}, "table_id": "id-1"}}"#
    )
}

fn load(c: &UnityCatalog) -> Result<Option<super::UnityTable>, KrishivError> {
    c.load("default", "marksheet")
}

#[test]
fn a_delta_table_has_ordered_columns_partitions_location_properties_and_a_bearer_token() {
    let server = serve(|r| match path(r) {
        "/api/2.1/unity-catalog/tables/unity.default.marksheet" => {
            Reply::Json(200, table("DELTA", "s3://bucket/warehouse/marksheet"))
        }
        _ => Reply::Json(404, "{}".into()),
    });
    let mut config = config_for(&server);
    config.token = Some(SecretString::new("tok-1"));
    let t = load(&catalog_with(config)).unwrap().unwrap();
    assert_eq!(t.format, UnityFormat::Delta);
    assert_eq!(t.location, "s3://bucket/warehouse/marksheet");
    assert_eq!(t.table_type.as_deref(), Some("MANAGED"));
    let names: Vec<_> = t.meta.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["id", "mark", "dt"], "ordered by position");
    assert!(!t.meta.columns[0].nullable && t.meta.columns[1].nullable);
    assert_eq!(t.meta.columns[0].data_type, DataType::Integer);
    assert_eq!(t.meta.columns[2].data_type, DataType::String);
    assert_eq!(t.partition_columns, ["dt"]);
    assert_eq!(t.meta.column_comments.get("mark").unwrap(), "the mark");
    assert_eq!(t.meta.ident.catalog, "uc");
    assert_eq!(t.meta.ident.namespace, ["default"]);
    assert_eq!(t.properties.get("delta.minReaderVersion").unwrap(), "1");
    assert!(!t.properties.contains_key("n"), "only string properties");
    let seen = server.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].method, "GET");
    assert_eq!(seen[0].header("authorization"), Some("Bearer tok-1"));
    assert_eq!(seen[0].header("accept-encoding"), Some("identity"));
    assert!(seen[0].body.is_empty(), "a read sends no body");
}

#[test]
fn parquet_and_csv_expose_format_and_location() {
    for (declared, want) in [("parquet", UnityFormat::Parquet), ("CSV", UnityFormat::Csv)] {
        let body = table(declared, "gs://b/t");
        let server = serve(move |_| Reply::Json(200, body.clone()));
        let t = load(&catalog_of(&server)).unwrap().unwrap();
        assert_eq!(t.format, want);
        assert_eq!(t.location, "gs://b/t");
    }
}

#[test]
fn listings_follow_pages_and_send_the_names_in_the_query() {
    let server = serve(|r| match (path(r), r.target.as_str()) {
        ("/api/2.1/unity-catalog/catalogs", _) => Reply::Json(
            200,
            r#"{"catalogs": [{"name": "unity"}, {"name": "other"}]}"#.into(),
        ),
        ("/api/2.1/unity-catalog/schemas", t) => {
            assert!(t.contains("catalog_name=unity"), "{t}");
            Reply::Json(
                200,
                r#"{"schemas": [{"name": "default"}, {"name": "s 2"}]}"#.into(),
            )
        }
        ("/api/2.1/unity-catalog/tables", t) if !t.contains("page_token") => Reply::Json(
            200,
            r#"{"tables": [{"name": "a"}, {"name": "b"}], "next_page_token": "p 2"}"#.into(),
        ),
        ("/api/2.1/unity-catalog/tables", t) => {
            assert!(
                t.contains("catalog_name=unity") && t.contains("schema_name=default"),
                "{t}"
            );
            assert!(
                t.contains("page_token=p+2") || t.contains("page_token=p%202"),
                "{t}"
            );
            Reply::Json(200, r#"{"tables": [{"name": "c"}]}"#.into())
        }
        _ => Reply::Json(404, "{}".into()),
    });
    let c = catalog_of(&server);
    assert_eq!(c.list_schemas().unwrap(), ["default", "s 2"]);
    assert_eq!(c.list_tables("default").unwrap(), ["a", "b", "c"]);
    assert_eq!(server.seen().len(), 3);
}

#[test]
fn an_empty_listing_may_omit_its_array() {
    let server = serve(|_| Reply::Json(200, "{}".into()));
    assert!(catalog_of(&server).list_schemas().unwrap().is_empty());
}

#[test]
fn a_listing_that_repeats_its_token_or_never_ends_is_stopped() {
    let server = serve(|_| {
        Reply::Json(
            200,
            r#"{"schemas": [{"name": "a"}], "next_page_token": "same"}"#.into(),
        )
    });
    let e = catalog_of(&server).list_schemas().unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);

    let n = std::sync::atomic::AtomicU64::new(0);
    let server = serve(move |_| {
        let i = n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Reply::Json(
            200,
            format!(r#"{{"schemas": [{{"name": "a"}}], "next_page_token": "t{i}"}}"#),
        )
    });
    let mut config = config_for(&server);
    config.limits.max_pages = 3;
    let e = catalog_with(config).list_schemas().unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);
    assert_eq!(server.seen().len(), 3);

    let server =
        serve(|_| Reply::Json(200, r#"{"schemas": [{"name": "a"}, {"name": "b"}]}"#.into()));
    let mut config = config_for(&server);
    config.limits.max_list_entries = 1;
    let e = catalog_with(config).list_schemas().unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);

    let long = "x".repeat(5000);
    let server = serve(move |_| {
        Reply::Json(
            200,
            format!(r#"{{"schemas": [], "next_page_token": "{long}"}}"#),
        )
    });
    assert!(catalog_of(&server).list_schemas().is_err());
}

#[test]
fn a_listed_name_that_would_shape_a_url_is_refused() {
    for bad in ["..", "a/b", "a.b", ""] {
        let body = format!(r#"{{"schemas": [{{"name": "{bad}"}}]}}"#);
        let server = serve(move |_| Reply::Json(200, body.clone()));
        assert!(catalog_of(&server).list_schemas().is_err(), "{bad:?}");
    }
}

#[test]
fn an_unknown_table_is_absent_and_the_contract_suite_passes() {
    let server = serve(|_| {
        Reply::Json(
            404,
            r#"{"error_code": "TABLE_DOES_NOT_EXIST", "message": "Table not found", "details": []}"#.into(),
        )
    });
    let c = catalog_of(&server);
    assert!(load(&c).unwrap().is_none());
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(krishiv_plan::analyzer::contract::read_only_contract(&c))
        .unwrap();
    assert!(c.as_write().is_none() && !c.capabilities().writes);
}

#[test]
fn a_404_that_is_not_about_the_table_is_a_failure_not_absence() {
    for body in ["not json", r#"{"error_code": "ENDPOINT_MOVED"}"#, ""] {
        let sent = body.to_owned();
        let server = serve(move |_| Reply::Json(404, sent.clone()));
        let e = load(&catalog_of(&server)).unwrap_err();
        assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure, "{body}");
    }
}

#[test]
fn an_error_status_names_the_catalog_and_table_and_not_the_servers_words_or_the_token() {
    let server = serve(|_| {
        Reply::Json(
            500,
            r#"{"error_code": "INTERNAL", "message": "secret=hunter2 at /etc/passwd"}"#.into(),
        )
    });
    let mut config = config_for(&server);
    config.token = Some(SecretString::new("tok-secret"));
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    let text = e.to_string();
    assert!(
        text.contains("`uc`") && text.contains("`marksheet`"),
        "{text}"
    );
    for leak in ["hunter2", "passwd", "tok-secret"] {
        assert!(!text.contains(leak), "{text}");
    }
}

#[test]
fn formats_other_than_delta_parquet_and_csv_are_table_operation_refusals() {
    for (format, note) in [
        ("ICEBERG", "Iceberg REST"),
        ("AVRO", "AVRO"),
        ("a b/c", "unrecognised"),
    ] {
        let body = table(format, "s3://b/t");
        let server = serve(move |_| Reply::Json(200, body.clone()));
        let e = load(&catalog_of(&server)).unwrap_err();
        assert_eq!(
            class(&e),
            ErrorClass::UnsupportedFeatureTableOperation,
            "{format}"
        );
        assert!(e.to_string().contains(note), "{e}");
    }
    let server = serve(|_| {
        Reply::Json(
            200,
            r#"{"columns": [], "storage_location": "s3://b/t"}"#.into(),
        )
    });
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::UnsupportedFeatureTableOperation);
}

#[test]
fn a_location_the_rules_refuse_is_a_catalog_fault_that_never_names_it() {
    for location in [
        "file:///etc/shadow",
        "s3://k:s3cret@bucket/t",
        "s3://b/../x",
        "relative/path",
        "",
    ] {
        let body = table("DELTA", location);
        let server = serve(move |_| Reply::Json(200, body.clone()));
        let e = load(&catalog_of(&server)).unwrap_err();
        assert_eq!(
            class(&e),
            ErrorClass::KrishivCatalogLocationRefused,
            "{location:?}"
        );
        let text = e.to_string();
        for leak in ["shadow", "s3cret", "relative"] {
            assert!(!text.contains(leak), "{text}");
        }
    }
}

#[test]
fn a_redirect_is_not_followed() {
    let elsewhere = serve(|_| Reply::Json(200, table("DELTA", "s3://b/t")));
    let target = format!("{}/x", elsewhere.base);
    let server = serve(move |_| {
        Reply::Raw(
            format!("HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .into_bytes(),
        )
    });
    let mut config = config_for(&server);
    config.token = Some(SecretString::new("tok"));
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(
        elsewhere.seen().is_empty(),
        "the redirect target was contacted"
    );
}

#[test]
fn a_stalled_server_is_cut_at_the_request_timeout() {
    let server = serve(|_| Reply::Stall(Duration::from_secs(5)));
    let mut config = config_for(&server);
    config.limits.request_timeout = Duration::from_millis(300);
    let start = Instant::now();
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(
        start.elapsed() < Duration::from_secs(3),
        "{:?}",
        start.elapsed()
    );

    // A trickle is cut by the deadline of the whole call, not by a per-read timeout.
    let server = serve(|_| Reply::Trickle(table("DELTA", "s3://b/t"), Duration::from_millis(100)));
    let mut config = config_for(&server);
    config.limits.request_timeout = Duration::from_millis(500);
    let start = Instant::now();
    assert!(load(&catalog_with(config)).is_err());
    assert!(
        start.elapsed() < Duration::from_secs(3),
        "{:?}",
        start.elapsed()
    );
}

#[test]
fn oversize_deep_compressed_and_wide_responses_are_refused() {
    // Oversize: the declared length and a body that outgrows its declared length alike are over the limit.
    let body = table("DELTA", "s3://b/t");
    let server = serve(move |_| Reply::Json(200, body.clone()));
    let mut config = config_for(&server);
    config.limits.max_response_bytes = 100;
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);

    let raw = format!(
        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{}",
        "x".repeat(10_000)
    );
    let server = serve(move |_| Reply::Raw(raw.clone().into_bytes()));
    let mut config = config_for(&server);
    config.limits.max_response_bytes = 4096;
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);

    // Depth 65 against the default 64.
    let deep = format!(
        r#"{{"properties": {{"x": {}{}}}}}"#,
        "[".repeat(65),
        "]".repeat(65)
    );
    let server = serve(move |_| Reply::Json(200, deep.clone()));
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);

    // A compressed body is refused, never inflated.
    let mut raw = b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 10\r\nConnection: close\r\n\r\n".to_vec();
    raw.extend_from_slice(&[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3]);
    let server = serve(move |_| Reply::Raw(raw.clone()));
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
}

#[test]
fn a_table_wider_than_max_columns_counts_nested_fields() {
    let body = table("DELTA", "s3://b/t");
    let server = serve(move |_| Reply::Json(200, body.clone()));
    let mut config = config_for(&server);
    config.limits.max_columns = 2;
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);
    assert!(e.to_string().contains("krishiv.catalog.maxColumns"), "{e}");

    // Two columns, but one is a struct of three fields: width 5.
    let body = r#"{"data_source_format": "DELTA", "storage_location": "s3://b/t", "columns": [
        {"name": "a", "type_text": "int", "position": 0},
        {"name": "s", "type_text": "struct<x:int,y:int,z:int>", "position": 1}]}"#;
    let server = serve(move |_| Reply::Json(200, body.into()));
    let mut config = config_for(&server);
    config.limits.max_columns = 4;
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);
}

fn cols(types: &[&str]) -> String {
    let cols: Vec<String> = types
        .iter()
        .enumerate()
        .map(|(i, t)| format!(r#"{{"name": "c{i}", "type_text": "{t}", "position": {i}}}"#))
        .collect();
    format!(
        r#"{{"data_source_format": "PARQUET", "storage_location": "s3://b/t", "columns": [{}]}}"#,
        cols.join(",")
    )
}

#[test]
fn column_types_char_varchar_unknown_and_hostile() {
    let server =
        serve(move |_| Reply::Json(200, cols(&["VARCHAR(5)", "decimal(10,2)", "array<int>"])));
    let t = load(&catalog_of(&server)).unwrap().unwrap();
    assert_eq!(t.meta.columns[0].data_type, DataType::String);
    assert_eq!(
        t.meta.columns[0].char_varchar.as_deref(),
        Some("varchar(5)")
    );
    assert_eq!(t.meta.columns.len(), 3);

    let server = serve(move |_| Reply::Json(200, cols(&["not_a_type"])));
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);

    let deep = format!("{}int{}", "array<".repeat(500), ">".repeat(500));
    let server = serve(move |_| Reply::Json(200, cols(&[&deep])));
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
}

#[test]
fn the_catalog_trait_folds_names_and_has_no_versions_temporary_or_nested_names() {
    let body = table("DELTA", "s3://b/t");
    let server = serve(move |_| Reply::Json(200, body.clone()));
    let c = catalog_of(&server);
    assert_eq!(
        (c.name(), c.epoch(), c.single_part_namespace()),
        ("uc", 0, true)
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let id = |ns: &[&str]| TableIdent {
        catalog: "uc".to_owned(),
        namespace: ns.iter().map(|s| (*s).to_owned()).collect(),
        name: "MarkSheet".to_owned(),
    };
    let form = NameForm {
        case_sensitive: false,
        qualified: true,
        temporary_only: false,
    };
    let load = |id: &TableIdent, v, form| rt.block_on(c.load_table(id, v, form)).unwrap();
    assert!(load(&id(&["default"]), Some(krishiv_plan::TableVersion(1)), form).is_none());
    assert!(load(&id(&["a", "b"]), None, form).is_none());
    assert!(
        load(
            &id(&["default"]),
            None,
            NameForm {
                temporary_only: true,
                ..form
            }
        )
        .is_none()
    );
    assert!(server.seen().is_empty(), "none of these reaches the server");
    let meta = load(&id(&["DEFAULT"]), None, form).unwrap();
    assert_eq!(meta.ident.name, "marksheet");
    assert_eq!(
        server.seen()[0].target,
        "/api/2.1/unity-catalog/tables/unity.default.marksheet"
    );
}

#[test]
fn a_name_with_odd_characters_is_one_encoded_path_segment() {
    let server = serve(|_| Reply::Json(404, r#"{"error_code": "TABLE_DOES_NOT_EXIST"}"#.into()));
    let c = catalog_of(&server);
    assert!(c.load("my schema", "t?x#y%").unwrap().is_none());
    assert_eq!(
        server.seen()[0].target,
        "/api/2.1/unity-catalog/tables/unity.my%20schema.t%3Fx%23y%25"
    );
    // Names that cannot exist or would reshape the URL never reach the server.
    for (s, t) in [("..", "t"), ("a/b", "t"), ("s", "a.b"), ("s", "")] {
        assert!(c.load(s, t).unwrap().is_none(), "{s:?}.{t:?}");
    }
    assert_eq!(server.seen().len(), 1);
}

#[test]
fn the_server_uri_may_carry_a_path_prefix() {
    let server = serve(|_| Reply::Json(404, r#"{"error_code": "TABLE_DOES_NOT_EXIST"}"#.into()));
    let mut config = config_for(&server);
    config.uri = url::Url::parse(&format!("{}/uc/", server.base)).unwrap();
    assert!(load(&catalog_with(config)).unwrap().is_none());
    assert_eq!(
        server.seen()[0].target,
        "/uc/api/2.1/unity-catalog/tables/unity.default.marksheet"
    );
}

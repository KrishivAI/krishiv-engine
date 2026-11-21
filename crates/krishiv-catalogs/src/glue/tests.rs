// SPDX-License-Identifier: Apache-2.0
//! End-to-end tests of `GlueCatalog` and `GlueClient` against a fake server that replays Glue JSON-1.1 answers: the
//! signed wire form, paging, absence, the format and location policies, and the hostile-catalog tests of design
//! P3-lakehouse §5.6 (redirect not followed, request timeout, parse limits, locations, no secret in an error).

use std::sync::Arc;
use std::time::Duration;

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::SecretString;
use krishiv_plan::analyzer::{Catalog, NameForm, TableIdent};
use krishiv_types::DataType;
use url::Url;

use super::{GlueCatalog, GlueConfig, GlueFormat};
use crate::http_test_server::{Reply, Seen, Server, serve};
use crate::location::RemoteLocationPolicy;

const SECRET: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";

fn action(seen: &Seen) -> &str {
    seen.header("x-amz-target")
        .and_then(|t| t.strip_prefix("AWSGlue."))
        .unwrap_or("")
}

fn config_for(server: &Server) -> GlueConfig {
    let mut c = GlueConfig::new("eu-west-1", "AKIDEXAMPLE", SecretString::new(SECRET)).unwrap();
    c.endpoint = Url::parse(&server.base).unwrap();
    c.allow_http = true;
    c
}

fn catalog_with(config: GlueConfig) -> GlueCatalog {
    GlueCatalog::new("glue", config, Arc::new(RemoteLocationPolicy::default())).unwrap()
}

fn catalog_of(server: &Server) -> GlueCatalog {
    catalog_with(config_for(server))
}

fn class(e: &KrishivError) -> ErrorClass {
    e.error_class()
}

/// A `GetTable` answer in the shape Glue sends.
fn table(params: &str, location: &str, input_format: &str) -> String {
    format!(
        r#"{{"Table": {{"Name": "orders", "DatabaseName": "sales", "TableType": "EXTERNAL_TABLE",
            "StorageDescriptor": {{
              "Columns": [
                {{"Name": "id", "Type": "bigint"}},
                {{"Name": "note", "Type": "varchar(10)", "Comment": "free text"}},
                {{"Name": "tags", "Type": "array<string>"}}
              ],
              "Location": "{location}", "InputFormat": "{input_format}",
              "SerdeInfo": {{"SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"}}
            }},
            "PartitionKeys": [{{"Name": "dt", "Type": "string"}}],
            "Parameters": {{{params}}}}}}}"#
    )
}

fn load(c: &GlueCatalog) -> Result<Option<super::GlueTable>, KrishivError> {
    c.load("sales", "orders")
}

fn not_found() -> Reply {
    Reply::Json(
        400,
        r#"{"__type": "com.amazonaws.glue#EntityNotFoundException", "Message": "Table not found"}"#
            .into(),
    )
}

#[test]
fn a_parquet_table_has_columns_partitions_location_and_a_signed_request() {
    let server = serve(|r| match action(r) {
        "GetTable" => Reply::Json(
            200,
            table(
                r#""spark.sql.sources.provider": "parquet", "n": 3"#,
                "s3://bucket/sales/orders",
                "x",
            ),
        ),
        _ => not_found(),
    });
    let mut config = config_for(&server);
    config.session_token = Some(SecretString::new("sess-tok"));
    config.catalog_id = Some("123456789012".into());
    let t = load(&catalog_with(config)).unwrap().unwrap();
    assert_eq!(t.format, GlueFormat::Parquet);
    assert_eq!(t.location, "s3://bucket/sales/orders");
    assert_eq!(t.table_type.as_deref(), Some("EXTERNAL_TABLE"));
    let names: Vec<_> = t.meta.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["id", "note", "tags", "dt"], "partition keys last");
    assert_eq!(t.meta.columns[0].data_type, DataType::Long);
    assert_eq!(
        t.meta.columns[1].char_varchar.as_deref(),
        Some("varchar(10)")
    );
    assert_eq!(t.partition_columns, ["dt"]);
    assert_eq!(t.meta.column_comments.get("note").unwrap(), "free text");
    assert_eq!(t.meta.ident.namespace, ["sales"]);
    assert!(!t.properties.contains_key("n"), "only string parameters");

    let seen = server.seen();
    assert_eq!(seen.len(), 1);
    let r = &seen[0];
    assert_eq!(r.method, "POST");
    assert_eq!(r.target, "/");
    assert_eq!(r.header("content-type"), Some("application/x-amz-json-1.1"));
    assert_eq!(r.header("accept-encoding"), Some("identity"));
    assert_eq!(r.header("x-amz-security-token"), Some("sess-tok"));
    let body: serde_json::Value = serde_json::from_str(&r.body).unwrap();
    assert_eq!(body["DatabaseName"], "sales");
    assert_eq!(body["Name"], "orders");
    assert_eq!(body["CatalogId"], "123456789012");
    let auth = r.header("authorization").unwrap();
    assert!(
        auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
            && auth.contains("/eu-west-1/glue/aws4_request, ")
            && auth.contains(
                "SignedHeaders=content-type;host;x-amz-date;x-amz-security-token;x-amz-target, "
            ),
        "{auth}"
    );
    assert!(!auth.contains(SECRET));
}

#[test]
fn the_signature_covers_the_body_and_the_date_the_request_carries() {
    use super::sigv4::{Signing, hex, hmac_sha256};
    let server = serve(|_| not_found());
    load(&catalog_of(&server)).unwrap();
    let r = &server.seen()[0];
    let date = r.header("x-amz-date").unwrap();
    assert_eq!(date.len(), 16, "{date}");
    let host = r.header("host").unwrap();
    let expected = Signing {
        method: "POST",
        host,
        service: "glue",
        region: "eu-west-1",
        date,
        headers: &[
            ("content-type", "application/x-amz-json-1.1"),
            ("x-amz-target", "AWSGlue.GetTable"),
        ],
        body: r.body.as_bytes(),
        access_key_id: "AKIDEXAMPLE",
        secret_access_key: SECRET,
    }
    .authorization();
    assert_eq!(r.header("authorization"), Some(expected.as_str()));
    // A changed body changes the signature.
    let other = Signing {
        method: "POST",
        host,
        service: "glue",
        region: "eu-west-1",
        date,
        headers: &[
            ("content-type", "application/x-amz-json-1.1"),
            ("x-amz-target", "AWSGlue.GetTable"),
        ],
        body: b"{}",
        access_key_id: "AKIDEXAMPLE",
        secret_access_key: SECRET,
    }
    .authorization();
    assert_ne!(other, expected);
    let _ = (hex(&hmac_sha256(b"k", b"d")), 0);
}

#[test]
fn delta_csv_and_iceberg_tables_expose_format_and_location() {
    for (params, want, location) in [
        (
            r#""spark.sql.sources.provider": "delta""#,
            GlueFormat::Delta,
            "s3://b/t",
        ),
        (
            r#""spark.sql.sources.provider": "CSV""#,
            GlueFormat::Csv,
            "s3://b/t",
        ),
        (
            r#""table_type": "ICEBERG", "metadata_location": "s3://b/t/metadata/00001.metadata.json""#,
            GlueFormat::Iceberg,
            "s3://b/t/metadata/00001.metadata.json",
        ),
    ] {
        let body = table(params, "s3://b/t", "x");
        let server = serve(move |_| Reply::Json(200, body.clone()));
        let t = load(&catalog_of(&server)).unwrap().unwrap();
        assert_eq!(t.format, want, "{params}");
        assert_eq!(t.location, location, "{params}");
    }
}

#[test]
fn a_hive_table_takes_its_format_from_the_input_format_or_serde() {
    let body = table(
        "",
        "s3://b/t",
        "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
    );
    let server = serve(move |_| Reply::Json(200, body.clone()));
    assert_eq!(
        load(&catalog_of(&server)).unwrap().unwrap().format,
        GlueFormat::Parquet
    );
}

#[test]
fn listings_follow_pages_and_send_the_names_in_the_body() {
    let server = serve(|r| {
        let body: serde_json::Value = serde_json::from_str(&r.body).unwrap();
        match (action(r), body.get("NextToken").and_then(|t| t.as_str())) {
            ("GetDatabases", _) => Reply::Json(
                200,
                r#"{"DatabaseList": [{"Name": "default"}, {"Name": "sales"}]}"#.into(),
            ),
            ("GetTables", None) => {
                assert_eq!(body["DatabaseName"], "sales");
                assert_eq!(body["MaxResults"], 100);
                Reply::Json(
                    200,
                    r#"{"TableList": [{"Name": "a"}, {"Name": "b"}], "NextToken": "p 2"}"#.into(),
                )
            }
            ("GetTables", Some("p 2")) => {
                Reply::Json(200, r#"{"TableList": [{"Name": "c"}]}"#.into())
            }
            _ => not_found(),
        }
    });
    let c = catalog_of(&server);
    assert_eq!(c.list_databases().unwrap(), ["default", "sales"]);
    assert_eq!(c.list_tables("sales").unwrap(), ["a", "b", "c"]);
    assert_eq!(server.seen().len(), 3);
}

#[test]
fn an_empty_listing_may_omit_its_array() {
    let server = serve(|_| Reply::Json(200, "{}".into()));
    assert!(catalog_of(&server).list_databases().unwrap().is_empty());
}

#[test]
fn a_listing_that_repeats_its_token_or_never_ends_is_stopped() {
    let server = serve(|_| {
        Reply::Json(
            200,
            r#"{"DatabaseList": [{"Name": "a"}], "NextToken": "same"}"#.into(),
        )
    });
    let e = catalog_of(&server).list_databases().unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);

    let n = std::sync::atomic::AtomicU64::new(0);
    let server = serve(move |_| {
        let i = n.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Reply::Json(
            200,
            format!(r#"{{"DatabaseList": [{{"Name": "a"}}], "NextToken": "t{i}"}}"#),
        )
    });
    let mut config = config_for(&server);
    config.limits.max_pages = 3;
    let e = catalog_with(config).list_databases().unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);
    assert_eq!(server.seen().len(), 3);

    let server = serve(|_| {
        Reply::Json(
            200,
            r#"{"DatabaseList": [{"Name": "a"}, {"Name": "b"}]}"#.into(),
        )
    });
    let mut config = config_for(&server);
    config.limits.max_list_entries = 1;
    let e = catalog_with(config).list_databases().unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);

    let long = "x".repeat(5000);
    let server = serve(move |_| {
        Reply::Json(
            200,
            format!(r#"{{"DatabaseList": [], "NextToken": "{long}"}}"#),
        )
    });
    assert!(catalog_of(&server).list_databases().is_err());
}

#[test]
fn an_unknown_table_is_absent_and_the_contract_suite_passes() {
    let server = serve(|_| not_found());
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
fn an_error_that_is_not_entity_not_found_is_a_failure_not_absence() {
    for (status, body) in [
        (
            400,
            r#"{"__type": "InvalidInputException", "Message": "x"}"#,
        ),
        (400, r#"{"__type": "AccessDeniedException"}"#),
        (400, "not json"),
        (400, ""),
        (404, r#"{"__type": "EntityNotFoundException"}"#),
        (500, r#"{"__type": "InternalServiceException"}"#),
        (403, r#"{"__type": "UnrecognizedClientException"}"#),
    ] {
        let sent = body.to_owned();
        let server = serve(move |_| Reply::Json(status, sent.clone()));
        let e = load(&catalog_of(&server)).unwrap_err();
        assert_eq!(
            class(&e),
            ErrorClass::KrishivCatalogFailure,
            "{status} {body}"
        );
    }
}

#[test]
fn an_error_status_names_the_catalog_and_table_and_not_the_servers_words_or_the_keys() {
    let server = serve(|_| {
        Reply::Json(
            500,
            r#"{"__type": "InternalServiceException", "Message": "secret=hunter2 at /etc/passwd"}"#
                .into(),
        )
    });
    let mut config = config_for(&server);
    config.session_token = Some(SecretString::new("tok-secret"));
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    let text = e.to_string();
    assert!(
        text.contains("`glue`") && text.contains("`orders`"),
        "{text}"
    );
    for leak in ["hunter2", "passwd", "tok-secret", SECRET, "AKIDEXAMPLE"] {
        assert!(!text.contains(leak), "{text}");
    }
}

#[test]
fn formats_and_providers_krishiv_does_not_read_are_table_operation_refusals() {
    for (params, input, note) in [
        (r#""spark.sql.sources.provider": "avro""#, "x", "avro"),
        (
            r#""spark.sql.sources.provider": "a b/c""#,
            "x",
            "unrecognised",
        ),
        (
            "",
            "org.apache.hadoop.mapred.TextInputFormat",
            "does not read",
        ),
        (r#""table_type": "ICEBERG""#, "x", "no metadata_location"),
        (
            r#""spark.sql.sources.provider": "parquet", "spark.sql.sources.schema.numParts": "1""#,
            "x",
            "table properties",
        ),
    ] {
        let body = table(params, "s3://b/t", input);
        let body = if input.contains("Text") {
            body.replace(
                "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
            )
        } else {
            body
        };
        let server = serve(move |_| Reply::Json(200, body.clone()));
        let e = load(&catalog_of(&server)).unwrap_err();
        assert_eq!(
            class(&e),
            ErrorClass::UnsupportedFeatureTableOperation,
            "{params}"
        );
        assert!(e.to_string().contains(note), "{e}");
    }
}

#[test]
fn views_are_refused_before_the_location_check() {
    let body = table(
        r#""spark.sql.sources.provider": "parquet""#,
        "file:///etc/x",
        "x",
    )
    .replace("EXTERNAL_TABLE", "VIRTUAL_VIEW");
    let server = serve(move |_| Reply::Json(200, body.clone()));
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::UnsupportedFeatureTableOperation);
    assert!(e.to_string().contains("VIRTUAL_VIEW"), "{e}");
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
        let body = table(r#""spark.sql.sources.provider": "delta""#, location, "x");
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
    let elsewhere = serve(|_| Reply::Json(200, "{}".into()));
    let target = format!("{}/x", elsewhere.base);
    let server = serve(move |_| {
        Reply::Raw(
            format!("HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .into_bytes(),
        )
    });
    let e = load(&catalog_of(&server)).unwrap_err();
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
    let started = std::time::Instant::now();
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[test]
fn oversize_deep_and_compressed_responses_are_refused() {
    let big = table(
        r#""spark.sql.sources.provider": "parquet""#,
        "s3://b/t",
        "x",
    );
    let server = serve(move |_| Reply::Json(200, big.clone()));
    let mut config = config_for(&server);
    config.limits.max_response_bytes = 100;
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);

    let deep = format!("{}1{}", "[".repeat(200), "]".repeat(200));
    let server = serve(move |_| Reply::Json(200, deep.clone()));
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);

    let server = serve(|_| {
        Reply::Raw(
            b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
                .to_vec(),
        )
    });
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
}

#[test]
fn a_table_wider_than_max_columns_is_refused() {
    let server = serve(|_| {
        Reply::Json(
            200,
            table(
                r#""spark.sql.sources.provider": "parquet""#,
                "s3://b/t",
                "x",
            ),
        )
    });
    let mut config = config_for(&server);
    config.limits.max_columns = 3;
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogLimitExceeded);
}

#[test]
fn an_unusable_column_type_stays_out_of_the_error() {
    let body = table(
        r#""spark.sql.sources.provider": "parquet""#,
        "s3://b/t",
        "x",
    )
    .replace("array<string>", "wibble<secret=hunter2>");
    let server = serve(move |_| Reply::Json(200, body.clone()));
    let e = load(&catalog_of(&server)).unwrap_err();
    assert!(!e.to_string().contains("hunter2"), "{e}");
}

#[test]
fn the_catalog_trait_folds_names_and_has_no_versions_temporary_or_nested_names() {
    let server = serve(|r| {
        let body: serde_json::Value = serde_json::from_str(&r.body).unwrap();
        // Glue stores lower-case names: the request must carry them so.
        assert_eq!(body["DatabaseName"], "sales");
        assert_eq!(body["Name"], "orders");
        Reply::Json(
            200,
            table(
                r#""spark.sql.sources.provider": "parquet""#,
                "s3://b/t",
                "x",
            ),
        )
    });
    let c = catalog_of(&server);
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let ident = |ns: &[&str], name: &str| TableIdent {
        catalog: "glue".into(),
        namespace: ns.iter().map(|s| (*s).to_owned()).collect(),
        name: name.to_owned(),
    };
    let form = NameForm {
        case_sensitive: true,
        qualified: true,
        temporary_only: false,
    };
    let meta = rt
        .block_on(c.load_table(&ident(&["Sales"], "Orders"), None, form))
        .unwrap();
    assert!(meta.is_some());
    assert!(c.single_part_namespace());
    for id in [ident(&[], "orders"), ident(&["a", "b"], "orders")] {
        assert!(
            rt.block_on(c.load_table(&id, None, form))
                .unwrap()
                .is_none()
        );
    }
    let temporary = NameForm {
        case_sensitive: true,
        qualified: true,
        temporary_only: true,
    };
    assert!(
        rt.block_on(c.load_table(&ident(&["sales"], "orders"), None, temporary))
            .unwrap()
            .is_none()
    );
    assert_eq!(server.seen().len(), 1, "only the first lookup reached Glue");
}

#[test]
fn a_server_that_trickles_bytes_is_cut_at_the_deadline() {
    let body = table(
        r#""spark.sql.sources.provider": "parquet""#,
        "s3://b/t",
        "x",
    );
    let server = serve(move |_| Reply::Trickle(body.clone(), Duration::from_millis(40)));
    let mut config = config_for(&server);
    config.limits.request_timeout = Duration::from_millis(300);
    let started = std::time::Instant::now();
    let e = load(&catalog_with(config)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[test]
fn an_error_type_header_alone_makes_a_table_absent() {
    let body = r#"{"Message": "Table not found"}"#;
    let raw = format!(
        "HTTP/1.1 400 Bad Request\r\ncontent-type: application/x-amz-json-1.1\r\nx-amzn-errortype: EntityNotFoundException:http://internal.amazon.com/coral/\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    let server = serve(move |_| Reply::Raw(raw.clone().into_bytes()));
    assert!(load(&catalog_of(&server)).unwrap().is_none());
}

#[test]
fn a_listing_that_is_not_an_object_is_a_failure() {
    let server = serve(|_| Reply::Json(200, "[]".into()));
    let e = catalog_of(&server).list_databases().unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
}

#[test]
fn a_spark_schema_property_that_is_not_a_string_is_a_failure() {
    let server = serve(|_| {
        Reply::Json(
            200,
            table(
                r#""spark.sql.sources.provider": "parquet", "spark.sql.sources.schema": 7"#,
                "s3://b/t",
                "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            ),
        )
    });
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
}

#[test]
fn a_malformed_varchar_length_is_a_failure() {
    let server = serve(|_| {
        Reply::Json(
            200,
            table(
                "",
                "s3://b/t",
                "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            )
            .replace("varchar(10)", "varchar(not_a_length)"),
        )
    });
    let e = load(&catalog_of(&server)).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivCatalogFailure);
}

#[test]
fn the_declaration_is_checked_at_construction() {
    let key = || SecretString::new(SECRET);
    assert!(GlueConfig::new("eu-west-1", "", key()).is_err());
    assert!(GlueConfig::new("eu-west-1", "AKID", SecretString::new("")).is_err());
    let cn = GlueConfig::new("cn-north-1", "AKID", key()).unwrap();
    assert_eq!(
        cn.endpoint.as_str(),
        "https://glue.cn-north-1.amazonaws.com.cn/"
    );
    let limits = super::GlueLimits {
        request_timeout: Duration::MAX,
        ..super::GlueLimits::default()
    };
    assert!(limits.validate().is_err());
    let mut limits = super::GlueLimits::default();
    limits.max_depth = 128;
    assert!(limits.validate().is_err());
    limits.max_depth = 127;
    assert!(limits.validate().is_ok());
}

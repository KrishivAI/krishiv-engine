// SPDX-License-Identifier: Apache-2.0
//! Catalog metadata end to end (design P3-T101 §5, §3.2): `GetCatalogs`, `GetDbSchemas`,
//! `GetTables`, `GetTableTypes`, the key commands, `GetXdbcTypeInfo`; transient sessions, access as `SHOW`, the
//! `include_schema` cap.

use arrow_array::{Array, BinaryArray, StringArray};
use arrow_flight::sql::{
    CommandGetCatalogs, CommandGetCrossReference, CommandGetDbSchemas, CommandGetExportedKeys,
    CommandGetImportedKeys, CommandGetPrimaryKeys, CommandGetTableTypes, CommandGetTables,
    CommandGetXdbcTypeInfo, XdbcDataType,
};
use arrow_schema::{DataType, Schema};

use krishiv_common::error::KrishivError;
use krishiv_session::{
    AccessDecision, AccessPolicy, BoxFuture, PolicyVersion, Principal, Privilege, Securable,
};

use super::*;

/// The class a refusal names.
fn class_of(e: &Status) -> Option<&str> {
    e.metadata()
        .get("x-krishiv-error-class")
        .and_then(|v| v.to_str().ok())
}

/// A listing's answer: its rows, and whether `GetFlightInfo` or `DoGet` set a cookie.
struct Listed {
    batches: Vec<RecordBatch>,
    info_cookie: Option<String>,
    fetch_cookie: Option<String>,
}

/// `GetFlightInfo(cmd)` then `DoGet` of its ticket. The info's schema must equal the rows'.
async fn listing<M: ProstMessageExt>(
    h: &mut Harness,
    token: &str,
    cookie: Option<&str>,
    cmd: &M,
) -> Result<Listed, Status> {
    let descriptor = FlightDescriptor::new_cmd(cmd.as_any().encode_to_vec());
    let response = h
        .client
        .get_flight_info(authed(descriptor, token, cookie))
        .await?;
    let info_cookie = set_cookie(response.metadata());
    let info = response.into_inner();
    let schema = info.clone().try_decode_schema().expect("info schema");
    let response = h
        .client
        .do_get(authed(ticket(&info), token, cookie))
        .await?;
    let fetch_cookie = set_cookie(response.metadata());
    let data = response
        .into_inner()
        .map_err(|s| FlightError::Tonic(Box::new(s)));
    let batches: Vec<RecordBatch> = FlightRecordBatchStream::new_from_flight_data(data)
        .try_collect()
        .await
        .map_err(|e| match e {
            FlightError::Tonic(s) => *s,
            other => Status::internal(other.to_string()),
        })?;
    for b in &batches {
        assert_eq!(
            b.schema().as_ref(),
            &schema,
            "the info's schema is the rows'"
        );
    }
    Ok(Listed {
        batches,
        info_cookie,
        fetch_cookie,
    })
}

/// Column `name` of every row, as strings (`""` for null).
fn column(batches: &[RecordBatch], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        let c = b.column_by_name(name).expect("column");
        let s = c.as_any().downcast_ref::<StringArray>().expect("utf8");
        for i in 0..s.len() {
            out.push(if s.is_null(i) {
                String::new()
            } else {
                s.value(i).to_owned()
            });
        }
    }
    out
}

/// `GetTables` rows as `catalog.schema.name:type`, sorted.
fn table_rows(batches: &[RecordBatch]) -> Vec<String> {
    let (c, s, n, t) = (
        column(batches, "catalog_name"),
        column(batches, "db_schema_name"),
        column(batches, "table_name"),
        column(batches, "table_type"),
    );
    let mut rows: Vec<String> = (0..n.len())
        .map(|i| format!("{}.{}.{}:{}", c[i], s[i], n[i], t[i]))
        .collect();
    rows.sort();
    rows
}

fn tables(
    catalog: Option<&str>,
    schema: Option<&str>,
    name: Option<&str>,
    types: &[&str],
    include_schema: bool,
) -> CommandGetTables {
    CommandGetTables {
        catalog: catalog.map(str::to_owned),
        db_schema_filter_pattern: schema.map(str::to_owned),
        table_name_filter_pattern: name.map(str::to_owned),
        table_types: types.iter().map(|t| (*t).to_owned()).collect(),
        include_schema,
    }
}

fn schemas(catalog: Option<&str>, pattern: Option<&str>) -> CommandGetDbSchemas {
    CommandGetDbSchemas {
        catalog: catalog.map(str::to_owned),
        db_schema_filter_pattern: pattern.map(str::to_owned),
    }
}

/// Runs `sqls` for `token` in one cookie session: its cookie.
async fn setup(h: &mut Harness, token: &str, sqls: &[&str]) -> String {
    let mut cookie: Option<String> = None;
    for sql in sqls {
        let (_, set) = info(h, token, cookie.as_deref(), sql).await.expect(sql);
        cookie = cookie.or(set);
    }
    cookie.expect("a cookie")
}

/// The catalog, as `SHOW` sees it: schemas and tables of `spark_catalog`, temporary views
/// without a catalog, patterns ignoring case, types filtered (§5).
#[tokio::test(flavor = "multi_thread")]
async fn listings_show_the_catalog() {
    let wh = tempfile::tempdir().expect("tempdir");
    let mut h = start_with(
        FlightSqlConfig::default(),
        Setup {
            factory: Some(grant_sessions(wh.path())),
            ..Setup::default()
        },
    )
    .await;
    let cookie = setup(
        &mut h,
        OPS,
        &[
            "CREATE TEMPORARY VIEW tv AS SELECT 2 AS y",
            "CREATE DATABASE db1",
            "CREATE TABLE db1.t (a INT, b STRING) USING parquet",
            "CREATE VIEW db1.v AS SELECT 1 AS x",
            "CREATE GLOBAL TEMPORARY VIEW gv AS SELECT 3 AS z",
        ],
    )
    .await;
    let c = Some(cookie.as_str());

    let l = listing(&mut h, OPS, c, &CommandGetCatalogs {})
        .await
        .expect("catalogs");
    assert_eq!(column(&l.batches, "catalog_name"), ["spark_catalog"]);

    let l = listing(&mut h, OPS, c, &schemas(None, None))
        .await
        .expect("schemas");
    // The temporary views' schemas come under the empty catalog: the local views' empty one (the
    // session has `tv`) and `global_temp`, so a schemas-then-tables walk reaches them.
    assert_eq!(
        column(&l.batches, "db_schema_name"),
        ["", "global_temp", "db1", "default"]
    );
    assert_eq!(
        column(&l.batches, "catalog_name"),
        ["", "", "spark_catalog", "spark_catalog"]
    );
    let l = listing(&mut h, OPS, c, &schemas(Some(""), Some("global%")))
        .await
        .expect("temporary schemas");
    assert_eq!(column(&l.batches, "db_schema_name"), ["global_temp"]);
    for (pattern, want) in [
        ("D%", vec!["db1", "default"]),
        ("db_", vec!["db1"]),
        ("x%", vec![]),
    ] {
        let l = listing(
            &mut h,
            OPS,
            c,
            &schemas(Some("SPARK_CATALOG"), Some(pattern)),
        )
        .await
        .expect(pattern);
        assert_eq!(column(&l.batches, "db_schema_name"), want, "{pattern}");
    }
    let l = listing(&mut h, OPS, c, &schemas(Some("lake"), None))
        .await
        .expect("lake");
    assert!(column(&l.batches, "db_schema_name").is_empty());

    let all = listing(&mut h, OPS, c, &tables(None, None, None, &[], false))
        .await
        .expect("tables");
    assert_eq!(
        table_rows(&all.batches),
        [
            "..tv:TEMPORARY VIEW",
            ".global_temp.gv:TEMPORARY VIEW",
            "spark_catalog.db1.t:TABLE",
            "spark_catalog.db1.v:VIEW",
        ]
    );
    for (cmd, want) in [
        (
            tables(None, None, None, &["view"], false),
            vec!["spark_catalog.db1.v:VIEW"],
        ),
        (
            tables(Some(""), None, None, &[], false),
            vec!["..tv:TEMPORARY VIEW", ".global_temp.gv:TEMPORARY VIEW"],
        ),
        (
            tables(None, None, Some("T"), &[], false),
            vec!["spark_catalog.db1.t:TABLE"],
        ),
        (
            tables(Some("spark_catalog"), Some("default"), None, &[], false),
            vec![],
        ),
        (
            tables(None, Some("glob%"), None, &[], false),
            vec![".global_temp.gv:TEMPORARY VIEW"],
        ),
    ] {
        let l = listing(&mut h, OPS, c, &cmd).await.expect("filtered");
        assert_eq!(table_rows(&l.batches), want, "{cmd:?}");
    }

    let l = listing(
        &mut h,
        OPS,
        c,
        &tables(None, Some("db1"), Some("t"), &[], true),
    )
    .await
    .expect("with schema");
    let bytes = l.batches[0]
        .column_by_name("table_schema")
        .expect("table_schema")
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("binary")
        .value(0)
        .to_vec();
    let message = arrow_flight::IpcMessage(prost::bytes::Bytes::from(bytes));
    let schema = Schema::try_from(message).expect("an IPC schema");
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    assert_eq!(
        fields,
        [
            ("a".to_owned(), DataType::Int32),
            ("b".to_owned(), DataType::Utf8)
        ]
    );

    let l = listing(&mut h, OPS, c, &CommandGetTableTypes {})
        .await
        .expect("types");
    let mut types = column(&l.batches, "table_type");
    types.sort();
    assert_eq!(types, ["TABLE", "TEMPORARY VIEW", "VIEW"]);
}

/// The key commands answer empty results with the specified schemas (§5): Krishiv has no keys.
#[tokio::test(flavor = "multi_thread")]
async fn key_commands_are_empty() {
    let mut h = start(FlightSqlConfig::default()).await;
    let cookie = setup(&mut h, OPS, &["CREATE TEMPORARY VIEW k AS SELECT 1 AS id"]).await;
    let c = Some(cookie.as_str());
    let pk = CommandGetPrimaryKeys {
        catalog: None,
        db_schema: None,
        table: "k".to_owned(),
    };
    let l = listing(&mut h, OPS, c, &pk).await.expect("primary keys");
    assert_eq!(rows(&l.batches), 0);
    let exported = CommandGetExportedKeys {
        catalog: Some("spark_catalog".to_owned()),
        db_schema: Some("default".to_owned()),
        table: "missing".to_owned(),
    };
    assert_eq!(
        rows(
            &listing(&mut h, OPS, c, &exported)
                .await
                .expect("exported")
                .batches
        ),
        0
    );
    let imported = CommandGetImportedKeys {
        catalog: None,
        db_schema: None,
        table: "k".to_owned(),
    };
    assert_eq!(
        rows(
            &listing(&mut h, OPS, c, &imported)
                .await
                .expect("imported")
                .batches
        ),
        0
    );
    let cross = CommandGetCrossReference {
        pk_catalog: None,
        pk_db_schema: None,
        pk_table: "k".to_owned(),
        fk_catalog: None,
        fk_db_schema: None,
        fk_table: "k".to_owned(),
    };
    assert_eq!(
        rows(
            &listing(&mut h, OPS, c, &cross)
                .await
                .expect("cross")
                .batches
        ),
        0
    );
}

/// A cookie-less listing runs in a transient session: no `set-cookie`, nothing left (§3.2); with
/// a cookie it reads that session's state and leaves the session as it was.
#[tokio::test(flavor = "multi_thread")]
async fn cookieless_listing_runs_in_a_transient_session() {
    let mut h = start(FlightSqlConfig::default()).await;
    for _ in 0..3 {
        let l = listing(&mut h, OPS, None, &tables(None, None, None, &[], false))
            .await
            .expect("tables");
        assert_eq!(l.info_cookie, None);
        assert_eq!(l.fetch_cookie, None);
        assert!(table_rows(&l.batches).is_empty());
    }
    wait_sessions(&h, 0).await;
    let cookie = setup(
        &mut h,
        OPS,
        &["CREATE TEMPORARY VIEW mine AS SELECT 1 AS n"],
    )
    .await;
    wait_sessions(&h, 1).await;
    let l = listing(
        &mut h,
        OPS,
        Some(&cookie),
        &tables(Some(""), None, None, &[], false),
    )
    .await
    .expect("tables");
    assert_eq!(table_rows(&l.batches), ["..mine:TEMPORARY VIEW"]);
    assert_eq!(
        l.fetch_cookie, None,
        "the cookie session is the client's already"
    );
    wait_sessions(&h, 1).await;
}

/// A listing is checked as the `SHOW` it stands for (ADR 0046 §3b): a principal refused `SHOW
/// TABLES IN db1` is refused the listing of `db1`'s tables, with the same status.
#[tokio::test(flavor = "multi_thread")]
async fn listings_are_checked_as_show() {
    let wh = tempfile::tempdir().expect("tempdir");
    let mut h = start_with(
        FlightSqlConfig::default(),
        Setup {
            factory: Some(grant_sessions(wh.path())),
            ..Setup::default()
        },
    )
    .await;
    setup(
        &mut h,
        OPS,
        &[
            "CREATE DATABASE db1",
            "CREATE TABLE db1.secret (card STRING) USING parquet",
        ],
    )
    .await;
    let show = match info(&mut h, OTHER, None, "SHOW TABLES IN db1").await {
        Ok((i, cookie)) => fetch(&mut h, OTHER, cookie.as_deref(), ticket(&i))
            .await
            .map(|b| rows(&b)),
        Err(e) => Err(e),
    }
    .expect_err("SHOW TABLES IN db1 is refused");
    let listed = listing(
        &mut h,
        OTHER,
        None,
        &tables(Some("spark_catalog"), Some("db1"), None, &[], false),
    )
    .await
    .err()
    .expect("the listing is refused too");
    for e in [&show, &listed] {
        assert_eq!(e.code(), Code::PermissionDenied, "{e:?}");
        assert_eq!(class_of(e), Some("KRISHIV_INSUFFICIENT_PERMISSIONS"));
    }
    // `SHOW NAMESPACES` with `spark_catalog` current: `Browse` on the catalog.
    let e = listing(&mut h, OTHER, None, &schemas(None, None))
        .await
        .err()
        .expect("schemas refused");
    assert_eq!(e.code(), Code::PermissionDenied, "{e:?}");
    // GetCatalogs reads no namespace: the catalogs this principal sees.
    let l = listing(&mut h, OTHER, None, &CommandGetCatalogs {})
        .await
        .expect("catalogs");
    assert_eq!(column(&l.batches, "catalog_name"), ["spark_catalog"]);
}

/// `include_schema` loads at most `maxSchemaTables` schemas: past it the listing is refused
/// before any is loaded (`INVALID_ARGUMENT`, narrow the pattern); without schemas no cap (§5).
#[tokio::test(flavor = "multi_thread")]
async fn include_schema_over_the_cap_is_invalid_argument() {
    let mut h = start(FlightSqlConfig {
        max_schema_tables: 1,
        ..FlightSqlConfig::default()
    })
    .await;
    let cookie = setup(
        &mut h,
        OPS,
        &[
            "CREATE TEMPORARY VIEW a1 AS SELECT 1 AS n",
            "CREATE TEMPORARY VIEW a2 AS SELECT 2 AS n",
        ],
    )
    .await;
    let c = Some(cookie.as_str());
    let e = listing(&mut h, OPS, c, &tables(None, None, Some("a%"), &[], true))
        .await
        .err()
        .expect("over the cap");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(class_of(&e), Some("KRISHIV_FLIGHT_SQL_SCHEMA_TABLES_LIMIT"));
    let l = listing(&mut h, OPS, c, &tables(None, None, Some("a%"), &[], false))
        .await
        .expect("no schemas, no cap");
    assert_eq!(rows(&l.batches), 2);
    let l = listing(&mut h, OPS, c, &tables(None, None, Some("a1"), &[], true))
        .await
        .expect("one schema");
    assert_eq!(rows(&l.batches), 1);
}

/// `GetXdbcTypeInfo` (design §5): one row per Spark type, filtered by `data_type`, without a
/// session.
#[tokio::test(flavor = "multi_thread")]
async fn xdbc_type_info_lists_the_spark_types() {
    let mut h = start(FlightSqlConfig::default()).await;
    let all = listing(
        &mut h,
        OPS,
        None,
        &CommandGetXdbcTypeInfo { data_type: None },
    )
    .await
    .expect("type info");
    assert_eq!(all.info_cookie, None);
    assert_eq!(all.fetch_cookie, None);
    let names = column(&all.batches, "type_name");
    for name in [
        "BOOLEAN",
        "TINYINT",
        "SMALLINT",
        "INT",
        "BIGINT",
        "FLOAT",
        "DOUBLE",
        "DECIMAL",
        "STRING",
        "BINARY",
        "DATE",
        "TIMESTAMP",
        "TIMESTAMP_NTZ",
        "INTERVAL YEAR TO MONTH",
        "INTERVAL DAY TO SECOND",
    ] {
        assert!(names.iter().any(|n| n == name), "{name} in {names:?}");
    }
    let ints = listing(
        &mut h,
        OPS,
        None,
        &CommandGetXdbcTypeInfo {
            data_type: Some(XdbcDataType::XdbcInteger as i32),
        },
    )
    .await
    .expect("filtered type info");
    assert_eq!(column(&ints.batches, "type_name"), ["INT"]);
    wait_sessions(&h, 0).await;
}

/// A policy for the metadata access tests: `ops` may do anything; `other` may only browse
/// (a viewer), and not namespace `hr` or `global_temp` or what is in them.
struct BrowseButHr;

impl AccessPolicy for BrowseButHr {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let shown = securable.to_string();
        let in_hr = shown
            .split('.')
            .any(|p| matches!(p.trim_matches('`'), "hr" | "global_temp"));
        let allowed = principal.name == "ops" || (privilege == Privilege::Browse && !in_hr);
        Box::pin(std::future::ready(Ok(if allowed {
            AccessDecision::Allow
        } else {
            AccessDecision::Deny {
                reason: "test policy".to_owned(),
            }
        })))
    }
}

/// Sessions over one runtime under [`BrowseButHr`].
fn browse_sessions(dir: &Path) -> Box<krishiv_connect_server::manager::SessionFactory> {
    let rt = krishiv_session::EngineRuntime::builder()
        .config(
            "spark.sql.warehouse.dir",
            dir.join("wh").to_str().expect("utf-8 path"),
        )
        .build()
        .expect("runtime");
    Box::new(move |caller, _transport, _| {
        Session::builder()
            .runtime(Arc::clone(&rt))
            .principal(caller.principal.clone())
            .access_policy(Arc::new(BrowseButHr) as Arc<dyn AccessPolicy>)
            .build()
    })
}

/// A viewer's `GetTables` with schemas (JDBC `getColumns`) answers each table's schema: it is
/// metadata, asked as `DESCRIBE` asks it (`Browse`), not a read (review finding 1).
#[tokio::test(flavor = "multi_thread")]
async fn a_viewer_gets_table_schemas() {
    let wh = tempfile::tempdir().expect("tempdir");
    let mut h = start_with(
        FlightSqlConfig::default(),
        Setup {
            factory: Some(browse_sessions(wh.path())),
            ..Setup::default()
        },
    )
    .await;
    setup(
        &mut h,
        OPS,
        &["CREATE TABLE default.t (a INT, b STRING) USING parquet"],
    )
    .await;
    let e = match info(&mut h, OTHER, None, "SELECT a FROM default.t").await {
        Ok((i, cookie)) => fetch(&mut h, OTHER, cookie.as_deref(), ticket(&i))
            .await
            .map(|b| rows(&b)),
        Err(e) => Err(e),
    }
    .expect_err("a viewer cannot read the table");
    assert_eq!(e.code(), Code::PermissionDenied, "{e:?}");
    let l = listing(
        &mut h,
        OTHER,
        None,
        &tables(Some("spark_catalog"), Some("default"), Some("t"), &[], true),
    )
    .await
    .expect("a viewer lists schemas");
    assert_eq!(table_rows(&l.batches), ["spark_catalog.default.t:TABLE"]);
    let bytes = l.batches[0]
        .column_by_name("table_schema")
        .expect("table_schema")
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("binary")
        .value(0)
        .to_vec();
    let message = arrow_flight::IpcMessage(prost::bytes::Bytes::from(bytes));
    let schema = Schema::try_from(message).expect("an IPC schema");
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["a", "b"]);
}

/// A namespace an unfiltered or pattern listing reaches but the caller may not browse is absent,
/// as `SHOW` hides it; naming it exactly is still refused (review finding 2).
#[tokio::test(flavor = "multi_thread")]
async fn a_hidden_namespace_is_absent_from_table_listings() {
    let wh = tempfile::tempdir().expect("tempdir");
    let mut h = start_with(
        FlightSqlConfig::default(),
        Setup {
            factory: Some(browse_sessions(wh.path())),
            ..Setup::default()
        },
    )
    .await;
    setup(
        &mut h,
        OPS,
        &[
            "CREATE DATABASE hr",
            "CREATE TABLE hr.salaries (s INT) USING parquet",
            "CREATE DATABASE sales",
            "CREATE TABLE sales.orders (o INT) USING parquet",
        ],
    )
    .await;
    for pattern in [None, Some("%"), Some("h%")] {
        let l = listing(
            &mut h,
            OTHER,
            None,
            &tables(None, pattern, None, &[], false),
        )
        .await
        .unwrap_or_else(|e| panic!("{pattern:?}: {e:?}"));
        let rows = table_rows(&l.batches);
        assert!(
            rows.iter().all(|r| !r.contains(".hr.")),
            "{pattern:?}: {rows:?}"
        );
        if pattern != Some("h%") {
            assert!(
                rows.contains(&"spark_catalog.sales.orders:TABLE".to_owned()),
                "{pattern:?}: {rows:?}"
            );
        }
    }
    let e = listing(
        &mut h,
        OTHER,
        None,
        &tables(None, Some("hr"), None, &[], false),
    )
    .await
    .err()
    .expect("an exact name is refused as SHOW TABLES IN hr is");
    assert_eq!(e.code(), Code::PermissionDenied, "{e:?}");
    assert_eq!(class_of(&e), Some("KRISHIV_INSUFFICIENT_PERMISSIONS"));
    // An exact `global_temp` (`_` is a wildcard, so `global\_temp`) answers as
    // `SHOW TABLES IN global_temp` does; the plain spelling is a pattern, so a refused namespace
    // it matches is absent.
    let show = match info(&mut h, OTHER, None, "SHOW TABLES IN global_temp").await {
        Ok((i, cookie)) => fetch(&mut h, OTHER, cookie.as_deref(), ticket(&i))
            .await
            .map(|b| rows(&b)),
        Err(e) => Err(e),
    };
    // The policy refuses `global_temp` to `other`, so the refusal path below is reached.
    assert!(
        show.is_err(),
        "SHOW TABLES IN global_temp is refused: {show:?}"
    );
    let exact = listing(
        &mut h,
        OTHER,
        None,
        &tables(None, Some(r"global\_temp"), None, &[], false),
    )
    .await;
    assert_eq!(
        show.as_ref().map_err(Status::code).err(),
        exact.as_ref().map_err(Status::code).err(),
        "SHOW {show:?} vs listing"
    );
    // With `global_temp` refused, an unfiltered `GetDbSchemas` answers (it hides the temporary
    // views' schemas, review of 969a350b4 nit 1) and lists the rest.
    let l = listing(&mut h, OTHER, None, &schemas(None, None))
        .await
        .expect("schemas");
    let names = column(&l.batches, "db_schema_name");
    assert!(names.iter().any(|n| n == "sales"), "{names:?}");
    assert!(!names.iter().any(|n| n == "global_temp"), "{names:?}");
}

/// A listing past `metadataTimeout` is `DEADLINE_EXCEEDED` with its class (design §6).
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_listing_times_out() {
    let mut h = start(FlightSqlConfig {
        metadata_timeout: std::time::Duration::from_nanos(1),
        ..FlightSqlConfig::default()
    })
    .await;
    let sqls: Vec<String> = (0..20)
        .map(|i| format!("CREATE TEMPORARY VIEW v{i} AS SELECT {i} AS a"))
        .collect();
    let refs: Vec<&str> = sqls.iter().map(String::as_str).collect();
    let cookie = setup(&mut h, OPS, &refs).await;
    let e = listing(
        &mut h,
        OPS,
        Some(&cookie),
        &tables(None, None, None, &[], true),
    )
    .await
    .err()
    .expect("timed out");
    assert_eq!(e.code(), Code::DeadlineExceeded, "{e:?}");
    assert_eq!(class_of(&e), Some("KRISHIV_FLIGHT_SQL_METADATA_TIMEOUT"));
}

// SPDX-License-Identifier: Apache-2.0
//! The `Catalog` relation over Spark Connect (CM group 9, PR-46; Spark's
//! `SparkConnectPlanner.transformCatalog`), end to end over a raw tonic client: every operation
//! answers the rows of the `Dataset` Spark builds (names, types, nullability), the calls that
//! change something change the session, errors keep their Spark class, and a catalog relation
//! under another relation reads like any other.
#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration-test helpers are outside #[test] fns"
)]
#![expect(
    clippy::disallowed_methods,
    reason = "the tests read back the server's status details and Arrow data"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::catalog::CatType;
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{ConnectServer, ConnectServerConfig, SessionLimits, SessionManager};
use krishiv_session::Session;
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "5b0e4c55-5ad0-4d1e-9a3c-3a1b0b2c7d77";

struct Running {
    channel: Channel,
    dir: tempfile::TempDir,
    _stop: oneshot::Sender<()>,
}

async fn start() -> Running {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8").to_owned();
    let factory = move |_: &krishiv_session::auth::Authenticated,
                        _: krishiv_session::Transport,
                        _: Option<krishiv_plan::analyzer::SparkProfile>| {
        let s = Session::new()?;
        s.conf().set("spark.sql.warehouse.dir", &warehouse)?;
        Ok(s)
    };
    let manager = Arc::new(SessionManager::new(
        Box::new(factory),
        SessionLimits::default(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = ConnectServer::new(manager, ConnectServerConfig::default());
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    Running {
        channel,
        dir,
        _stop: stop,
    }
}

fn cat(kind: CatType) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(RelType::Catalog(pb::Catalog {
            cat_type: Some(kind),
        })),
    }
}

fn sql(text: &str) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(RelType::Sql(pb::Sql {
            query: text.to_owned(),
            ..Default::default()
        })),
    }
}

/// Runs `relation` and returns its batches, or the status.
async fn run(run: &Running, relation: pb::Relation) -> Result<Vec<RecordBatch>, tonic::Status> {
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let plan = pb::Plan {
        op_type: Some(OpType::Root(Box::new(relation))),
    };
    let mut live = grpc
        .server_streaming(
            tonic::Request::new(pb::ExecutePlanRequest {
                session_id: SESSION.to_owned(),
                plan: Some(plan),
                ..Default::default()
            }),
            "/spark.connect.SparkConnectService/ExecutePlan"
                .parse()
                .expect("path"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await?
        .into_inner();
    let mut out = Vec::new();
    while let Some(m) = live.message().await? {
        if let Some(ResponseType::ArrowBatch(b)) = m.response_type {
            out.extend(
                arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(b.data), None)
                    .expect("ipc")
                    .collect::<Result<Vec<_>, _>>()
                    .expect("batches"),
            );
        }
    }
    Ok(out)
}

async fn ok(r: &Running, relation: pb::Relation) -> Vec<RecordBatch> {
    run(r, relation).await.unwrap_or_else(|e| panic!("{e:?}"))
}

/// Runs a statement as a `sql_command`, as `spark.sql()` does for DDL and DML.
async fn ddl(r: &Running, text: &str) {
    let mut grpc = tonic::client::Grpc::new(r.channel.clone());
    grpc.ready().await.expect("ready");
    let plan = pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(pb::command::CommandType::SqlCommand(Box::new(
                pb::SqlCommand {
                    input: Some(sql(text)),
                    ..Default::default()
                },
            ))),
        }))),
    };
    let mut live = grpc
        .server_streaming(
            tonic::Request::new(pb::ExecutePlanRequest {
                session_id: SESSION.to_owned(),
                plan: Some(plan),
                ..Default::default()
            }),
            "/spark.connect.SparkConnectService/ExecutePlan"
                .parse()
                .expect("path"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await
        .unwrap_or_else(|e| panic!("{text}: {e:?}"))
        .into_inner();
    while live
        .message()
        .await
        .unwrap_or_else(|e| panic!("{text}: {e:?}"))
        .is_some()
    {}
}

/// The strings of column `name` over `batches`.
fn strings(batches: &[RecordBatch], name: &str) -> Vec<Option<String>> {
    let mut out = Vec::new();
    for b in batches {
        let column = b
            .column_by_name(name)
            .unwrap_or_else(|| panic!("column {name}"));
        let text = column.as_string::<i32>();
        out.extend((0..text.len()).map(|i| (!text.is_null(i)).then(|| text.value(i).to_owned())));
    }
    out
}

fn names(batches: &[RecordBatch], name: &str) -> Vec<String> {
    strings(batches, name).into_iter().flatten().collect()
}

fn bools(batches: &[RecordBatch], name: &str) -> Vec<bool> {
    batches
        .iter()
        .flat_map(|b| {
            b.column_by_name(name)
                .expect("column")
                .as_boolean()
                .iter()
                .flatten()
                .collect::<Vec<_>>()
        })
        .collect()
}

fn rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

#[derive(Clone, PartialEq, prost::Message)]
struct RpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, repeated, tag = "3")]
    details: Vec<prost_types::Any>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct ErrorInfo {
    #[prost(string, tag = "1")]
    reason: String,
    #[prost(string, tag = "2")]
    domain: String,
    #[prost(btree_map = "string, string", tag = "3")]
    metadata: BTreeMap<String, String>,
}

fn error_class(status: &tonic::Status) -> String {
    let rpc = RpcStatus::decode(status.details()).expect("google.rpc.Status");
    ErrorInfo::decode(rpc.details[0].value.as_slice())
        .expect("ErrorInfo")
        .metadata
        .get("errorClass")
        .cloned()
        .unwrap_or_default()
}

fn current_database() -> pb::Relation {
    cat(CatType::CurrentDatabase(pb::CurrentDatabase {}))
}

fn set_database(db: &str) -> pb::Relation {
    cat(CatType::SetCurrentDatabase(pb::SetCurrentDatabase {
        db_name: db.to_owned(),
    }))
}

fn create_database(db: &str) -> pb::Relation {
    cat(CatType::CreateDatabase(pb::CreateDatabase {
        db_name: db.to_owned(),
        if_not_exists: false,
        properties: BTreeMap::from([("team".to_owned(), "krishiv".to_owned())]),
    }))
}

fn int_struct(columns: &[&str]) -> pb::DataType {
    use pb::data_type::{Integer, Kind, Struct, StructField};
    pb::DataType {
        kind: Some(Kind::Struct(Struct {
            fields: columns
                .iter()
                .map(|c| StructField {
                    name: (*c).to_owned(),
                    data_type: Some(pb::DataType {
                        kind: Some(Kind::Integer(Integer::default())),
                    }),
                    nullable: true,
                    metadata: None,
                })
                .collect(),
            type_variation_reference: 0,
        })),
    }
}

fn create_table(name: &str, schema: Option<pb::DataType>) -> pb::Relation {
    cat(CatType::CreateTable(pb::CreateTable {
        table_name: name.to_owned(),
        path: None,
        source: Some("parquet".to_owned()),
        description: Some("made over connect".to_owned()),
        schema,
        options: BTreeMap::new(),
    }))
}

fn list_tables(db: Option<&str>, pattern: Option<&str>) -> pb::Relation {
    cat(CatType::ListTables(pb::ListTables {
        db_name: db.map(str::to_owned),
        pattern: pattern.map(str::to_owned),
    }))
}

fn table_exists(db: Option<&str>, name: &str) -> pb::Relation {
    cat(CatType::TableExists(pb::TableExists {
        table_name: name.to_owned(),
        db_name: db.map(str::to_owned),
    }))
}

#[tokio::test]
async fn the_current_database_and_catalog_are_one_string_row() {
    let r = start().await;
    let b = ok(&r, current_database()).await;
    assert_eq!(names(&b, "value"), ["default"]);
    assert_eq!(b[0].schema().field(0).data_type(), &DataType::Utf8);
    let b = ok(&r, cat(CatType::CurrentCatalog(pb::CurrentCatalog {}))).await;
    assert_eq!(names(&b, "value"), ["spark_catalog"]);
}

#[tokio::test]
async fn a_set_call_changes_the_session_and_answers_no_rows() {
    let r = start().await;
    assert_eq!(rows(&ok(&r, create_database("db1")).await), 0);
    let b = ok(&r, set_database("db1")).await;
    assert_eq!(rows(&b), 0);
    // Spark's empty relation: a `value` column that is not nullable.
    let field = b[0]
        .schema()
        .field_with_name("value")
        .expect("value")
        .clone();
    assert!(!field.is_nullable());
    assert_eq!(names(&ok(&r, current_database()).await, "value"), ["db1"]);
    // The statement path sees the change too.
    let b = ok(&r, sql("SELECT current_database() AS d")).await;
    assert_eq!(names(&b, "d"), ["db1"]);
}

#[tokio::test]
async fn databases_list_get_exist_and_drop() {
    let r = start().await;
    ok(&r, create_database("alpha")).await;
    ok(&r, create_database("beta")).await;
    let b = ok(
        &r,
        cat(CatType::ListDatabases(pb::ListDatabases { pattern: None })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["alpha", "beta", "default"]);
    let schema = b[0].schema();
    let fields: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(fields, ["name", "catalog", "description", "locationUri"]);
    let b = ok(
        &r,
        cat(CatType::ListDatabases(pb::ListDatabases {
            pattern: Some("al*".to_owned()),
        })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["alpha"]);
    let b = ok(
        &r,
        cat(CatType::GetDatabase(pb::GetDatabase {
            db_name: "beta".to_owned(),
        })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["beta"]);
    assert_eq!(names(&b, "catalog"), ["spark_catalog"]);
    let exists = |db: &str| {
        cat(CatType::DatabaseExists(pb::DatabaseExists {
            db_name: db.to_owned(),
        }))
    };
    assert_eq!(bools(&ok(&r, exists("alpha")).await, "value"), [true]);
    ok(
        &r,
        cat(CatType::DropDatabase(pb::DropDatabase {
            db_name: "alpha".to_owned(),
            if_exists: false,
            cascade: false,
        })),
    )
    .await;
    assert_eq!(bools(&ok(&r, exists("alpha")).await, "value"), [false]);
    let e = run(
        &r,
        cat(CatType::GetDatabase(pb::GetDatabase {
            db_name: "alpha".to_owned(),
        })),
    )
    .await
    .expect_err("dropped");
    assert_eq!(error_class(&e), "SCHEMA_NOT_FOUND");
}

#[tokio::test]
async fn tables_are_created_listed_described_and_dropped() {
    let r = start().await;
    let b = ok(&r, create_table("t1", Some(int_struct(&["a", "b"])))).await;
    assert_eq!(rows(&b), 0);
    ddl(&r, "CREATE TABLE t2 (x INT) USING parquet").await;
    ddl(&r, "CREATE TEMPORARY VIEW v AS SELECT 1 AS one").await;
    let b = ok(&r, list_tables(None, None)).await;
    assert_eq!(names(&b, "name"), ["t1", "t2", "v"]);
    assert_eq!(bools(&b, "isTemporary"), [false, false, true]);
    assert_eq!(names(&b, "tableType"), ["MANAGED", "MANAGED", "TEMPORARY"]);
    assert!(matches!(
        b[0].column_by_name("namespace")
            .expect("namespace")
            .data_type(),
        DataType::List(_)
    ));
    let b = ok(&r, list_tables(None, Some("t*"))).await;
    assert_eq!(names(&b, "name"), ["t1", "t2"]);
    let b = ok(&r, list_tables(Some("default"), Some("t1"))).await;
    assert_eq!(names(&b, "name"), ["t1"]);
    assert_eq!(
        strings(&b, "description"),
        [Some("made over connect".to_owned())]
    );
    assert_eq!(
        bools(&ok(&r, table_exists(None, "t1")).await, "value"),
        [true]
    );
    assert_eq!(
        bools(&ok(&r, table_exists(Some("default"), "t1")).await, "value"),
        [true]
    );
    assert_eq!(
        bools(&ok(&r, table_exists(None, "nope")).await, "value"),
        [false]
    );
    let b = ok(
        &r,
        cat(CatType::GetTable(pb::GetTable {
            table_name: "t1".to_owned(),
            db_name: Some("default".to_owned()),
        })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["t1"]);
    ok(
        &r,
        cat(CatType::DropTable(pb::DropTable {
            table_name: "t1".to_owned(),
            if_exists: false,
            purge: false,
        })),
    )
    .await;
    assert_eq!(
        bools(&ok(&r, table_exists(None, "t1")).await, "value"),
        [false]
    );
    // `if_exists` makes a missing table a no-op, as `DROP TABLE IF EXISTS`.
    ok(
        &r,
        cat(CatType::DropTable(pb::DropTable {
            table_name: "t1".to_owned(),
            if_exists: true,
            purge: false,
        })),
    )
    .await;
    let e = run(
        &r,
        cat(CatType::DropTable(pb::DropTable {
            table_name: "t1".to_owned(),
            if_exists: false,
            purge: false,
        })),
    )
    .await
    .expect_err("missing");
    assert_eq!(error_class(&e), "TABLE_OR_VIEW_NOT_FOUND");
}

#[tokio::test]
async fn columns_describe_the_table() {
    let r = start().await;
    ok(&r, create_table("t", Some(int_struct(&["a", "b"])))).await;
    let b = ok(
        &r,
        cat(CatType::ListColumns(pb::ListColumns {
            table_name: "t".to_owned(),
            db_name: None,
        })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["a", "b"]);
    assert_eq!(names(&b, "dataType"), ["int", "int"]);
    assert_eq!(bools(&b, "nullable"), [true, true]);
    assert_eq!(bools(&b, "isPartition"), [false, false]);
    // With a database, the same table through its two-part name.
    let b = ok(
        &r,
        cat(CatType::ListColumns(pb::ListColumns {
            table_name: "t".to_owned(),
            db_name: Some("default".to_owned()),
        })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["a", "b"]);
    let e = run(
        &r,
        cat(CatType::ListColumns(pb::ListColumns {
            table_name: "missing".to_owned(),
            db_name: None,
        })),
    )
    .await
    .expect_err("missing");
    assert_eq!(error_class(&e), "TABLE_OR_VIEW_NOT_FOUND");
}

#[tokio::test]
async fn views_are_listed_apart_and_dropped() {
    let r = start().await;
    ddl(&r, "CREATE TABLE t (x INT) USING parquet").await;
    ddl(&r, "CREATE VIEW pv AS SELECT 1 AS one").await;
    ddl(&r, "CREATE TEMPORARY VIEW tv AS SELECT 2 AS two").await;
    let b = ok(
        &r,
        cat(CatType::ListViews(pb::ListViews {
            db_name: None,
            pattern: None,
        })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["pv", "tv"]);
    ok(
        &r,
        cat(CatType::DropView(pb::DropView {
            view_name: "pv".to_owned(),
            if_exists: false,
        })),
    )
    .await;
    let dropped = |name: &str| {
        cat(CatType::DropTempView(pb::DropTempView {
            view_name: name.to_owned(),
        }))
    };
    assert_eq!(bools(&ok(&r, dropped("tv")).await, "value"), [true]);
    assert_eq!(bools(&ok(&r, dropped("tv")).await, "value"), [false]);
    let global = cat(CatType::DropGlobalTempView(pb::DropGlobalTempView {
        view_name: "none".to_owned(),
    }));
    assert_eq!(bools(&ok(&r, global).await, "value"), [false]);
    let b = ok(
        &r,
        cat(CatType::ListViews(pb::ListViews {
            db_name: Some("default".to_owned()),
            pattern: None,
        })),
    )
    .await;
    assert_eq!(rows(&b), 0);
}

#[tokio::test]
async fn functions_are_listed_got_and_checked() {
    let r = start().await;
    let b = ok(
        &r,
        cat(CatType::ListFunctions(pb::ListFunctions {
            db_name: None,
            pattern: Some("abs".to_owned()),
        })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["abs"]);
    assert_eq!(bools(&b, "isTemporary"), [true]);
    let b = ok(
        &r,
        cat(CatType::GetFunction(pb::GetFunction {
            function_name: "abs".to_owned(),
            db_name: None,
        })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["abs"]);
    let exists = |f: &str| {
        cat(CatType::FunctionExists(pb::FunctionExists {
            function_name: f.to_owned(),
            db_name: None,
        }))
    };
    assert_eq!(bools(&ok(&r, exists("abs")).await, "value"), [true]);
    assert_eq!(bools(&ok(&r, exists("no_such_fn")).await, "value"), [false]);
    // A database that is not there is an error before the listing.
    let e = run(
        &r,
        cat(CatType::ListFunctions(pb::ListFunctions {
            db_name: Some("nope".to_owned()),
            pattern: None,
        })),
    )
    .await
    .expect_err("no database");
    assert_eq!(error_class(&e), "SCHEMA_NOT_FOUND");
}

#[tokio::test]
async fn catalogs_list_by_pattern_and_switch() {
    let r = start().await;
    let b = ok(
        &r,
        cat(CatType::ListCatalogs(pb::ListCatalogs { pattern: None })),
    )
    .await;
    assert_eq!(names(&b, "name"), ["spark_catalog"]);
    assert_eq!(strings(&b, "description"), [None]);
    let b = ok(
        &r,
        cat(CatType::ListCatalogs(pb::ListCatalogs {
            pattern: Some("zz*".to_owned()),
        })),
    )
    .await;
    assert_eq!(rows(&b), 0);
    ok(
        &r,
        cat(CatType::SetCurrentCatalog(pb::SetCurrentCatalog {
            catalog_name: "spark_catalog".to_owned(),
        })),
    )
    .await;
}

#[tokio::test]
async fn table_text_and_properties_come_from_their_statements() {
    let r = start().await;
    ddl(
        &r,
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES ('k' = 'v')",
    )
    .await;
    let b = ok(
        &r,
        cat(CatType::GetTableProperties(pb::GetTableProperties {
            table_name: "t".to_owned(),
        })),
    )
    .await;
    let keys = names(&b, "key");
    let values = strings(&b, "value");
    let at = keys.iter().position(|k| k == "k").expect("the property");
    assert_eq!(values[at].as_deref(), Some("v"));
    let b = ok(
        &r,
        cat(CatType::GetCreateTableString(pb::GetCreateTableString {
            table_name: "t".to_owned(),
            as_serde: false,
        })),
    )
    .await;
    let text = names(&b, "value");
    assert_eq!(text.len(), 1);
    assert!(text[0].starts_with("CREATE TABLE"), "{}", text[0]);
}

#[tokio::test]
async fn caching_is_the_sessions_and_a_viewer_free_call_runs() {
    let r = start().await;
    ddl(&r, "CREATE TABLE t (a INT) USING parquet").await;
    let cached = || {
        cat(CatType::IsCached(pb::IsCached {
            table_name: "t".to_owned(),
        }))
    };
    assert_eq!(bools(&ok(&r, cached()).await, "value"), [false]);
    ok(
        &r,
        cat(CatType::CacheTable(pb::CacheTable {
            table_name: "t".to_owned(),
            storage_level: None,
        })),
    )
    .await;
    assert_eq!(bools(&ok(&r, cached()).await, "value"), [true]);
    ok(
        &r,
        cat(CatType::UncacheTable(pb::UncacheTable {
            table_name: "t".to_owned(),
        })),
    )
    .await;
    assert_eq!(bools(&ok(&r, cached()).await, "value"), [false]);
    // With a storage level the table is cached at that level.
    ok(
        &r,
        cat(CatType::CacheTable(pb::CacheTable {
            table_name: "t".to_owned(),
            storage_level: Some(pb::StorageLevel {
                use_disk: true,
                use_memory: false,
                use_off_heap: false,
                deserialized: false,
                replication: 1,
            }),
        })),
    )
    .await;
    assert_eq!(bools(&ok(&r, cached()).await, "value"), [true]);
    ok(&r, cat(CatType::ClearCache(pb::ClearCache {}))).await;
    assert_eq!(bools(&ok(&r, cached()).await, "value"), [false]);
    ok(
        &r,
        cat(CatType::RefreshTable(pb::RefreshTable {
            table_name: "t".to_owned(),
        })),
    )
    .await;
    ok(
        &r,
        cat(CatType::RefreshByPath(pb::RefreshByPath {
            path: r.dir.path().join("none").to_str().expect("utf8").to_owned(),
        })),
    )
    .await;
}

#[tokio::test]
async fn table_maintenance_calls_run_their_statements() {
    let r = start().await;
    ddl(&r, "CREATE TABLE t (a INT) USING parquet").await;
    ddl(&r, "INSERT INTO t VALUES (1), (2)").await;
    ok(
        &r,
        cat(CatType::AnalyzeTable(pb::AnalyzeTable {
            table_name: "t".to_owned(),
            no_scan: true,
        })),
    )
    .await;
    ok(
        &r,
        cat(CatType::TruncateTable(pb::TruncateTable {
            table_name: "t".to_owned(),
        })),
    )
    .await;
    let b = ok(&r, sql("SELECT count(*) AS n FROM t")).await;
    assert_eq!(
        b[0].column_by_name("n")
            .expect("n")
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0),
        0
    );
    let e = run(
        &r,
        cat(CatType::ListPartitions(pb::ListPartitions {
            table_name: "t".to_owned(),
        })),
    )
    .await
    .expect_err("unpartitioned");
    assert!(!error_class(&e).is_empty());
    ddl(
        &r,
        "CREATE TABLE p (a INT, k STRING) USING parquet PARTITIONED BY (k)",
    )
    .await;
    let b = ok(
        &r,
        cat(CatType::ListPartitions(pb::ListPartitions {
            table_name: "p".to_owned(),
        })),
    )
    .await;
    assert_eq!(rows(&b), 0);
    // The session refuses `ALTER TABLE ... RECOVER PARTITIONS` for this table; the refusal
    // reaches the client with its class.
    let e = run(
        &r,
        cat(CatType::RecoverPartitions(pb::RecoverPartitions {
            table_name: "p".to_owned(),
        })),
    )
    .await
    .expect_err("recover partitions is not supported yet");
    assert_eq!(error_class(&e), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[tokio::test]
async fn a_catalog_relation_reads_under_another_relation() {
    let r = start().await;
    ddl(&r, "CREATE TABLE a (x INT) USING parquet").await;
    ddl(&r, "CREATE TABLE b (x INT) USING parquet").await;
    let limited = pb::Relation {
        common: None,
        rel_type: Some(RelType::Limit(Box::new(pb::Limit {
            input: Some(Box::new(list_tables(None, None))),
            limit: 1,
        }))),
    };
    assert_eq!(rows(&ok(&r, limited).await), 1);
}

#[tokio::test]
async fn an_unset_operation_is_invalid_one_of_field_not_set() {
    let r = start().await;
    let e = run(
        &r,
        pb::Relation {
            common: None,
            rel_type: Some(RelType::Catalog(pb::Catalog { cat_type: None })),
        },
    )
    .await
    .expect_err("no operation");
    assert_eq!(
        error_class(&e),
        "CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SET"
    );
}

#[tokio::test]
async fn a_schema_that_is_not_a_struct_is_an_assertion_failure() {
    let r = start().await;
    let schema = pb::DataType {
        kind: Some(pb::data_type::Kind::Integer(
            pb::data_type::Integer::default(),
        )),
    };
    let e = run(&r, create_table("t", Some(schema)))
        .await
        .expect_err("not a struct");
    assert_eq!(error_class(&e), "CONNECT_INVALID_PLAN.ASSERTION_FAILURE");
}

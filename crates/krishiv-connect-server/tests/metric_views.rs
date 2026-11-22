// SPDX-License-Identifier: Apache-2.0
//! Metric views (P3 4.2-MV, design note `P3-metric-views.md` §6) through Spark Connect: a view made
//! with `spark.sql()`, then read by DataFrame plans: `groupBy(..).agg(expr("measure(m)"))`, a
//! `union` of two such reads of one view, and a plain read with no aggregate, which analysis
//! accepts and planning refuses (Spark's `INTERNAL_ERROR`).
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{ConnectServer, ConnectServerConfig, SessionLimits, SessionManager};
use krishiv_session::Session;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "0d6e1f3a-5b7c-4c2d-9e8f-1a2b3c4d5e6f";
const CREATE: &str = "CREATE VIEW mv WITH METRICS LANGUAGE YAML AS $$
version: 0.1
source: SELECT * FROM VALUES ('a', 1), ('a', 2), ('b', 5) AS t(region, n)
dimensions:
  - name: region
    expr: region
measures:
  - name: total
    expr: sum(n)
  - name: rows
    expr: count(1)
$$";

struct Running {
    channel: Channel,
    _stop: oneshot::Sender<()>,
}

async fn start() -> Running {
    let manager = Arc::new(SessionManager::from_base(
        Session::new().expect("session"),
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
        _stop: stop,
    }
}

fn relation(rel: RelType) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(rel),
    }
}

/// `spark.sql(text)`, a command.
fn sql(text: &str) -> pb::Plan {
    #[expect(deprecated, reason = "the field every client version sends")]
    let command = pb::SqlCommand {
        sql: text.to_owned(),
        ..Default::default()
    };
    pb::Plan {
        op_type: Some(OpType::Command(Box::new(pb::Command {
            command_type: Some(pb::command::CommandType::SqlCommand(Box::new(command))),
        }))),
    }
}

fn table(name: &str) -> pb::Relation {
    relation(RelType::Read(pb::Read {
        is_streaming: false,
        read_type: Some(pb::read::ReadType::NamedTable(pb::read::NamedTable {
            unparsed_identifier: name.to_owned(),
            options: Default::default(),
        })),
    }))
}

fn expr(text: &str) -> pb::Expression {
    pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::ExpressionString(
            pb::expression::ExpressionString {
                expression: text.to_owned(),
            },
        )),
    }
}

/// `spark.table("mv").groupBy("region").agg(expr("measure(total)"), expr("measure(rows)"))`.
fn by_region() -> pb::Relation {
    relation(RelType::Aggregate(Box::new(pb::Aggregate {
        input: Some(Box::new(table("mv"))),
        group_type: pb::aggregate::GroupType::Groupby as i32,
        grouping_expressions: vec![expr("region")],
        aggregate_expressions: vec![expr("measure(total)"), expr("measure(rows)")],
        pivot: None,
        grouping_sets: Vec::new(),
    })))
}

/// Runs `rel` and returns its rows as text, `NULL` for a null, sorted.
async fn run(run: &Running, rel: pb::Relation) -> Result<Vec<Vec<String>>, tonic::Status> {
    execute(
        run,
        pb::Plan {
            op_type: Some(OpType::Root(Box::new(rel))),
        },
    )
    .await
}

async fn execute(run: &Running, plan: pb::Plan) -> Result<Vec<Vec<String>>, tonic::Status> {
    let request = pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        user_context: Some(pb::UserContext {
            user_id: "someone".to_owned(),
            ..Default::default()
        }),
        plan: Some(plan),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(run.channel.clone());
    grpc.ready().await.expect("ready");
    let mut stream = grpc
        .server_streaming(
            tonic::Request::new(request),
            "/spark.connect.SparkConnectService/ExecutePlan"
                .parse()
                .expect("path"),
            tonic_prost::ProstCodec::<pb::ExecutePlanRequest, pb::ExecutePlanResponse>::default(),
        )
        .await?
        .into_inner();
    let mut out = Vec::new();
    while let Some(response) = stream.message().await? {
        if let Some(ResponseType::ArrowBatch(batch)) = response.response_type {
            for b in read_ipc(&batch.data) {
                out.extend(text_rows(&b));
            }
        }
    }
    out.sort();
    Ok(out)
}

fn read_ipc(data: &[u8]) -> Vec<RecordBatch> {
    arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(data.to_vec()), None)
        .expect("ipc")
        .collect::<Result<Vec<_>, _>>()
        .expect("batches")
}

fn text_rows(batch: &RecordBatch) -> Vec<Vec<String>> {
    (0..batch.num_rows())
        .map(|row| {
            batch
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(row) {
                        "NULL".to_owned()
                    } else {
                        match c.data_type() {
                            DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
                            DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
                            other => format!("<{other}>"),
                        }
                    }
                })
                .collect()
        })
        .collect()
}

fn strings(rows: &[&[&str]]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|r| r.iter().map(|s| (*s).to_owned()).collect())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dataframe_aggregate_reads_the_measures_of_a_metric_view() {
    let r = start().await;
    execute(&r, sql(CREATE)).await.expect("create");
    let rows = run(&r, by_region()).await.expect("aggregate");
    assert_eq!(rows, strings(&[&["a", "3", "2"], &["b", "5", "1"]]));
    // `df.union(df)` of two aggregated reads of the view: each read gets its own ids.
    let union = relation(RelType::SetOp(Box::new(pb::SetOperation {
        left_input: Some(Box::new(by_region())),
        right_input: Some(Box::new(by_region())),
        set_op_type: pb::set_operation::SetOpType::Union as i32,
        is_all: Some(true),
        by_name: Some(false),
        allow_missing_columns: Some(false),
    })));
    let rows = run(&r, union).await.expect("union");
    assert_eq!(
        rows,
        strings(&[
            &["a", "3", "2"],
            &["a", "3", "2"],
            &["b", "5", "1"],
            &["b", "5", "1"]
        ])
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_read_with_no_aggregate_fails_at_planning_as_sparks_does() {
    let r = start().await;
    execute(&r, sql(CREATE)).await.expect("create");
    let status = run(&r, table("mv")).await.expect_err("no aggregate");
    // The class and SQLSTATE are Spark's; the server withholds an internal error's text from
    // clients (design note §5's `spark_internal_reproduced` marker, which would show Spark's text,
    // is not built yet).
    assert!(
        status.message().starts_with("[INTERNAL_ERROR]") && status.message().contains("XX000"),
        "{}",
        status.message()
    );
}

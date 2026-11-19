// SPDX-License-Identifier: Apache-2.0
//! The relation arms of PR-13 (Sample, Deduplicate, Hint, RepartitionByExpression, Unpivot) run
//! through `ExecutePlan` end to end, with the rows a stock Spark client expects (P3-T012).
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
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

const SESSION: &str = "5c1f6a52-0b6e-4a39-8d3a-7f6f1b0e2a11";
const VALUES: &str = "SELECT * FROM VALUES (1, 10, 100), (1, 10, 100), (2, 20, CAST(NULL AS INT)) \
                      AS t(id, a, b)";

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

fn source() -> pb::Relation {
    relation(RelType::Sql(pb::Sql {
        query: VALUES.to_owned(),
        ..Default::default()
    }))
}

fn attr(name: &str) -> pb::Expression {
    pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::UnresolvedAttribute(
            pb::expression::UnresolvedAttribute {
                unparsed_identifier: name.to_owned(),
                plan_id: None,
                is_metadata_column: None,
            },
        )),
    }
}

/// Runs `plan` and returns its rows as text, `NULL` for a null, sorted.
async fn run(run: &Running, rel: pb::Relation) -> Result<Vec<Vec<String>>, tonic::Status> {
    let request = pb::ExecutePlanRequest {
        session_id: SESSION.to_owned(),
        user_context: Some(pb::UserContext {
            user_id: "someone".to_owned(),
            ..Default::default()
        }),
        plan: Some(pb::Plan {
            op_type: Some(OpType::Root(Box::new(rel))),
        }),
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
                            DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
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

fn error_class(status: &tonic::Status) -> String {
    // The class is the bracketed prefix of the message, as Spark's clients read it.
    let message = status.message();
    message
        .strip_prefix('[')
        .and_then(|m| m.split_once(']'))
        .map(|(class, _)| class.to_owned())
        .unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deduplicate_over_all_columns_drops_repeated_rows() {
    let server = start().await;
    let dedup = relation(RelType::Deduplicate(Box::new(pb::Deduplicate {
        input: Some(Box::new(source())),
        column_names: vec![],
        all_columns_as_keys: Some(true),
        within_watermark: None,
    })));
    assert_eq!(
        run(&server, dedup).await.expect("runs"),
        strings(&[&["1", "10", "100"], &["2", "20", "NULL"]])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deduplicate_without_keys_is_a_spark_error_class() {
    let server = start().await;
    let dedup = relation(RelType::Deduplicate(Box::new(pb::Deduplicate {
        input: Some(Box::new(source())),
        column_names: vec![],
        all_columns_as_keys: None,
        within_watermark: None,
    })));
    let status = run(&server, dedup).await.expect_err("refused");
    assert_eq!(
        error_class(&status),
        "CONNECT_INVALID_PLAN.DEDUPLICATE_REQUIRES_COLUMNS_OR_ALL"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hint_leaves_the_rows_alone() {
    let server = start().await;
    let hint = relation(RelType::Hint(Box::new(pb::Hint {
        input: Some(Box::new(source())),
        name: "broadcast".to_owned(),
        parameters: vec![],
    })));
    assert_eq!(
        run(&server, hint).await.expect("runs"),
        strings(&[
            &["1", "10", "100"],
            &["1", "10", "100"],
            &["2", "20", "NULL"]
        ])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repartition_by_expression_keeps_every_row() {
    let server = start().await;
    let repartition = relation(RelType::RepartitionByExpression(Box::new(
        pb::RepartitionByExpression {
            input: Some(Box::new(source())),
            partition_exprs: vec![attr("id")],
            num_partitions: None,
        },
    )));
    assert_eq!(
        run(&server, repartition).await.expect("runs"),
        strings(&[
            &["1", "10", "100"],
            &["1", "10", "100"],
            &["2", "20", "NULL"]
        ])
    );
}

fn sample(lower: f64, upper: f64) -> pb::Relation {
    relation(RelType::Sample(Box::new(pb::Sample {
        input: Some(Box::new(source())),
        lower_bound: lower,
        upper_bound: upper,
        with_replacement: Some(false),
        seed: Some(7),
        deterministic_order: false,
    })))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sample_of_everything_or_nothing_is_exact() {
    let server = start().await;
    assert_eq!(
        run(&server, sample(0.0, 1.0)).await.expect("runs"),
        strings(&[
            &["1", "10", "100"],
            &["1", "10", "100"],
            &["2", "20", "NULL"]
        ])
    );
    assert!(
        run(&server, sample(0.0, 0.0))
            .await
            .expect("runs")
            .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sample_fraction_above_one_is_refused() {
    let server = start().await;
    let status = run(&server, sample(0.0, 1.5)).await.expect_err("refused");
    assert!(
        status
            .message()
            .contains("requirement failed: Sampling fraction (1.5) must be on interval [0, 1] without replacement"),
        "{}",
        status.message()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_seeded_sample_repeats() {
    let server = start().await;
    let first = run(&server, sample(0.0, 0.5)).await.expect("runs");
    let second = run(&server, sample(0.0, 0.5)).await.expect("runs");
    assert_eq!(first, second);
    // Bernoulli keeps a sub-multiset of the input rows, each at most as often as it occurs.
    let all = ["1|10|100", "1|10|100", "2|20|NULL"];
    for row in first.iter().map(|r| r.join("|")) {
        let kept = first.iter().filter(|r| r.join("|") == row).count();
        let present = all.iter().filter(|a| **a == row).count();
        assert!((1..=present).contains(&kept), "{row}: {first:?}");
    }
}

fn unpivot(values: Option<Vec<pb::Expression>>) -> pb::Relation {
    relation(RelType::Unpivot(Box::new(pb::Unpivot {
        input: Some(Box::new(source())),
        ids: vec![attr("id")],
        values: values.map(|values| pb::unpivot::Values { values }),
        variable_column_name: "k".to_owned(),
        value_column_name: "v".to_owned(),
    })))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unpivot_makes_one_row_per_value_column_and_keeps_nulls() {
    let server = start().await;
    let got = run(&server, unpivot(Some(vec![attr("a"), attr("b")])))
        .await
        .expect("runs");
    assert_eq!(
        got,
        strings(&[
            &["1", "a", "10"],
            &["1", "a", "10"],
            &["1", "b", "100"],
            &["1", "b", "100"],
            &["2", "a", "20"],
            &["2", "b", "NULL"],
        ])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unpivot_value_that_is_an_id_is_refused_not_misprojected() {
    let server = start().await;
    let status = run(&server, unpivot(Some(vec![attr("id"), attr("b")])))
        .await
        .expect_err("refused");
    assert_eq!(error_class(&status), "KRISHIV_UNSUPPORTED_FEATURE");
}

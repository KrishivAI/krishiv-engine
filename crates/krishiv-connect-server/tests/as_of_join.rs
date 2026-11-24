// SPDX-License-Identifier: Apache-2.0
//! Spark Connect's `AsOfJoin` relation (relation 39, PySpark's `joinAsOf`) end to end over a raw
//! tonic client, against Spark 4.3.0's answers (`tests/fixtures/spark-4.3.0/connect-asof.json`,
//! `tests/scripts/gen_connect_asof_goldens.py`): backward and forward, exact and strict, inner and
//! left, no key, the operands swapped (the flipped match); the nearest direction, a tolerance and
//! `using_columns` are refused for now.
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]
#![expect(
    clippy::disallowed_methods,
    reason = "the tests read back the server's status details and Arrow data"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_response::ResponseType;
use krishiv_connect::proto::v4_3::expression::{self, ExprType};
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect_server::{ConnectServer, ConnectServerConfig, SessionLimits, SessionManager};
use krishiv_session::Session;
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

const SESSION: &str = "5a1d2c3b-4e5f-4a6b-8c7d-9e0f1a2b3c4d";

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

async fn execute(
    channel: &Channel,
    rel: pb::Relation,
) -> Result<Vec<pb::ExecutePlanResponse>, tonic::Status> {
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
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.expect("ready");
    let mut s = grpc
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
    while let Some(r) = s.message().await? {
        out.push(r);
    }
    Ok(out)
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

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".to_owned();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        t => format!("<{t}>"),
    }
}

/// Every row of the responses' Arrow batches as `a|b|...`, sorted.
fn rows(responses: &[pb::ExecutePlanResponse]) -> Vec<String> {
    let mut out = Vec::new();
    for r in responses {
        let Some(ResponseType::ArrowBatch(b)) = &r.response_type else {
            continue;
        };
        let reader =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(b.data.to_vec()), None)
                .expect("ipc");
        for batch in reader {
            let batch: RecordBatch = batch.expect("batch");
            for i in 0..batch.num_rows() {
                let row: Vec<String> = batch
                    .columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), i))
                    .collect();
                out.push(row.join("|"));
            }
        }
    }
    out.sort();
    out
}

fn relation(rel: RelType) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(rel),
    }
}

fn expr(kind: ExprType) -> pb::Expression {
    pb::Expression {
        common: None,
        expr_type: Some(kind),
    }
}

fn attr(name: &str) -> pb::Expression {
    expr(ExprType::UnresolvedAttribute(
        expression::UnresolvedAttribute {
            unparsed_identifier: name.to_owned(),
            plan_id: None,
            is_metadata_column: None,
        },
    ))
}

fn func(name: &str, args: Vec<pb::Expression>) -> pb::Expression {
    expr(ExprType::UnresolvedFunction(
        expression::UnresolvedFunction {
            function_name: name.to_owned(),
            arguments: args,
            is_distinct: false,
            is_user_defined_function: false,
            is_internal: None,
        },
    ))
}

fn int(v: i32) -> pb::Expression {
    expr(ExprType::Literal(expression::Literal {
        literal_type: Some(expression::literal::LiteralType::Integer(v)),
        data_type: None,
    }))
}

/// A `VALUES` relation of the fixture's rows (`columns` names them).
fn values(side: &serde_json::Value, alias: &str) -> pb::Relation {
    let columns: Vec<&str> = side["columns"]
        .as_array()
        .expect("columns")
        .iter()
        .map(|c| c.as_str().expect("name"))
        .collect();
    let rows: Vec<String> = side["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .map(|r| {
            let cells: Vec<String> = r
                .as_array()
                .expect("row")
                .iter()
                .map(|v| match v {
                    serde_json::Value::Null => "NULL".to_owned(),
                    serde_json::Value::String(s) => format!("'{s}'"),
                    other => other.to_string(),
                })
                .collect();
            format!("({})", cells.join(", "))
        })
        .collect();
    relation(RelType::Sql(pb::Sql {
        query: format!(
            "SELECT * FROM VALUES {} AS {alias}({})",
            rows.join(", "),
            columns.join(", ")
        ),
        ..Default::default()
    }))
}

fn as_of_join(doc: &serde_json::Value, case: &serde_json::Value) -> pb::Relation {
    let swapped = case["swapped"].as_bool().expect("swapped");
    let (left_as_of, right_as_of) = if swapped {
        (attr("rt"), attr("lt"))
    } else {
        (attr("lt"), attr("rt"))
    };
    let using: Vec<String> = case["using"]
        .as_array()
        .map(|u| {
            u.iter()
                .map(|c| c.as_str().expect("using").to_owned())
                .collect()
        })
        .unwrap_or_default();
    let join_expr = (case["keyed"].as_bool().expect("keyed") && using.is_empty())
        .then(|| func("==", vec![attr("lk"), attr("rk")]));
    relation(RelType::AsOfJoin(Box::new(pb::AsOfJoin {
        left: Some(Box::new(values(&doc["left"], "l"))),
        right: Some(Box::new(values(&doc["right"], "r"))),
        left_as_of: Some(left_as_of),
        right_as_of: Some(right_as_of),
        join_expr,
        using_columns: using,
        join_type: case["how"].as_str().expect("how").to_owned(),
        tolerance: case["tolerance"]
            .as_i64()
            .map(|t| int(i32::try_from(t).expect("tolerance"))),
        allow_exact_matches: case["allow_exact"].as_bool().expect("allow_exact"),
        direction: case["direction"].as_str().expect("direction").to_owned(),
    })))
}

#[tokio::test]
async fn as_of_joins_give_sparks_rows() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.3.0/connect-asof.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json");
    assert_eq!(doc["spark"], "4.3.0");
    let s = start().await;
    let cases = doc["cases"].as_array().expect("cases");
    assert!(cases.len() >= 14, "the fixture's cases");
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let got = execute(&s.channel, as_of_join(&doc, case)).await;
        match case["krishiv"].as_str().expect("krishiv") {
            "rows" => {
                let want: Vec<String> = case["rows"]
                    .as_array()
                    .expect("rows")
                    .iter()
                    .map(|r| r.as_str().expect("row").to_owned())
                    .collect();
                let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(rows(&got), want, "{name}");
            }
            _ => {
                let Err(e) = got else {
                    panic!("{name}: refused for now, but it ran");
                };
                assert_eq!(error_class(&e), "KRISHIV_UNSUPPORTED_FEATURE", "{name}");
            }
        }
    }
}

/// Spark evaluates a Connect as-of join's distance on every passing pair (`RewriteAsOfJoin`'s
/// `min_by`), backward `L - R` as well as forward `R - L`: under ANSI an overflow on any passing
/// row raises, where the SQL scan would not (Spark 4.3.0, the fixture's `overflow` cases).
#[tokio::test]
async fn as_of_join_distances_overflow_where_sparks_do() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.3.0/connect-asof.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json");
    let s = start().await;
    let cases = doc["overflow"].as_array().expect("overflow cases");
    assert!(cases.len() >= 5, "the fixture's overflow cases");
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let side = |values: &serde_json::Value, column: &str| {
            let rows: Vec<serde_json::Value> = values
                .as_array()
                .expect("values")
                .iter()
                .map(|v| serde_json::Value::Array(vec![v.clone()]))
                .collect();
            serde_json::json!({ "columns": [column], "rows": rows })
        };
        let rel = relation(RelType::AsOfJoin(Box::new(pb::AsOfJoin {
            left: Some(Box::new(values(&side(&case["left"], "lt"), "l"))),
            right: Some(Box::new(values(&side(&case["right"], "rt"), "r"))),
            left_as_of: Some(attr("lt")),
            right_as_of: Some(attr("rt")),
            join_expr: None,
            using_columns: Vec::new(),
            join_type: "inner".to_owned(),
            tolerance: None,
            allow_exact_matches: true,
            direction: case["direction"].as_str().expect("direction").to_owned(),
        })));
        let got = execute(&s.channel, rel).await;
        match case["error"].as_str() {
            Some(class) => {
                let Err(e) = got else {
                    panic!("{name}: Spark raises {class}, Krishiv ran");
                };
                assert_eq!(error_class(&e), class, "{name}");
            }
            None => {
                let want: Vec<String> = case["rows"]
                    .as_array()
                    .expect("rows")
                    .iter()
                    .map(|r| r.as_str().expect("row").to_owned())
                    .collect();
                let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(rows(&got), want, "{name}");
            }
        }
    }
}

/// Sets `spark.sql.ansi.enabled=false` on the test session, and the session zone as in the Spark
/// runs, whatever the host's (a zone whose offset ever decreases refuses a Connect as-of join on
/// `TIMESTAMP` values).
async fn ansi_off(s: &Running) {
    set_conf(
        s,
        &[
            ("spark.sql.ansi.enabled", "false"),
            ("spark.sql.session.timeZone", "UTC"),
        ],
    )
    .await;
}

/// Sets `pairs` on the test session through the `Config` RPC.
async fn set_conf(s: &Running, pairs: &[(&str, &str)]) {
    use pb::config_request::operation::OpType as ConfOp;
    let request = pb::ConfigRequest {
        session_id: SESSION.to_owned(),
        user_context: Some(pb::UserContext {
            user_id: "someone".to_owned(),
            ..Default::default()
        }),
        operation: Some(pb::config_request::Operation {
            op_type: Some(ConfOp::Set(pb::config_request::Set {
                pairs: pairs
                    .iter()
                    .map(|(key, value)| pb::KeyValue {
                        key: (*key).to_owned(),
                        value: Some((*value).to_owned()),
                    })
                    .collect(),
                silent: None,
            })),
        }),
        ..Default::default()
    };
    let mut grpc = tonic::client::Grpc::new(s.channel.clone());
    grpc.ready().await.expect("ready");
    grpc.unary(
        tonic::Request::new(request),
        "/spark.connect.SparkConnectService/Config"
            .parse()
            .expect("path"),
        tonic_prost::ProstCodec::<pb::ConfigRequest, pb::ConfigResponse>::default(),
    )
    .await
    .expect("config set");
}

/// A keyless inner Connect as-of join of `SELECT * FROM VALUES <left> AS l(lt)` and the same of
/// `right` as `r(rt)`, with an exact match allowed.
fn keyless_as_of(left: &str, right: &str, direction: &str) -> pb::Relation {
    let side = |rows: &str, alias: &str, column: &str| {
        relation(RelType::Sql(pb::Sql {
            query: format!("SELECT * FROM VALUES {rows} AS {alias}({column})"),
            ..Default::default()
        }))
    };
    relation(RelType::AsOfJoin(Box::new(pb::AsOfJoin {
        left: Some(Box::new(side(left, "l", "lt"))),
        right: Some(Box::new(side(right, "r", "rt"))),
        left_as_of: Some(attr("lt")),
        right_as_of: Some(attr("rt")),
        join_expr: None,
        using_columns: Vec::new(),
        join_type: "inner".to_owned(),
        tolerance: None,
        allow_exact_matches: true,
        direction: direction.to_owned(),
    })))
}

/// Without ANSI an integral distance wraps, so Spark's `min_by` over the distances can pick
/// another row than the nearest as-of value (left `2147483647`, right `-2` and `2147483646`:
/// Spark picks `-2`). The merge walks the values' order, so the Connect form refuses there.
#[tokio::test]
async fn as_of_join_distance_that_can_wrap_without_ansi_is_refused() {
    let s = start().await;
    ansi_off(&s).await;
    let rel = keyless_as_of(
        &format!("({})", i32::MAX),
        &format!("(-2), ({})", i32::MAX - 1),
        "backward",
    );
    let Err(e) = execute(&s.channel, rel).await else {
        panic!("a wrapping distance ran without ANSI");
    };
    assert_eq!(error_class(&e), "KRISHIV_UNSUPPORTED_FEATURE");
}

/// Spark's `min_by` ranks floating-point distances, which order unlike the values at the
/// infinities (`-Infinity - -Infinity` is NaN, above every number), so a Connect as-of join on
/// floating-point values is refused, with ANSI and without.
#[tokio::test]
async fn a_connect_as_of_join_on_floating_point_values_is_refused() {
    let s = start().await;
    for ansi in [true, false] {
        if !ansi {
            ansi_off(&s).await;
        }
        for direction in ["backward", "forward"] {
            let rel = keyless_as_of("(5.5D), (2.0D)", "(1.0D), (4.0D), (9.0D)", direction);
            let Err(e) = execute(&s.channel, rel).await else {
                panic!("a floating-point as-of join ran (ansi {ansi}, {direction})");
            };
            assert_eq!(
                error_class(&e),
                "KRISHIV_UNSUPPORTED_FEATURE",
                "ansi {ansi}, {direction}"
            );
        }
    }
}

/// Spark's `min_by` ranks `TIMESTAMP` distances taken as local date-times in the session zone, so
/// across a DST fallback a later right row can be nearer than its value says (Spark 4.3.0,
/// `America/New_York`, 2026-11-01: 06:15Z is 45 minutes after 04:30Z there, 05:45Z 75): the
/// Connect form is refused in that zone, both directions, and runs in UTC.
#[tokio::test]
async fn a_connect_timestamp_as_of_join_across_daylight_saving_is_refused() {
    let s = start().await;
    let ts = |utc: &str| format!("CAST('2026-11-01 {utc}Z' AS TIMESTAMP)");
    let (left, right) = (
        format!("({})", ts("04:30:00")),
        format!("({}), ({})", ts("05:45:00"), ts("06:15:00")),
    );
    set_conf(&s, &[("spark.sql.session.timeZone", "America/New_York")]).await;
    for direction in ["backward", "forward"] {
        let Err(e) = execute(&s.channel, keyless_as_of(&left, &right, direction)).await else {
            panic!("a {direction} as-of join across the DST fallback ran");
        };
        assert_eq!(
            error_class(&e),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{direction}"
        );
    }
    set_conf(&s, &[("spark.sql.session.timeZone", "UTC")]).await;
    // In UTC the nearer value is also the nearer distance: forward takes 05:45Z.
    let text = |name: &str| {
        expr(ExprType::Cast(Box::new(expression::Cast {
            expr: Some(Box::new(attr(name))),
            eval_mode: 0,
            cast_to_type: Some(expression::cast::CastToType::TypeStr("string".to_owned())),
        })))
    };
    let rel = relation(RelType::Project(Box::new(pb::Project {
        input: Some(Box::new(keyless_as_of(&left, &right, "forward"))),
        expressions: vec![text("lt"), text("rt")],
    })));
    let got = execute(&s.channel, rel).await.expect("forward in UTC runs");
    assert_eq!(rows(&got), ["2026-11-01 04:30:00|2026-11-01 05:45:00"]);
}

/// A value type, its left and right rows, and Spark's rows backward and forward.
type Case = (
    &'static str,
    String,
    String,
    [&'static str; 2],
    [&'static str; 2],
);

/// Timestamp, date and exact decimal distances do not wrap without ANSI (a datetime subtraction
/// raises as with ANSI, an exact decimal cannot overflow), so those Connect as-of joins run and
/// give Spark's rows (Spark 4.3.0, `spark.sql.ansi.enabled=false`,
/// `spark.sql.session.timeZone=UTC`, `_joinAsOf` over the same rows).
#[tokio::test]
async fn as_of_joins_on_non_integral_values_run_without_ansi() {
    let s = start().await;
    ansi_off(&s).await;
    let ts = |sec: u32| format!("TIMESTAMP'2024-01-01 00:00:0{sec}'");
    let date = |day: u32| format!("DATE'2024-01-0{day}'");
    let cases: [Case; 3] = [
        (
            "timestamp",
            format!("({}), ({})", ts(5), ts(2)),
            format!("({}), ({}), ({})", ts(1), ts(4), ts(9)),
            [
                "2024-01-01 00:00:02|2024-01-01 00:00:01",
                "2024-01-01 00:00:05|2024-01-01 00:00:04",
            ],
            [
                "2024-01-01 00:00:02|2024-01-01 00:00:04",
                "2024-01-01 00:00:05|2024-01-01 00:00:09",
            ],
        ),
        (
            "date",
            format!("({}), ({})", date(5), date(2)),
            format!("({}), ({}), ({})", date(1), date(4), date(9)),
            ["2024-01-02|2024-01-01", "2024-01-05|2024-01-04"],
            ["2024-01-02|2024-01-04", "2024-01-05|2024-01-09"],
        ),
        (
            "decimal",
            "(5.50BD), (2.00BD)".to_owned(),
            "(1.00BD), (4.00BD), (9.00BD)".to_owned(),
            ["2.00|1.00", "5.50|4.00"],
            ["2.00|4.00", "5.50|9.00"],
        ),
    ];
    let text = |name: &str| {
        expr(ExprType::Cast(Box::new(expression::Cast {
            expr: Some(Box::new(attr(name))),
            eval_mode: 0,
            cast_to_type: Some(expression::cast::CastToType::TypeStr("string".to_owned())),
        })))
    };
    for (name, left, right, backward, forward) in cases {
        for (direction, want) in [("backward", backward), ("forward", forward)] {
            // The values come back as text, printed as Spark prints them.
            let rel = relation(RelType::Project(Box::new(pb::Project {
                input: Some(Box::new(keyless_as_of(&left, &right, direction))),
                expressions: vec![text("lt"), text("rt")],
            })));
            let got = execute(&s.channel, rel)
                .await
                .unwrap_or_else(|e| panic!("{name} {direction}: {e}"));
            assert_eq!(rows(&got), want, "{name} {direction}");
        }
    }
}

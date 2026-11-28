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
    start_with(None).await
}

async fn start_with(conf: Option<(&'static str, &'static str)>) -> Running {
    let factory = move |_: &krishiv_session::auth::Authenticated, _: krishiv_session::Transport| {
        let s = Session::new()?;
        if let Some((k, v)) = conf {
            s.conf().set(k, v)?;
        }
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
                            DataType::Float64 => c
                                .as_primitive::<arrow_array::types::Float64Type>()
                                .value(row)
                                .to_string(),
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

// ---- na.fill / na.drop / na.replace / stat.cov / stat.corr / stat.sampleBy, run end to end ----
// Expected rows follow Spark's `DataFrameNaFunctions` / `StatFunctions` (v4.2.0 / v4.3.0 sources) and
// the quoted Spark 4.0 run: `(1,2),(NULL,3),(3,7),(4,NULL)` gives cov -1/3.

fn sql(query: &str) -> pb::Relation {
    relation(RelType::Sql(pb::Sql {
        query: query.to_owned(),
        ..Default::default()
    }))
}

fn lit(kind: pb::expression::literal::LiteralType) -> pb::expression::Literal {
    pb::expression::Literal {
        literal_type: Some(kind),
        data_type: None,
    }
}

fn long(v: i64) -> pb::expression::Literal {
    lit(pb::expression::literal::LiteralType::Long(v))
}

fn double(v: f64) -> pb::expression::Literal {
    lit(pb::expression::literal::LiteralType::Double(v))
}

fn text(v: &str) -> pb::expression::Literal {
    lit(pb::expression::literal::LiteralType::String(v.to_owned()))
}

const MIXED: &str = "SELECT * FROM VALUES (CAST(NULL AS STRING), CAST(NULL AS INT), CAST(NULL AS DOUBLE)), \
                     ('a', 5, CAST('NaN' AS DOUBLE)) AS t(name, age, score)";
const PAIRS: &str = "SELECT * FROM VALUES (CAST(1 AS DOUBLE), CAST(2 AS DOUBLE)), (CAST(NULL AS DOUBLE), CAST(3 AS DOUBLE)), \
                     (CAST(3 AS DOUBLE), CAST(7 AS DOUBLE)), (CAST(4 AS DOUBLE), CAST(NULL AS DOUBLE)) AS t(x, y)";

fn fill(query: &str, cols: &[&str], values: Vec<pb::expression::Literal>) -> pb::Relation {
    relation(RelType::FillNa(Box::new(pb::NaFill {
        input: Some(Box::new(sql(query))),
        cols: cols.iter().map(|c| (*c).to_owned()).collect(),
        values,
    })))
}

fn drop_na(cols: &[&str], min: Option<i32>) -> pb::Relation {
    relation(RelType::DropNa(Box::new(pb::NaDrop {
        input: Some(Box::new(sql(MIXED))),
        cols: cols.iter().map(|c| (*c).to_owned()).collect(),
        min_non_nulls: min,
    })))
}

fn replace(
    query: &str,
    cols: &[&str],
    pairs: Vec<(pb::expression::Literal, pb::expression::Literal)>,
) -> pb::Relation {
    relation(RelType::Replace(Box::new(pb::NaReplace {
        input: Some(Box::new(sql(query))),
        cols: cols.iter().map(|c| (*c).to_owned()).collect(),
        replacements: pairs
            .into_iter()
            .map(|(old, new)| pb::na_replace::Replacement {
                old_value: Some(old),
                new_value: Some(new),
            })
            .collect(),
    })))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn na_fill_one_value_fills_the_numeric_columns_and_nan() {
    let server = start().await;
    // fill(0L): INT and DOUBLE columns, NaN included; the string column is left alone.
    let rows = run(&server, fill(MIXED, &[], vec![long(0)]))
        .await
        .expect("runs");
    assert_eq!(rows, strings(&[&["NULL", "0", "0"], &["a", "5", "0"]]));
    // fill("x"): only the string column.
    let rows = run(&server, fill(MIXED, &[], vec![text("x")]))
        .await
        .expect("runs");
    assert_eq!(rows, strings(&[&["a", "5", "NaN"], &["x", "NULL", "NULL"]]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn na_fill_per_column_values_cast_to_each_columns_type() {
    let server = start().await;
    // fillMap {"name": 0, "age": "5"}: "0" in the string column, 5 in the INT column, `score` untouched.
    let rows = run(
        &server,
        fill(MIXED, &["name", "age"], vec![long(0), text("5")]),
    )
    .await
    .expect("runs");
    assert_eq!(rows, strings(&[&["0", "5", "NULL"], &["a", "5", "NaN"]]));
    // A column named twice takes the last value (Spark builds a Map).
    let rows = run(
        &server,
        fill(MIXED, &["age", "age"], vec![long(1), long(2)]),
    )
    .await
    .expect("runs");
    assert_eq!(rows, strings(&[&["NULL", "2", "NULL"], &["a", "5", "NaN"]]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn na_fill_resolves_the_named_columns() {
    let server = start().await;
    let status = run(&server, fill(MIXED, &["agee"], vec![long(0)]))
        .await
        .expect_err("unknown column");
    assert_eq!(error_class(&status), "UNRESOLVED_COLUMN.WITH_SUGGESTION");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn na_drop_counts_values_that_are_neither_null_nor_nan() {
    let server = start().await;
    // any: the first row has NULLs, the second a NaN in `score`.
    assert!(
        run(&server, drop_na(&[], None))
            .await
            .expect("runs")
            .is_empty()
    );
    // A subset leaves NaN out of it.
    assert_eq!(
        run(&server, drop_na(&["name", "age"], None))
            .await
            .expect("runs"),
        strings(&[&["a", "5", "NaN"]])
    );
    // thresh 2 over all three columns: the NaN row has two non-null values.
    assert_eq!(
        run(&server, drop_na(&[], Some(2))).await.expect("runs"),
        strings(&[&["a", "5", "NaN"]])
    );
    // thresh 0 keeps every row; a threshold above the subset size keeps none.
    assert_eq!(
        run(&server, drop_na(&[], Some(0)))
            .await
            .expect("runs")
            .len(),
        2
    );
    assert!(
        run(&server, drop_na(&["name"], Some(2)))
            .await
            .expect("runs")
            .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn na_replace_compares_the_key_as_written() {
    let server = start().await;
    const NUMBERS: &str =
        "SELECT * FROM VALUES (1, 'a'), (2, 'b'), (3, CAST(NULL AS STRING)) AS t(n, s)";
    // 1.5 never equals the INT 1: nothing changes (it used to become 1 and match).
    let rows = run(
        &server,
        replace(NUMBERS, &[], vec![(double(1.5), double(9.0))]),
    )
    .await
    .expect("runs");
    assert_eq!(rows, strings(&[&["1", "a"], &["2", "b"], &["3", "NULL"]]));
    // 1.0 = 1: the INT column gets 9.
    let rows = run(
        &server,
        replace(NUMBERS, &["n"], vec![(double(1.0), double(9.0))]),
    )
    .await
    .expect("runs");
    assert_eq!(rows, strings(&[&["2", "b"], &["3", "NULL"], &["9", "a"]]));
    // A string key reaches the string column only; a repeated key keeps its last pair.
    let rows = run(
        &server,
        replace(
            NUMBERS,
            &[],
            vec![(text("a"), text("y")), (text("a"), text("z"))],
        ),
    )
    .await
    .expect("runs");
    assert_eq!(rows, strings(&[&["1", "z"], &["2", "b"], &["3", "NULL"]]));
    // No pairs still resolves the named columns.
    let status = run(&server, replace(NUMBERS, &["nope"], vec![]))
        .await
        .expect_err("unknown column");
    assert_eq!(error_class(&status), "UNRESOLVED_COLUMN.WITH_SUGGESTION");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn na_replace_matches_a_nan_key() {
    let server = start().await;
    let rows = run(
        &server,
        replace(MIXED, &["score"], vec![(double(f64::NAN), double(0.0))]),
    )
    .await
    .expect("runs");
    assert_eq!(
        rows,
        strings(&[&["NULL", "NULL", "NULL"], &["a", "5", "0"]])
    );
}

fn stat(rel: RelType) -> pb::Relation {
    relation(rel)
}

fn number(rows: &[Vec<String>]) -> f64 {
    assert_eq!(rows.len(), 1, "one row: {rows:?}");
    rows[0][0].parse().expect("a number")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cov_and_corr_read_null_as_zero() {
    let server = start().await;
    let cov = stat(RelType::Cov(Box::new(pb::StatCov {
        input: Some(Box::new(sql(PAIRS))),
        col1: "x".to_owned(),
        col2: "y".to_owned(),
    })));
    // The quoted Spark 4.0 run: -1/3.
    let got = number(&run(&server, cov).await.expect("runs"));
    assert!((got - (-1.0 / 3.0)).abs() < 1e-12, "{got}");
    let corr = stat(RelType::Corr(Box::new(pb::StatCorr {
        input: Some(Box::new(sql(PAIRS))),
        col1: "x".to_owned(),
        col2: "y".to_owned(),
        method: None,
    })));
    // sxy = -1, sxx = 10, syy = 26 over the zero-filled columns.
    let got = number(&run(&server, corr).await.expect("runs"));
    assert!((got - (-1.0 / 260.0_f64.sqrt())).abs() < 1e-12, "{got}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cov_and_corr_below_two_rows_are_zero_and_nan() {
    let server = start().await;
    for query in [
        "SELECT * FROM VALUES (CAST(1 AS DOUBLE), CAST(2 AS DOUBLE)) AS t(x, y)",
        "SELECT * FROM VALUES (CAST(1 AS DOUBLE), CAST(2 AS DOUBLE)) AS t(x, y) WHERE x > 5",
    ] {
        let cov = stat(RelType::Cov(Box::new(pb::StatCov {
            input: Some(Box::new(sql(query))),
            col1: "x".to_owned(),
            col2: "y".to_owned(),
        })));
        assert_eq!(run(&server, cov).await.expect("runs"), strings(&[&["0"]]));
        let corr = stat(RelType::Corr(Box::new(pb::StatCorr {
            input: Some(Box::new(sql(query))),
            col1: "x".to_owned(),
            col2: "y".to_owned(),
            method: None,
        })));
        assert_eq!(
            run(&server, corr).await.expect("runs"),
            strings(&[&["NaN"]])
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cov_of_a_column_that_is_not_a_number_is_an_error_not_zero() {
    let server = start().await;
    let cov = stat(RelType::Cov(Box::new(pb::StatCov {
        input: Some(Box::new(sql(
            "SELECT * FROM VALUES ('abc', 1.0D), ('def', 2.0D) AS t(x, y)",
        ))),
        col1: "x".to_owned(),
        col2: "y".to_owned(),
    })));
    let status = run(&server, cov).await.expect_err("not a number");
    assert_eq!(error_class(&status), "CAST_INVALID_INPUT");
}

fn sample_by(fractions: Vec<(pb::expression::Literal, f64)>) -> pb::Relation {
    relation(RelType::SampleBy(Box::new(pb::StatSampleBy {
        input: Some(Box::new(sql(
            "SELECT * FROM VALUES (1, 'a'), (1, 'b'), (2, 'c'), (CAST(NULL AS INT), 'd') AS t(k, v)",
        ))),
        col: Some(attr("k")),
        fractions: fractions
            .into_iter()
            .map(|(stratum, fraction)| pb::stat_sample_by::Fraction {
                stratum: Some(stratum),
                fraction,
            })
            .collect(),
        seed: Some(7),
    })))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sample_by_with_fractions_zero_and_one_is_deterministic() {
    let server = start().await;
    // Stratum 1 whole, stratum 2 none; the NULL stratum and unlisted strata get 0.0.
    let rows = run(&server, sample_by(vec![(long(1), 1.0), (long(2), 0.0)]))
        .await
        .expect("runs");
    assert_eq!(rows, strings(&[&["1", "a"], &["1", "b"]]));
    // A stratum listed twice keeps its last fraction.
    let rows = run(&server, sample_by(vec![(long(1), 0.0), (long(1), 1.0)]))
        .await
        .expect("runs");
    assert_eq!(rows, strings(&[&["1", "a"], &["1", "b"]]));
}

fn null_of_long() -> pb::expression::Literal {
    lit(pb::expression::literal::LiteralType::Null(pb::DataType {
        kind: Some(pb::data_type::Kind::Long(pb::data_type::Long::default())),
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn na_replace_compares_and_converts_numbers_as_doubles() {
    let server = start().await;
    const BIG: &str =
        "SELECT * FROM VALUES (CAST(9007199254740993 AS BIGINT)), (CAST(5 AS BIGINT)) AS t(n)";
    // (a) Spark compares double(col) = 9.007199254740992E15, true for 2^53 + 1: the row becomes 0.
    let rows = run(
        &server,
        replace(BIG, &[], vec![(long(9_007_199_254_740_992), long(0))]),
    )
    .await
    .expect("runs");
    assert_eq!(rows, strings(&[&["0"], &["5"]]));
    // (c) The new value goes through a double: 2^53 + 1 is stored as 2^53.
    let rows = run(
        &server,
        replace(BIG, &[], vec![(long(5), long(9_007_199_254_740_993))]),
    )
    .await
    .expect("runs");
    assert_eq!(
        rows,
        strings(&[&["9007199254740992"], &["9007199254740993"]])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn na_replace_with_a_null_new_value_writes_null() {
    let server = start().await;
    const NUMBERS: &str = "SELECT * FROM VALUES (1, 'a'), (2, 'b') AS t(n, s)";
    let rows = run(
        &server,
        replace(NUMBERS, &["n"], vec![(long(1), null_of_long())]),
    )
    .await
    .expect("runs");
    assert_eq!(rows, strings(&[&["2", "b"], &["NULL", "a"]]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn na_drop_how_all_keeps_a_row_with_any_value() {
    let server = start().await;
    const ROWS: &str = "SELECT * FROM VALUES (CAST(NULL AS STRING), CAST(NULL AS INT)), \
                        (CAST(NULL AS STRING), 3), ('a', 5) AS t(name, age)";
    // `how="all"` is min_non_nulls = 1.
    let drop = relation(RelType::DropNa(Box::new(pb::NaDrop {
        input: Some(Box::new(sql(ROWS))),
        cols: vec![],
        min_non_nulls: Some(1),
    })));
    assert_eq!(
        run(&server, drop).await.expect("runs"),
        strings(&[&["NULL", "3"], &["a", "5"]])
    );
}

fn project(input: pb::Relation, expressions: Vec<pb::Expression>) -> pb::Relation {
    relation(RelType::Project(Box::new(pb::Project {
        input: Some(Box::new(input)),
        expressions,
    })))
}

/// `left + right AS name`.
fn plus(left: &str, right: i32, name: &str) -> pb::Expression {
    use pb::expression::{Alias, ExprType, Literal, UnresolvedFunction, literal::LiteralType};
    let literal = pb::Expression {
        common: None,
        expr_type: Some(ExprType::Literal(Literal {
            literal_type: Some(LiteralType::Integer(right)),
            data_type: None,
        })),
    };
    let call = pb::Expression {
        common: None,
        expr_type: Some(ExprType::UnresolvedFunction(UnresolvedFunction {
            function_name: "+".to_owned(),
            arguments: vec![attr(left), literal],
            is_distinct: false,
            is_user_defined_function: false,
            is_internal: None,
        })),
    };
    pb::Expression {
        common: None,
        expr_type: Some(ExprType::Alias(Box::new(Alias {
            expr: Some(Box::new(call)),
            name: vec![name.to_owned()],
            metadata: None,
        }))),
    }
}

fn zip(left: pb::Relation, right: pb::Relation) -> pb::Relation {
    relation(RelType::Zip(Box::new(pb::Zip {
        left: Some(Box::new(left)),
        right: Some(Box::new(right)),
    })))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_puts_the_columns_of_two_projections_of_one_source_side_by_side() {
    let server = start().await;
    let left = project(source(), vec![attr("id"), plus("a", 1, "a1")]);
    let right = project(source(), vec![attr("b"), plus("a", 2, "a2")]);
    assert_eq!(
        run(&server, zip(left, right)).await.expect("runs"),
        strings(&[
            &["1", "11", "100", "12"],
            &["1", "11", "100", "12"],
            &["2", "21", "NULL", "22"],
        ])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_of_a_source_with_a_projection_of_it_keeps_both_sets_of_columns() {
    let server = start().await;
    let right = project(source(), vec![plus("id", 100, "big")]);
    assert_eq!(
        run(&server, zip(source(), right)).await.expect("runs"),
        strings(&[
            &["1", "10", "100", "101"],
            &["1", "10", "100", "101"],
            &["2", "20", "NULL", "102"],
        ])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_of_different_sources_is_not_mergeable() {
    let server = start().await;
    let other = relation(RelType::Sql(pb::Sql {
        query: "SELECT 1 AS x".to_owned(),
        ..Default::default()
    }));
    let status = run(&server, zip(source(), other))
        .await
        .expect_err("refused");
    assert_eq!(error_class(&status), "ZIP_PLANS_NOT_MERGEABLE");
}

fn call(function: &str, arg: pb::Expression, name: &str) -> pb::Expression {
    use pb::expression::{Alias, ExprType, UnresolvedFunction};
    let call = pb::Expression {
        common: None,
        expr_type: Some(ExprType::UnresolvedFunction(UnresolvedFunction {
            function_name: function.to_owned(),
            arguments: vec![arg],
            is_distinct: false,
            is_user_defined_function: false,
            is_internal: None,
        })),
    };
    pb::Expression {
        common: None,
        expr_type: Some(ExprType::Alias(Box::new(Alias {
            expr: Some(Box::new(call)),
            name: vec![name.to_owned()],
            metadata: None,
        }))),
    }
}

fn aggregate(
    input: pb::Relation,
    keys: Vec<pb::Expression>,
    aggs: Vec<pb::Expression>,
) -> pb::Relation {
    relation(RelType::Aggregate(Box::new(pb::Aggregate {
        input: Some(Box::new(input)),
        group_type: pb::aggregate::GroupType::Groupby as i32,
        grouping_expressions: keys,
        aggregate_expressions: aggs,
        pivot: None,
        grouping_sets: Vec::new(),
    })))
}

fn pivot_without_values() -> pb::Relation {
    relation(RelType::Aggregate(Box::new(pb::Aggregate {
        input: Some(Box::new(source())),
        group_type: pb::aggregate::GroupType::Pivot as i32,
        grouping_expressions: vec![attr("id")],
        aggregate_expressions: vec![call("sum", attr("b"), "s")],
        pivot: Some(pb::aggregate::Pivot {
            col: Some(attr("a")),
            values: Vec::new(),
        }),
        grouping_sets: Vec::new(),
    })))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pivot_without_values_takes_the_sorted_distinct_values_of_its_column() {
    let server = start().await;
    let got = run(&server, pivot_without_values()).await.expect("runs");
    // Spark: `collectPivotValues` gives 10, 20; the columns are named by the values.
    assert_eq!(
        got,
        strings(&[&["1", "200", "NULL"], &["2", "NULL", "NULL"]])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pivot_with_more_values_than_pivot_max_values_is_refused() {
    let server = start_with(Some(("spark.sql.pivotMaxValues", "1"))).await;
    let status = run(&server, pivot_without_values())
        .await
        .expect_err("refused");
    // A legacy class has no bracketed prefix; the message is Spark's `_LEGACY_ERROR_TEMP_1324`.
    assert!(
        status
            .message()
            .starts_with("The pivot column a has more than 1 distinct values"),
        "{status:?}"
    );
}

fn renamed(input: pb::Relation, from: &str, to: &str) -> pb::Relation {
    relation(RelType::WithColumnsRenamed(Box::new(
        pb::WithColumnsRenamed {
            input: Some(Box::new(input)),
            rename_columns_map: [(from.to_owned(), to.to_owned())].into(),
            renames: Vec::new(),
        },
    )))
}

fn dropped(input: pb::Relation, column: &str) -> pb::Relation {
    relation(RelType::Drop(Box::new(pb::Drop {
        input: Some(Box::new(input)),
        columns: Vec::new(),
        column_names: vec![column.to_owned()],
    })))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_sees_through_a_rename_and_a_drop_on_either_side() {
    let server = start().await;
    // DataFrameZipSuite "withColumnRenamed on both sides" and a drop against a select.
    let both = zip(renamed(source(), "id", "id1"), renamed(source(), "a", "a1"));
    assert_eq!(
        run(&server, both).await.expect("runs"),
        strings(&[
            &["1", "10", "100", "1", "10", "100"],
            &["1", "10", "100", "1", "10", "100"],
            &["2", "20", "NULL", "2", "20", "NULL"],
        ])
    );
    let drop_select = zip(dropped(source(), "b"), project(source(), vec![attr("a")]));
    assert_eq!(
        run(&server, drop_select).await.expect("runs"),
        strings(&[&["1", "10", "10"], &["1", "10", "10"], &["2", "20", "20"]])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_layers_a_chain_of_projections_and_keeps_a_shared_producer_once() {
    let server = start().await;
    // Several layers on one side: `a2` reads `a1`, which reads the base.
    let left = project(
        project(source(), vec![attr("id"), plus("a", 1, "a1")]),
        vec![plus("a1", 1, "a2")],
    );
    let right = project(source(), vec![plus("a", 5, "a5")]);
    assert_eq!(
        run(&server, zip(left, right)).await.expect("runs"),
        strings(&[&["12", "15"], &["12", "15"], &["22", "25"]])
    );
    // The same producer under two names is computed once and keeps both names' values.
    let x = project(source(), vec![plus("a", 1, "x")]);
    let y = project(source(), vec![plus("a", 1, "y")]);
    assert_eq!(
        run(&server, zip(x, y)).await.expect("runs"),
        strings(&[&["11", "11"], &["11", "11"], &["21", "21"]])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_refuses_an_aggregate_against_the_plan_it_aggregates() {
    let server = start().await;
    let maxima = || vec![call("max", attr("id"), "m"), call("max", attr("a"), "n")];
    let left = aggregate(source(), Vec::new(), maxima());
    let right = project(source(), vec![attr("id"), attr("a")]);
    let status = run(&server, zip(left, right)).await.expect_err("refused");
    assert_eq!(error_class(&status), "ZIP_PLANS_NOT_MERGEABLE");
    // Two aggregates over a grouped DataFrame: the outer one is not the grouped DataFrame itself.
    let grouped = || {
        aggregate(
            source(),
            vec![attr("id")],
            vec![call("sum", attr("a"), "s")],
        )
    };
    let outer = aggregate(
        grouped(),
        Vec::new(),
        vec![call("max", attr("id"), "id"), call("max", attr("s"), "s")],
    );
    let inner = project(grouped(), vec![attr("id"), attr("s")]);
    let status = run(&server, zip(outer, inner)).await.expect_err("refused");
    assert_eq!(error_class(&status), "ZIP_PLANS_NOT_MERGEABLE");
}

/// `rand() AS name`.
fn rand_as(name: &str) -> pb::Expression {
    use pb::expression::{Alias, ExprType, UnresolvedFunction};
    pb::Expression {
        common: None,
        expr_type: Some(ExprType::Alias(Box::new(Alias {
            expr: Some(Box::new(pb::Expression {
                common: None,
                expr_type: Some(ExprType::UnresolvedFunction(UnresolvedFunction {
                    function_name: "rand".to_owned(),
                    arguments: Vec::new(),
                    is_distinct: false,
                    is_user_defined_function: false,
                    is_internal: None,
                })),
            })),
            name: vec![name.to_owned()],
            metadata: None,
        }))),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_computes_a_shared_unseeded_producer_once() {
    let server = start().await;
    // `P = select(rand() AS r)`; `zip(P.select(r + r AS x), P)`: both sides share P's alias, so `x` reads
    // the one `r` the output carries.
    let shared = project(source(), vec![rand_as("r")]);
    let left = project(shared.clone(), vec![add_columns("r", "r", "x")]);
    let rows = run(&server, zip(left, shared)).await.expect("runs");
    assert_eq!(rows.len(), 3, "{rows:?}");
    for r in &rows {
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(
            (cell_number(&r[0]) - 2.0 * cell_number(&r[1])).abs() < 1e-12,
            "{r:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_keeps_independent_nondeterministic_producers_apart() {
    let server = start().await;
    // Equal bodies, no shared parent: Spark gives each `rand()` its own seed, so `y` is not read from `x`.
    let left = project(source(), vec![rand_as("x")]);
    let right = project(source(), vec![rand_as("y")]);
    let rows = run(&server, zip(left, right)).await.expect("runs");
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|r| r[0] != r[1]), "{rows:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_of_a_zero_column_projection_keeps_the_other_side() {
    let server = start().await;
    let left = project(source(), Vec::new());
    let right = project(source(), vec![attr("id")]);
    assert_eq!(
        run(&server, zip(left, right)).await.expect("runs"),
        strings(&[&["1"], &["1"], &["2"]])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_keeps_a_reference_to_an_input_dataframe_resolvable() {
    let server = start().await;
    let with_id = |mut r: pb::Relation, id: i64| {
        r.common = Some(pb::RelationCommon {
            plan_id: Some(id),
            ..Default::default()
        });
        r
    };
    let left = with_id(project(source(), vec![attr("id"), attr("a")]), 7);
    let right = project(source(), vec![attr("b")]);
    let by_plan = pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::UnresolvedAttribute(
            pb::expression::UnresolvedAttribute {
                unparsed_identifier: "a".to_owned(),
                plan_id: Some(7),
                is_metadata_column: None,
            },
        )),
    };
    assert_eq!(
        run(&server, project(zip(left, right), vec![by_plan]))
            .await
            .expect("runs"),
        strings(&[&["10"], &["10"], &["20"]])
    );
}

/// `rand(seed) AS name`.
fn rand_seeded_as(name: &str, seed: i32) -> pb::Expression {
    use pb::expression::{Alias, ExprType, Literal, UnresolvedFunction, literal::LiteralType};
    pb::Expression {
        common: None,
        expr_type: Some(ExprType::Alias(Box::new(Alias {
            expr: Some(Box::new(pb::Expression {
                common: None,
                expr_type: Some(ExprType::UnresolvedFunction(UnresolvedFunction {
                    function_name: "rand".to_owned(),
                    arguments: vec![pb::Expression {
                        common: None,
                        expr_type: Some(ExprType::Literal(Literal {
                            literal_type: Some(LiteralType::Integer(seed)),
                            data_type: None,
                        })),
                    }],
                    is_distinct: false,
                    is_user_defined_function: false,
                    is_internal: None,
                })),
            })),
            name: vec![name.to_owned()],
            metadata: None,
        }))),
    }
}

/// `left + right AS name` over two columns.
fn add_columns(left: &str, right: &str, name: &str) -> pb::Expression {
    use pb::expression::{Alias, ExprType, UnresolvedFunction};
    pb::Expression {
        common: None,
        expr_type: Some(ExprType::Alias(Box::new(Alias {
            expr: Some(Box::new(pb::Expression {
                common: None,
                expr_type: Some(ExprType::UnresolvedFunction(UnresolvedFunction {
                    function_name: "+".to_owned(),
                    arguments: vec![attr(left), attr(right)],
                    is_distinct: false,
                    is_user_defined_function: false,
                    is_internal: None,
                })),
            })),
            name: vec![name.to_owned()],
            metadata: None,
        }))),
    }
}

fn cell_number(cell: &str) -> f64 {
    cell.parse().expect("a number")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_merges_one_seeded_producer_and_keeps_different_seeds_apart() {
    let server = start().await;
    // Different seeds are different producers.
    let rows = run(
        &server,
        zip(
            project(source(), vec![rand_seeded_as("x", 1)]),
            project(source(), vec![rand_seeded_as("y", 2)]),
        ),
    )
    .await
    .expect("runs");
    assert_eq!(rows.len(), 3, "{rows:?}");
    assert!(rows.iter().all(|r| r[0] != r[1]), "{rows:?}");
    // The same seeded producer on both sides is computed once: `x = r + r` reads the merged `r`.
    let left = project(source(), vec![rand_seeded_as("r", 7)]);
    let right = project(
        project(source(), vec![rand_seeded_as("r", 7)]),
        vec![add_columns("r", "r", "x")],
    );
    let rows = run(&server, zip(left, right)).await.expect("runs");
    assert_eq!(rows.len(), 3);
    for r in &rows {
        assert!(
            (cell_number(&r[1]) - 2.0 * cell_number(&r[0])).abs() < 1e-12,
            "{r:?}"
        );
    }
    // A chain `withColumn("r", rand(7)).withColumn("x", r + r)` zipped with a plain select.
    let chain = project(
        project(source(), vec![attr("id"), rand_seeded_as("r", 7)]),
        vec![attr("r"), add_columns("r", "r", "x")],
    );
    let rows = run(&server, zip(chain, project(source(), vec![attr("id")])))
        .await
        .expect("runs");
    assert_eq!(rows.len(), 3, "{rows:?}");
    assert!(rows.iter().all(|r| r.len() == 3), "{rows:?}");
    assert_eq!(
        rows.iter().map(|r| r[2].as_str()).collect::<Vec<_>>(),
        ["1", "1", "2"]
    );
    for r in &rows {
        assert!(
            (cell_number(&r[1]) - 2.0 * cell_number(&r[0])).abs() < 1e-12,
            "{r:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_resolves_a_right_reference_to_a_deduplicated_column_by_its_own_name() {
    let server = start().await;
    // `l` computes `a + 1` as `x` in an inner layer only; `r["y"]` is the same value under another name.
    let inner = project(source(), vec![attr("id"), plus("a", 1, "x")]);
    let left = project(inner, vec![attr("id")]);
    let mut right = project(source(), vec![plus("a", 1, "y")]);
    right.common = Some(pb::RelationCommon {
        plan_id: Some(9),
        ..Default::default()
    });
    let by_plan = pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::UnresolvedAttribute(
            pb::expression::UnresolvedAttribute {
                unparsed_identifier: "y".to_owned(),
                plan_id: Some(9),
                is_metadata_column: None,
            },
        )),
    };
    assert_eq!(
        run(&server, project(zip(left, right), vec![by_plan]))
            .await
            .expect("runs"),
        strings(&[&["11"], &["11"], &["21"]])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zip_merges_two_independent_range_inputs() {
    let server = start().await;
    let range = || {
        relation(RelType::Range(pb::Range {
            start: Some(0),
            end: 3,
            step: 1,
            num_partitions: None,
        }))
    };
    assert_eq!(
        run(&server, zip(range(), range())).await.expect("runs"),
        strings(&[&["0", "0"], &["1", "1"], &["2", "2"]])
    );
}

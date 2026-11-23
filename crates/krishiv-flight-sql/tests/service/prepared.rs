// SPDX-License-Identifier: Apache-2.0
//! Prepared statements end to end (design P3-T101 §4.4, §3.2): create, bind, execute, update,
//! close, the handle cap.

use arrow_flight::FlightData;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::sql::{
    ActionClosePreparedStatementRequest, ActionCreatePreparedStatementRequest,
    ActionCreatePreparedStatementResult, Any, CommandPreparedStatementQuery,
    CommandPreparedStatementUpdate, DoPutPreparedStatementResult, DoPutUpdateResult,
};
use arrow_schema::{DataType, Schema};
use krishiv_flight_sql::handles::HandleLimits;

use super::*;

/// The class a refusal names.
fn class_of(e: &Status) -> Option<&str> {
    e.metadata()
        .get("x-krishiv-error-class")
        .and_then(|v| v.to_str().ok())
}

/// `dataset_schema` / `parameter_schema` bytes as a schema (`None` when empty).
fn schema_of(bytes: &[u8]) -> Option<Schema> {
    if bytes.is_empty() {
        return None;
    }
    let message = arrow_flight::IpcMessage(prost::bytes::Bytes::copy_from_slice(bytes));
    Some(Schema::try_from(message).expect("an IPC schema"))
}

/// `CreatePreparedStatement(sql)`: the result and the session id set.
async fn create(
    h: &mut Harness,
    token: &str,
    cookie: Option<&str>,
    sql: &str,
) -> Result<(ActionCreatePreparedStatementResult, Option<String>), Status> {
    let request = ActionCreatePreparedStatementRequest {
        query: sql.to_owned(),
        transaction_id: None,
    };
    let action = Action::new("CreatePreparedStatement", request.as_any().encode_to_vec());
    let response = h.client.do_action(authed(action, token, cookie)).await?;
    let id = set_cookie(response.metadata());
    let results: Vec<_> = response.into_inner().try_collect().await?;
    assert_eq!(results.len(), 1);
    let any = Any::decode(results[0].body.as_ref()).expect("Any");
    let result = any
        .unpack::<ActionCreatePreparedStatementResult>()
        .expect("unpack")
        .expect("a create result");
    Ok((result, id))
}

/// `ClosePreparedStatement(handle)`.
async fn close(h: &mut Harness, token: &str, handle: &[u8]) -> Result<(), Status> {
    let request = ActionClosePreparedStatementRequest {
        prepared_statement_handle: handle.to_vec().into(),
    };
    let action = Action::new("ClosePreparedStatement", request.as_any().encode_to_vec());
    let response = h.client.do_action(authed(action, token, None)).await?;
    let results: Vec<_> = response.into_inner().try_collect().await?;
    assert!(results.is_empty());
    Ok(())
}

/// The `DoPut` messages of `cmd` with `batches` as its bound rows.
async fn put_rows<M: ProstMessageExt>(cmd: &M, batches: Vec<RecordBatch>) -> Vec<FlightData> {
    let descriptor = FlightDescriptor::new_cmd(cmd.as_any().encode_to_vec());
    let encoded = FlightDataEncoderBuilder::new()
        .with_flight_descriptor(Some(descriptor.clone()))
        .build(futures::stream::iter(batches.into_iter().map(Ok)));
    let data: Vec<FlightData> = encoded.try_collect().await.expect("encode");
    if data.is_empty() {
        return vec![FlightData::new().with_descriptor(descriptor)];
    }
    data
}

/// `DoPut(CommandPreparedStatementQuery)` binding `batches`: the handle it answers.
async fn bind(
    h: &mut Harness,
    token: &str,
    handle: &[u8],
    batches: Vec<RecordBatch>,
) -> Result<Vec<u8>, Status> {
    let cmd = CommandPreparedStatementQuery {
        prepared_statement_handle: handle.to_vec().into(),
    };
    let data = put_rows(&cmd, batches).await;
    let response = h
        .client
        .do_put(authed(futures::stream::iter(data), token, None))
        .await?;
    let results: Vec<_> = response.into_inner().try_collect().await?;
    assert_eq!(results.len(), 1);
    let result =
        DoPutPreparedStatementResult::decode(results[0].app_metadata.as_ref()).expect("result");
    Ok(result.prepared_statement_handle.expect("a handle").to_vec())
}

/// `DoPut(CommandPreparedStatementUpdate)` with `batches`: the record count.
async fn run_update(
    h: &mut Harness,
    token: &str,
    handle: &[u8],
    batches: Vec<RecordBatch>,
) -> Result<i64, Status> {
    let cmd = CommandPreparedStatementUpdate {
        prepared_statement_handle: handle.to_vec().into(),
    };
    let data = put_rows(&cmd, batches).await;
    let response = h
        .client
        .do_put(authed(futures::stream::iter(data), token, None))
        .await?;
    let results: Vec<_> = response.into_inner().try_collect().await?;
    assert_eq!(results.len(), 1);
    Ok(DoPutUpdateResult::decode(results[0].app_metadata.as_ref())
        .expect("result")
        .record_count)
}

/// `GetFlightInfo(CommandPreparedStatementQuery)`.
async fn prepared_info(h: &mut Harness, token: &str, handle: &[u8]) -> Result<FlightInfo, Status> {
    let cmd = CommandPreparedStatementQuery {
        prepared_statement_handle: handle.to_vec().into(),
    };
    let descriptor = FlightDescriptor::new_cmd(cmd.as_any().encode_to_vec());
    Ok(h.client
        .get_flight_info(authed(descriptor, token, None))
        .await?
        .into_inner())
}

/// One row of an `BIGINT` and a `STRING` per value pair.
fn row(values: &[(i64, &str)]) -> RecordBatch {
    RecordBatch::try_from_iter([
        (
            "a",
            Arc::new(Int64Array::from(
                values.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            )) as arrow_array::ArrayRef,
        ),
        (
            "b",
            Arc::new(StringArray::from(
                values.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
            )),
        ),
    ])
    .expect("batch")
}

/// One `BIGINT` column.
fn longs(values: &[i64]) -> RecordBatch {
    RecordBatch::try_from_iter([(
        "a",
        Arc::new(Int64Array::from(values.to_vec())) as arrow_array::ArrayRef,
    )])
    .expect("batch")
}

/// The first column of a fetch as `BIGINT`s.
fn first_longs(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let c = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("BIGINT");
            c.values().to_vec()
        })
        .collect()
}

/// A parameterized query: its `dataset_schema` is that of the `NULL`-bound analysis (a client
/// reads an empty one as an update), each marker a nullable parameter of its analyzed type; bound
/// and fetched, rebound and fetched again, without a cookie (a session holding a prepared handle
/// outlives a cookie-less fetch, §3.2); closed twice (the second is OK), then gone.
#[tokio::test(flavor = "multi_thread")]
async fn prepared_query_with_parameters() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (created, cookie) = create(&mut h, OPS, None, "SELECT ? + 1 AS n, ? AS s")
        .await
        .expect("create");
    assert!(
        cookie.is_some(),
        "a create marks its provisional session changed"
    );
    let columns = schema_of(&created.dataset_schema).expect("a query's columns");
    let names: Vec<_> = columns.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["n", "s"]);
    assert_eq!(
        columns.field(1).data_type(),
        &DataType::Null,
        "`? AS s` is untyped"
    );
    // `? + 1` types its marker `INT` (note §4.4, amended 2026-11-20); `? AS s` is untyped,
    // advertised as a string.
    let params = schema_of(&created.parameter_schema).expect("parameter schema");
    let types: Vec<_> = params
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    assert_eq!(types, [DataType::Int32, DataType::Utf8]);
    assert!(params.fields().iter().all(|f| f.is_nullable()));
    let handle = created.prepared_statement_handle.to_vec();
    for (n, s) in [(41, "x"), (1, "it's")] {
        let echoed = bind(&mut h, OPS, &handle, vec![row(&[(n, s)])])
            .await
            .expect("bind");
        assert_eq!(echoed, handle);
        let info = prepared_info(&mut h, OPS, &handle).await.expect("info");
        let batches = fetch(&mut h, OPS, None, ticket(&info))
            .await
            .expect("cookie-less fetch: the session holds a prepared handle");
        assert_eq!(first_longs(&batches), [n + 1]);
        let text = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("STRING");
        assert_eq!(text.value(0), s);
    }
    assert_eq!(h.flight.session_count(), 1);
    close(&mut h, OPS, &handle).await.expect("close");
    close(&mut h, OPS, &handle)
        .await
        .expect("closing a gone handle is OK");
    let e = prepared_info(&mut h, OPS, &handle)
        .await
        .expect_err("closed");
    assert_eq!(e.code(), Code::NotFound, "{e:?}");

    // Without parameters a query's columns are announced at create.
    let (plain, _) = create(&mut h, OPS, None, "SELECT 1 AS one")
        .await
        .expect("create");
    let schema = schema_of(&plain.dataset_schema).expect("dataset schema");
    assert_eq!(schema.field(0).name(), "one");
    assert_eq!(schema_of(&plain.parameter_schema), None);
}

/// A query binds one row; the columns must match the markers; another owner's handle is
/// unknown to it, and its close leaves the handle alone; a malformed body or handle is
/// `INVALID_ARGUMENT`.
#[tokio::test(flavor = "multi_thread")]
async fn prepared_bindings_are_checked() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (created, _) = create(&mut h, OPS, None, "SELECT ? AS a, ? AS b")
        .await
        .expect("create");
    let handle = created.prepared_statement_handle.to_vec();
    let e = bind(&mut h, OPS, &handle, vec![row(&[(1, "x"), (2, "y")])])
        .await
        .expect_err("two rows for a query");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(class_of(&e), Some("KRISHIV_FLIGHT_SQL_INVALID_PARAMETERS"));
    let e = bind(&mut h, OPS, &handle, vec![longs(&[1])])
        .await
        .expect_err("one column for two markers");
    assert_eq!(class_of(&e), Some("KRISHIV_FLIGHT_SQL_INVALID_PARAMETERS"));
    let (command, _) = create(&mut h, OPS, None, "SET spark.sql.shuffle.partitions = ?")
        .await
        .expect("create a command");
    let e = bind(
        &mut h,
        OPS,
        &command.prepared_statement_handle,
        vec![longs(&[1, 2])],
    )
    .await
    .expect_err("a command binds one row too (a batch is an update)");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(class_of(&e), Some("KRISHIV_FLIGHT_SQL_INVALID_PARAMETERS"));
    let e = prepared_info(&mut h, OPS, &handle)
        .await
        .expect_err("unbound markers");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(class_of(&e), Some("UNBOUND_SQL_PARAMETER"));

    let e = prepared_info(&mut h, OTHER, &handle)
        .await
        .expect_err("another owner's handle");
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
    close(&mut h, OTHER, &handle)
        .await
        .expect("closing what one does not hold is OK");
    bind(&mut h, OPS, &handle, vec![row(&[(1, "x")])])
        .await
        .expect("still the owner's");

    let e = h
        .client
        .do_action(authed(
            Action::new("CreatePreparedStatement", b"not a protobuf".to_vec()),
            OPS,
            None,
        ))
        .await
        .expect_err("a malformed body");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    let e = close(&mut h, OPS, &[1, 2, 3])
        .await
        .expect_err("a malformed handle");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    let e = prepared_info(&mut h, OPS, &[1, 2, 3])
        .await
        .expect_err("a malformed handle");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
}

/// A prepared update runs once per bound row, in order, the counts summed (`-1`: a `SET VAR`
/// reports none); more rows than `maxBatchRows` is `INVALID_ARGUMENT` and runs none; a
/// prepared query sent as an update is refused unrun.
#[tokio::test(flavor = "multi_thread")]
async fn prepared_update_batch_limit() {
    let mut h = start(FlightSqlConfig {
        max_batch_rows: 3,
        ..FlightSqlConfig::default()
    })
    .await;
    let (_, cookie) = updates::update(&mut h, OPS, None, "DECLARE VARIABLE v BIGINT DEFAULT 0")
        .await
        .expect("declare");
    let cookie = cookie.expect("a variable marks the session changed");
    let (created, set) = create(&mut h, OPS, Some(&cookie), "SET VAR v = v + ?")
        .await
        .expect("create");
    assert_eq!(set, None, "a presented cookie sets none");
    assert_eq!(schema_of(&created.dataset_schema), None, "a command");
    let handle = created.prepared_statement_handle.to_vec();
    let count = run_update(&mut h, OPS, &handle, vec![longs(&[1, 2]), longs(&[3])])
        .await
        .expect("three rows");
    assert_eq!(count, 0, "no run reports a count");
    let e = run_update(&mut h, OPS, &handle, vec![longs(&[10, 10, 10, 10])])
        .await
        .expect_err("over maxBatchRows");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(class_of(&e), Some("KRISHIV_FLIGHT_SQL_INVALID_PARAMETERS"));
    let (q, _) = info(&mut h, OPS, Some(&cookie), "SELECT v")
        .await
        .expect("info");
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect("fetch");
    assert_eq!(
        first_longs(&batches),
        [6],
        "1 + 2 + 3; the refused rows ran none"
    );

    let (query, _) = create(&mut h, OPS, Some(&cookie), "SELECT ?")
        .await
        .expect("create");
    let e = run_update(
        &mut h,
        OPS,
        &query.prepared_statement_handle,
        vec![longs(&[1])],
    )
    .await
    .expect_err("a query");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(class_of(&e), Some("KRISHIV_FLIGHT_SQL_QUERY_AS_UPDATE"));
}

/// Prepared handles are never evicted (design §4.1, v4 review m5): at the session's cap with
/// only prepared handles a create is refused, and both handles still run.
#[tokio::test(flavor = "multi_thread")]
async fn prepared_handles_not_evicted_refused_at_cap() {
    let mut h = start(FlightSqlConfig {
        handles: HandleLimits {
            per_session: 2,
            ..HandleLimits::default()
        },
        ..FlightSqlConfig::default()
    })
    .await;
    let (first, cookie) = create(&mut h, OPS, None, "SELECT 1 AS a")
        .await
        .expect("first");
    let cookie = cookie.expect("cookie");
    let (second, _) = create(&mut h, OPS, Some(&cookie), "SELECT 2 AS a")
        .await
        .expect("second");
    let e = create(&mut h, OPS, Some(&cookie), "SELECT 3 AS a")
        .await
        .expect_err("at the cap with only prepared handles");
    assert_eq!(e.code(), Code::ResourceExhausted, "{e:?}");
    assert_eq!(class_of(&e), Some("KRISHIV_FLIGHT_SQL_HANDLE_LIMIT"));
    for created in [&first, &second] {
        close(&mut h, OPS, &created.prepared_statement_handle)
            .await
            .expect("both still there to close");
    }
    let (third, _) = create(&mut h, OPS, Some(&cookie), "SELECT 3 AS a")
        .await
        .expect("room again");
    let info = prepared_info(&mut h, OPS, &third.prepared_statement_handle)
        .await
        .expect("info");
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&info))
        .await
        .expect("fetch");
    let a = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("INT");
    assert_eq!(a.value(0), 3);
}

/// A query a `NULL` cannot analyze (`LIMIT ?`: Spark's `INVALID_LIMIT_LIKE_EXPRESSION`) is
/// prepared with its markers tried as `1`, announces its columns, and runs with its binding.
#[tokio::test(flavor = "multi_thread")]
async fn prepared_limit_parameter() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (created, _) = create(&mut h, OPS, None, "SELECT id FROM range(10) LIMIT ?")
        .await
        .expect("a paging query prepares");
    let columns = schema_of(&created.dataset_schema).expect("a query's columns");
    assert_eq!(columns.field(0).name(), "id");
    let handle = created.prepared_statement_handle.to_vec();
    // `LIMIT` takes an `INT` (a `BIGINT` is Spark's `INVALID_LIMIT_LIKE_EXPRESSION.DATA_TYPE`).
    let three = RecordBatch::try_from_iter([(
        "a",
        Arc::new(Int32Array::from(vec![3])) as arrow_array::ArrayRef,
    )])
    .expect("batch");
    bind(&mut h, OPS, &handle, vec![three]).await.expect("bind");
    let info = prepared_info(&mut h, OPS, &handle).await.expect("info");
    let batches = fetch(&mut h, OPS, None, ticket(&info))
        .await
        .expect("fetch");
    assert_eq!(first_longs(&batches), [0, 1, 2]);
    let e = create(
        &mut h,
        OPS,
        None,
        "SELECT * FROM no_such_table WHERE id = ?",
    )
    .await
    .expect_err("fails with NULL and with 1: the NULL analysis' error");
    assert_eq!(class_of(&e), Some("TABLE_OR_VIEW_NOT_FOUND"), "{e:?}");
}

/// A batch whose second row fails after the first ran says how many ran, with the failing row's
/// own class (review of 4d7f3235f finding 5); the first row's effect stays.
#[tokio::test(flavor = "multi_thread")]
async fn prepared_update_partly_ran() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (_, cookie) = updates::update(&mut h, OPS, None, "DECLARE VARIABLE v BIGINT DEFAULT 0")
        .await
        .expect("declare");
    let cookie = cookie.expect("cookie");
    updates::update(
        &mut h,
        OPS,
        Some(&cookie),
        "SET spark.sql.ansi.enabled = true",
    )
    .await
    .expect("ANSI");
    let (created, _) = create(
        &mut h,
        OPS,
        Some(&cookie),
        "SET VAR v = v + CAST(? AS BIGINT)",
    )
    .await
    .expect("create");
    let strings = RecordBatch::try_from_iter([(
        "a",
        Arc::new(StringArray::from(vec!["1", "x", "2"])) as arrow_array::ArrayRef,
    )])
    .expect("batch");
    let e = run_update(
        &mut h,
        OPS,
        &created.prepared_statement_handle,
        vec![strings],
    )
    .await
    .expect_err("row 2 is not a number");
    assert_eq!(class_of(&e), Some("CAST_INVALID_INPUT"), "{e:?}");
    assert!(
        e.message().contains("1 of 3 parameter rows ran"),
        "{}",
        e.message()
    );
    let (q, _) = info(&mut h, OPS, Some(&cookie), "SELECT v")
        .await
        .expect("info");
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect("fetch");
    assert_eq!(first_longs(&batches), [1], "row 1 ran, row 3 did not");
}

/// A prepared handle's TTL runs from the end of its last use, not from its create (review of
/// 4d7f3235f finding 1), and a sweep while an update runs leaves it. Expiry is checked against
/// the real clock at each lookup, so the TTL leaves room for a loaded run: a bind ttl/2 after the
/// create, and the use right after the slow update ends, each land well inside it.
#[tokio::test(flavor = "multi_thread")]
async fn prepared_handle_ttl_runs_from_last_use() {
    let ttl = Duration::from_secs(2);
    let mut h = start(FlightSqlConfig {
        handles: HandleLimits {
            ttl,
            ..HandleLimits::default()
        },
        ..FlightSqlConfig::default()
    })
    .await;
    let (_, cookie) = updates::update(&mut h, OPS, None, "DECLARE VARIABLE v BIGINT DEFAULT 0")
        .await
        .expect("declare");
    let cookie = cookie.expect("cookie");
    let (created, _) = create(&mut h, OPS, Some(&cookie), "SELECT ? AS a")
        .await
        .expect("create");
    let handle = created.prepared_statement_handle.to_vec();
    let created_at = Instant::now();
    tokio::time::sleep(ttl / 2).await;
    bind(&mut h, OPS, &handle, vec![longs(&[1])])
        .await
        .expect("bind inside the TTL");
    let used_at = Instant::now();
    h.flight
        .sweep_at(created_at + ttl + Duration::from_millis(50));
    assert!(
        used_at + ttl > created_at + ttl + Duration::from_millis(50),
        "the sweep falls past create + TTL, inside last use + TTL"
    );
    prepared_info(&mut h, OPS, &handle)
        .await
        .expect("used since its create: still there");

    // A slow prepared update: a sweep far past the TTL while it runs leaves its handle.
    let (slow, _) = create(
        &mut h,
        OPS,
        Some(&cookie),
        "SET VAR v = (SELECT max(id) FROM range(0, 200000000))",
    )
    .await
    .expect("create");
    let mut client = h.client.clone();
    let cmd = CommandPreparedStatementUpdate {
        prepared_statement_handle: slow.prepared_statement_handle.clone(),
    };
    let data = put_rows(&cmd, Vec::new()).await;
    let running = tokio::spawn(async move {
        let response = client
            .do_put(authed(futures::stream::iter(data), OPS, None))
            .await?;
        response
            .into_inner()
            .try_collect::<Vec<_>>()
            .await
            .map(|r| r.len())
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    while h.manager.busy() == 0 && !running.is_finished() {
        assert!(Instant::now() < deadline, "the update never started");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    h.flight.sweep_at(Instant::now() + Duration::from_secs(60));
    assert!(!running.is_finished(), "the update is still running");
    assert_eq!(running.await.expect("task").expect("ran"), 1);
    // Used right after the update ended (its TTL runs from there), before anything slower.
    run_update(&mut h, OPS, &slow.prepared_statement_handle, Vec::new())
        .await
        .expect("idle only since the update ended: still there");
    let (q, _) = info(&mut h, OPS, Some(&cookie), "SELECT v")
        .await
        .expect("info");
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect("fetch");
    assert_eq!(first_longs(&batches), [199_999_999]);
}

/// The advertised types of the markers of `sql` prepared by `token` in `cookie`'s session.
async fn parameter_types(
    h: &mut Harness,
    token: &str,
    cookie: Option<&str>,
    sql: &str,
) -> Vec<DataType> {
    let (created, _) = create(h, token, cookie, sql).await.expect(sql);
    let params = schema_of(&created.parameter_schema).expect("parameter schema");
    assert!(params.fields().iter().all(|f| f.is_nullable()), "{sql}");
    params
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect()
}

/// Each marker is advertised with the type its context gives it (note §4.4, amended
/// 2026-11-20; the JDBC driver binds a value into a vector of that type): a query's from its
/// analysis (comparisons, `IN`, a written `CAST`, inside a subquery; `LIMIT ?` an `INT`; an
/// untyped marker a string), an `INSERT ... VALUES` cell's from its target column, by position
/// or column list. A caller that may not write the table learns none of its types.
#[tokio::test(flavor = "multi_thread")]
async fn parameter_types_come_from_the_statement() {
    let wh = tempfile::tempdir().expect("tempdir");
    let mut h = start_with(
        FlightSqlConfig::default(),
        Setup {
            factory: Some(grant_sessions(wh.path())),
            ..Setup::default()
        },
    )
    .await;
    let mut cookie: Option<String> = None;
    for sql in [
        "CREATE DATABASE db1",
        "CREATE TABLE db1.t (a INT, d DATE, s STRING, x DECIMAL(10,2)) USING parquet",
    ] {
        let (_, set) = info(&mut h, OPS, cookie.as_deref(), sql).await.expect(sql);
        cookie = cookie.or(set);
    }
    let c = cookie.as_deref();
    let decimal = DataType::Decimal128(10, 2);
    for (sql, want) in [
        (
            "SELECT a FROM db1.t WHERE d <= ? AND a = ? AND x > ? AND s LIKE ?",
            vec![
                DataType::Date32,
                DataType::Int32,
                decimal.clone(),
                DataType::Utf8,
            ],
        ),
        (
            "SELECT CAST(? AS BIGINT) AS b, ? AS u",
            vec![DataType::Int64, DataType::Utf8],
        ),
        (
            "SELECT a FROM db1.t WHERE a IN (?, ?) OR a IN (SELECT a FROM db1.t WHERE d = ?)",
            vec![DataType::Int32, DataType::Int32, DataType::Date32],
        ),
        ("SELECT a FROM db1.t LIMIT ?", vec![DataType::Int32]),
        (
            "INSERT INTO db1.t VALUES (?, ?, 'z', ?)",
            vec![DataType::Int32, DataType::Date32, decimal.clone()],
        ),
        (
            "INSERT INTO db1.t (d, a) VALUES (?, ?), ((?), 1)",
            vec![DataType::Date32, DataType::Int32, DataType::Date32],
        ),
        // Not a whole cell: untyped.
        (
            "INSERT INTO db1.t SELECT ? + 1, NULL, NULL, NULL",
            vec![DataType::Utf8],
        ),
    ] {
        assert_eq!(parameter_types(&mut h, OPS, c, sql).await, want, "{sql}");
    }
    // `other` may not write db1.t: its markers stay untyped.
    assert_eq!(
        parameter_types(&mut h, OTHER, None, "INSERT INTO db1.t VALUES (?, ?, ?, ?)").await,
        [
            DataType::Utf8,
            DataType::Utf8,
            DataType::Utf8,
            DataType::Utf8
        ]
    );
}

/// Design P3-T066 §5.1: while the server drains, every Flight SQL call that starts new work is
/// refused with `KRISHIV_SERVER_DRAINING` before any session work: `CreatePreparedStatement`, a
/// prepared handle made before the drain run through `GetFlightInfo` or either `DoPut`, and
/// `DoPut(CommandStatementUpdate)`. `ClosePreparedStatement` ends existing work and is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_draining_server_refuses_prepared_and_update_work_but_closes_handles() {
    use std::sync::atomic::Ordering::Release;
    let mut h = start(FlightSqlConfig::default()).await;
    let (created, cookie) = create(&mut h, OPS, None, "SELECT 1 AS x")
        .await
        .expect("create before the drain");
    let handle = created.prepared_statement_handle.to_vec();
    h.runtime.set_draining(true);
    h.draining.store(true, Release);
    let draining = |e: &Status, what: &str| {
        assert_eq!(e.code(), Code::FailedPrecondition, "{what}: {e:?}");
        assert_eq!(class_of(e), Some("KRISHIV_SERVER_DRAINING"), "{what}");
    };
    let e = create(&mut h, OPS, cookie.as_deref(), "SELECT 2")
        .await
        .expect_err("create");
    draining(&e, "CreatePreparedStatement");
    let e = prepared_info(&mut h, OPS, &handle)
        .await
        .expect_err("get_flight_info");
    draining(&e, "GetFlightInfo of a prepared handle");
    let e = bind(&mut h, OPS, &handle, Vec::new())
        .await
        .expect_err("bind");
    draining(&e, "DoPut(CommandPreparedStatementQuery)");
    let e = run_update(&mut h, OPS, &handle, Vec::new())
        .await
        .expect_err("prepared update");
    draining(&e, "DoPut(CommandPreparedStatementUpdate)");
    let e = updates::update(&mut h, OPS, cookie.as_deref(), "CREATE VIEW v AS SELECT 1")
        .await
        .expect_err("update");
    draining(&e, "DoPut(CommandStatementUpdate)");
    close(&mut h, OPS, &handle)
        .await
        .expect("ClosePreparedStatement while draining");
    assert_eq!(h.flight.held(), 0, "a refused call holds no ticket");
}

// SPDX-License-Identifier: Apache-2.0
//! `DoPut(CommandStatementUpdate)` end to end (design P3-T101 §4.3, §16 M2).

use arrow_flight::FlightData;
use arrow_flight::sql::{CommandStatementIngest, CommandStatementUpdate, DoPutUpdateResult};

use super::*;

/// The `FlightData` that opens a `DoPut` of `cmd`.
pub(super) fn put_of<M: ProstMessageExt>(cmd: &M) -> FlightData {
    FlightData::new().with_descriptor(FlightDescriptor::new_cmd(cmd.as_any().encode_to_vec()))
}

/// `DoPut(CommandStatementUpdate{sql})` carrying a `traceparent`: the record count.
pub(super) async fn update_with_trace(
    h: &mut Harness,
    token: &str,
    cookie: &str,
    sql: &str,
    traceparent: &str,
) -> Result<i64, Status> {
    let cmd = CommandStatementUpdate {
        query: sql.to_owned(),
        transaction_id: None,
    };
    let mut request = authed(futures::stream::iter([put_of(&cmd)]), token, Some(cookie));
    request
        .metadata_mut()
        .insert("traceparent", traceparent.parse().expect("header"));
    let response = h.client.do_put(request).await?;
    let results: Vec<_> = response.into_inner().try_collect().await?;
    let result = DoPutUpdateResult::decode(results[0].app_metadata.as_ref()).expect("result");
    Ok(result.record_count)
}

/// `DoPut(CommandStatementUpdate{sql})`: the record count and the session id set.
pub(super) async fn update(
    h: &mut Harness,
    token: &str,
    cookie: Option<&str>,
    sql: &str,
) -> Result<(i64, Option<String>), Status> {
    let cmd = CommandStatementUpdate {
        query: sql.to_owned(),
        transaction_id: None,
    };
    let response = h
        .client
        .do_put(authed(futures::stream::iter([put_of(&cmd)]), token, cookie))
        .await?;
    let id = set_cookie(response.metadata());
    let results: Vec<_> = response.into_inner().try_collect().await?;
    assert_eq!(results.len(), 1, "one PutResult");
    let result = DoPutUpdateResult::decode(results[0].app_metadata.as_ref()).expect("result");
    Ok((result.record_count, id))
}

/// DDL and DML answer `0` (Spark reports no count for them; `-1` would make the JDBC driver ask for a result set); the statement runs once, so a
/// second create of the same view fails; a cookie session keeps what it made.
#[tokio::test(flavor = "multi_thread")]
async fn update_counts() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (count, cookie) = update(
        &mut h,
        OPS,
        None,
        "CREATE TEMPORARY VIEW u AS SELECT 7 AS n",
    )
    .await
    .expect("create");
    assert_eq!(count, 0, "no count is 0 (Avatica reads -1 as a result set)");
    let cookie = cookie.expect("a temporary view marks the session changed: cookie");
    let e = update(
        &mut h,
        OPS,
        Some(&cookie),
        "CREATE TEMPORARY VIEW u AS SELECT 8 AS n",
    )
    .await
    .expect_err("ran once: the view exists");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    let (q, _) = info(&mut h, OPS, Some(&cookie), "SELECT n FROM u")
        .await
        .expect("the view is in the cookie session");
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect("fetch");
    assert_eq!(rows(&batches), 1);
    let (count, set) = update(
        &mut h,
        OPS,
        Some(&cookie),
        "SET spark.sql.ansi.enabled = true",
    )
    .await
    .expect("SET");
    assert_eq!((count, set), (0, None), "a presented cookie sets none");
}

/// A query sent as an update is refused before it runs (design §4.3, N2), and its session is
/// given back.
#[tokio::test(flavor = "multi_thread")]
async fn query_as_update_invalid_argument() {
    let mut h = start(FlightSqlConfig::default()).await;
    let e = update(&mut h, OPS, None, "SELECT 1")
        .await
        .expect_err("a query");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    assert_eq!(
        e.metadata()
            .get("x-krishiv-error-class")
            .and_then(|v| v.to_str().ok()),
        Some("KRISHIV_FLIGHT_SQL_QUERY_AS_UPDATE")
    );
    wait_sessions(&h, 0).await;
    let e = update(&mut h, OPS, None, &"-".repeat(2 << 20))
        .await
        .expect_err("over maxStatementBytes");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
}

/// Cookie-less updates that leave no session state run in transient sessions (§16 M2): no
/// cookie, nothing held, so a loop of them never meets `maxSessionsPerPrincipal`. One that does
/// (a `SET`) keeps its provisional session and answers with its cookie.
#[tokio::test(flavor = "multi_thread")]
async fn cookieless_update_loop_stays_under_session_cap() {
    let mut h = start_with(
        FlightSqlConfig::default(),
        Setup {
            limits: SessionLimits {
                max_sessions_per_principal: 2,
                ..SessionLimits::default()
            },
            ..Setup::default()
        },
    )
    .await;
    for i in 0..5 {
        let sql = format!("SHOW TABLES LIKE 'x{i}'");
        let (_, set) = update(&mut h, OPS, None, &sql)
            .await
            .expect("under the cap");
        assert_eq!(set, None, "transient: no cookie");
        wait_sessions(&h, 0).await;
    }
    let (_, set) = update(&mut h, OPS, None, "SET spark.sql.ansi.enabled = false")
        .await
        .expect("SET");
    let cookie = set.expect("a SET keeps its session: cookie");
    assert_eq!(h.flight.session_count(), 1);
    h.flight.sweep_at(Instant::now() + Duration::from_secs(120));
    assert_eq!(
        h.flight.session_count(),
        1,
        "a changed provisional session outlives provisionalTtl"
    );
    let (q, _) = info(&mut h, OPS, Some(&cookie), "SET spark.sql.ansi.enabled")
        .await
        .expect("info");
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect("fetch");
    let v = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("STRING");
    assert_eq!(v.value(0), "false");
}

/// The gate classifies a `DoPut` by its first message (§16 m5): none or no descriptor is
/// `INVALID_ARGUMENT`, a command not served (ingest) `UNIMPLEMENTED`, both before session work.
#[tokio::test(flavor = "multi_thread")]
async fn doput_gate_refuses_before_session_work() {
    let mut h = start(FlightSqlConfig::default()).await;
    let e = h
        .client
        .do_put(authed(futures::stream::empty::<FlightData>(), OPS, None))
        .await
        .expect_err("no first message");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    let e = h
        .client
        .do_put(authed(
            futures::stream::iter([FlightData::new()]),
            OPS,
            None,
        ))
        .await
        .expect_err("no descriptor");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    let ingest = CommandStatementIngest::default();
    let e = h
        .client
        .do_put(authed(futures::stream::iter([put_of(&ingest)]), OPS, None))
        .await
        .expect_err("ingest is not served");
    assert_eq!(e.code(), Code::Unimplemented, "{e:?}");
    assert_eq!(h.flight.session_count(), 0, "no session work");
}

/// An update needs a credential like every RPC.
#[tokio::test(flavor = "multi_thread")]
async fn update_requires_auth() {
    let mut h = start(FlightSqlConfig::default()).await;
    let cmd = CommandStatementUpdate {
        query: "SET a = 1".to_owned(),
        transaction_id: None,
    };
    let e = h
        .client
        .do_put(Request::new(futures::stream::iter([put_of(&cmd)])))
        .await
        .expect_err("no token");
    assert_eq!(e.code(), Code::Unauthenticated, "{e:?}");
}

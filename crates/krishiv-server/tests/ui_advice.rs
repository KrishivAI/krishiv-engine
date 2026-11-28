// SPDX-License-Identifier: Apache-2.0
//! The server console over a real runtime: the Advisor page's rows come from the query history, and a query
//! whose id two sessions share is ended in the named session only.
#![cfg(feature = "session")]
#![expect(
    clippy::unwrap_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_server::embedded::EmbeddedSource;
use krishiv_server::internal::{Row, SystemTable, UiSource};
use krishiv_server::session_source::SessionSource;
use krishiv_session::{EngineRuntime, QueryOptions, Session};

/// A cross join no machine finishes soon.
const ENDLESS: &str =
    "SELECT count(*) FROM range(0, 1000000000) a CROSS JOIN range(0, 1000000000) b";
/// The operation id both clients chose.
const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";

async fn run(session: &Session, sql: &str) {
    session.sql(sql).await.unwrap().collect().await.unwrap();
}

/// The sessions that run a query under [`OP`] now.
async fn running_in(source: &SessionSource) -> Vec<String> {
    let mut of: Vec<String> = source
        .rows(SystemTable::Queries)
        .await
        .unwrap()
        .iter()
        .filter(|r| r["query_id"] == OP && r["state"] == "running")
        .filter_map(|r| r["session_id"].as_str().map(str::to_owned))
        .collect();
    of.sort_unstable();
    of
}

#[tokio::test]
async fn the_advice_of_a_history_with_a_planted_filter_column_is_listed_and_none_without_a_history()
{
    let none = SessionSource::new(EngineRuntime::builder().build().unwrap()).unwrap();
    let rows = none.rows(SystemTable::PartitionAdvice).await.unwrap();
    assert!(rows.is_empty(), "no history, no advice: {rows:?}");

    let (history, warehouse) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let rt = EngineRuntime::builder()
        .config("krishiv.history.path", history.path().to_str().unwrap())
        .build()
        .unwrap();
    let client = Session::builder()
        .runtime(Arc::clone(&rt))
        .config(
            "spark.sql.warehouse.dir",
            warehouse.path().to_str().unwrap(),
        )
        .build()
        .unwrap();
    run(
        &client,
        "CREATE TABLE ev USING parquet AS \
         SELECT id, CAST(id % 365 AS INT) AS d, CAST(id % 3 AS INT) AS c FROM range(0, 20000)",
    )
    .await;
    for i in 0..10 {
        let sql = format!("SELECT sum(id) FROM ev WHERE d = {}", i * 3);
        run(&client, &sql).await;
    }
    run(&client, "SELECT sum(id) FROM ev WHERE c = 1").await;
    // The writer runs off the query's thread and stores in order: once the last query is stored, all are.
    client
        .sql_with(
            "SELECT sum(id) FROM ev WHERE c = 2",
            QueryOptions::new().operation_id(OP),
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let start = Instant::now();
    while rt.stored_profile(OP).unwrap().is_none() {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the profile was never stored"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let source = SessionSource::new(rt).unwrap();
    let rows = source.rows(SystemTable::PartitionAdvice).await.unwrap();
    let of_ev: Vec<&Row> = rows
        .iter()
        .filter(|r| r["table_name"] == "spark_catalog.default.ev")
        .collect();
    assert_eq!(of_ev.len(), 2, "{rows:?}");
    assert_eq!(of_ev[0]["rank"], 1);
    assert_eq!(of_ev[0]["column_name"], "d");
    assert_eq!(of_ev[0]["column_queries"], 10);
    assert_eq!(of_ev[0]["table_queries"], 12);
    assert_eq!(of_ev[0]["partitions"], 365);
    // The optimizer's estimate: the share of the table's queries that filter on the column (10 of 12) times
    // the share of partitions such a query skips (all but one of 365).
    let pruned = of_ev[0]["pruned_fraction"].as_f64().unwrap();
    let expected = (10.0 / 12.0) * (1.0 - 1.0 / 365.0);
    assert!((pruned - expected).abs() < 1e-9, "{pruned} != {expected}");
    assert_eq!(of_ev[1]["column_name"], "c");
    assert_eq!(of_ev[1]["rank"], 2);

    // The console of one session lists none of it, even the session whose queries the advice is folded from.
    let embedded = EmbeddedSource::new(&client).unwrap();
    let rows = embedded.rows(SystemTable::PartitionAdvice).await.unwrap();
    assert!(rows.is_empty(), "{rows:?}");
}

#[tokio::test]
async fn a_query_whose_id_two_sessions_share_is_ended_in_the_named_session_only() {
    let rt = EngineRuntime::builder().build().unwrap();
    let source = SessionSource::new(Arc::clone(&rt)).unwrap();
    let ann = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let bob = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let task = |s: &Session| {
        let s = s.clone();
        tokio::spawn(async move {
            s.sql_with(ENDLESS, QueryOptions::new().operation_id(OP))
                .await?
                .collect()
                .await
        })
    };
    let (of_ann, of_bob) = (task(&ann), task(&bob));
    let mut both = [ann.id().to_owned(), bob.id().to_owned()];
    both.sort_unstable();
    let start = Instant::now();
    while running_in(&source).await != both {
        assert!(start.elapsed() < Duration::from_secs(20), "never ran");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    assert!(
        !source
            .cancel_in_session(OP, "no-such-session")
            .await
            .unwrap()
    );
    assert!(
        !source
            .cancel_in_session("not-a-query-id", ann.id())
            .await
            .unwrap()
    );
    assert!(source.cancel_in_session(OP, ann.id()).await.unwrap());
    let failed = tokio::time::timeout(Duration::from_secs(20), of_ann)
        .await
        .expect("the query ends once it is cancelled")
        .unwrap()
        .unwrap_err();
    assert_eq!(
        failed.error_class().name(),
        "KRISHIV_QUERY_KILLED",
        "{failed}"
    );
    assert_eq!(
        running_in(&source).await,
        [bob.id().to_owned()],
        "the other session's query of the same id still runs"
    );
    assert!(!of_bob.is_finished());
    // Ended once: the same request again finds nothing running there.
    assert!(!source.cancel_in_session(OP, ann.id()).await.unwrap());

    assert!(source.cancel_in_session(OP, bob.id()).await.unwrap());
    let failed = tokio::time::timeout(Duration::from_secs(20), of_bob)
        .await
        .expect("the query ends once it is cancelled")
        .unwrap()
        .unwrap_err();
    assert_eq!(failed.error_class().name(), "KRISHIV_QUERY_KILLED");
}

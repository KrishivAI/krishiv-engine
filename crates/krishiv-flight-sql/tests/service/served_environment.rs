// SPDX-License-Identifier: Apache-2.0
//! A Flight SQL client of a served runtime (`krishiv server`: `DeclMode::Server`) never reads the
//! server's environment through `${env:NAME}` / `${system:NAME}` (rev-f2 code health 0.3, finding
//! I-1); the startup key `krishiv.server.sql.substituteEnvironment` restores Spark's substitution.

use arrow_array::Array;
use arrow_array::cast::AsArray;
use krishiv_session::EngineRuntime;
use krishiv_session::catalog_decl::DeclMode;
use krishiv_session::catalog_registry::CatalogSetup;

use super::*;

/// A harness whose base session is on a server-mode runtime with `config`.
async fn served(config: &[(&str, &str)]) -> Harness {
    let runtime = config
        .iter()
        .fold(
            EngineRuntime::builder().catalogs(CatalogSetup::new(DeclMode::Server)),
            |b, (k, v)| b.config(*k, *v),
        )
        .build()
        .expect("runtime");
    let base = Session::builder()
        .runtime(runtime)
        .build()
        .expect("session");
    start_with(
        FlightSqlConfig::default(),
        Setup {
            base: Some(base),
            ..Setup::default()
        },
    )
    .await
}

/// What a view made by `CREATE TEMPORARY VIEW .. AS SELECT '<reference>'` holds, read back in the
/// same cookie session.
async fn stored(h: &mut Harness, reference: &str) -> String {
    let sql = format!("CREATE TEMPORARY VIEW v AS SELECT '{reference}' AS s");
    let (created, cookie) = info(h, OPS, None, &sql).await.expect("create");
    let cookie = cookie.expect("cookie");
    fetch(h, OPS, Some(&cookie), ticket(&created))
        .await
        .expect("fetch create");
    let (q, _) = info(h, OPS, Some(&cookie), "SELECT s FROM v")
        .await
        .expect("info");
    let batches = fetch(h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect("fetch");
    let column = batches[0].column(0);
    assert_eq!(column.len(), 1);
    match column.as_string_opt::<i32>() {
        Some(a) => a.value(0).to_owned(),
        None => column.as_string_view().value(0).to_owned(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_flight_sql_client_never_reads_the_servers_environment() {
    let mut h = served(&[]).await;
    assert_eq!(stored(&mut h, "${env:PATH}").await, "${env:PATH}");
    let mut h = served(&[]).await;
    assert_eq!(stored(&mut h, "${system:PATH}").await, "${system:PATH}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_operator_restores_sparks_substitution_for_flight_sql() {
    let mut h = served(&[("krishiv.server.sql.substituteEnvironment", "true")]).await;
    let path = std::env::var("PATH").expect("PATH is set");
    assert_eq!(stored(&mut h, "${env:PATH}").await, path);
}

#[tokio::test(flavor = "multi_thread")]
async fn set_session_options_cannot_turn_the_environment_on() {
    use krishiv_flight_sql::session_options::ErrorValue;
    let mut h = served(&[]).await;
    let (errors, cookie) = super::session_options::set_options(
        &mut h,
        None,
        vec![(
            "krishiv.server.sql.substituteEnvironment",
            super::session_options::text("true"),
        )],
    )
    .await;
    assert_eq!(
        errors.get("krishiv.server.sql.substituteEnvironment"),
        Some(&(ErrorValue::Error as i32)),
        "{errors:?}"
    );
    let cookie = cookie.expect("cookie");
    let (q, _) = info(&mut h, OPS, Some(&cookie), "SELECT '${env:PATH}' AS s")
        .await
        .expect("info");
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect("fetch");
    let column = batches[0].column(0);
    let value = match column.as_string_opt::<i32>() {
        Some(a) => a.value(0).to_owned(),
        None => column.as_string_view().value(0).to_owned(),
    };
    assert_eq!(value, "${env:PATH}");
}

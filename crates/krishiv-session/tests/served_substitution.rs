// SPDX-License-Identifier: Apache-2.0
//! A served session (a `krishiv server` runtime, `DeclMode::Server`) never reads the server's
//! environment through `${env:NAME}` / `${system:NAME}` (rev-f2 code health 0.3, finding I-1):
//! the reference stays as written, no client setting turns it back on, and only the startup key
//! `krishiv.server.sql.substituteEnvironment` restores Spark's behaviour. Embedded sessions keep it.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::PolicyVersion;
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::catalog_decl::DeclMode;
use krishiv_session::catalog_registry::CatalogSetup;
use krishiv_session::{EngineRuntime, Principal, PrincipalKind, Session};

/// `PATH` is set wherever the tests run, so its value is what a leak would show.
const ENV_REF: &str = "${env:PATH}";
const SYSTEM_REF: &str = "${system:PATH}";

fn runtime(mode: DeclMode, pairs: &[(&str, &str)]) -> Arc<EngineRuntime> {
    let mut b = EngineRuntime::builder().catalogs(CatalogSetup::new(mode));
    for (k, v) in pairs {
        b = b.config(*k, *v);
    }
    b.build().unwrap()
}

/// A caller's session on `rt`, as `krishiv server` builds one for each signed-in client.
fn caller_session(rt: &Arc<EngineRuntime>) -> Session {
    let base = Session::builder().runtime(Arc::clone(rt)).build().unwrap();
    base.new_session_as(
        Principal::new("u", PrincipalKind::User),
        Arc::new(RolePolicy::new(
            Some(Role::Admin),
            Role::Admin,
            PolicyVersion(1),
        )),
    )
}

fn run(s: &Session, sql: &str) -> Result<Vec<arrow_array::RecordBatch>, KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await })
}

/// The first value of the string column `column` (offsets or views).
fn first(column: &dyn Array) -> String {
    if let Some(a) = column.as_string_opt::<i32>() {
        return a.value(0).to_owned();
    }
    if let Some(a) = column.as_string_opt::<i64>() {
        return a.value(0).to_owned();
    }
    match column.as_string_view_opt() {
        Some(a) => a.value(0).to_owned(),
        None => panic!("not a string column: {}", column.data_type()),
    }
}

/// The one string value `SELECT '<text>'` returns.
fn selected(s: &Session, text: &str) -> String {
    let batches = run(s, &format!("SELECT '{text}' AS v")).unwrap();
    assert_eq!(batches[0].num_rows(), 1);
    first(batches[0].column(0))
}

fn path() -> String {
    std::env::var("PATH").unwrap()
}

#[test]
fn a_served_statement_does_not_read_the_process_environment() {
    let rt = runtime(DeclMode::Server, &[]);
    for s in [
        caller_session(&rt),
        Session::builder().runtime(Arc::clone(&rt)).build().unwrap(),
    ] {
        assert_eq!(selected(&s, ENV_REF), ENV_REF);
        assert_eq!(selected(&s, SYSTEM_REF), SYSTEM_REF);
        // Configuration values are still substituted, and one cannot smuggle the environment in.
        run(&s, &format!("SET k1 = {ENV_REF}")).unwrap();
        assert_eq!(selected(&s, "${k1}"), ENV_REF);
        // Turning substitution on again (it is on by default) does not reach the environment.
        run(&s, "SET spark.sql.variable.substitute = true").unwrap();
        assert_eq!(selected(&s, ENV_REF), ENV_REF);
        // EXPLAIN substitutes through the same path.
        let explained = run(&s, &format!("EXPLAIN SELECT '{ENV_REF}'")).unwrap();
        let plan = first(explained[0].column(0));
        assert!(!plan.contains(&path()), "{plan}");
    }
}

#[test]
fn no_client_setting_turns_the_environment_back_on() {
    let rt = runtime(DeclMode::Server, &[]);
    let s = caller_session(&rt);
    let e = run(&s, "SET krishiv.server.sql.substituteEnvironment = true").unwrap_err();
    assert_eq!(e.error_class().name(), "CANNOT_MODIFY_STATIC_CONFIG", "{e}");
    assert_eq!(selected(&s, ENV_REF), ENV_REF);
    let built = Session::builder()
        .runtime(Arc::clone(&rt))
        .config("krishiv.server.sql.substituteEnvironment", "true")
        .build();
    let e = built.err().unwrap();
    assert_eq!(e.error_class().name(), "CANNOT_MODIFY_STATIC_CONFIG", "{e}");
}

#[test]
fn the_operator_may_restore_sparks_substitution_at_startup() {
    let rt = runtime(
        DeclMode::Server,
        &[("krishiv.server.sql.substituteEnvironment", "true")],
    );
    assert_eq!(selected(&caller_session(&rt), ENV_REF), path());
}

#[test]
fn an_embedded_session_reads_the_environment_as_spark_does() {
    let rt = runtime(DeclMode::Embedded, &[]);
    let s = Session::builder().runtime(rt).build().unwrap();
    assert_eq!(selected(&s, ENV_REF), path());
    assert_eq!(selected(&s, SYSTEM_REF), path());
    assert_eq!(selected(&Session::new().unwrap(), ENV_REF), path());
}

// SPDX-License-Identifier: Apache-2.0
//! `SetSessionOptions`, `GetSessionOptions` and `CloseSession` end to end (design P3-T101 §3.2).

use std::collections::HashMap;

use krishiv_flight_sql::session_options::{
    CLOSE_SESSION, CloseSessionResult, CloseStatus, ErrorValue, GET_SESSION_OPTIONS,
    GetSessionOptionsResult, OptionValue, SET_SESSION_OPTIONS, SessionOptionValue,
    SetSessionOptionsRequest, SetSessionOptionsResult, StringListValue,
};

use super::*;

/// A string option.
fn text(v: &str) -> SessionOptionValue {
    SessionOptionValue {
        option_value: Some(OptionValue::StringValue(v.to_owned())),
    }
}

/// The one answer of action `kind` with `body`, and the session id it set.
async fn action(
    h: &mut Harness,
    cookie: Option<&str>,
    kind: &str,
    body: Vec<u8>,
) -> Result<(Vec<u8>, Option<String>), Status> {
    let response = h
        .client
        .do_action(authed(Action::new(kind, body), OPS, cookie))
        .await?;
    let id = set_cookie(response.metadata());
    let results: Vec<_> = response.into_inner().try_collect().await?;
    assert_eq!(results.len(), 1, "one result");
    Ok((results[0].body.to_vec(), id))
}

/// `SetSessionOptions(options)`: the refused options and the session id set.
async fn set_options(
    h: &mut Harness,
    cookie: Option<&str>,
    options: Vec<(&str, SessionOptionValue)>,
) -> (HashMap<String, i32>, Option<String>) {
    let request = SetSessionOptionsRequest {
        session_options: options
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    };
    let (body, id) = action(h, cookie, SET_SESSION_OPTIONS, request.encode_to_vec())
        .await
        .expect("set options");
    let result = SetSessionOptionsResult::decode(body.as_slice()).expect("result");
    let errors = result
        .errors
        .into_iter()
        .map(|(k, e)| (k, e.value))
        .collect();
    (errors, id)
}

/// `GetSessionOptions`: the string options.
async fn get_options(h: &mut Harness, cookie: Option<&str>) -> HashMap<String, String> {
    let (body, id) = action(h, cookie, GET_SESSION_OPTIONS, Vec::new())
        .await
        .expect("get options");
    assert_eq!(id, None, "a read sets no cookie");
    GetSessionOptionsResult::decode(body.as_slice())
        .expect("result")
        .session_options
        .into_iter()
        .filter_map(|(k, v)| match v.option_value {
            Some(OptionValue::StringValue(s)) => Some((k, s)),
            _ => None,
        })
        .collect()
}

/// The JDBC driver's first call on a connection with `catalog=` is `SetSessionOptions`: it
/// makes a cookie session that keeps the catalog and schema for the statements after it.
#[tokio::test(flavor = "multi_thread")]
async fn jdbc_first_call_set_session_options_keeps_catalog() {
    // `ops` is an admin under the grant policy, so it may create the namespace.
    let wh = tempfile::tempdir().expect("tempdir");
    let mut h = start_with(
        FlightSqlConfig::default(),
        Setup {
            factory: Some(grant_sessions(wh.path())),
            ..Setup::default()
        },
    )
    .await;
    updates::update(&mut h, OPS, None, "CREATE NAMESPACE IF NOT EXISTS so_ns")
        .await
        .expect("namespace");
    let (errors, cookie) = set_options(
        &mut h,
        None,
        vec![
            ("catalog", text("spark_catalog")),
            ("schema", text("so_ns")),
        ],
    )
    .await;
    assert!(errors.is_empty(), "{errors:?}");
    let cookie = cookie.expect("a set marks the session changed: cookie");
    let options = get_options(&mut h, Some(&cookie)).await;
    assert_eq!(
        options.get("catalog").map(String::as_str),
        Some("spark_catalog")
    );
    assert_eq!(options.get("schema").map(String::as_str), Some("so_ns"));
    let (q, _) = info(
        &mut h,
        OPS,
        Some(&cookie),
        "SELECT current_database() AS db",
    )
    .await
    .expect("info");
    let batches = fetch(&mut h, OPS, Some(&cookie), ticket(&q))
        .await
        .expect("fetch");
    let db = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("STRING");
    assert_eq!(db.value(0), "so_ns");
    let fresh = get_options(&mut h, None).await;
    assert_eq!(
        fresh.get("schema").map(String::as_str),
        Some("default"),
        "a cookie-less read runs in a transient session"
    );
}

/// A `Startup` key is refused (`ERROR`), as `Config.set` refuses it; an unknown schema and a
/// list value are `INVALID_VALUE`; the other options of the call still apply; a set key is
/// listed unless `Config` would redact it; a valueless option unsets.
#[tokio::test(flavor = "multi_thread")]
async fn set_session_options_refuses_static_keys() {
    let mut h = start(FlightSqlConfig::default()).await;
    let list = SessionOptionValue {
        option_value: Some(OptionValue::StringListValue(StringListValue {
            values: vec!["a".to_owned()],
        })),
    };
    let (errors, cookie) = set_options(
        &mut h,
        None,
        vec![
            ("krishiv.resultCache.maxBytes", text("1")),
            ("schema", text("no_such_namespace")),
            ("spark.sql.session.timeZone", list),
            (
                "spark.sql.ansi.enabled",
                SessionOptionValue {
                    option_value: Some(OptionValue::BoolValue(false)),
                },
            ),
            ("my.app.secret", text("hidden")),
        ],
    )
    .await;
    let cookie = cookie.expect("cookie");
    let expected: HashMap<String, i32> = [
        ("krishiv.resultCache.maxBytes", ErrorValue::Error),
        ("schema", ErrorValue::InvalidValue),
        ("spark.sql.session.timeZone", ErrorValue::InvalidValue),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v as i32))
    .collect();
    assert_eq!(errors, expected);
    let options = get_options(&mut h, Some(&cookie)).await;
    assert_eq!(
        options.get("krishiv.sql.ansi.enabled").map(String::as_str),
        Some("false"),
        "listed under its canonical key, as Connect's `Config.getAll` lists it"
    );
    assert!(
        !options.contains_key("my.app.secret"),
        "redacted: {options:?}"
    );
    assert!(!options.contains_key("krishiv.resultCache.maxBytes"));
    let (errors, _) = set_options(
        &mut h,
        Some(&cookie),
        vec![(
            "spark.sql.ansi.enabled",
            SessionOptionValue { option_value: None },
        )],
    )
    .await;
    assert!(errors.is_empty(), "{errors:?}");
    let options = get_options(&mut h, Some(&cookie)).await;
    assert!(!options.contains_key("krishiv.sql.ansi.enabled"), "unset");

    let e = h
        .client
        .do_action(authed(
            Action::new(SET_SESSION_OPTIONS, vec![0xff, 0xff]),
            OPS,
            None,
        ))
        .await
        .expect_err("a malformed request");
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
}

/// `CloseSession` releases the cookie's session with its handles; without a cookie, or with
/// another caller's, it answers `CLOSED` and leaves everything alone.
#[tokio::test(flavor = "multi_thread")]
async fn close_session_releases_handles() {
    let mut h = start(FlightSqlConfig::default()).await;
    let (_, cookie) =
        set_options(&mut h, None, vec![("spark.sql.ansi.enabled", text("true"))]).await;
    let cookie = cookie.expect("cookie");
    let (_q, _) = info(&mut h, OPS, Some(&cookie), "SELECT 1")
        .await
        .expect("a handle in the session");
    assert_eq!(h.flight.session_count(), 1);
    assert_eq!(h.flight.handle_count(), 1);
    let closed = |body: Vec<u8>| {
        CloseSessionResult::decode(body.as_slice())
            .expect("result")
            .status
            == CloseStatus::Closed as i32
    };
    let response = h
        .client
        .do_action(authed(
            Action::new(CLOSE_SESSION, Vec::new()),
            OTHER,
            Some(&cookie),
        ))
        .await
        .expect("another caller's close");
    let results: Vec<_> = response.into_inner().try_collect().await.expect("results");
    assert!(closed(results[0].body.to_vec()));
    assert_eq!(h.flight.session_count(), 1, "another caller's: left alone");
    let (body, _) = action(&mut h, None, CLOSE_SESSION, Vec::new())
        .await
        .expect("close without a cookie");
    assert!(closed(body));
    assert_eq!(h.flight.session_count(), 1);
    let (body, _) = action(&mut h, Some(&cookie), CLOSE_SESSION, Vec::new())
        .await
        .expect("close");
    assert!(closed(body));
    assert_eq!(h.flight.session_count(), 0);
    assert_eq!(h.flight.handle_count(), 0, "its handles went with it");
}

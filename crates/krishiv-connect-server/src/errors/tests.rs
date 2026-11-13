// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use krishiv_common::error::{
    DivideByZero, ErrorQueryContext, KrishivError, KrishivSpillIoError, ParseSyntaxError,
};
use krishiv_connect::proto::v4_3::fetch_error_details_response::query_context::ContextType;

use super::{CACHE_SIZE, CACHE_TTL, ErrorCache, MAX_CHAIN, details};

fn error(n: usize) -> KrishivError {
    DivideByZero {
        config: format!("\"c{n}\""),
    }
    .into()
}

#[test]
fn an_error_is_fetched_once() {
    let mut cache = ErrorCache::default();
    let now = Instant::now();
    cache.put_at("a".to_owned(), error(1), now);
    assert_eq!(cache.take_at("a", now), Some(error(1)));
    assert_eq!(cache.take_at("a", now), None);
    assert_eq!(cache.take_at("unknown", now), None);
}

#[test]
fn the_cache_keeps_the_twenty_most_recent() {
    let mut cache = ErrorCache::default();
    let now = Instant::now();
    for n in 0..CACHE_SIZE + 5 {
        cache.put_at(format!("e{n}"), error(n), now);
    }
    assert_eq!(cache.len(), CACHE_SIZE);
    for n in 0..5 {
        assert_eq!(cache.take_at(&format!("e{n}"), now), None, "e{n}");
    }
    for n in 5..CACHE_SIZE + 5 {
        assert_eq!(cache.take_at(&format!("e{n}"), now), Some(error(n)));
    }
}

#[test]
fn an_entry_expires_sixty_seconds_after_it_was_put() {
    assert_eq!(CACHE_SIZE, 20);
    assert_eq!(CACHE_TTL, Duration::from_secs(60));
    let mut cache = ErrorCache::default();
    let t0 = Instant::now();
    cache.put_at("a".to_owned(), error(1), t0);
    cache.put_at("b".to_owned(), error(2), t0 + Duration::from_secs(30));
    let late = t0 + CACHE_TTL + Duration::from_millis(1);
    assert_eq!(cache.take_at("a", late), None);
    assert_eq!(cache.take_at("b", late), Some(error(2)));
    // A put purges what expired.
    cache.put_at("c".to_owned(), error(3), t0);
    cache.put_at("d".to_owned(), error(4), late + CACHE_TTL);
    assert_eq!(cache.len(), 1);
}

#[test]
fn details_of_an_error() {
    let e: KrishivError = ParseSyntaxError {
        error: "'FROM'".to_owned(),
        hint: String::new(),
    }
    .into();
    let e = e.with_query_context(ErrorQueryContext::sql("FROM", 7, 10));
    let r = details(&e);
    assert_eq!(r.root_error_idx, Some(0));
    assert!(r.session_id.is_empty() && r.server_side_session_id.is_empty());
    assert_eq!(r.errors.len(), 1);
    let root = &r.errors[0];
    assert_eq!(root.message, e.to_string());
    assert_eq!(
        root.error_type_hierarchy[..2],
        [
            "org.apache.spark.sql.catalyst.parser.ParseException",
            "org.apache.spark.sql.AnalysisException"
        ]
    );
    assert_eq!(
        root.error_type_hierarchy.last().map(String::as_str),
        Some("java.lang.Object")
    );
    assert!(root.stack_trace.is_empty());
    assert_eq!(root.cause_idx, None);
    let t = root.spark_throwable.as_ref().unwrap();
    assert_eq!(t.error_class.as_deref(), Some("PARSE_SYNTAX_ERROR"));
    assert_eq!(t.sql_state.as_deref(), Some("42601"));
    assert_eq!(
        t.message_parameters.get("error").map(String::as_str),
        Some("'FROM'")
    );
    assert_eq!(t.query_contexts.len(), 1);
    let c = &t.query_contexts[0];
    assert_eq!(c.context_type, ContextType::Sql as i32);
    assert_eq!((c.start_index, c.stop_index), (7, 10));
    assert_eq!(c.fragment, "FROM");
    assert!(c.object_type.is_empty() && c.object_name.is_empty());
}

#[test]
fn the_cause_chain_is_cut_at_five_and_redacted_per_cause() {
    // Six errors deep: root and five causes.
    let mut e = error(5);
    for n in (0..5).rev() {
        e = error(n).with_cause(e);
    }
    let r = details(&e);
    assert_eq!(r.errors.len(), MAX_CHAIN);
    for (i, err) in r.errors.iter().enumerate() {
        let want = (i + 1 < MAX_CHAIN).then(|| i32::try_from(i + 1).unwrap());
        assert_eq!(err.cause_idx, want, "{i}");
        assert!(
            err.message.contains(&format!("\"c{i}\"")),
            "{}",
            err.message
        );
    }
    // An operator-only cause (it can name internal paths) is sent with its
    // parameters redacted, as its own status would be.
    let inner: KrishivError = KrishivSpillIoError {
        operation: "write".to_owned(),
        path: "/srv/krishiv/spill/secret-dir".to_owned(),
        reason: "disk full".to_owned(),
    }
    .into();
    let r = details(&error(0).with_cause(inner));
    assert_eq!(r.errors.len(), 2);
    let text = format!("{:?}", r.errors[1].message);
    assert!(!text.contains("secret-dir"), "{text}");
    let params = &r.errors[1]
        .spark_throwable
        .as_ref()
        .unwrap()
        .message_parameters;
    assert!(
        params.values().all(|v| !v.contains("secret-dir")),
        "{params:?}"
    );
}

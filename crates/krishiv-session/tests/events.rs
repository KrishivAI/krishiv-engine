// SPDX-License-Identifier: Apache-2.0
//! The query event stream of an engine runtime (P3 E9, design `P3-E9-query-events.md` §10): every path into
//! `QueryLog` yields its events, gap-free, and the final usage is the `system.queries` row's.
#![cfg(test)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::executor::block_on;
use krishiv_session::events::{EngineEvent, EventKindTag, EventSink, QueryEvent, QueryEventKind};
use krishiv_session::{EngineRuntime, Session};

#[derive(Default)]
struct Collect(Mutex<Vec<QueryEvent>>);

impl EventSink for Collect {
    fn name(&self) -> &str {
        "collect"
    }

    fn deliver(&self, batch: &[Arc<EngineEvent>]) {
        let mut all = self.0.lock().unwrap();
        for e in batch {
            let EngineEvent::Query(q) = &**e else {
                continue;
            };
            all.push(q.clone());
        }
    }
}

impl Collect {
    /// The events of the query that end in a terminal event, once one arrived.
    fn wait_for_end(&self) -> Vec<QueryEvent> {
        let until = Instant::now() + Duration::from_secs(20);
        loop {
            let all = self.0.lock().unwrap().clone();
            if all
                .iter()
                .any(|e| e.kind.tag().name() == "finished" || e.kind.tag().name() == "cancelled")
                || Instant::now() > until
            {
                return all;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn runtime(extra: &[(&str, &str)]) -> (Arc<EngineRuntime>, Arc<Collect>) {
    let sink = Arc::new(Collect::default());
    let mut b = EngineRuntime::builder().event_sink(Arc::clone(&sink) as Arc<dyn EventSink>);
    for (k, v) in extra {
        b = b.config(*k, *v);
    }
    (b.build().unwrap(), sink)
}

fn session(rt: &Arc<EngineRuntime>) -> Session {
    Session::builder().runtime(Arc::clone(rt)).build().unwrap()
}

fn assert_gap_free(events: &[QueryEvent]) {
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e.seq, i as u64, "{events:?}");
    }
}

#[test]
fn a_finished_query_yields_submitted_started_finished() {
    let (rt, sink) = runtime(&[]);
    let s = session(&rt);
    block_on(
        block_on(s.sql("SELECT id FROM range(0, 10)"))
            .unwrap()
            .collect(),
    )
    .unwrap();
    let events = sink.wait_for_end();
    assert_gap_free(&events);
    let kinds: Vec<_> = events.iter().map(|e| e.kind.tag().name()).collect();
    assert_eq!(kinds.first(), Some(&"submitted"), "{kinds:?}");
    assert_eq!(kinds.last(), Some(&"finished"), "{kinds:?}");
    assert!(kinds.contains(&"started"), "{kinds:?}");
    let QueryEventKind::Finished(f) = &events.last().unwrap().kind else {
        panic!("not finished");
    };
    assert_eq!(f.state, "finished");
    assert_eq!(f.usage.record_id.1, events[0].query.query_id);
    assert_eq!(f.usage.record_id.2, events.last().unwrap().seq);
    assert!(f.usage.wall_ns.is_some());
    assert_eq!(events[0].query.compute_id, rt.compute_id());
}

#[test]
fn a_refused_statement_is_submitted_then_failed_at_once() {
    let (rt, sink) = runtime(&[]);
    let s = session(&rt);
    assert!(block_on(s.sql("SELECT * FROM no_such_table_x")).is_err());
    let events = sink.wait_for_end();
    assert_gap_free(&events);
    let kinds: Vec<_> = events.iter().map(|e| e.kind.tag().name()).collect();
    assert_eq!(kinds, ["submitted", "finished"], "{events:?}");
    let QueryEventKind::Finished(f) = &events[1].kind else {
        panic!();
    };
    assert_eq!(f.state, "failed");
    assert!(f.error_class.is_some());
}

#[test]
fn text_carries_no_literal_under_the_mask_policy_and_is_cut() {
    let (rt, sink) = runtime(&[
        ("krishiv.system.queryText.literals", "mask"),
        ("krishiv.events.textMaxBytes", "20"),
    ]);
    let s = session(&rt);
    block_on(
        block_on(s.sql("SELECT 'sekrit-value-1234567890' AS c"))
            .unwrap()
            .collect(),
    )
    .unwrap();
    let events = sink.wait_for_end();
    let QueryEventKind::Submitted { text, .. } = &events[0].kind else {
        panic!("{events:?}");
    };
    let text = text.as_deref().unwrap();
    assert!(!text.contains("sekrit"), "{text}");
    assert!(text.len() <= 20, "{text}");
    let all = events
        .iter()
        .map(|e| e.to_json().to_string())
        .collect::<String>();
    assert!(!all.contains("sekrit"));
}

#[test]
fn text_carries_no_literal_under_the_default_or_keep_policy() {
    for conf in [
        &[][..],
        &[("krishiv.system.queryText.literals", "keep")][..],
    ] {
        let (rt, sink) = runtime(conf);
        let s = session(&rt);
        block_on(
            block_on(s.sql("SELECT 'sekrit-value-1234567890' AS c"))
                .unwrap()
                .collect(),
        )
        .unwrap();
        let events = sink.wait_for_end();
        let all = events
            .iter()
            .map(|e| e.to_json().to_string())
            .collect::<String>();
        assert!(!all.contains("sekrit"), "{conf:?}: {all}");
    }
}

#[test]
fn no_sink_means_an_inactive_bus() {
    let rt = EngineRuntime::builder().build().unwrap();
    assert_eq!(rt.event_queue_depth(), 0);
    let _ = EventKindTag::Queued;
}

#[test]
fn the_json_file_sink_writes_one_document_per_event() {
    let dir = std::env::temp_dir().join(format!("krishiv-ev-it-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let prefix = dir.join("ev");
    let rt = EngineRuntime::builder()
        .config("krishiv.events.file.path", prefix.to_str().unwrap())
        .build()
        .unwrap();
    let s = session(&rt);
    block_on(block_on(s.sql("SELECT 1")).unwrap().collect()).unwrap();
    let until = Instant::now() + Duration::from_secs(20);
    let text = loop {
        let text = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .collect::<String>();
        if text.contains("\"kind\":\"finished\"") || Instant::now() > until {
            break text;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(text.lines().count() >= 2, "{text}");
    for line in text.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["schema"], "krishiv.event.v1");
    }
    std::fs::remove_dir_all(&dir).ok();
}

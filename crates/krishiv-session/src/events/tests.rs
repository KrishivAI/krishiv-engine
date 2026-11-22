// SPDX-License-Identifier: Apache-2.0
//! Tests of the query event stream.

use super::*;
use std::sync::atomic::AtomicUsize;

fn event(kind: QueryEventKind, seq: u64) -> EngineEvent {
    EngineEvent::Query(QueryEvent {
        version: SCHEMA_VERSION,
        seq,
        emitted_at_ms: 0,
        query: QueryRef {
            query_id: "q".into(),
            session_id: "s".into(),
            principal: "p".into(),
            delegated_by: None,
            compute_id: "c".into(),
            trace_id: None,
            tags: vec![("k".into(), "v".into())],
        },
        kind,
    })
}

fn usage(seq: u64) -> UsageRecord {
    UsageRecord {
        record_id: ("s".into(), "q".into(), seq),
        kind: UsageKind::Final,
        principal: "p".into(),
        delegated_by: None,
        session_id: "s".into(),
        compute_id: "c".into(),
        workload_class: None,
        resource_group: None,
        statement_class: None,
        window_start_ms: 1,
        window_end_ms: 2,
        wall_ns: Some(5),
        queued_ns: None,
        cpu_ns: None,
        bytes_scanned: None,
        rows_read: None,
        files_read: None,
        bytes_written: None,
        rows_written: None,
        files_written: None,
        spill_bytes: None,
        spill_files: None,
        peak_memory_bytes: None,
        result_cache_hit: false,
    }
}

fn finished(seq: u64) -> EngineEvent {
    event(
        QueryEventKind::Finished(Box::new(FinishedEvent {
            state: "finished".into(),
            usage: usage(seq),
            error_class: None,
            statement_class: Some("READ".into()),
            result_cache_hit: false,
        })),
        seq,
    )
}

/// A sink that holds every batch until released.
struct Gate {
    open: Mutex<bool>,
    wake: Condvar,
    seen: Mutex<Vec<EventKindTag>>,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            open: Mutex::new(false),
            wake: Condvar::new(),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.wake.notify_all();
    }
}

impl EventSink for Gate {
    fn name(&self) -> &str {
        "gate"
    }

    fn deliver(&self, batch: &[Arc<EngineEvent>]) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.wake.wait(open).unwrap();
        }
        self.seen
            .lock()
            .unwrap()
            .extend(batch.iter().map(|e| e.tag()));
    }
}

#[test]
fn a_blocked_sink_never_blocks_emit_and_drops_progress_first() {
    let bus = EventBus::new("c".into(), 100, 4);
    let gate = Gate::new();
    bus.register(Arc::clone(&gate) as Arc<dyn EventSink>);
    // The thread takes the first event and blocks inside `deliver`; the queue then holds 4.
    bus.emit(event(QueryEventKind::Queued, 0));
    for _ in 0..200 {
        if bus.queue_depth() == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    for i in 0..3 {
        bus.emit(event(QueryEventKind::Progress(usage(i)), i));
    }
    bus.emit(finished(3));
    // Full now: a terminal event pushes out a Progress, a further Progress is dropped itself.
    bus.emit(finished(4));
    bus.emit(event(QueryEventKind::Progress(usage(5)), 5));
    assert_eq!(bus.dropped("gate", EventKindTag::Progress), 2);
    assert_eq!(bus.dropped("gate", EventKindTag::Finished), 0);
    gate.release();
    bus.shutdown(Duration::from_secs(5));
    let seen = gate.seen.lock().unwrap().clone();
    assert_eq!(
        seen.iter()
            .filter(|t| **t == EventKindTag::Finished)
            .count(),
        2
    );
}

#[test]
fn a_sink_that_never_returns_is_left_behind_at_shutdown() {
    let bus = EventBus::new("c".into(), 100, 10);
    let gate = Gate::new();
    bus.register(Arc::clone(&gate) as Arc<dyn EventSink>);
    // The thread takes this one and blocks in `deliver`; what is queued after it is left over.
    bus.emit(event(QueryEventKind::Queued, 0));
    for _ in 0..400 {
        if bus.queue_depth() == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    bus.emit(event(QueryEventKind::Queued, 1));
    bus.emit(event(QueryEventKind::Queued, 2));
    let began = std::time::Instant::now();
    bus.shutdown(Duration::from_millis(100));
    assert!(began.elapsed() < Duration::from_secs(3));
    assert!(bus.dropped("gate", EventKindTag::Queued) >= 1);
    gate.release();
}

#[test]
fn an_inactive_bus_is_inactive_until_a_sink_registers() {
    let bus = EventBus::default();
    assert!(!bus.is_active());
    bus.emit(finished(0)); // no sink: nothing happens
    assert_eq!(bus.dropped_total(), 0);
}

#[test]
fn a_sink_may_filter_kinds() {
    struct OnlyEnds(AtomicUsize);
    impl EventSink for OnlyEnds {
        fn name(&self) -> &str {
            "ends"
        }
        fn accept(&self, kind: EventKindTag) -> bool {
            kind.terminal()
        }
        fn deliver(&self, batch: &[Arc<EngineEvent>]) {
            self.0.fetch_add(batch.len(), Ordering::SeqCst);
        }
    }
    let bus = EventBus::new("c".into(), 100, 10);
    let sink = Arc::new(OnlyEnds(AtomicUsize::new(0)));
    bus.register(Arc::clone(&sink) as Arc<dyn EventSink>);
    bus.emit(event(QueryEventKind::Queued, 0));
    bus.emit(finished(1));
    bus.shutdown(Duration::from_secs(5));
    assert_eq!(sink.0.load(Ordering::SeqCst), 1);
}

#[test]
fn json_has_the_v1_shape_for_every_kind() {
    let kinds = [
        QueryEventKind::Submitted {
            statement_class: Some("READ".into()),
            text: Some("SELECT ?".into()),
        },
        QueryEventKind::Queued,
        QueryEventKind::Started {
            resource_group: Some("g".into()),
            queued_ns: 7,
        },
        QueryEventKind::Progress(usage(1)),
        QueryEventKind::Cancelled {
            reason: Some("timeout".into()),
            usage: usage(2),
        },
    ];
    let mut docs: Vec<Value> = kinds
        .into_iter()
        .enumerate()
        .map(|(i, k)| event(k, i as u64).to_json())
        .collect();
    docs.push(finished(9).to_json());
    for doc in &docs {
        assert_eq!(doc["schema"], SCHEMA_NAME);
        assert_eq!(doc["minor"], 0);
        assert_eq!(doc["query"]["query_id"], "q");
        assert_eq!(doc["query"]["tags"]["k"], "v");
        assert!(doc["kind"].is_string() && doc["seq"].is_u64());
    }
    assert_eq!(docs[0]["text"], "SELECT ?");
    assert_eq!(docs[2]["queued_ns"], 7);
    assert_eq!(docs[3]["usage"]["fidelity"], "exact");
    assert_eq!(docs[4]["reason"], "timeout");
    assert_eq!(docs[5]["usage"]["record_id"]["seq"], 9);
    // No error message member exists on any kind.
    assert!(docs.iter().all(|d| d.get("error_message").is_none()));
}

#[test]
fn text_is_cut_at_a_character_boundary() {
    assert_eq!(cut("héllo", 2), "h");
    assert_eq!(cut("héllo", 3), "hé");
    assert_eq!(cut("abc", 10), "abc");
}

#[test]
fn the_json_sink_rotates_and_keeps_the_newest_segments() {
    let dir = std::env::temp_dir().join(format!("krishiv-events-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let sink = JsonLinesSink::new(dir.join("ev"), 400, 2);
    let batch: Vec<_> = (0..30).map(|i| Arc::new(finished(i))).collect();
    sink.deliver(&batch);
    sink.flush();
    assert_eq!(sink.failures(), 0);
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(names.len(), 2, "{names:?}");
    // Every line of every kept segment is one complete document.
    for n in &names {
        for line in std::fs::read_to_string(dir.join(n)).unwrap().lines() {
            let v: Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["schema"], SCHEMA_NAME);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_restarted_json_sink_continues_after_the_highest_segment() {
    let dir = std::env::temp_dir().join(format!("krishiv-events-restart-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("ev.000007.jsonl"), "old\n").unwrap();
    let sink = JsonLinesSink::new(dir.join("ev"), 1_000_000, 5);
    sink.deliver(&[Arc::new(finished(0))]);
    sink.flush();
    assert!(dir.join("ev.000008.jsonl").exists());
    assert_eq!(
        std::fs::read_to_string(dir.join("ev.000007.jsonl")).unwrap(),
        "old\n"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn row_marks_name_the_transitions() {
    let a = RowMark {
        state: "running",
        started: false,
        ended: false,
    };
    let queued = RowMark {
        state: "queued",
        ..a
    };
    assert_eq!(a.since(queued).collect::<Vec<_>>(), [Transition::Queued]);
    let started = RowMark {
        state: "running",
        started: true,
        ended: false,
    };
    assert_eq!(
        queued.since(started).collect::<Vec<_>>(),
        [Transition::Started]
    );
    let ended = RowMark {
        state: "finished",
        started: true,
        ended: true,
    };
    assert_eq!(
        started.since(ended).collect::<Vec<_>>(),
        [Transition::Ended]
    );
    // A write that fails after its stream ends once more, with its new state.
    let failed = RowMark {
        state: "failed",
        ..ended
    };
    assert_eq!(ended.since(failed).collect::<Vec<_>>(), [Transition::Ended]);
    assert_eq!(ended.since(ended).count(), 0);
}

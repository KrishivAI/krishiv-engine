// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the `system` catalog: row layouts, masking, table names.

use super::*;

#[test]
fn an_error_message_is_masked_by_quotes_not_by_sql_tokens() {
    let mask = TextPolicy {
        max_bytes: 4096,
        mask: true,
    };
    let keep = TextPolicy {
        max_bytes: 4096,
        mask: false,
    };
    let m = "The value 'abc' of the type \"STRING\" can't be cast -- see 'x'";
    assert_eq!(
        mask.apply_message(m),
        "The value '?' of the type \"STRING\" can't be cast -- see '?'"
    );
    assert_eq!(keep.apply_message(m), m);
    // An unclosed quote hides nothing after it.
    assert_eq!(mask.apply_message("it's fine"), "it's fine");
}

#[test]
fn a_backquoted_value_is_masked_but_a_backquoted_name_stays() {
    let mask = TextPolicy {
        max_bytes: 4096,
        mask: true,
    };
    let got = mask.apply_message(
        "The variant value `\"secret_canary\"` cannot be cast; table `db`.`t_1` is missing.",
    );
    assert!(!got.contains("secret_canary"), "{got}");
    assert!(got.contains("`db`.`t_1`"), "{got}");
}

#[test]
fn an_embedded_backquote_does_not_end_a_masked_value() {
    let mask = TextPolicy {
        max_bytes: 4096,
        mask: true,
    };
    let got =
        mask.apply_message("The variant value `\"x`secret_canary\"` cannot be cast to `INT`.");
    assert!(!got.contains("secret_canary"), "{got}");
    assert!(got.contains("`INT`"), "{got}");
}

#[test]
fn a_backquote_before_whitespace_inside_a_masked_value_does_not_end_it() {
    let mask = TextPolicy {
        max_bytes: 4096,
        mask: true,
    };
    let got =
        mask.apply_message("The variant value `\"x` secret_canary\"` cannot be cast to `INT`.");
    assert_eq!(got, "The variant value `?` cannot be cast to `INT`.");
    // An escaped double quote in the rendered value does not close its JSON string.
    let got =
        mask.apply_message("The variant value `\"x\\\"` secret_canary\"` cannot be cast to `INT`.");
    assert_eq!(got, "The variant value `?` cannot be cast to `INT`.");
}

#[test]
fn a_raw_value_parameter_is_masked_and_a_long_value_is_linear() {
    let mask = TextPolicy {
        max_bytes: 4096,
        mask: true,
    };
    let got = mask.apply_message_with(
        "Malformed records: {\"a\":\"secret_canary\", in parsing.",
        &[("badRecord", "{\"a\":\"secret_canary\",".to_owned())],
    );
    assert_eq!(got, "Malformed records: ? in parsing.");
    // 40,000 backquote-space pairs inside one open JSON string: one pass, not a rescan per backquote.
    let long = format!("The value `\"{}` cannot be cast.", "x` ".repeat(40_000));
    let got = mask.apply_message(&long);
    assert!(!got.contains("x` x`"));
}

#[test]
fn scientific_notation_and_other_bare_numbers_are_masked_in_a_message() {
    let mask = TextPolicy {
        max_bytes: 4096,
        mask: true,
    };
    let got = mask.apply_message(
        "The value 1e400 and 1.5E-3 and 7e+2 and v2 and 42P01x cannot be cast. SQLSTATE: 22003",
    );
    assert_eq!(
        got,
        "The value ? and ? and ? and v2 and 42P01x cannot be cast. SQLSTATE: 22003"
    );
}

#[test]
fn an_escaped_quote_does_not_end_a_masked_value() {
    let mask = TextPolicy {
        max_bytes: 4096,
        mask: true,
    };
    let m = "The value 'x\\' secret_canary' of the type \"STRING\" cannot be cast.";
    let got = mask.apply_message(m);
    assert!(!got.contains("secret_canary"), "{got}");
    assert_eq!(
        mask.apply_message("a 'b\\\\' then 'c' end"),
        "a '?' then '?' end"
    );
}

#[test]
fn capabilities_has_one_row_per_fact() {
    let rows = capability_rows();
    let mut pairs: Vec<(&str, &str)> = rows
        .iter()
        .map(|r| (r[0].as_str(), r[1].as_str()))
        .collect();
    pairs.sort_unstable();
    assert!(
        pairs.windows(2).all(|w| w[0] != w[1]),
        "one row per (kind, name): {pairs:?}"
    );
    let one = |kind: &str, name: &str| -> [String; 4] {
        let found: Vec<_> = rows
            .iter()
            .filter(|r| r[0] == kind && r[1] == name)
            .collect();
        assert_eq!(found.len(), 1, "{kind}/{name}");
        found[0].clone()
    };
    let engine = one("build", "engine");
    assert_eq!(engine[2], "krishiv");
    assert_eq!(engine[3], env!("CARGO_PKG_VERSION"));
    let cloud = one("feature", "cloud");
    assert_eq!(
        cloud[2],
        if cfg!(feature = "cloud") {
            "enabled"
        } else {
            "disabled"
        }
    );
    let memory: Arc<dyn MemoryPool> =
        krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
    let sources = Sources {
        queries: Arc::new(QueryLog::default()),
        io: krishiv_io::IoScheduler::new(krishiv_io::IoConfig::default()),
        relations: crate::cache::RelationCache::new(&memory),
        metrics: Arc::new(krishiv_exec::MetricsRegistry::new()),
        memory,
        monitor: Arc::new(Monitor::new(
            Arc::new(Principal::local_user()),
            Arc::new(krishiv_plan::analyzer::AllowAll),
        )),
        sessions: Arc::default(),
        groups: None,
        advice: Arc::new(crate::partition_advice::AdviceCell::new(
            None,
            krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig::default()).unwrap(),
        )),
        catalogs: Default::default(),
        information: None,
    };
    assert!(provider("capabilities", &sources).unwrap().is_some());
    assert!(provider("nope", &sources).unwrap().is_none());
}

#[test]
fn a_finished_query_can_be_turned_into_a_failed_one() {
    let log = QueryLog::default();
    let id = log.begin(Some("INSERT"), Some("write"));
    log.finish(
        &id,
        &Finished {
            state: "finished",
            wall_ns: 5,
            peak_memory_bytes: 7,
            error_class: None,
            operators: Vec::new(),
            totals: None,
            hints: None,
        },
    );
    let error: KrishivError = krishiv_common::error::PathNotFound {
        path: "x".to_owned(),
    }
    .into();
    log.fail(&id, &error);
    let row = &log.snapshot(false)[0];
    assert_eq!(row.state, "failed");
    assert_eq!(row.error_class.as_deref(), Some("PATH_NOT_FOUND"));
    assert_eq!(row.wall_ns, Some(5));
    // A row evicted meanwhile is just gone.
    log.fail("unknown", &error);
    let cancelled: KrishivError = krishiv_common::error::OperationCanceled {}.into();
    log.fail(&id, &cancelled);
    let row = &log.snapshot(false)[0];
    assert_eq!(row.state, "cancelled");
    assert_eq!(row.error_class, None);
}

#[test]
fn concurrent_tags_never_exceed_the_limit() {
    let log = QueryLog::default();
    let taken: usize = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..MAX_TAGS * 2)
            .map(|i| {
                let log = &log;
                scope.spawn(move || log.try_set_tag(&format!("k{i}"), "v"))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| usize::from(h.join().unwrap_or(false)))
            .sum()
    });
    assert_eq!(taken, MAX_TAGS);
    assert_eq!(log.tags().len(), MAX_TAGS);
    // A known key can still change; an unknown one is refused until a tag is removed.
    assert!(log.try_set_tag("k0", "other") || log.tags().iter().all(|(k, _)| k != "k0"));
    assert!(!log.try_set_tag("new", "v"));
    let (key, _) = log.tags()[0].clone();
    log.remove_tag(&key);
    assert!(log.try_set_tag("new", "v"));
}

#[test]
fn a_replayed_query_is_a_finished_row() {
    let log = QueryLog::default();
    log.set_trace(Some(RemoteParent {
        trace_id: "trace".to_owned(),
        span_id: "span".to_owned(),
        flags: 1,
    }));
    assert!(log.try_set_tag("team", "etl"));
    let id = log.record_replayed(TextPolicy::default(), Some("SELECT 1"), Some("READ"), None);
    let row = &log.snapshot(false)[0];
    assert_eq!(row.id, id);
    assert_eq!(row.state, "finished");
    assert_eq!(row.statement_class, Some("READ"));
    assert_eq!(row.tags, [("team".to_owned(), "etl".to_owned())]);
    assert_eq!(row.error_class, None);
}

#[test]
fn an_error_is_masked_by_the_policy_the_query_started_under() {
    let log = QueryLog::default();
    let mask = TextPolicy {
        mask: true,
        ..TextPolicy::default()
    };
    let id = log.begin_as(mask, Some("SELECT 'a'"), None, None).unwrap();
    // Another statement of the session runs under `keep` before this one fails.
    let _ = log.begin_as(TextPolicy::default(), Some("SELECT 'b'"), None, None);
    log.note_error(
        &id,
        &KrishivError::from(InternalError {
            message: "bad value 'secret'".to_owned(),
        }),
    );
    let rows = log.snapshot(false);
    let row = rows.iter().find(|r| r.id == id).unwrap();
    let message = row.error_message.clone().unwrap();
    assert!(!message.contains("secret"), "{message}");
}

#[test]
fn a_query_pool_is_named_by_its_history_id_and_found_again() {
    let log = QueryLog::default();
    let id = log.begin(Some("SELECT 1"), None);
    let pool = pool_name(&id);
    assert_eq!(query_of_path(&format!("process/{pool}")), Some(id.clone()));
    assert_eq!(
        query_of_path(&format!("process/{pool}/hash-join")),
        Some(id)
    );
    assert_eq!(
        query_of_path(&format!("process/session:abc/{pool}/sort")),
        query_of_path(&format!("process/{pool}"))
    );
    assert_eq!(query_of_path("process/session:abc"), None);
    assert_eq!(query_of_path("process/shuffle"), None);
    assert_eq!(query_of_path("process"), None);
}

#[test]
fn query_ids_are_uuid_v7_and_unique() {
    let log = QueryLog::default();
    let a = log.begin(Some("SELECT 1"), Some("read"));
    let b = log.begin(None, None);
    assert_ne!(a, b);
    for id in [&a, &b] {
        let p: Vec<&str> = id.split('-').collect();
        assert_eq!(
            p.iter().map(|x| x.len()).collect::<Vec<_>>(),
            [8, 4, 4, 4, 12],
            "{id}"
        );
        assert!(p[2].starts_with('7') && "89ab".contains(&p[3][..1]), "{id}");
    }
    log.finish(
        &a,
        &Finished {
            state: "finished",
            wall_ns: 5,
            peak_memory_bytes: 7,
            error_class: None,
            operators: Vec::new(),
            totals: None,
            hints: None,
        },
    );
    let rows = log.snapshot(false);
    assert_eq!((rows[0].state, rows[1].state), ("finished", "running"));
}

#[test]
fn a_history_copy_shares_the_operator_rows() {
    let log = QueryLog::default();
    let id = log.begin(None, None);
    let op = OperatorRow {
        query_id: id.clone(),
        session_id: String::new(),
        operator_id: 0,
        node_id: None,
        name: "Scan".to_owned(),
        role: "source",
        drivers: 1,
        rows_in: 0,
        rows_out: 3,
        batches_in: 0,
        batches_out: 1,
        compute_ns: 0,
        max_driver_compute_ns: 0,
        blocked_io_ns: 0,
        blocked_memory_ns: 0,
        blocked_sink_ns: 0,
        peak_reserved_bytes: 0,
        spills: 0,
        spilled_bytes: 0,
        spill_files: 0,
        spill_write_bytes: 0,
        pipelines: vec![0],
        extra: Vec::new(),
        workers: 1,
    };
    log.finish(
        &id,
        &Finished {
            state: "finished",
            wall_ns: 1,
            peak_memory_bytes: 0,
            error_class: None,
            operators: vec![op],
            totals: None,
            hints: None,
        },
    );
    let (a, b) = (log.snapshot(false), log.snapshot(false));
    assert!(Arc::ptr_eq(&a[0].operators, &b[0].operators));
    let ops = log.operators(false);
    assert_eq!((ops.len(), ops[0].rows_out), (1, 3));
}

#[test]
fn history_is_bounded_and_long_text_is_cut_on_a_char() {
    let log = QueryLog::default();
    for _ in 0..(crate::history::PRINCIPAL_ROWS + 5) {
        let id = log.begin(None, None);
        log.finish(
            &id,
            &Finished {
                state: "finished",
                wall_ns: 1,
                peak_memory_bytes: 0,
                error_class: None,
                operators: Vec::new(),
                totals: None,
                hints: None,
            },
        );
    }
    assert_eq!(log.snapshot(false).len(), crate::history::PRINCIPAL_ROWS);
    let long = "é".repeat(TEXT_BYTES);
    log.begin(Some(&long), None);
    let last = log.snapshot(false).pop().unwrap();
    assert!(last.truncated && last.text.unwrap().len() <= TEXT_BYTES);
}

#[test]
fn baggage_is_bounded_and_malformed_entries_are_skipped() {
    let got = parse_baggage("a=1;prop=x, nokey, =v, k = v2 ,c\u{7}=1");
    assert_eq!(
        got,
        [
            ("a".to_owned(), "1".to_owned()),
            ("k".to_owned(), "v2".to_owned())
        ]
    );
    let many = (0..100)
        .map(|i| format!("k{i}=v"))
        .collect::<Vec<_>>()
        .join(",");
    assert_eq!(parse_baggage(&many).len(), MAX_BAGGAGE_ENTRIES);
    let big = format!("a={}", "x".repeat(MAX_BAGGAGE_BYTES));
    assert!(parse_baggage(&big).is_empty());
    let p = parse_remote_parent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        .expect("valid");
    assert_eq!((p.span_id.as_str(), p.flags), ("00f067aa0ba902b7", 1));
}

#[test]
fn traceparent_is_checked_as_w3c_says() {
    let ok = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    assert_eq!(
        parse_remote_parent(ok).map(|p| p.trace_id).as_deref(),
        Some("4bf92f3577b34da6a3ce929d0e0e4736")
    );
    for bad in [
        "",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
        "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
    ] {
        assert!(parse_remote_parent(bad).is_none(), "{bad}");
    }
    assert!(parse_remote_parent(&format!("01-{}-00f067aa0ba902b7-01-more", &ok[3..35])).is_some());
}

#[test]
fn names_are_matched_without_case() {
    assert!(is_system("SYSTEM"));
    assert!(is_system_table(&[
        "System".to_owned(),
        "Capabilities".to_owned()
    ]));
    assert!(!is_system_table(&["system".to_owned(), "x".to_owned()]));
    assert!(!is_system_table(&["system".to_owned()]));
}

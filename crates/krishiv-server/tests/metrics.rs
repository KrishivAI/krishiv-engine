// SPDX-License-Identifier: Apache-2.0
//! Integration test (feature `metrics-prometheus`): `GET /metrics` serves
//! the registry snapshot as Prometheus text, is absent when disabled, and
//! never changes a counter.
#![cfg(feature = "metrics-prometheus")]
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_exec::{
    BUCKETS, CATALOGUE, MetricKind, MetricSource, MetricsRegistry, QueryOutcome, Status,
    WorkloadClass,
};
use krishiv_server::metrics::{SnapshotProvider, router_with_metrics};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

fn full_registry() -> Arc<MetricsRegistry> {
    let registry = Arc::new(MetricsRegistry::new());
    registry.query_begin(WorkloadClass::Interactive);
    registry.fold(&QueryOutcome {
        class: WorkloadClass::Interactive,
        status: Status::Succeeded,
        error_class: None,
        duration_secs: 0.25,
        metrics: &krishiv_exec::QueryMetrics::default(),
        arbitration_spills: 2,
        arbitration_aborts: 1,
        cache_hits: 3,
        cache_misses: 1,
    });
    registry.fold(&QueryOutcome {
        class: WorkloadClass::Batch,
        status: Status::Failed,
        error_class: Some("DIVIDE_BY_ZERO"),
        duration_secs: 0.5,
        metrics: &krishiv_exec::QueryMetrics::default(),
        arbitration_spills: 0,
        arbitration_aborts: 0,
        cache_hits: 0,
        cache_misses: 0,
    });
    for class in ["interactive", "batch", "background", "agent"] {
        registry
            .add_gauge(
                "krishiv.query.active",
                vec![("class".to_owned(), class.to_owned())],
                || Some(0),
            )
            .unwrap();
    }
    registry
        .add_gauge("krishiv.memory.reserved", Vec::new(), || Some(8))
        .unwrap();
    registry
        .add_gauge("krishiv.memory.limit", Vec::new(), || Some(16))
        .unwrap();
    registry
        .add_gauge("krishiv.spill.disk.used", Vec::new(), || Some(0))
        .unwrap();
    registry
        .add_gauge(
            "krishiv.cache.size",
            vec![("cache".to_owned(), "metadata".to_owned())],
            || Some(4),
        )
        .unwrap();
    registry
        .add_gauge(
            "krishiv.build.info",
            vec![
                ("version".to_owned(), "0.1.0".to_owned()),
                ("git_sha".to_owned(), "abc".to_owned()),
            ],
            || Some(1),
        )
        .unwrap();
    registry
}

fn provider(registry: &Arc<MetricsRegistry>) -> SnapshotProvider {
    let registry = Arc::clone(registry);
    Arc::new(move || registry.snapshot())
}

async fn serve(router: axum::Router) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await;
    });
    addr
}

async fn get(addr: std::net::SocketAddr, path: &str) -> String {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    tokio::time::timeout(std::time::Duration::from_secs(10), s.read_to_end(&mut out))
        .await
        .expect("read timed out")
        .unwrap();
    String::from_utf8(out).unwrap()
}

fn body(response: &str) -> &str {
    response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or("")
}

/// One parsed sample line.
#[derive(Debug)]
struct Sample {
    name: String,
    labels: Vec<(String, String)>,
    value: f64,
}

/// One parsed metric: HELP/TYPE lines plus samples.
#[derive(Debug, Default)]
struct Metric {
    help: bool,
    type_line: Option<String>,
    samples: Vec<Sample>,
}

/// Strictly parses what the route emits: HELP/TYPE/sample lines only, no
/// timestamps, no other comments.
fn parse(text: &str) -> Result<BTreeMap<String, Metric>, String> {
    let mut metrics: BTreeMap<String, Metric> = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        let line_no = index + 1;
        if let Some(rest) = line.strip_prefix("# HELP ") {
            let (name, _) = rest
                .split_once(' ')
                .ok_or_else(|| format!("line {line_no}: HELP without text"))?;
            metrics.entry(base_of(name)).or_default().help = true;
        } else if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, kind) = rest
                .split_once(' ')
                .ok_or_else(|| format!("line {line_no}: TYPE without kind"))?;
            if !matches!(kind, "counter" | "gauge" | "histogram") {
                return Err(format!("line {line_no}: bad TYPE {kind:?}"));
            }
            let entry = metrics.entry(base_of(name)).or_default();
            if entry.type_line.replace(kind.to_owned()).is_some() {
                return Err(format!("line {line_no}: duplicate TYPE for {name}"));
            }
        } else if line.starts_with('#') {
            return Err(format!("line {line_no}: unexpected comment"));
        } else {
            let sample = parse_sample(line, line_no)?;
            metrics
                .entry(base_of(&sample.name))
                .or_default()
                .samples
                .push(sample);
        }
    }
    Ok(metrics)
}

fn base_of(name: &str) -> String {
    for suffix in ["_bucket", "_count", "_sum"] {
        if let Some(base) = name.strip_suffix(suffix) {
            return base.to_owned();
        }
    }
    name.to_owned()
}

fn parse_sample(line: &str, line_no: usize) -> Result<Sample, String> {
    let bad = |why: &str| format!("line {line_no}: {why} in {line:?}");
    let at = line.find(['{', ' ']).ok_or_else(|| bad("no value"))?;
    let (name, rest) = line.split_at(at);
    let (labels, value) = if let Some(braced) = rest.strip_prefix('{') {
        // The closing brace is the first `}` outside a quoted value:
        // values may hold `,`/`}`/`=` themselves.
        let end = labels_end(braced).ok_or_else(|| bad("unclosed labels"))?;
        let value = braced[end + 1..].trim_start_matches(' ');
        if value.is_empty() || value.contains(' ') {
            return Err(bad("bad value"));
        }
        (parse_labels(&braced[..end], line_no)?, value)
    } else {
        let value = rest.trim_start_matches(' ');
        if value.is_empty() || value.contains(' ') {
            return Err(bad("bad value"));
        }
        (Vec::new(), value)
    };
    Ok(Sample {
        name: name.to_owned(),
        labels,
        value: value
            .parse::<f64>()
            .map_err(|_| bad("value is not a float"))?,
    })
}

/// The index of the `}` closing a label set: the first `}` outside a
/// quoted value (quotes honor backslash escapes).
fn labels_end(braced: &str) -> Option<usize> {
    let mut in_quotes = false;
    let mut escaped = false;
    for (index, c) in braced.char_indices() {
        if in_quotes {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_quotes = false;
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == '}' {
            return Some(index);
        }
    }
    None
}

fn parse_labels(text: &str, line_no: usize) -> Result<Vec<(String, String)>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for pair in split_labels(text) {
        let (name, quoted) = pair
            .split_once('=')
            .ok_or_else(|| format!("line {line_no}: bad label {pair:?}"))?;
        let value = quoted
            .strip_prefix('"')
            .and_then(|q| q.strip_suffix('"'))
            .ok_or_else(|| format!("line {line_no}: label {name} is not quoted"))?;
        out.push((name.to_owned(), unescape(value, line_no)?));
    }
    Ok(out)
}

/// Splits a label set on top-level commas: values may hold commas.
fn split_labels(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    let mut escaped = false;
    for (index, c) in text.char_indices() {
        if in_quotes {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_quotes = false;
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == ',' {
            parts.push(&text[start..index]);
            start = index + 1;
        }
    }
    parts.push(&text[start..]);
    parts
}

fn unescape(value: &str, line_no: usize) -> Result<String, String> {
    let mut out = String::new();
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('n') => out.push('\n'),
            other => return Err(format!("line {line_no}: bad escape \\{other:?}")),
        }
    }
    Ok(out)
}

fn label<'a>(sample: &'a Sample, name: &str) -> Option<&'a str> {
    sample
        .labels
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.as_str())
}

#[tokio::test]
async fn metrics_serves_every_sourced_catalogue_metric() {
    let addr = serve(router_with_metrics(true, provider(&full_registry()))).await;
    let response = get(addr, "/metrics").await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(
        response
            .to_lowercase()
            .contains("content-type: text/plain; version=0.0.4"),
        "{response}"
    );
    let parsed = parse(body(&response)).unwrap();
    for def in CATALOGUE {
        if def.source == MetricSource::NoSource {
            assert!(
                !parsed.contains_key(&def.prometheus_name()),
                "{} emits without a source",
                def.name
            );
            continue;
        }
        let metric = parsed.get(&def.prometheus_name()).unwrap_or_else(|| {
            panic!(
                "{} ({}) missing from:\n{}",
                def.name,
                def.prometheus_name(),
                body(&response)
            )
        });
        assert!(metric.help, "{} has no HELP", def.name);
        assert_eq!(
            metric.type_line.as_deref(),
            Some(def.kind.as_str()),
            "{} has a bad TYPE",
            def.name
        );
        assert!(!metric.samples.is_empty(), "{} has no samples", def.name);
    }
    // Histogram buckets are cumulative and end in +Inf.
    let duration = &parsed["krishiv_query_duration_seconds"];
    let buckets: Vec<&Sample> = duration
        .samples
        .iter()
        .filter(|s| {
            s.name == "krishiv_query_duration_seconds_bucket"
                && label(s, "status") == Some("succeeded")
        })
        .collect();
    assert_eq!(buckets.len(), BUCKETS.len() + 1);
    let mut previous = 0.0;
    for sample in &buckets {
        assert!(sample.value >= previous, "buckets are not cumulative");
        previous = sample.value;
    }
    assert_eq!(label(buckets[buckets.len() - 1], "le"), Some("+Inf"));
    let count = duration
        .samples
        .iter()
        .find(|s| {
            s.name == "krishiv_query_duration_seconds_count"
                && label(s, "status") == Some("succeeded")
        })
        .unwrap();
    assert_eq!(buckets[buckets.len() - 1].value, count.value);
    assert!(duration.samples.iter().any(|s| {
        s.name == "krishiv_query_duration_seconds_sum" && label(s, "status") == Some("succeeded")
    }));
    // Counters use the OpenMetrics `_total` name.
    assert!(parsed.contains_key("krishiv_query_count_total"));
    // The base routes still serve beside /metrics.
    let version = get(addr, "/api/v1/version").await;
    assert!(version.starts_with("HTTP/1.1 200"), "{version}");
}

#[tokio::test]
async fn metrics_is_absent_when_disabled() {
    let addr = serve(router_with_metrics(false, provider(&full_registry()))).await;
    let response = get(addr, "/metrics").await;
    assert!(response.starts_with("HTTP/1.1 404"), "{response}");
    assert!(
        response
            .to_lowercase()
            .contains("content-type: application/json"),
        "{response}"
    );
    assert!(!body(&response).contains("# HELP"), "{response}");
}

#[tokio::test]
async fn scraping_does_not_change_counters() {
    let registry = full_registry();
    let before = registry.snapshot();
    let addr = serve(router_with_metrics(true, provider(&registry))).await;
    let first = get(addr, "/metrics").await;
    let second = get(addr, "/metrics").await;
    assert!(first.starts_with("HTTP/1.1 200"), "{first}");
    assert_eq!(body(&first), body(&second));
    assert_eq!(registry.snapshot(), before);
    assert_eq!(registry.fold_count(), 2);
}

#[tokio::test]
async fn label_values_escape() {
    // Quotes, backslashes, newlines, commas, braces and `=`: every
    // separator the text format gives meaning.
    let nasty = "Q\"U\\OTE\nX,A=B}C{D";
    let registry = Arc::new(MetricsRegistry::new());
    registry.fold(&QueryOutcome {
        class: WorkloadClass::Interactive,
        status: Status::Failed,
        error_class: Some(nasty),
        duration_secs: 0.0,
        metrics: &krishiv_exec::QueryMetrics::default(),
        arbitration_spills: 0,
        arbitration_aborts: 0,
        cache_hits: 0,
        cache_misses: 0,
    });
    let addr = serve(router_with_metrics(true, provider(&registry))).await;
    let response = get(addr, "/metrics").await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let text = body(&response).to_owned();
    assert!(text.contains("Q\\\"U\\\\OTE\\nX,A=B}C{D"), "{text}");
    let parsed = parse(&text).unwrap();
    let count = &parsed["krishiv_query_count_total"];
    let failed = count
        .samples
        .iter()
        .find(|s| label(s, "status") == Some("failed"))
        .unwrap();
    assert_eq!(label(failed, "error_class"), Some(nasty));
    assert_eq!(
        parsed["krishiv_query_duration_seconds"]
            .type_line
            .as_deref(),
        Some(MetricKind::Histogram.as_str())
    );
}

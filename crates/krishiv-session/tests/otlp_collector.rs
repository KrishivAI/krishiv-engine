// SPDX-License-Identifier: Apache-2.0
//! OTLP export against a collector (P2-T116 "OTLP collector test", P2-T125):
//! a local HTTP server stands in for the collector, records what the
//! exporters POST and the test decodes the protobuf bodies.
#![cfg(test)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::executor::block_on;
use krishiv_exec::MetricsRegistry;
use krishiv_session::Session;
use krishiv_session::otlp::{OtlpConfig, OtlpMetrics, OtlpTracing};
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::any_value::Value;
use prost::Message as _;
use tracing_subscriber::layer::SubscriberExt as _;

/// What the collector received: (path, headers lowercased, body).
type Requests = Arc<Mutex<Vec<(String, Vec<(String, String)>, Vec<u8>)>>>;

/// A collector that answers every POST with 200 and keeps the requests.
fn collector() -> (String, Requests) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let seen = Requests::default();
    let store = Arc::clone(&seen);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let store = Arc::clone(&store);
            std::thread::spawn(move || serve(stream, &store));
        }
    });
    (url, seen)
}

fn serve(mut stream: std::net::TcpStream, store: &Requests) {
    let Ok(read) = stream.try_clone() else { return };
    let mut reader = BufReader::new(read);
    // Keep-alive: answer requests until the client closes.
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let path = line
            .split_whitespace()
            .nth(1)
            .unwrap_or_default()
            .to_owned();
        let mut headers = Vec::new();
        let mut length = 0usize;
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                break;
            }
            if let Some((k, v)) = h.trim().split_once(':') {
                let (k, v) = (k.trim().to_ascii_lowercase(), v.trim().to_owned());
                if k == "content-length" {
                    length = v.parse().unwrap_or(0);
                }
                headers.push((k, v));
            }
        }
        let mut body = vec![0; length];
        if reader.read_exact(&mut body).is_err() {
            return;
        }
        if let Ok(mut s) = store.lock() {
            s.push((path, headers, body));
        }
        let ok =
            "HTTP/1.1 200 OK\r\ncontent-type: application/x-protobuf\r\ncontent-length: 0\r\n\r\n";
        if stream.write_all(ok.as_bytes()).is_err() {
            return;
        }
    }
}

/// Waits until `pick` finds something in the requests.
fn wait_for<T>(seen: &Requests, pick: impl Fn(&str, &[u8]) -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let found = seen
            .lock()
            .ok()
            .and_then(|s| s.iter().find_map(|(p, _, b)| pick(p, b)));
        if let Some(found) = found {
            return found;
        }
        assert!(
            Instant::now() < deadline,
            "the collector got nothing it wanted"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn hex_bytes(h: &str) -> Vec<u8> {
    (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex"))
        .collect()
}

#[test]
fn query_spans_reach_the_collector_with_the_database_attributes() {
    let (url, seen) = collector();
    let mut config = OtlpConfig::new(&url);
    config.service_name = "krishiv-test".into();
    config.headers.push(("x-api-key".into(), "k".into()));
    let tracing = OtlpTracing::new(&config).expect("exporter");
    let subscriber = tracing_subscriber::registry().with(tracing.layer());
    // Actions run on the session's executor threads, so the subscriber is global.
    tracing::subscriber::set_global_default(subscriber).expect("one subscriber in this binary");

    let s = Session::new().expect("session");
    // T068: the caller's trace arrives as headers; a hostile baggage is bounded.
    assert!(s.set_inbound_trace(
        Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        Some("user=ann, broken, k=v"),
    ));
    let df = block_on(s.sql("SELECT 'secret-token' AS x, 1 + 1 AS y")).expect("plans");
    block_on(df.collect()).expect("runs");
    tracing.flush().expect("flush");

    // The request that holds the query span: a batch the exporter scheduled
    // earlier may carry other spans.
    let request = wait_for(&seen, |path, body| {
        let request = (path == "/v1/traces")
            .then(|| ExportTraceServiceRequest::decode(body).ok())
            .flatten()?;
        request
            .resource_spans
            .iter()
            .flat_map(|r| &r.scope_spans)
            .flat_map(|s| &s.spans)
            .any(|s| s.name == "krishiv.query")
            .then_some(request)
    });
    let resource = request.resource_spans[0]
        .resource
        .as_ref()
        .expect("resource");
    assert!(resource.attributes.iter().any(|a| a.key == "service.name"
        && matches!(&a.value.as_ref().and_then(|v| v.value.clone()),
            Some(Value::StringValue(v)) if v == "krishiv-test")));
    let span = request
        .resource_spans
        .iter()
        .flat_map(|r| &r.scope_spans)
        .flat_map(|s| &s.spans)
        .find(|s| s.name == "krishiv.query")
        .expect("a krishiv.query span");
    let attribute = |key: &str| {
        span.attributes.iter().find(|a| a.key == key).and_then(|a| {
            match a.value.as_ref()?.value.as_ref()? {
                Value::StringValue(v) => Some(v.clone()),
                _ => None,
            }
        })
    };
    // The engine span is a child of the caller's span, in the caller's trace.
    assert_eq!(
        span.trace_id,
        hex_bytes("4bf92f3577b34da6a3ce929d0e0e4736"),
        "the query span joins the client's trace"
    );
    assert_eq!(span.parent_span_id, hex_bytes("00f067aa0ba902b7"));
    assert_eq!(attribute("db.system.name").as_deref(), Some("krishiv"));
    assert_eq!(attribute("db.operation.name").as_deref(), Some("SELECT"));
    let text = attribute("db.query.text").expect("sanitized text");
    assert!(!text.contains("secret-token"), "literal left in {text}");
    // The configured header went out with the request.
    let sent = seen.lock().expect("lock");
    assert!(
        sent.iter()
            .any(|(_, h, _)| h.iter().any(|(k, v)| k == "x-api-key" && v == "k"))
    );
}

#[test]
fn engine_metrics_reach_the_collector() {
    use opentelemetry_proto::tonic::metrics::v1::metric::Data;
    let (url, seen) = collector();
    // The query-end fold is on only with `krishiv.metrics.enabled`.
    let session = Session::builder()
        .config("krishiv.metrics.enabled", "true")
        .build()
        .expect("session");
    // A finished query puts a counter and a histogram in the registry.
    let df = block_on(session.sql("SELECT 1 AS x")).expect("plans");
    block_on(df.collect()).expect("runs");
    let pusher = OtlpMetrics::start(
        &OtlpConfig::new(&url),
        Duration::from_millis(50),
        move || session.metrics(),
    )
    .expect("pusher");
    // A push that carries a sum and a histogram as well as the gauges.
    let request = wait_for(&seen, |path, body| {
        let request = (path == "/v1/metrics")
            .then(|| ExportMetricsServiceRequest::decode(body).ok())
            .flatten()?;
        let kinds: Vec<_> = request
            .resource_metrics
            .iter()
            .flat_map(|r| &r.scope_metrics)
            .flat_map(|s| &s.metrics)
            .filter_map(|m| m.data.as_ref())
            .collect();
        (kinds.iter().any(|d| matches!(d, Data::Sum(_)))
            && kinds.iter().any(|d| matches!(d, Data::Histogram(_))))
        .then_some(request)
    });
    let sent: Vec<_> = request
        .resource_metrics
        .iter()
        .flat_map(|r| &r.scope_metrics)
        .flat_map(|s| &s.metrics)
        .collect();
    assert!(sent.iter().any(|m| m.name == "krishiv.build.info"));
    // T125 section 6.4: the names, units and descriptions are the catalogue's.
    let catalogue = MetricsRegistry::catalogue();
    for m in &sent {
        let def = catalogue
            .iter()
            .find(|d| d.name == m.name)
            .unwrap_or_else(|| panic!("{} is not in the catalogue", m.name));
        assert_eq!(
            (m.unit.as_str(), m.description.as_str()),
            (def.unit, def.description)
        );
    }
    // ...and the resource names the worker and the instance.
    let resource = request.resource_metrics[0]
        .resource
        .as_ref()
        .expect("resource");
    let value = |key: &str| {
        resource
            .attributes
            .iter()
            .find(|a| a.key == key)
            .and_then(|a| match a.value.as_ref()?.value.as_ref()? {
                Value::StringValue(v) => Some(v.clone()),
                _ => None,
            })
    };
    assert_eq!(value("worker.id").as_deref(), Some("local"));
    assert!(value("service.instance.id").is_some());
    drop(pusher);
}

#[test]
fn an_unreachable_collector_never_fails_the_query() {
    // A port that was open a moment ago and is closed now: connections are refused.
    let closed = {
        let l = TcpListener::bind("127.0.0.1:0").expect("bind");
        format!("http://{}", l.local_addr().expect("addr"))
    };
    let session = Session::new().expect("session");
    let s = session.clone();
    let pusher = OtlpMetrics::start(
        &OtlpConfig::new(closed),
        Duration::from_millis(20),
        move || s.metrics(),
    )
    .expect("starts");
    let df = block_on(session.sql("SELECT 1")).expect("plans");
    assert_eq!(block_on(df.collect()).expect("runs").len(), 1);
    // Let a few pushes fail, then stop: the drop does not hang.
    std::thread::sleep(Duration::from_millis(100));
    let started = Instant::now();
    drop(pusher);
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn two_pushers_are_two_series() {
    let (url, seen) = collector();
    let (a, b) = (Session::new().expect("a"), Session::new().expect("b"));
    let config = OtlpConfig::new(&url);
    let _pa =
        OtlpMetrics::start(&config, Duration::from_millis(30), move || a.metrics()).expect("a");
    let _pb =
        OtlpMetrics::start(&config, Duration::from_millis(30), move || b.metrics()).expect("b");
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let instances: std::collections::BTreeSet<String> = seen
            .lock()
            .expect("lock")
            .iter()
            .filter(|(p, _, _)| p == "/v1/metrics")
            .filter_map(|(_, _, body)| ExportMetricsServiceRequest::decode(body.as_slice()).ok())
            .flat_map(|r| r.resource_metrics)
            .filter_map(|r| r.resource)
            .flat_map(|r| r.attributes)
            .filter(|a| a.key == "service.instance.id")
            .filter_map(|a| match a.value?.value? {
                Value::StringValue(v) => Some(v),
                _ => None,
            })
            .collect();
        if instances.len() >= 2 {
            return;
        }
        assert!(Instant::now() < deadline, "saw instances {instances:?}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

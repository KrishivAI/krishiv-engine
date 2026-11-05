// SPDX-License-Identifier: Apache-2.0
//! OTLP export (feature `telemetry-otlp`, P2-T116 / P2-T125): the
//! `krishiv.query` spans and the engine metrics, sent to a collector over
//! OTLP/HTTP with protobuf bodies (`http-proto`; rustls, no gRPC stack).
//!
//! * Spans: [`OtlpTracing`] owns a batch span exporter; its
//!   [`layer`](OtlpTracing::layer) goes into the application's `tracing`
//!   subscriber, so the `krishiv.query` spans leave the process.
//! * Metrics: [`encode_metrics`] turns a [`MetricsSnapshot`] into an OTLP
//!   `ExportMetricsServiceRequest` (cumulative sums, gauges, explicit-bucket
//!   histograms) and [`OtlpMetrics`] pushes the session's snapshot on an
//!   interval from a thread of its own. The encoder is pure and handwritten
//!   over the snapshot, like the Prometheus one: the SDK cannot be handed
//!   pre-aggregated histograms.
//!
//! The module is behind the `telemetry-otlp` feature and, like `unstable-ir`, is
//! not covered by semver: it exposes OpenTelemetry 0.33 and
//! tracing-opentelemetry 0.34 types.
//!
//! Nothing here runs unless the application builds one of these; the
//! default embedded build does not compile this module.

use std::collections::HashMap;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use krishiv_common::error::{InvalidParameterValueString, KrishivError};
use krishiv_exec::{MetricsRegistry, MetricsSnapshot};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{InstrumentationScope, KeyValue};
use opentelemetry_http::HttpClient as _;
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig as _, WithHttpConfig as _};
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::common::v1::{
    AnyValue, InstrumentationScope as ProtoScope, KeyValue as ProtoKeyValue, any_value,
};
use opentelemetry_proto::tonic::metrics::v1::{
    AggregationTemporality, Gauge, Histogram, HistogramDataPoint, Metric, NumberDataPoint,
    ResourceMetrics, ScopeMetrics, Sum, metric, number_data_point,
};
use opentelemetry_proto::tonic::resource::v1::Resource as ProtoResource;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider};
use prost::Message as _;
use tracing::Subscriber;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::registry::LookupSpan;

/// Where and how to export.
///
/// This module is behind the `telemetry-otlp` feature and, like `unstable-ir`,
/// is not covered by semver: it exposes OpenTelemetry 0.33 and
/// tracing-opentelemetry 0.34 types.
#[derive(Clone)]
pub struct OtlpConfig {
    /// The collector's base URL, e.g. `http://localhost:4318`; `/v1/traces`
    /// and `/v1/metrics` are appended.
    pub endpoint: String,
    /// `service.name` of the resource.
    pub service_name: String,
    /// `worker.id` of the resource (T125 section 6.4): the process that
    /// exports. `local` for an embedded engine.
    pub worker_id: String,
    /// `cluster.id` of the resource, when the engine runs in one.
    pub cluster_id: Option<String>,
    /// Extra request headers (an API key, for instance). Each name and value
    /// must be a valid HTTP header; values are sent as written (the trace
    /// exporter also reads `OTEL_EXPORTER_OTLP_*` from the environment, the
    /// metrics pusher does not).
    pub headers: Vec<(String, String)>,
    /// Per-request timeout. Dropping an exporter waits for a request in
    /// flight, so a drop can block for up to this long.
    pub timeout: Duration,
}

impl std::fmt::Debug for OtlpConfig {
    // Neither the URL (it may carry credentials) nor header values are printed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtlpConfig")
            .field("service_name", &self.service_name)
            .field("worker_id", &self.worker_id)
            .field("cluster_id", &self.cluster_id)
            .field(
                "headers",
                &self.headers.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            )
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl OtlpConfig {
    /// A config for the collector at `endpoint`, service `krishiv`, worker
    /// `local`, a 10 s timeout.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            service_name: "krishiv".to_owned(),
            worker_id: "local".to_owned(),
            cluster_id: None,
            headers: Vec::new(),
            timeout: Duration::from_secs(10),
        }
    }

    fn url(&self, signal: &str) -> String {
        format!("{}/v1/{signal}", self.endpoint.trim_end_matches('/'))
    }

    /// Refuses a config before anything is built, with fixed messages: the
    /// endpoint and the headers may carry credentials and are never echoed.
    fn check(&self) -> Result<(), KrishivError> {
        let refuse = |parameter: &str, why: &str| {
            Err(InvalidParameterValueString {
                parameter: parameter.to_owned(),
                function_name: "OtlpConfig".to_owned(),
                invalid_value: why.to_owned(),
            }
            .into())
        };
        let well_formed = ["traces", "metrics"].iter().all(|signal| {
            let url = self.url(signal);
            (url.starts_with("http://") || url.starts_with("https://"))
                && url
                    .parse::<http::Uri>()
                    .is_ok_and(|u| u.scheme().is_some() && u.authority().is_some())
        });
        if !well_formed {
            return refuse("endpoint", "(expected a valid http:// or https:// URL)");
        }
        for (name, value) in &self.headers {
            if http::HeaderName::from_bytes(name.as_bytes()).is_err()
                || http::HeaderValue::from_str(value).is_err()
            {
                return refuse("headers", "(a header name or value is not valid HTTP)");
            }
        }
        Ok(())
    }
}

fn export_error(what: &str, why: &str) -> KrishivError {
    InvalidParameterValueString {
        parameter: what.to_owned(),
        function_name: "krishiv_session::otlp".to_owned(),
        invalid_value: why.to_owned(),
    }
    .into()
}

/// Runs `f` on a thread of its own: the blocking HTTP client may not be
/// built or dropped on a thread that drives an async runtime.
fn off_runtime<T: Send>(f: impl FnOnce() -> T + Send) -> Option<T> {
    std::thread::scope(|s| s.spawn(f).join().ok())
}

/// The span exporter and its tracer provider. Dropping it flushes and shuts
/// the exporter down.
#[derive(Debug)]
pub struct OtlpTracing {
    provider: Option<SdkTracerProvider>,
    tracer: SdkTracer,
}

impl OtlpTracing {
    /// Builds a batch span exporter to `config.endpoint`.
    ///
    /// # Errors
    /// `INVALID_PARAMETER_VALUE.STRING` for an endpoint that is not an
    /// `http(s)` URL or a header the HTTP client refuses.
    pub fn new(config: &OtlpConfig) -> Result<Self, KrishivError> {
        config.check()?;
        // The trace exporter url-decodes header values (and drops one that decodes to
        // invalid HTTP): encode `%` so a value goes out as written.
        let headers: HashMap<String, String> = config
            .headers
            .iter()
            .map(|(k, v)| (k.clone(), v.replace('%', "%25")))
            .collect();
        let built = off_runtime(|| {
            SpanExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_endpoint(config.url("traces"))
                .with_headers(headers)
                .with_timeout(config.timeout)
                .build()
        })
        .ok_or_else(|| export_error("exporter", "the exporter thread panicked"))?;
        // A fixed message: the library's text may repeat the endpoint.
        let exporter =
            built.map_err(|_| export_error("exporter", "the span exporter could not be built"))?;
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(resource(config, &crate::telemetry::new_session_id()))
            .build();
        let tracer = provider.tracer_with_scope(
            InstrumentationScope::builder("krishiv")
                .with_version(env!("CARGO_PKG_VERSION"))
                .build(),
        );
        Ok(Self {
            provider: Some(provider),
            tracer,
        })
    }

    /// The `tracing` layer that sends spans to the collector: add it to the
    /// subscriber with `.with(tracing.layer())`. Keep `self` alive as long as
    /// the subscriber: after the drop the exporter is shut down and spans are
    /// lost without notice.
    #[must_use]
    pub fn layer<S>(&self) -> OpenTelemetryLayer<S, SdkTracer>
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        tracing_opentelemetry::layer().with_tracer(self.tracer.clone())
    }

    /// Sends the spans buffered so far and waits for the answer.
    ///
    /// # Errors
    /// `INVALID_PARAMETER_VALUE.STRING` when the collector could not be reached.
    pub fn flush(&self) -> Result<(), KrishivError> {
        match &self.provider {
            Some(p) => p
                .force_flush()
                .map_err(|e| export_error("flush", &e.to_string())),
            None => Ok(()),
        }
    }
}

impl Drop for OtlpTracing {
    fn drop(&mut self) {
        if let Some(p) = self.provider.take() {
            // Off any async runtime: the blocking client drops here.
            let _ = off_runtime(move || p.shutdown());
        }
    }
}

/// The resource attributes of one exporter: `instance` tells two exporters of
/// one process (two sessions) apart, so their cumulative series do not collide.
fn resource_attributes(config: &OtlpConfig, instance: &str) -> Vec<(&'static str, String)> {
    let mut attributes = vec![
        ("service.name", config.service_name.clone()),
        ("service.version", env!("CARGO_PKG_VERSION").to_owned()),
        ("service.instance.id", instance.to_owned()),
        ("worker.id", config.worker_id.clone()),
    ];
    if let Some(cluster) = &config.cluster_id {
        attributes.push(("cluster.id", cluster.clone()));
    }
    attributes
}

fn resource(config: &OtlpConfig, instance: &str) -> Resource {
    Resource::builder_empty()
        .with_attributes(
            resource_attributes(config, instance)
                .into_iter()
                .map(|(k, v)| KeyValue::new(k, v)),
        )
        .build()
}

fn string_attr(key: &str, value: &str) -> ProtoKeyValue {
    ProtoKeyValue {
        key: key.to_owned(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(value.to_owned())),
        }),
        ..ProtoKeyValue::default()
    }
}

fn nanos(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

fn labels(pairs: &[(String, String)]) -> Vec<ProtoKeyValue> {
    pairs.iter().map(|(k, v)| string_attr(k, v)).collect()
}

fn number(attributes: Vec<ProtoKeyValue>, value: f64, start: u64, now: u64) -> NumberDataPoint {
    NumberDataPoint {
        attributes,
        start_time_unix_nano: start,
        time_unix_nano: now,
        value: Some(number_data_point::Value::AsDouble(value)),
        ..NumberDataPoint::default()
    }
}

/// `snapshot` as one OTLP metrics request. Counters are cumulative monotonic
/// sums, gauges keep their `worker_id` as an attribute, histograms carry the
/// catalogue's explicit bounds with per-bucket (not cumulative) counts, as
/// OTLP wants them. Series whose name is not in the catalogue are skipped,
/// as the Prometheus encoder skips them. `now` stamps every point; `start`
/// is when the exporter began (the start of its cumulative series) and
/// `instance` its `service.instance.id`, so that two exporters of one process
/// are two series. The resource carries `service.*`, `worker.id` and, when
/// configured, `cluster.id`.
#[must_use]
pub fn encode_metrics(
    snapshot: &MetricsSnapshot,
    config: &OtlpConfig,
    instance: &str,
    start: SystemTime,
    now: SystemTime,
) -> ExportMetricsServiceRequest {
    let (start, now) = (nanos(start), nanos(now));
    let catalogue = MetricsRegistry::catalogue();
    let describe = |name: &str| catalogue.iter().find(|d| d.name == name);
    let mut metrics = Vec::new();

    for group in snapshot.counters.chunk_by(|a, b| a.name == b.name) {
        let Some(def) = describe(group[0].name) else {
            continue;
        };
        metrics.push(Metric {
            name: def.name.to_owned(),
            description: def.description.to_owned(),
            unit: def.unit.to_owned(),
            data: Some(metric::Data::Sum(Sum {
                data_points: group
                    .iter()
                    .map(|s| number(labels(&s.labels), s.value, start, now))
                    .collect(),
                aggregation_temporality: AggregationTemporality::Cumulative as i32,
                is_monotonic: true,
            })),
            ..Metric::default()
        });
    }
    for group in snapshot.gauges.chunk_by(|a, b| a.name == b.name) {
        let Some(def) = describe(group[0].name) else {
            continue;
        };
        metrics.push(Metric {
            name: def.name.to_owned(),
            description: def.description.to_owned(),
            unit: def.unit.to_owned(),
            data: Some(metric::Data::Gauge(Gauge {
                data_points: group
                    .iter()
                    .map(|s| {
                        let mut attributes = labels(&s.labels);
                        attributes.push(string_attr("worker_id", &s.worker_id));
                        number(attributes, s.value, 0, now)
                    })
                    .collect(),
            })),
            ..Metric::default()
        });
    }
    for group in snapshot.histograms.chunk_by(|a, b| a.name == b.name) {
        let Some(def) = describe(group[0].name) else {
            continue;
        };
        metrics.push(Metric {
            name: def.name.to_owned(),
            description: def.description.to_owned(),
            unit: def.unit.to_owned(),
            data: Some(metric::Data::Histogram(Histogram {
                data_points: group
                    .iter()
                    .map(|s| {
                        let mut previous = 0;
                        let mut bucket_counts: Vec<u64> = s
                            .buckets
                            .iter()
                            .map(|b| {
                                let n = b.count.saturating_sub(previous);
                                previous = b.count;
                                n
                            })
                            .collect();
                        bucket_counts.push(s.count.saturating_sub(previous));
                        // Equal by construction, even when the registry was read
                        // while a query folded: collectors reject a point whose
                        // buckets do not add up to its count.
                        let count = bucket_counts.iter().sum();
                        HistogramDataPoint {
                            attributes: labels(&s.labels),
                            start_time_unix_nano: start,
                            time_unix_nano: now,
                            count,
                            sum: Some(s.sum),
                            bucket_counts,
                            explicit_bounds: s.buckets.iter().map(|b| b.le).collect(),
                            ..HistogramDataPoint::default()
                        }
                    })
                    .collect(),
                aggregation_temporality: AggregationTemporality::Cumulative as i32,
            })),
            ..Metric::default()
        });
    }

    ExportMetricsServiceRequest {
        resource_metrics: vec![ResourceMetrics {
            resource: Some(ProtoResource {
                attributes: resource_attributes(config, instance)
                    .iter()
                    .map(|(k, v)| string_attr(k, v))
                    .collect(),
                ..ProtoResource::default()
            }),
            scope_metrics: vec![ScopeMetrics {
                scope: Some(ProtoScope {
                    name: "krishiv".to_owned(),
                    version: env!("CARGO_PKG_VERSION").to_owned(),
                    ..ProtoScope::default()
                }),
                metrics,
                ..ScopeMetrics::default()
            }],
            ..ResourceMetrics::default()
        }],
    }
}

/// Sends one metrics request now. Errors carry fixed text: the HTTP
/// library's own message repeats the URL, which may hold credentials.
fn push(
    client: &reqwest::blocking::Client,
    config: &OtlpConfig,
    instance: &str,
    start: SystemTime,
    snapshot: &MetricsSnapshot,
) -> Result<(), KrishivError> {
    let body = encode_metrics(snapshot, config, instance, start, SystemTime::now()).encode_to_vec();
    let mut request =
        http::Request::post(config.url("metrics")).header("content-type", "application/x-protobuf");
    for (k, v) in &config.headers {
        let mut value = http::HeaderValue::from_str(v)
            .map_err(|_| export_error("request", "a header value is not valid"))?;
        value.set_sensitive(true);
        request = request.header(k, value);
    }
    let request = request
        .body(body.into())
        .map_err(|_| export_error("request", "the request could not be built"))?;
    let response = futures::executor::block_on(client.send_bytes(request)).map_err(|e| {
        let timed_out = e
            .downcast_ref::<reqwest::Error>()
            .is_some_and(reqwest::Error::is_timeout);
        export_error(
            "push",
            if timed_out {
                "the collector did not answer in time"
            } else {
                "the collector could not be reached"
            },
        )
    })?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(export_error(
            "push",
            &format!("the collector answered {}", response.status()),
        ))
    }
}

/// Pushes the engine metrics to the collector every `interval`, and once more
/// when dropped, from a thread of its own. Dropping it waits for that thread,
/// so a drop can block for up to the config's `timeout` when the collector
/// does not answer.
#[derive(Debug)]
pub struct OtlpMetrics {
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl OtlpMetrics {
    /// Starts the pusher. `snapshot` is called once per push (for a session,
    /// `move || session.metrics()`); it must be cheap. The series start now:
    /// a pusher started later than the counters is a reset to a backend.
    ///
    /// # Errors
    /// `INVALID_PARAMETER_VALUE.STRING` for an endpoint that is not an
    /// `http(s)` URL, a header that is not valid HTTP, or when the thread or
    /// the HTTP client cannot start.
    pub fn start(
        config: &OtlpConfig,
        interval: Duration,
        snapshot: impl Fn() -> MetricsSnapshot + Send + 'static,
    ) -> Result<Self, KrishivError> {
        config.check()?;
        if interval.is_zero() {
            return Err(export_error("interval", "(must be longer than zero)"));
        }
        let config = config.clone();
        let (instance, start) = (crate::telemetry::new_session_id(), SystemTime::now());
        let (stop, rx) = mpsc::channel::<()>();
        let (ready_tx, ready) = mpsc::channel::<bool>();
        let thread = std::thread::Builder::new()
            .name("krishiv-otlp-metrics".to_owned())
            .spawn(move || {
                // Built here, never on the caller's thread: the blocking
                // client may not be built or dropped on an async runtime.
                let client = reqwest::blocking::Client::builder()
                    .timeout(config.timeout)
                    .build();
                let _ = ready_tx.send(client.is_ok());
                let Ok(client) = client else { return };
                loop {
                    let stopped = matches!(
                        rx.recv_timeout(interval),
                        Ok(()) | Err(RecvTimeoutError::Disconnected)
                    );
                    // A failed push is retried at the next tick: telemetry
                    // never fails a query.
                    if let Err(e) = push(&client, &config, &instance, start, &snapshot()) {
                        tracing::debug!(error = %e, "OTLP metrics push failed");
                    }
                    if stopped {
                        break;
                    }
                }
            })
            .map_err(|_| export_error("thread", "the pusher thread could not start"))?;
        if !ready.recv().unwrap_or(false) {
            let _ = thread.join();
            return Err(export_error("client", "the HTTP client could not be built"));
        }
        Ok(Self {
            stop: Some(stop),
            thread: Some(thread),
        })
    }
}

impl Drop for OtlpMetrics {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(t) = self.thread.take()
            && t.join().is_err()
        {
            tracing::warn!("the OTLP metrics pusher stopped early: its snapshot function panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_exec::{CounterSample, GaugeSample, HistogramBucket, HistogramSample};

    fn snapshot() -> MetricsSnapshot {
        MetricsSnapshot {
            counters: vec![CounterSample {
                name: "krishiv.query.count",
                unit: "{query}",
                labels: vec![("status".into(), "ok".into())],
                value: 3.0,
            }],
            gauges: vec![GaugeSample {
                name: "krishiv.build.info",
                unit: "1",
                labels: Vec::new(),
                worker_id: "w1".into(),
                value: 1.0,
            }],
            histograms: vec![HistogramSample {
                name: "krishiv.query.duration",
                unit: "s",
                labels: Vec::new(),
                buckets: vec![
                    HistogramBucket { le: 0.1, count: 1 },
                    HistogramBucket { le: 1.0, count: 4 },
                ],
                count: 5,
                sum: 2.5,
            }],
        }
    }

    #[test]
    fn a_snapshot_encodes_as_sums_gauges_and_histograms() {
        let mut config = OtlpConfig::new("http://localhost:4318");
        config.cluster_id = Some("c1".into());
        let req = encode_metrics(&snapshot(), &config, "inst-1", UNIX_EPOCH, UNIX_EPOCH);
        let scope = &req.resource_metrics[0].scope_metrics[0];
        let names: Vec<_> = scope.metrics.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "krishiv.query.count",
                "krishiv.build.info",
                "krishiv.query.duration"
            ]
        );
        let Some(metric::Data::Histogram(h)) = &scope.metrics[2].data else {
            panic!("a histogram")
        };
        let p = &h.data_points[0];
        // Cumulative 1, 4, 5 become per-bucket 1, 3, 1 (the last is the overflow bucket).
        assert_eq!(p.bucket_counts, [1, 3, 1]);
        assert_eq!(p.explicit_bounds, [0.1, 1.0]);
        assert_eq!((p.count, p.sum), (5, Some(2.5)));
        let Some(metric::Data::Gauge(g)) = &scope.metrics[1].data else {
            panic!("a gauge")
        };
        assert!(
            g.data_points[0]
                .attributes
                .iter()
                .any(|a| a.key == "worker_id")
        );
    }

    #[test]
    fn the_resource_names_the_service_the_instance_and_the_worker() {
        let mut config = OtlpConfig::new("http://localhost:4318");
        config.worker_id = "w7".into();
        config.cluster_id = Some("c1".into());
        let req = encode_metrics(&snapshot(), &config, "inst-1", UNIX_EPOCH, UNIX_EPOCH);
        let attrs: Vec<(String, String)> = req.resource_metrics[0]
            .resource
            .as_ref()
            .map(|r| {
                r.attributes
                    .iter()
                    .filter_map(|a| match a.value.as_ref()?.value.as_ref()? {
                        any_value::Value::StringValue(v) => Some((a.key.clone(), v.clone())),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (k, v) in [
            ("service.name", "krishiv"),
            ("service.instance.id", "inst-1"),
            ("worker.id", "w7"),
            ("cluster.id", "c1"),
        ] {
            assert!(
                attrs.iter().any(|(a, b)| a == k && b == v),
                "{k} in {attrs:?}"
            );
        }
    }

    #[test]
    fn the_series_start_is_the_one_given_not_a_process_global() {
        let config = OtlpConfig::new("http://localhost:4318");
        let (a, b) = (
            UNIX_EPOCH + Duration::from_secs(5),
            UNIX_EPOCH + Duration::from_secs(9),
        );
        let now = UNIX_EPOCH + Duration::from_secs(20);
        for start in [a, b] {
            let req = encode_metrics(&snapshot(), &config, "i", start, now);
            let Some(metric::Data::Sum(sum)) =
                &req.resource_metrics[0].scope_metrics[0].metrics[0].data
            else {
                panic!("a sum")
            };
            assert_eq!(sum.data_points[0].start_time_unix_nano, nanos(start));
            assert!(sum.data_points[0].start_time_unix_nano <= sum.data_points[0].time_unix_nano);
        }
    }

    #[test]
    fn a_config_is_refused_with_text_that_never_holds_the_url_or_a_header_value() {
        for endpoint in [
            "collector:4318?key=secret",
            "http://host:4318/?key=secret value",
            "https://",
        ] {
            let e = OtlpTracing::new(&OtlpConfig::new(endpoint)).unwrap_err();
            assert!(!e.to_string().contains("secret"), "{endpoint}: {e}");
        }
        let mut config = OtlpConfig::new("http://localhost:4318");
        config.headers.push(("x-key".into(), "se\ncret".into()));
        let e = OtlpMetrics::start(&config, Duration::from_secs(1), MetricsSnapshot::default)
            .unwrap_err();
        assert!(!e.to_string().contains("cret"), "{e}");
    }

    #[test]
    fn debug_shows_header_names_only() {
        let mut config = OtlpConfig::new("http://user:pw@localhost:4318");
        config.headers.push(("x-api-key".into(), "hunter2".into()));
        let shown = format!("{config:?}");
        assert!(shown.contains("x-api-key"));
        assert!(
            !shown.contains("hunter2") && !shown.contains("pw@"),
            "{shown}"
        );
    }
}

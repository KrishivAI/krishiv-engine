---
description: "Turn on JSON logs and OTLP trace export from a Rust application, and scrape Krishiv's metrics."
---

# Observability from Rust

Krishiv's JSON logs and OTLP export are library APIs in the `krishiv-session` crate. Nothing starts them for you: an
application installs them into its own [`tracing`](https://docs.rs/tracing) subscriber. The `krishiv` command line
installs the JSON logs when `KRISHIV_LOG=json` is set.

## JSON logs

Enable the `telemetry-logs` feature and call `install_json_logs`:

```toml
[dependencies]
krishiv-session = { version = "0.3", features = ["telemetry-logs"] }
```

```rust,ignore
// Writes one JSON object per event to standard error, at INFO and above,
// at most 20 events per call site per second.
krishiv_session::logging::install_json_logs(20)?;
```

Each line has `timestamp`, `level`, `fields`, `target` and the span list, with the query and session ids on the
`krishiv.query` span. Statement text is sanitized, and the planner's warnings (which can quote literals) are never
written. To send the logs elsewhere or combine them with other layers, use `logging::json_layer(per_second, writer)`
on your own `tracing_subscriber::registry()`. `install_json_logs` returns an error if a global subscriber is already set.

## OTLP traces

Enable `telemetry-otlp`. Spans go to an OTLP/HTTP collector (protobuf bodies, rustls); the endpoint is the
collector's base URL and `/v1/traces` is appended.

```toml
krishiv-session = { version = "0.3", features = ["telemetry-otlp"] }
tracing-subscriber = "0.3"
```

```rust,ignore
use krishiv_session::otlp::{OtlpConfig, OtlpTracing};
use tracing_subscriber::prelude::*;

let mut config = OtlpConfig::new("http://localhost:4318");
config.service_name = "my-app".into();
let tracing = OtlpTracing::new(&config)?;
tracing_subscriber::registry().with(tracing.layer()).init();

// ... run queries: each one is a `krishiv.query` span ...

tracing.flush()?; // before the process exits
```

`OtlpConfig` also takes `headers` (for an API key) and a timeout. `otlp::OtlpMetrics` pushes the session's metrics
snapshot to the collector on an interval. The module exposes OpenTelemetry types, so it is not covered by semver.

## Metrics

Metric names, labels and units are in [metrics v1](../reference/metrics-v1.md). Query-level numbers are in
[EXPLAIN ANALYZE](../reference/explain-analyze.md).

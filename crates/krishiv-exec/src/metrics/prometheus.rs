// SPDX-License-Identifier: Apache-2.0
//! Prometheus text exposition 0.0.4 over a [`MetricsSnapshot`].
//!
//! The rendering is mechanical, exactly the one the catalogue promises:
//! dots to underscores, `_seconds` or `_bytes` for those units, `_total`
//! for counters; histograms expose the base name with the usual `_bucket`
//! (with an `le` label, plus `+Inf`), `_count` and `_sum` suffixes. Every
//! metric that has a series carries its `# HELP` (the catalogue
//! description) and `# TYPE` lines. The encoder is handwritten over the
//! snapshot: no encoder crate, no global recorder.
//!
//! [`render`] is pure: it only reads the snapshot, so scraping never
//! changes a counter. Gauges keep their `worker_id` as a label (the text
//! format has no resource attributes): one sample per worker stays one
//! series instead of collapsing into duplicates. Series whose dotted name
//! is not in the catalogue are skipped: a registry-built snapshot never
//! holds one.

use super::registry::{CATALOGUE, MetricDef, MetricsSnapshot};

/// Renders `snapshot` as Prometheus text exposition format 0.0.4: counters
/// (with `_total`), gauges (with a `worker_id` label) and histograms (with
/// cumulative `_bucket` lines, `+Inf`, `_count` and `_sum`), each group
/// preceded by its `# HELP` and `# TYPE` lines. Catalogue entries without
/// a series emit nothing.
#[must_use]
pub fn render(snapshot: &MetricsSnapshot) -> String {
    let mut out = String::new();
    for group in snapshot.counters.chunk_by(|a, b| a.name == b.name) {
        let Some(def) = catalogue(group[0].name) else {
            continue;
        };
        push_help(&mut out, def);
        for sample in group {
            out.push_str(&def.prometheus_name());
            push_labels(&mut out, &sample.labels, None);
            out.push(' ');
            push_value(&mut out, sample.value);
            out.push('\n');
        }
    }
    for group in snapshot.gauges.chunk_by(|a, b| a.name == b.name) {
        let Some(def) = catalogue(group[0].name) else {
            continue;
        };
        push_help(&mut out, def);
        for sample in group {
            out.push_str(&def.prometheus_name());
            push_labels(&mut out, &sample.labels, Some(sample.worker_id.as_str()));
            out.push(' ');
            push_value(&mut out, sample.value);
            out.push('\n');
        }
    }
    for group in snapshot.histograms.chunk_by(|a, b| a.name == b.name) {
        let Some(def) = catalogue(group[0].name) else {
            continue;
        };
        let base = def.prometheus_name();
        push_help(&mut out, def);
        for sample in group {
            for bucket in &sample.buckets {
                out.push_str(&base);
                out.push_str("_bucket");
                push_histogram_labels(&mut out, &sample.labels, &format_le(bucket.le));
                out.push(' ');
                push_count(&mut out, bucket.count);
                out.push('\n');
            }
            out.push_str(&base);
            out.push_str("_bucket");
            push_histogram_labels(&mut out, &sample.labels, "+Inf");
            out.push(' ');
            push_count(&mut out, sample.count);
            out.push('\n');
            out.push_str(&base);
            out.push_str("_count");
            push_labels(&mut out, &sample.labels, None);
            out.push(' ');
            push_count(&mut out, sample.count);
            out.push('\n');
            out.push_str(&base);
            out.push_str("_sum");
            push_labels(&mut out, &sample.labels, None);
            out.push(' ');
            push_value(&mut out, sample.sum);
            out.push('\n');
        }
    }
    out
}

/// The catalogue entry for `name`, if any.
fn catalogue(name: &str) -> Option<&'static MetricDef> {
    CATALOGUE.iter().find(|def| def.name == name)
}

/// Pushes the `# HELP` and `# TYPE` lines of `def`.
fn push_help(out: &mut String, def: &MetricDef) {
    out.push_str("# HELP ");
    out.push_str(&def.prometheus_name());
    out.push(' ');
    push_escaped_help(out, def.description);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(&def.prometheus_name());
    out.push(' ');
    out.push_str(def.kind.as_str());
    out.push('\n');
}

/// Pushes `{k="v",...}` for `labels` (plus `worker_id` when given); nothing
/// when both are absent.
fn push_labels(out: &mut String, labels: &[(String, String)], worker_id: Option<&str>) {
    if labels.is_empty() && worker_id.is_none() {
        return;
    }
    out.push('{');
    let mut first = true;
    for (name, value) in labels {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(name);
        out.push_str("=\"");
        push_escaped_label(out, value);
        out.push('"');
    }
    if let Some(worker_id) = worker_id {
        if !first {
            out.push(',');
        }
        out.push_str("worker_id=\"");
        push_escaped_label(out, worker_id);
        out.push('"');
    }
    out.push('}');
}

/// Pushes `{k="v",...,le="bound"}` for a histogram bucket line.
fn push_histogram_labels(out: &mut String, labels: &[(String, String)], le: &str) {
    out.push('{');
    let mut first = true;
    for (name, value) in labels {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(name);
        out.push_str("=\"");
        push_escaped_label(out, value);
        out.push('"');
    }
    if !first {
        out.push(',');
    }
    out.push_str("le=\"");
    out.push_str(le);
    out.push_str("\"}");
}

/// Pushes a sample value: finite floats in full precision, so a parsed
/// value compares exactly.
fn push_value(out: &mut String, value: f64) {
    out.push_str(&format!("{value}"));
}

/// Pushes an exact count.
fn push_count(out: &mut String, count: u64) {
    out.push_str(&format!("{count}"));
}

/// Formats a bucket bound: `0.001`, `10`, `300` (never an exponent: the
/// bounds are small decimals).
fn format_le(le: f64) -> String {
    format!("{le}")
}

/// Pushes a label value with the exposition escaping: backslash, quote and
/// newline.
fn push_escaped_label(out: &mut String, value: &str) {
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
}

/// Pushes help text with the exposition escaping: backslash and newline.
fn push_escaped_help(out: &mut String, text: &str) {
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::collections::BTreeMap;

    use crate::executor::WorkloadClass;
    use crate::metrics::QueryMetrics;

    use super::super::registry::{
        BUCKETS, CounterSample, GaugeSample, HistogramBucket, HistogramSample, MetricKind,
        MetricSource, MetricsRegistry, QueryOutcome, Status,
    };
    use super::*;

    /// A folded registry holding one succeeded query plus every scrape
    /// gauge, so the snapshot carries each catalogue metric that has a
    /// source.
    fn full_registry() -> crate::error::Result<MetricsRegistry> {
        let registry = MetricsRegistry::new();
        let metrics = QueryMetrics::default();
        registry.query_begin(WorkloadClass::Interactive);
        registry.fold(&QueryOutcome {
            class: WorkloadClass::Interactive,
            status: Status::Succeeded,
            error_class: None,
            duration_secs: 0.25,
            metrics: &metrics,
            arbitration_spills: 2,
            arbitration_aborts: 1,
            cache_hits: 3,
            cache_misses: 1,
        });
        for class in ["interactive", "batch", "background", "agent"] {
            registry.add_gauge(
                "krishiv.query.active",
                vec![("class".to_owned(), class.to_owned())],
                || Some(0),
            )?;
        }
        registry.add_gauge("krishiv.memory.reserved", Vec::new(), || Some(8))?;
        registry.add_gauge("krishiv.memory.limit", Vec::new(), || Some(16))?;
        registry.add_gauge("krishiv.spill.disk.used", Vec::new(), || Some(0))?;
        registry.add_gauge(
            "krishiv.cache.size",
            vec![("cache".to_owned(), "metadata".to_owned())],
            || Some(4),
        )?;
        registry.add_gauge(
            "krishiv.build.info",
            vec![
                ("version".to_owned(), "0.1.0".to_owned()),
                ("git_sha".to_owned(), "abc".to_owned()),
            ],
            || Some(1),
        )?;
        Ok(registry)
    }

    /// One parsed sample line.
    #[derive(Debug, PartialEq)]
    struct Sample {
        name: String,
        labels: Vec<(String, String)>,
        value: String,
    }

    /// One parsed metric: its HELP/TYPE lines and samples.
    #[derive(Debug, Default)]
    struct Metric {
        help: Option<String>,
        type_line: Option<String>,
        samples: Vec<Sample>,
    }

    /// Parses what [`render`] emits, strictly: anything else is an error.
    /// No timestamps, no `# EOF`, no bare `#` comments.
    fn parse(text: &str) -> Result<BTreeMap<String, Metric>, String> {
        let mut metrics: BTreeMap<String, Metric> = BTreeMap::new();
        for (index, line) in text.lines().enumerate() {
            let line_no = index + 1;
            if let Some(rest) = line.strip_prefix("# HELP ") {
                let (name, help) = rest
                    .split_once(' ')
                    .ok_or_else(|| format!("line {line_no}: HELP without text"))?;
                check_name(name, line_no)?;
                metrics.entry(base_of(name)).or_default().help = Some(help.to_owned());
            } else if let Some(rest) = line.strip_prefix("# TYPE ") {
                let (name, kind) = rest
                    .split_once(' ')
                    .ok_or_else(|| format!("line {line_no}: TYPE without kind"))?;
                check_name(name, line_no)?;
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
        for (name, metric) in &metrics {
            if metric.help.is_none() {
                return Err(format!("{name}: missing HELP"));
            }
            if metric.type_line.is_none() {
                return Err(format!("{name}: missing TYPE"));
            }
        }
        Ok(metrics)
    }

    /// The metric a sample belongs to: histogram `_bucket`/`_count`/`_sum`
    /// fold onto the base name.
    fn base_of(name: &str) -> String {
        for suffix in ["_bucket", "_count", "_sum"] {
            if let Some(base) = name.strip_suffix(suffix) {
                return base.to_owned();
            }
        }
        name.to_owned()
    }

    fn check_name(name: &str, line_no: usize) -> Result<(), String> {
        let mut chars = name.chars();
        let first = chars
            .next()
            .ok_or_else(|| format!("line {line_no}: empty name"))?;
        if !matches!(first, 'a'..='z' | 'A'..='Z' | '_' | ':') {
            return Err(format!("line {line_no}: bad name {name:?}"));
        }
        if !chars.all(|c| matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | ':')) {
            return Err(format!("line {line_no}: bad name {name:?}"));
        }
        Ok(())
    }

    fn parse_sample(line: &str, line_no: usize) -> Result<Sample, String> {
        let bad = |why: &str| format!("line {line_no}: {why} in {line:?}");
        let (name, rest) = match line.find(['{', ' ']) {
            Some(at) => line.split_at(at),
            None => return Err(bad("no value")),
        };
        check_name(name, line_no)?;
        let (labels, value) = if let Some(braced) = rest.strip_prefix('{') {
            // The closing brace is the first `}` outside a quoted value:
            // values may hold `,`/`}`/`=` themselves.
            let end = labels_end(braced).ok_or_else(|| bad("unclosed labels"))?;
            let labels = parse_labels(&braced[..end], line_no)?;
            let value = braced[end + 1..].trim_start_matches(' ');
            if value.is_empty() {
                return Err(bad("no value"));
            }
            (labels, value)
        } else {
            let value = rest.trim_start_matches(' ');
            if value.is_empty() {
                return Err(bad("no value"));
            }
            (Vec::new(), value)
        };
        if value.split_whitespace().count() != 1 {
            return Err(bad("trailing text after value"));
        }
        value
            .parse::<f64>()
            .map_err(|_| bad("value is not a float"))?;
        Ok(Sample {
            name: name.to_owned(),
            labels,
            value: value.to_owned(),
        })
    }

    /// The index just past the `}` closing a label set: the first `}`
    /// outside a quoted value (quotes honor backslash escapes).
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
            check_name(name, line_no)?;
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

    fn labels_of(sample: &Sample, name: &str) -> Option<String> {
        sample
            .labels
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
    }

    #[test]
    fn empty_snapshot_renders_nothing() {
        assert_eq!(render(&MetricsSnapshot::default()), "");
    }

    #[test]
    fn output_round_trips_and_covers_every_sourced_metric() {
        let text = render(&full_registry().unwrap().snapshot());
        let parsed = parse(&text).unwrap();
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
                    "{} ({}) missing from:\n{text}",
                    def.name,
                    def.prometheus_name()
                )
            });
            assert_eq!(metric.type_line.as_deref(), Some(def.kind.as_str()));
            assert!(
                metric.help.as_ref().is_some_and(|h| !h.is_empty()),
                "{} has an empty HELP",
                def.name
            );
            assert!(!metric.samples.is_empty(), "{} has no samples", def.name);
            if def.kind == MetricKind::Counter {
                assert!(
                    def.prometheus_name().ends_with("_total"),
                    "{} renders without _total",
                    def.name
                );
                for sample in &metric.samples {
                    assert_eq!(sample.name, def.prometheus_name());
                }
            }
        }
        // Histograms expose cumulative buckets, +Inf, count and sum.
        let duration = &parsed["krishiv_query_duration_seconds"];
        let bucketed: Vec<&Sample> = duration
            .samples
            .iter()
            .filter(|s| s.name == "krishiv_query_duration_seconds_bucket")
            .collect();
        assert_eq!(bucketed.len(), BUCKETS.len() + 1);
        let mut previous = 0.0f64;
        for (sample, bound) in bucketed.iter().zip(
            BUCKETS
                .iter()
                .map(|b| format!("{b}"))
                .chain(std::iter::once("+Inf".to_owned())),
        ) {
            assert_eq!(labels_of(sample, "le").as_deref(), Some(bound.as_str()));
            let count: f64 = sample.value.parse().unwrap();
            assert!(count >= previous, "buckets are not cumulative");
            previous = count;
        }
        let inf = bucketed.last().unwrap();
        let count_line = duration
            .samples
            .iter()
            .find(|s| s.name == "krishiv_query_duration_seconds_count")
            .unwrap();
        assert_eq!(inf.value, count_line.value);
        assert!(duration.samples.iter().any(|s| {
            s.name == "krishiv_query_duration_seconds_sum"
                && s.value.parse::<f64>().unwrap() == 0.25
        }));
        // Gauges keep one series per worker.
        let build = &parsed["krishiv_build_info"];
        assert_eq!(build.type_line.as_deref(), Some("gauge"));
        assert_eq!(build.samples.len(), 1);
        assert!(labels_of(&build.samples[0], "worker_id").is_some());
    }

    #[test]
    fn label_values_escape_and_round_trip() {
        // The value holds every separator the text format gives meaning:
        // quotes, backslashes, newlines, commas, braces and `=`.
        let nasty = "WEIRD\"CLASS\\NAME\nHERE,A=B}C{D";
        let registry = MetricsRegistry::new();
        let metrics = QueryMetrics::default();
        registry.fold(&QueryOutcome {
            class: WorkloadClass::Batch,
            status: Status::Failed,
            error_class: Some(nasty),
            duration_secs: 0.0,
            metrics: &metrics,
            arbitration_spills: 0,
            arbitration_aborts: 0,
            cache_hits: 0,
            cache_misses: 0,
        });
        registry
            .add_gauge(
                "krishiv.build.info",
                vec![
                    ("version".to_owned(), "1.0\"\\\n".to_owned()),
                    ("git_sha".to_owned(), "abc".to_owned()),
                ],
                || Some(1),
            )
            .unwrap();
        let text = render(&registry.snapshot());
        assert!(
            text.contains("WEIRD\\\"CLASS\\\\NAME\\nHERE,A=B}C{D"),
            "{text}"
        );
        let parsed = parse(&text).unwrap();
        let count = &parsed["krishiv_query_count_total"];
        let failed = count
            .samples
            .iter()
            .find(|s| labels_of(s, "status").as_deref() == Some("failed"))
            .unwrap();
        assert_eq!(labels_of(failed, "error_class").as_deref(), Some(nasty));
        let build = &parsed["krishiv_build_info"];
        assert_eq!(
            labels_of(&build.samples[0], "version").as_deref(),
            Some("1.0\"\\\n")
        );
    }

    #[test]
    fn rendering_skips_names_outside_the_catalogue() {
        let mut snapshot = full_registry().unwrap().snapshot();
        snapshot.counters.push(CounterSample {
            name: "krishiv.nope",
            unit: "1",
            labels: Vec::new(),
            value: 1.0,
        });
        snapshot.gauges.push(GaugeSample {
            name: "krishiv.nope",
            unit: "1",
            labels: Vec::new(),
            worker_id: "7".to_owned(),
            value: 1.0,
        });
        snapshot.histograms.push(HistogramSample {
            name: "krishiv.nope",
            unit: "s",
            labels: Vec::new(),
            buckets: BUCKETS
                .iter()
                .map(|le| HistogramBucket { le: *le, count: 0 })
                .collect(),
            count: 0,
            sum: 0.0,
        });
        let text = render(&snapshot);
        assert!(!text.contains("krishiv_nope"), "{text}");
        parse(&text).unwrap();
    }

    #[test]
    fn rendering_is_deterministic_and_read_only() {
        let registry = full_registry().unwrap();
        let first = render(&registry.snapshot());
        let second = render(&registry.snapshot());
        assert_eq!(first, second);
        assert_eq!(registry.fold_count(), 1);
        let reparsed = parse(&second).unwrap();
        assert_eq!(
            reparsed["krishiv_query_scan_rows_total"].samples[0].value,
            "0"
        );
    }
}

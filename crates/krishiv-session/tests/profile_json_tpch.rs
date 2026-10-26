// SPDX-License-Identifier: Apache-2.0
//! P2-T093b: every TPC-H query's `EXPLAIN ANALYZE` JSON profile validates
//! against the checked-in schema, round-trips through `from_json`, and its
//! per-operator rows/time/memory totals equal what the text renderer prints
//! (parsed numbers are compared, with room for the display's rounding).
#![cfg(test)]

mod common;

use std::collections::BTreeMap;

use futures::executor::block_on;
use krishiv_exec::{ExecutorConfig, TokioExecutor};
use krishiv_session::{QueryProfile, Session, TextOptions};
use serde_json::Value;

use common::query;

/// Orders in the data set: 160 lineitems, like the other session TPC-H tests.
const ORDERS: usize = 40;

/// One compute thread, so batch boundaries do not depend on scheduling.
/// `view_strings` selects `Utf8View` string columns (the engine's literal
/// type); without it the tables hold `Utf8`, on which string equality does
/// not run.
fn session(view_strings: bool) -> Session {
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: 1,
        io_threads: 1,
        blocking_threads: 1,
        io_blocking_threads: 2,
    })
    .unwrap();
    let s = Session::builder().executor(executor).build().unwrap();
    if view_strings {
        // `Utf8View` strings (the engine's literal type): string equality
        // does not run on the `Utf8` tables `register` makes.
        common::register_with(&s, ORDERS, true);
    } else {
        common::register(&s, ORDERS);
    }
    s
}

/// The checked-in JSON Schema for version 1 profile documents.
fn schema_doc() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/reference/query-profile-v1.schema.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// A minimal JSON Schema validator covering the validation keywords the
/// profile schema file uses: `type` (object, array, string, boolean,
/// integer, null), `const`, `enum`, `required`, `properties`,
/// `additionalProperties: false`, `items` (one schema for every element),
/// `oneOf`, `minimum`, `maximum`, `minItems` and `$ref` (a local `#/...`
/// pointer, used for `$defs/u64` and `$defs/u32`). The rest of the file is
/// documentation (`title`, `description`) or lookup tables (`$defs`), which
/// need no validation. Keep this list in sync with the schema file: a new
/// validation keyword there would be ignored here, not rejected.
fn validate(schema: &Value, doc: &Value, root: &Value, path: &str) -> Result<(), String> {
    let bad = |m: &str| format!("{path}: {m}");
    // A `$ref` names the schema; its siblings here are only `description`.
    let schema = match schema.get("$ref") {
        Some(reference) => {
            let reference = reference
                .as_str()
                .ok_or_else(|| bad("$ref is not a string"))?;
            let target = resolve(root, reference)
                .ok_or_else(|| bad(&format!("cannot resolve {reference}")))?;
            validate(target, doc, root, path)?;
            schema
        }
        None => schema,
    };
    if let Some(want) = schema.get("type").and_then(Value::as_str) {
        let ok = match want {
            "object" => doc.is_object(),
            "array" => doc.is_array(),
            "string" => doc.is_string(),
            "boolean" => doc.is_boolean(),
            "integer" => doc.is_u64() || doc.is_i64(),
            "null" => doc.is_null(),
            other => return Err(bad(&format!("unsupported type {other}"))),
        };
        if !ok {
            return Err(bad(&format!("is not {want}: {doc}")));
        }
    }
    if let Some(want) = schema.get("const")
        && want != doc
    {
        return Err(bad(&format!("is not {want}: {doc}")));
    }
    if let Some(one_of) = schema.get("enum").and_then(Value::as_array)
        && !one_of.iter().any(|v| v == doc)
    {
        return Err(bad(&format!("is not one of {one_of:?}: {doc}")));
    }
    if let Some(o) = doc.as_object() {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for key in required {
                let key = key
                    .as_str()
                    .ok_or_else(|| bad("required is not a string"))?;
                if !o.contains_key(key) {
                    return Err(bad(&format!("missing member {key:?}")));
                }
            }
        }
        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for (key, sub) in properties {
                if let Some(v) = o.get(key) {
                    validate(sub, v, root, &format!("{path}/{key}"))?;
                }
            }
            if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
                for key in o.keys() {
                    if !properties.contains_key(key) {
                        return Err(bad(&format!("unexpected member {key:?}")));
                    }
                }
            }
        }
    }
    if let Some(a) = doc.as_array() {
        if let Some(items) = schema.get("items") {
            for (i, v) in a.iter().enumerate() {
                validate(items, v, root, &format!("{path}[{i}]"))?;
            }
        }
        let len = u64::try_from(a.len()).map_err(|_| bad("the array is too long"))?;
        if let Some(min) = schema.get("minItems").and_then(Value::as_u64)
            && len < min
        {
            return Err(bad(&format!("has fewer than {min} items")));
        }
    }
    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        let mut matched = 0;
        let mut reasons = Vec::new();
        for sub in one_of {
            match validate(sub, doc, root, path) {
                Ok(()) => matched += 1,
                Err(e) => reasons.push(e),
            }
        }
        if matched != 1 {
            return Err(bad(&format!(
                "matches {matched} of {} oneOf branches ({})",
                one_of.len(),
                reasons.join("; ")
            )));
        }
    }
    if schema.get("minimum").is_some() || schema.get("maximum").is_some() {
        let here: Option<i128> = doc
            .as_u64()
            .map(i128::from)
            .or_else(|| doc.as_i64().map(i128::from));
        if let Some(here) = here {
            if let Some(min) = schema.get("minimum").and_then(Value::as_u64)
                && here < i128::from(min)
            {
                return Err(bad(&format!("{here} is below minimum {min}")));
            }
            if let Some(max) = schema.get("maximum").and_then(Value::as_u64)
                && here > i128::from(max)
            {
                return Err(bad(&format!("{here} is above maximum {max}")));
            }
        }
    }
    Ok(())
}

/// Follows a local `#/...` pointer from the schema root.
fn resolve<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    let pointer = reference.strip_prefix('#')?;
    if pointer.is_empty() {
        return Some(root);
    }
    let mut v = root;
    for part in pointer.strip_prefix('/')?.split('/') {
        let part = part.replace("~1", "/").replace("~0", "~");
        v = v.get(&part)?;
    }
    Some(v)
}

/// The validator above handles exactly these keywords; annotations carry no
/// validation. Fails on any other keyword so a schema addition without
/// validator support is caught here, not silently ignored.
#[test]
fn validator_supports_every_schema_keyword() {
    fn visit(node: &Value, path: &str) {
        let Some(obj) = node.as_object() else {
            return;
        };
        for (key, value) in obj {
            match key.as_str() {
                // No validation and no subschemas: annotations (`$schema`,
                // `$id`, `title`, `description`), data (`const`, `enum`,
                // `required`) and leaf checks (`type`,
                // `additionalProperties`, `minItems`, `$ref`).
                "$schema"
                | "$id"
                | "title"
                | "description"
                | "const"
                | "enum"
                | "required"
                | "type"
                | "additionalProperties"
                | "minItems"
                | "$ref" => {}
                "$defs" => {
                    let defs = value
                        .as_object()
                        .unwrap_or_else(|| panic!("{path}/$defs is not an object: {value}"));
                    for (name, sub) in defs {
                        visit(sub, &format!("{path}/$defs/{name}"));
                    }
                }
                // `properties` maps names to schemas; the names are data.
                "properties" => {
                    let props = value
                        .as_object()
                        .unwrap_or_else(|| panic!("{path}/properties is not an object: {value}"));
                    for (name, sub) in props {
                        visit(sub, &format!("{path}/properties/{name}"));
                    }
                }
                // Subschemas to descend into.
                "items" => visit(value, &format!("{path}/items")),
                "oneOf" => {
                    let branches = value
                        .as_array()
                        .unwrap_or_else(|| panic!("{path}/oneOf is not an array: {value}"));
                    for (i, sub) in branches.iter().enumerate() {
                        visit(sub, &format!("{path}/oneOf[{i}]"));
                    }
                }
                "minimum" | "maximum" => {
                    assert!(
                        value.as_u64().is_some(),
                        "{path}/{key} is not a non-negative integer: {value}"
                    );
                }
                other => panic!("{path}: validator does not support keyword {other:?}"),
            }
        }
    }

    let schema = schema_doc();
    visit(&schema, "#");
    // Every `$ref` in the file resolves to a schema the validator can see.
    fn check_refs(node: &Value, root: &Value, path: &str) {
        if let Some(obj) = node.as_object() {
            for (key, value) in obj {
                if key == "$ref" {
                    let reference = value
                        .as_str()
                        .unwrap_or_else(|| panic!("{path}/$ref is not a string: {value}"));
                    assert!(
                        resolve(root, reference).is_some(),
                        "{path}: cannot resolve {reference:?}"
                    );
                } else if key == "properties" || key == "$defs" {
                    for (name, sub) in value.as_object().unwrap() {
                        check_refs(sub, root, &format!("{path}/{key}/{name}"));
                    }
                } else if key == "items" {
                    check_refs(value, root, &format!("{path}/items"));
                } else if key == "oneOf" {
                    for (i, sub) in value.as_array().unwrap().iter().enumerate() {
                        check_refs(sub, root, &format!("{path}/oneOf[{i}]"));
                    }
                }
            }
        }
    }
    check_refs(&schema, &schema, "#");
}

/// A time the text prints, with the unit it was printed in (the display's
/// rounding quantum follows the unit).
#[derive(Clone, Copy, Debug)]
struct TextTime {
    ns: u64,
    unit: TimeUnit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimeUnit {
    S,
    Ms,
    Us,
}

/// Half of the display's last digit: the most a printed time may be off.
fn time_tol(unit: TimeUnit) -> u64 {
    match unit {
        TimeUnit::S => 500_000,
        TimeUnit::Ms => 50_000,
        TimeUnit::Us => 50,
    }
}

/// Parses `Printer::time`: `508.6 ms`, `937.7 µs`, `1.285 s`.
fn parse_time(text: &str) -> TextTime {
    let (num, unit) = text.rsplit_once(' ').unwrap();
    let unit = match unit {
        "s" => TimeUnit::S,
        "ms" => TimeUnit::Ms,
        "µs" | "μs" | "us" => TimeUnit::Us,
        other => panic!("unknown time unit in {text:?} ({other})"),
    };
    let mult: u64 = match unit {
        TimeUnit::S => 1_000_000_000,
        TimeUnit::Ms => 1_000_000,
        TimeUnit::Us => 1_000,
    };
    let (whole, frac) = match num.split_once('.') {
        Some((w, f)) => (w, f),
        None => (num, ""),
    };
    let mut ns = whole.parse::<u64>().unwrap() * mult;
    if !frac.is_empty() {
        let scale = 10_u64.pow(u32::try_from(frac.len()).unwrap());
        ns += frac.parse::<u64>().unwrap() * mult / scale;
    }
    TextTime { ns, unit }
}

/// A size the text prints, with the unit it was printed in.
#[derive(Clone, Copy, Debug)]
struct TextBytes {
    bytes: u64,
    unit: ByteUnit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ByteUnit {
    B,
    KiB,
    MiB,
    GiB,
    TiB,
}

/// Half of the display's last digit (rounded up).
fn bytes_tol(unit: ByteUnit) -> u64 {
    match unit {
        ByteUnit::B => 0,
        ByteUnit::KiB => 52,
        ByteUnit::MiB => 52_429,
        ByteUnit::GiB => 53_687_092,
        ByteUnit::TiB => 54_975_581_389,
    }
}

/// Parses `human_bytes`: `1023 B`, `1.5 KiB`, `9.9 MiB`.
fn parse_bytes(text: &str) -> TextBytes {
    let (num, unit) = text.rsplit_once(' ').unwrap();
    let (mult, unit) = match unit {
        "B" => (1_u64, ByteUnit::B),
        "KiB" => (1 << 10, ByteUnit::KiB),
        "MiB" => (1 << 20, ByteUnit::MiB),
        "GiB" => (1 << 30, ByteUnit::GiB),
        "TiB" => (1 << 40, ByteUnit::TiB),
        other => panic!("unknown size unit in {text:?} ({other})"),
    };
    // Exact below one KiB; decimal above it.
    if unit == ByteUnit::B {
        return TextBytes {
            bytes: num.parse::<u64>().unwrap(),
            unit,
        };
    }
    let (whole, frac) = match num.split_once('.') {
        Some((w, f)) => (w, f),
        None => (num, ""),
    };
    let mut bytes = u128::from(whole.parse::<u64>().unwrap()) * u128::from(mult);
    if !frac.is_empty() {
        let scale = 10_u128.pow(u32::try_from(frac.len()).unwrap());
        bytes += u128::from(frac.parse::<u64>().unwrap()) * u128::from(mult) / scale;
    }
    TextBytes {
        bytes: u64::try_from(bytes).unwrap_or(u64::MAX),
        unit,
    }
}

/// One operator line of the text: everything the JSON must agree with.
/// Times and sizes keep their printed unit for the rounding tolerance;
/// `morsels` has no text form and is not parsed.
#[derive(Debug)]
struct TextOp {
    role: String,
    name: String,
    pipelines: Vec<u32>,
    rows_in: Option<(u64, u64)>,
    rows_out: Option<(u64, u64)>,
    compute_ns: TextTime,
    max_compute_ns: TextTime,
    drivers: u64,
    io_ns: Option<(TextTime, TextTime)>,
    mem_wait_ns: Option<TextTime>,
    sink_wait_ns: Option<TextTime>,
    peak_mem: Option<TextBytes>,
    spilled: Option<(u64, TextBytes)>,
    spill_files: Option<(u64, TextBytes)>,
    extras: String,
}

#[derive(Debug)]
struct TextHeader {
    wall: TextTime,
    pipelines: u64,
    drivers: u64,
    threads: u64,
    workers: u64,
    peak_mem: Option<TextBytes>,
    /// `None` is the text's `none` (unlimited).
    limit: Option<Option<TextBytes>>,
}

#[derive(Debug)]
struct TextPipeline {
    wall: TextTime,
    drivers: u64,
    threads: u64,
    ops: Vec<u32>,
}

/// The operator lines of the text: the part after `·`, or a `Result:`
/// line, which has no `·`.
fn operator_line(line: &str) -> Option<&str> {
    if let Some(at) = line.find('·') {
        return Some(line[at + '·'.len_utf8()..].trim_start());
    }
    let trimmed = line.trim_start();
    for role in ["source", "operator", "sink"] {
        if let Some(rest) = trimmed.strip_prefix(role)
            && rest.starts_with(char::is_whitespace)
        {
            return Some(trimmed);
        }
    }
    None
}

fn parse_rows(part: &str) -> (u64, u64) {
    // `in 80 rows (1 batch)` / `out 6 rows (2 batches)`.
    let (rows, batches) = part.split_once(" rows (").unwrap();
    let batches = batches.strip_suffix(')').unwrap();
    let (count, _) = batches.split_once(' ').unwrap();
    (rows.parse::<u64>().unwrap(), count.parse::<u64>().unwrap())
}

fn parse_compute(part: &str) -> (TextTime, TextTime, u64) {
    // `compute 1.2 ms (max 0.8 ms, 2 drivers)`.
    let rest = part.strip_prefix("compute ").unwrap();
    let (time, rest) = rest.split_once(" (max ").unwrap();
    let (max, drivers) = rest.strip_suffix(')').unwrap().split_once(", ").unwrap();
    let (count, _) = drivers.split_once(' ').unwrap();
    (
        parse_time(time),
        parse_time(max),
        count.parse::<u64>().unwrap(),
    )
}

fn parse_io(part: &str) -> (TextTime, TextTime) {
    // `io 0.4 ms (max 0.3 ms)`.
    let rest = part.strip_prefix("io ").unwrap();
    let (time, rest) = rest.split_once(" (max ").unwrap();
    (
        parse_time(time),
        parse_time(rest.strip_suffix(')').unwrap()),
    )
}

fn parse_op(line: &str) -> (u32, TextOp) {
    // `sink   #2 HashAggregate p2: in 80 rows (1 batch); compute ...`.
    let hash = line.find('#').unwrap();
    let after = &line[hash + 1..];
    let end = after.find(char::is_whitespace).unwrap();
    let id: u32 = after[..end].parse::<u32>().unwrap();
    let rest = after[end..].trim_start();
    // The pipes end at the first `: ` (extra values may hold their own).
    let (head, parts) = rest.split_once(": ").unwrap();
    let (name, pipes) = match head.rsplit_once(' ') {
        Some((name, pipes)) => (name, pipes),
        None => (head, ""),
    };
    if !pipes.is_empty() {
        assert!(
            pipes.split(',').all(|p| p.starts_with('p')),
            "pipes of {line:?}"
        );
    }
    let pipelines = if pipes.is_empty() {
        Vec::new()
    } else {
        pipes
            .split(',')
            .map(|p| p[1..].parse::<u32>().unwrap())
            .collect()
    };
    let mut op = TextOp {
        role: line.split_whitespace().next().unwrap().to_owned(),
        name: name.to_owned(),
        pipelines,
        rows_in: None,
        rows_out: None,
        compute_ns: TextTime {
            ns: 0,
            unit: TimeUnit::Us,
        },
        max_compute_ns: TextTime {
            ns: 0,
            unit: TimeUnit::Us,
        },
        drivers: 0,
        io_ns: None,
        mem_wait_ns: None,
        sink_wait_ns: None,
        peak_mem: None,
        spilled: None,
        spill_files: None,
        extras: String::new(),
    };
    let mut saw_compute = false;
    for part in parts.split("; ") {
        if let Some(rest) = part.strip_prefix("in ") {
            op.rows_in = Some(parse_rows(rest));
        } else if let Some(rest) = part.strip_prefix("out ") {
            op.rows_out = Some(parse_rows(rest));
        } else if part.starts_with("compute ") {
            let (time, max, drivers) = parse_compute(part);
            op.compute_ns = time;
            op.max_compute_ns = max;
            op.drivers = drivers;
            saw_compute = true;
        } else if part.starts_with("io ") {
            op.io_ns = Some(parse_io(part));
        } else if let Some(rest) = part.strip_prefix("memory wait ") {
            op.mem_wait_ns = Some(parse_time(rest));
        } else if let Some(rest) = part.strip_prefix("sink wait ") {
            op.sink_wait_ns = Some(parse_time(rest));
        } else if let Some(rest) = part.strip_prefix("peak mem ") {
            op.peak_mem = Some(parse_bytes(rest));
        } else if let Some(rest) = part.strip_prefix("spilled ") {
            // `spilled 2x 3.0 KiB`.
            let (count, bytes) = rest.split_once('x').unwrap();
            op.spilled = Some((
                count.trim_end().parse::<u64>().unwrap(),
                parse_bytes(bytes.trim_start()),
            ));
        } else if let Some(rest) = part.strip_prefix("spill files ") {
            // `spill files 1 (2.0 KiB written)`.
            let (count, bytes) = rest.split_once(' ').unwrap();
            let bytes = bytes.strip_prefix('(').unwrap();
            let bytes = bytes.strip_suffix(" written)").unwrap();
            op.spill_files = Some((count.parse::<u64>().unwrap(), parse_bytes(bytes)));
        } else if part.contains('=') {
            assert!(op.extras.is_empty(), "two extra blocks in {line:?}");
            op.extras = part.to_owned();
        } else {
            panic!("cannot parse operator part {part:?} in {line:?}");
        }
    }
    assert!(saw_compute, "no compute part in {line:?}");
    (id, op)
}

/// Parses the header, the operator lines and the pipeline lines.
fn parse_text(
    text: &str,
) -> (
    TextHeader,
    BTreeMap<u32, TextOp>,
    BTreeMap<u32, TextPipeline>,
) {
    let mut lines = text.lines();
    assert_eq!(lines.next().unwrap(), "== Physical Plan (analyzed) ==");
    let header = lines.next().unwrap();
    // `wall 508.6 ms, 7 pipelines, 54 drivers, 12 threads (most in one
    // pipeline), 1 worker`.
    let head: Vec<&str> = header.split(", ").collect();
    assert_eq!(head.len(), 5, "{header:?}");
    let count = |part: &str| part.split_once(' ').unwrap().0.parse::<u64>().unwrap();
    let mut header = TextHeader {
        wall: parse_time(head[0].strip_prefix("wall ").unwrap()),
        pipelines: count(head[1]),
        drivers: count(head[2]),
        threads: count(head[3]),
        workers: count(head[4]),
        peak_mem: None,
        limit: None,
    };
    assert!(
        head[1].split_once(' ').unwrap().1.starts_with("pipeline"),
        "{header:?}"
    );
    assert!(
        head[2].split_once(' ').unwrap().1.starts_with("driver"),
        "{header:?}"
    );
    assert!(
        head[3].split_once(' ').unwrap().1.starts_with("thread"),
        "{header:?}"
    );
    assert!(
        head[4].split_once(' ').unwrap().1.starts_with("worker"),
        "{header:?}"
    );
    let mut ops = BTreeMap::new();
    let mut pipelines = BTreeMap::new();
    let mut in_pipelines = false;
    for line in lines {
        if line == "Pipelines:" {
            in_pipelines = true;
            continue;
        }
        if in_pipelines {
            // `   p2: wall 125.5 ms, 12 drivers, 12 threads: #16 ...`.
            let line = line.trim_start();
            let (head, ops_text) = line.rsplit_once(": ").unwrap();
            let (id, rest) = head.split_once(':').unwrap();
            let id: u32 = id.strip_prefix('p').unwrap().parse::<u32>().unwrap();
            let wall: Vec<&str> = rest.trim_start().split(", ").collect();
            assert_eq!(wall.len(), 3, "{line:?}");
            let ops_here = ops_text
                .split(" -> ")
                .map(|o| o.strip_prefix('#').unwrap().split_once(' ').unwrap().0)
                .map(|n| n.parse::<u32>().unwrap())
                .collect();
            pipelines.insert(
                id,
                TextPipeline {
                    wall: parse_time(wall[0].strip_prefix("wall ").unwrap()),
                    drivers: count(wall[1]),
                    threads: count(wall[2]),
                    ops: ops_here,
                },
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("memory: peak reserved ") {
            let (peak, limit) = rest.split_once(", limit ").unwrap();
            header.peak_mem = Some(parse_bytes(peak));
            header.limit = Some(if limit == "none" {
                None
            } else {
                Some(parse_bytes(limit))
            });
            continue;
        }
        if line == "(probe lengths not collected)" || line == "Result:" {
            continue;
        }
        if let Some(op) = operator_line(line) {
            let (id, parsed) = parse_op(op);
            assert!(ops.insert(id, parsed).is_none(), "operator {id} twice");
        }
    }
    (header, ops, pipelines)
}

fn close(tag: &str, what: &str, expected_ns: u64, actual: TextTime) {
    let diff = expected_ns.abs_diff(actual.ns);
    assert!(
        diff <= time_tol(actual.unit),
        "{tag}: {what}: JSON {expected_ns}ns vs text {}ns (diff {diff}ns)",
        actual.ns,
    );
}

fn close_bytes(tag: &str, what: &str, expected: u64, actual: TextBytes) {
    let diff = expected.abs_diff(actual.bytes);
    assert!(
        diff <= bytes_tol(actual.unit),
        "{tag}: {what}: JSON {expected}B vs text {}B (diff {diff}B)",
        actual.bytes,
    );
}

/// `in`/`out 80 rows (1 batch)`: absent in the text means the JSON pair is
/// zero.
fn check_rows(tag: &str, id: u32, m: &Value, out: bool, parsed: Option<(u64, u64)>) {
    let (rows_key, batches_key) = if out {
        ("rows_out", "batches_out")
    } else {
        ("rows_in", "batches_in")
    };
    match parsed {
        Some((rows, batches)) => {
            assert_eq!(
                rows,
                m[rows_key].as_u64().unwrap(),
                "{tag}: operator {id} {rows_key}"
            );
            assert_eq!(
                batches,
                m[batches_key].as_u64().unwrap(),
                "{tag}: operator {id} {batches_key}"
            );
        }
        None => {
            assert_eq!(
                m[rows_key].as_u64().unwrap(),
                0,
                "{tag}: operator {id} {rows_key}"
            );
            assert_eq!(
                m[batches_key].as_u64().unwrap(),
                0,
                "{tag}: operator {id} {batches_key}"
            );
        }
    }
}

/// An optional text time (`memory wait`, `sink wait`): absent means the
/// JSON nanos are zero.
fn check_time_opt(tag: &str, id: u32, what: &str, parsed: Option<TextTime>, json_ns: u64) {
    match parsed {
        Some(t) => close(tag, &format!("operator {id} {what}"), json_ns, t),
        None => assert_eq!(json_ns, 0, "{tag}: operator {id} {what}"),
    }
}

/// An optional text size: absent means the JSON bytes are zero.
fn check_bytes_opt(tag: &str, id: u32, what: &str, parsed: Option<TextBytes>, json: u64) {
    match parsed {
        Some(b) => close_bytes(tag, &format!("operator {id} {what}"), json, b),
        None => assert_eq!(json, 0, "{tag}: operator {id} {what}"),
    }
}

/// `spilled 2x 3.0 KiB` against `spills` and `spilled_bytes`.
fn check_spilled(tag: &str, id: u32, m: &Value, parsed: Option<(u64, TextBytes)>) {
    match parsed {
        Some((n, bytes)) => {
            assert_eq!(
                n,
                m["spills"].as_u64().unwrap(),
                "{tag}: operator {id} spills"
            );
            close_bytes(
                tag,
                &format!("operator {id} spilled bytes"),
                m["spilled_bytes"].as_u64().unwrap(),
                bytes,
            );
        }
        None => {
            assert_eq!(
                m["spills"].as_u64().unwrap(),
                0,
                "{tag}: operator {id} spills"
            );
            assert_eq!(
                m["spilled_bytes"].as_u64().unwrap(),
                0,
                "{tag}: operator {id} spilled bytes"
            );
        }
    }
}

/// `spill files 1 (2.0 KiB written)` against `spill_files` and
/// `spill_write_bytes`.
fn check_spill_files(tag: &str, id: u32, m: &Value, parsed: Option<(u64, TextBytes)>) {
    match parsed {
        Some((n, bytes)) => {
            assert_eq!(
                n,
                m["spill_files"].as_u64().unwrap(),
                "{tag}: operator {id} spill files"
            );
            close_bytes(
                tag,
                &format!("operator {id} spill write bytes"),
                m["spill_write_bytes"].as_u64().unwrap(),
                bytes,
            );
        }
        None => {
            assert_eq!(
                m["spill_files"].as_u64().unwrap(),
                0,
                "{tag}: operator {id} spill files"
            );
            assert_eq!(
                m["spill_write_bytes"].as_u64().unwrap(),
                0,
                "{tag}: operator {id} spill write bytes"
            );
        }
    }
}

/// `io 0.4 ms (max 0.3 ms)`: absent (a non-source) means both JSON counters
/// are zero.
fn check_io(tag: &str, id: u32, m: &Value, parsed: Option<(TextTime, TextTime)>) {
    match parsed {
        Some((io, max)) => {
            close(
                tag,
                &format!("operator {id} io"),
                m["blocked_on_io_ns"].as_u64().unwrap(),
                io,
            );
            close(
                tag,
                &format!("operator {id} max io"),
                m["max_driver_blocked_on_io_ns"].as_u64().unwrap(),
                max,
            );
        }
        None => {
            assert_eq!(
                m["blocked_on_io_ns"].as_u64().unwrap(),
                0,
                "{tag}: operator {id} io"
            );
            assert_eq!(
                m["max_driver_blocked_on_io_ns"].as_u64().unwrap(),
                0,
                "{tag}: operator {id} max io"
            );
        }
    }
}

/// The extra metrics of one operator, in the JSON's order, against the
/// text's trailing `name=value` block.
fn check_extras(tag: &str, id: u32, json: &Value, extras: &str) {
    let items = json.as_array().unwrap();
    let mut rest = extras;
    for (i, item) in items.iter().enumerate() {
        let name = item["name"].as_str().unwrap();
        let needle = format!("{name}=");
        let at = rest.find(&needle).unwrap_or_else(|| {
            panic!("{tag}: operator {id}: text has no {needle:?} in {extras:?}")
        });
        assert!(
            at == 0 || rest[..at].ends_with(' '),
            "{tag}: operator {id}: {needle:?} is not an item start in {extras:?}"
        );
        rest = &rest[at + needle.len()..];
        let next = items
            .get(i + 1)
            .map(|n| format!(" {}=", n["name"].as_str().unwrap()));
        let (mut value, tail) = match &next {
            Some(next) => match rest.find(next) {
                Some(at) => (&rest[..at], &rest[at + 1..]),
                None => panic!("{tag}: operator {id}: text has no {next:?} after {name}="),
            },
            None => (rest, ""),
        };
        // The text decorates two values with their ratio.
        for marker in [" (load ", " (avg "] {
            if let Some(at) = value.rfind(marker) {
                value = &value[..at];
            }
        }
        match item["kind"].as_str().unwrap() {
            "count" | "max" => {
                let want = item["value"].as_u64().unwrap();
                if name.ends_with("_bytes") {
                    close_bytes(
                        tag,
                        &format!("operator {id} extra {name}"),
                        want,
                        parse_bytes(value),
                    );
                } else if name.ends_with("_ns") {
                    close(
                        tag,
                        &format!("operator {id} extra {name}"),
                        want,
                        parse_time(value),
                    );
                } else {
                    assert_eq!(
                        value.parse::<u64>().unwrap(),
                        want,
                        "{tag}: operator {id} extra {name}"
                    );
                }
            }
            "text" => assert_eq!(
                value,
                item["value"].as_str().unwrap(),
                "{tag}: operator {id} extra {name}"
            ),
            kind => panic!("{tag}: operator {id}: unknown extra kind {kind}"),
        }
        rest = tail;
    }
    assert!(
        rest.is_empty(),
        "{tag}: operator {id}: text has trailing extras {rest:?}"
    );
}

/// A JSON id: the schema bounds it to `u32`.
fn u32_of(v: &Value) -> u32 {
    u32::try_from(v.as_u64().unwrap()).unwrap()
}

/// Every number the text prints for one operator equals the JSON's number
/// (times and sizes within the display's rounding).
fn check_text_matches_json(tag: &str, text: &str, doc: &Value) {
    let (header, ops, pipelines) = parse_text(text);
    let query = &doc["query"];
    let u64_of = |v: &Value| v.as_u64().unwrap();
    close(tag, "wall", u64_of(&query["wall_time_ns"]), header.wall);
    assert_eq!(
        header.pipelines,
        u64::try_from(doc["pipelines"].as_array().unwrap().len()).unwrap(),
        "{tag}: pipelines"
    );
    assert_eq!(header.drivers, u64_of(&query["drivers"]), "{tag}: drivers");
    assert_eq!(header.workers, u64_of(&query["workers"]), "{tag}: workers");
    let most_threads = doc["pipelines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| u64_of(&p["threads"]))
        .max()
        .unwrap_or(0);
    assert_eq!(header.threads, most_threads, "{tag}: threads");
    match (header.peak_mem, query.get("memory")) {
        (Some(peak), Some(mem)) if !mem.is_null() => {
            close_bytes(
                tag,
                "peak reserved",
                u64_of(&mem["peak_reserved_bytes"]),
                peak,
            );
            match (header.limit, mem.get("limit_bytes")) {
                (Some(Some(limit)), Some(json)) if !json.is_null() => {
                    close_bytes(tag, "limit", u64_of(json), limit);
                }
                (limit, json) => {
                    assert!(
                        limit.is_none_or(|l| l.is_none()) && json.is_none_or(Value::is_null),
                        "{tag}: limit: text {limit:?} vs JSON {json:?}"
                    );
                }
            }
        }
        (peak, mem) => {
            assert!(
                peak.is_none() && mem.is_none_or(Value::is_null),
                "{tag}: memory: text {peak:?} vs JSON {mem:?}"
            );
        }
    }

    let json_ops: BTreeMap<u32, &Value> = doc["operators"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (u32_of(&e["id"]), e))
        .collect();
    assert_eq!(
        ops.keys().collect::<Vec<_>>(),
        json_ops.keys().collect::<Vec<_>>(),
        "{tag}: operators in text vs JSON"
    );
    for (id, op) in &ops {
        let json = json_ops[id];
        assert_eq!(
            op.role,
            json["role"].as_str().unwrap(),
            "{tag}: operator {id} role"
        );
        assert_eq!(
            op.name,
            json["name"].as_str().unwrap(),
            "{tag}: operator {id} name"
        );
        let mut want_pipes: Vec<u32> = json["pipelines"]
            .as_array()
            .unwrap()
            .iter()
            .map(u32_of)
            .collect();
        want_pipes.sort_unstable();
        let mut have_pipes = op.pipelines.clone();
        have_pipes.sort_unstable();
        assert_eq!(have_pipes, want_pipes, "{tag}: operator {id} pipelines");
        let m = &json["metrics"];
        check_rows(tag, *id, m, false, op.rows_in);
        check_rows(tag, *id, m, true, op.rows_out);
        assert_eq!(
            op.drivers,
            u64_of(&m["drivers"]),
            "{tag}: operator {id} drivers"
        );
        close(
            tag,
            &format!("operator {id} compute"),
            u64_of(&m["compute_time_ns"]),
            op.compute_ns,
        );
        close(
            tag,
            &format!("operator {id} max compute"),
            u64_of(&m["max_driver_compute_time_ns"]),
            op.max_compute_ns,
        );
        check_io(tag, *id, m, op.io_ns);
        check_time_opt(
            tag,
            *id,
            "memory wait",
            op.mem_wait_ns,
            u64_of(&m["blocked_on_memory_ns"]),
        );
        check_time_opt(
            tag,
            *id,
            "sink wait",
            op.sink_wait_ns,
            u64_of(&m["blocked_on_sink_ns"]),
        );
        check_bytes_opt(
            tag,
            *id,
            "peak mem",
            op.peak_mem,
            u64_of(&m["peak_reserved_bytes"]),
        );
        check_spilled(tag, *id, m, op.spilled);
        check_spill_files(tag, *id, m, op.spill_files);
        check_extras(tag, *id, &json["extra"], &op.extras);
    }

    let json_pipes: BTreeMap<u32, &Value> = doc["pipelines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (u32_of(&p["id"]), p))
        .collect();
    assert_eq!(
        pipelines.keys().collect::<Vec<_>>(),
        json_pipes.keys().collect::<Vec<_>>(),
        "{tag}: pipelines in text vs JSON"
    );
    for (id, pipe) in &pipelines {
        let json = json_pipes[id];
        let wall = json["ended_ns"].as_u64().unwrap() - json["started_ns"].as_u64().unwrap();
        close(tag, &format!("pipeline {id} wall"), wall, pipe.wall);
        assert_eq!(
            pipe.drivers,
            u64_of(&json["drivers"]),
            "{tag}: pipeline {id} drivers"
        );
        assert_eq!(
            pipe.threads,
            u64_of(&json["threads"]),
            "{tag}: pipeline {id} threads"
        );
        let mut want: Vec<u32> = json_ops
            .iter()
            .filter(|(_, e)| {
                e["pipelines"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|p| u32_of(p) == *id)
            })
            .map(|(oid, _)| *oid)
            .collect();
        want.sort_unstable();
        let mut have = pipe.ops.clone();
        have.sort_unstable();
        assert_eq!(have, want, "{tag}: pipeline {id} operators");
    }
}

#[test]
fn all_tpch_profiles_validate_and_match_their_text() {
    let s = session(true);
    let schema = schema_doc();
    for q in 1..=22 {
        let tag = format!("q{q:02}");
        let df = block_on(s.sql(&query(q))).unwrap_or_else(|e| panic!("{tag}: sql: {e}"));
        let (_, profile) =
            block_on(df.collect_with_profile()).unwrap_or_else(|e| panic!("{tag}: analyze: {e}"));
        // The JSON round-trips exactly.
        let text = profile.to_json();
        let back = QueryProfile::from_json(&text).unwrap();
        assert_eq!(back, profile, "{tag}: JSON round trip");
        // ... validates against the checked-in schema ...
        let doc: Value = serde_json::from_str(&text).unwrap();
        validate(&schema, &doc, &schema, "#")
            .unwrap_or_else(|e| panic!("{tag}: schema validation: {e}"));
        // ... and its numbers equal the text's.
        check_text_matches_json(&tag, &profile.to_text(TextOptions::default()), &doc);
    }
}

/// Under a tight memory limit the profile reports spill bytes; the spilling
/// run's JSON still validates and matches its text (this exercises the
/// `spilled`, `spill files` and spill extra metrics with nonzero values).
#[test]
fn spilling_profile_validates_and_matches_its_text() {
    let dir = tempfile::tempdir().unwrap();
    // The session creates the spill root (mode 0700) below the temp dir.
    let spill = dir.path().join("spill");
    let mut b = Session::builder().config("krishiv.spill.dir", spill.to_str().unwrap());
    b = b.config("krishiv.memory.limit", "2m");
    let s = b.build().unwrap();
    // Enough data to overflow a 2 MiB pool: Q18's grouped lineitem.
    common::register_with(&s, 16_000, true);
    let df = block_on(s.sql(&query(18))).unwrap();
    let (_, profile) = block_on(df.collect_with_profile()).unwrap();
    let spilled: u64 = profile
        .metrics
        .operators
        .values()
        .map(|e| e.metrics.spill_write_bytes)
        .sum();
    assert!(spilled > 0, "the profile reports spill bytes");
    let text = profile.to_json();
    let back = QueryProfile::from_json(&text).unwrap();
    assert_eq!(back, profile, "spilling profile round trip");
    let doc: Value = serde_json::from_str(&text).unwrap();
    let schema = schema_doc();
    validate(&schema, &doc, &schema, "#")
        .unwrap_or_else(|e| panic!("spilling profile schema validation: {e}"));
    check_text_matches_json(
        "q18-spilling",
        &profile.to_text(TextOptions::default()),
        &doc,
    );
}

/// A document that breaks the schema fails validation; one that breaks the
/// producer's invariants fails to load. (`additionalProperties` is the
/// exception the other way: the reader ignores members added within v1, so
/// an older build still reads a newer document.)
#[test]
fn corrupt_documents_fail_validation() {
    // Q1 needs no string equality, so the `Utf8` tables do.
    let s = session(false);
    let df = block_on(s.sql(&query(1))).unwrap();
    let profile = block_on(df.explain_analyze()).unwrap();
    let schema = schema_doc();
    let good = profile.to_json_value();
    // An operator with extra metrics (the result sink has none).
    let with_extra = good["operators"]
        .as_array()
        .unwrap()
        .iter()
        .position(|o| !o["extra"].as_array().unwrap().is_empty())
        .unwrap();
    type Edit = Box<dyn Fn(&mut Value)>;

    let schema_cases: Vec<(&str, Edit, &str, Option<&str>)> = vec![
        (
            "format is not",
            Box::new(|v| v["format"] = "x".into()),
            "#/format: is not",
            Some("format is not krishiv.query-profile"),
        ),
        (
            "version 2 is not supported",
            Box::new(|v| v["version"] = 2.into()),
            "#/version: is not 1: 2",
            Some("version 2 is not supported (this build reads 1)"),
        ),
        (
            "missing member \"query\"",
            Box::new(|v| {
                v.as_object_mut().unwrap().remove("query");
            }),
            "#: missing member \"query\"",
            Some("missing member \"query\""),
        ),
        (
            "wall_time_ns is not an unsigned integer",
            Box::new(|v| v["query"]["wall_time_ns"] = (-1).into()),
            "#/query/wall_time_ns: -1 is below minimum 0",
            Some("wall_time_ns is not an unsigned integer"),
        ),
        (
            "drivers is not an unsigned integer",
            Box::new(|v| v["query"]["drivers"] = "many".into()),
            "#/query/drivers: is not integer",
            Some("drivers is not an unsigned integer"),
        ),
        (
            "unexpected member",
            Box::new(|v| v["bogus"] = 1.into()),
            "#: unexpected member \"bogus\"",
            None,
        ),
        (
            "unknown operator role",
            Box::new(|v| v["operators"][0]["role"] = "x".into()),
            "#/operators[0]/role: is not one of",
            Some("unknown operator role \"x\""),
        ),
        (
            "unknown metric kind",
            Box::new(move |v| v["operators"][with_extra]["extra"][0]["kind"] = "avg".into()),
            "/extra[0]: matches 0 of 2 oneOf branches",
            Some("unknown metric kind \"avg\""),
        ),
        (
            "a pipeline id exceeds u32",
            Box::new(|v| v["pipelines"][0]["id"] = (u64::from(u32::MAX) + 1).into()),
            "#/pipelines[0]/id: 4294967296 is above maximum 4294967295",
            Some("id exceeds u32"),
        ),
        (
            "node is not a u32 or null",
            Box::new(|v| v["operators"][0]["node"] = "0".into()),
            "#/operators[0]/node: matches 0 of 2 oneOf branches",
            Some("node is not an unsigned integer"),
        ),
        (
            "limit_bytes is not a u64 or null",
            Box::new(|v| v["query"]["memory"]["limit_bytes"] = "unlimited".into()),
            "#/query/memory/limit_bytes: matches 0 of 2 oneOf branches",
            Some("limit_bytes is not an unsigned integer"),
        ),
    ];
    for (want, edit, validate_want, reader_want) in schema_cases {
        let mut v = good.clone();
        edit(&mut v);
        let validator_err = validate(&schema, &v, &schema, "#")
            .err()
            .unwrap_or_else(|| panic!("{want} validates"));
        assert!(
            validator_err.contains(validate_want),
            "{want}: validator error {validator_err:?} does not contain {validate_want:?}"
        );
        match reader_want {
            // Within-v1 growth: the reader ignores what it does not know.
            None => assert!(
                QueryProfile::from_json(&v.to_string()).is_ok(),
                "{want} does not load"
            ),
            Some(reader_want) => {
                let err = QueryProfile::from_json(&v.to_string())
                    .err()
                    .unwrap_or_else(|| panic!("accepted a document with {want}"));
                let message = err.to_string();
                assert!(message.starts_with("invalid query profile: "), "{message}");
                assert!(
                    message.contains(reader_want),
                    "{want}: reader error {message:?} does not contain {reader_want:?}"
                );
            }
        }
    }

    // The schema cannot see ids, name uniqueness or the plan's shape; the
    // reader still must.
    let reader_cases: Vec<(&str, Edit, &str)> = vec![
        (
            "extra metric listed twice",
            Box::new(move |v| {
                let first = v["operators"][with_extra]["extra"][0].clone();
                v["operators"][with_extra]["extra"]
                    .as_array_mut()
                    .unwrap()
                    .push(first);
            }),
            "extra metric \"groups\" listed twice",
        ),
        (
            "operator 0 listed twice",
            Box::new(|v| {
                let first = v["operators"][0].clone();
                v["operators"].as_array_mut().unwrap().push(first);
            }),
            "operator 0 listed twice",
        ),
        (
            "pipeline 0 listed twice",
            Box::new(|v| {
                let first = v["pipelines"][0].clone();
                v["pipelines"].as_array_mut().unwrap().push(first);
            }),
            "pipeline 0 listed twice",
        ),
        (
            "a plan input is not a later node",
            Box::new(|v| v["plan"][0]["children"] = serde_json::json!([0])),
            "a plan input is not a later node",
        ),
        (
            "the input of two nodes",
            Box::new(|v| {
                v["plan"][2]["children"] = serde_json::json!([3]);
                v["plan"][1]["children"] = serde_json::json!([2, 3]);
            }),
            "a plan node is the input of two nodes",
        ),
    ];
    for (want, edit, reader_want) in reader_cases {
        let mut v = good.clone();
        edit(&mut v);
        let err = QueryProfile::from_json(&v.to_string())
            .err()
            .unwrap_or_else(|| panic!("accepted a document with {want}"));
        let message = err.to_string();
        assert!(message.starts_with("invalid query profile: "), "{message}");
        assert!(
            message.contains(reader_want),
            "{want}: reader error {message:?} does not contain {reader_want:?}"
        );
    }
}

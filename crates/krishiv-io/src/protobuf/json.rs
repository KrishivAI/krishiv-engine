// SPDX-License-Identifier: Apache-2.0
//! protobuf-java's `JsonFormat.printer().omittingInsignificantWhitespace().preservingProtoFieldNames()`, for the `Any` fields that
//! `convert.any.fields.to.json` prints: field names as declared, fields in number order, no default values (a proto3 scalar that is
//! its zero is not printed), 64-bit integers as strings, `Timestamp` as RFC 3339, `Duration` as `1.500s`, the wrappers as their
//! value, an `Any` as `{"@type": url, ...}` (the value under `"value"` for a well-known type). The type registry is the whole
//! descriptor set. `Struct`, `Value`, `ListValue` and `FieldMask` are refused.

use super::desc::{FieldDescriptor, Kind, MessageDescriptor};

use super::DescriptorSet;
use super::decode::Failure;
use super::wire::{Parsed, Raw, Slot, collect};
use crate::error::unsupported;

/// The JSON text of the `google.protobuf.Any` message in `bytes`.
pub(crate) fn any_to_json(
    set: &DescriptorSet,
    bytes: &[u8],
    enums_as_ints: bool,
    depth: u32,
) -> Result<String, Failure> {
    let mut out = String::new();
    let p = Printer { set, enums_as_ints };
    let any = set
        .message_by_full_name("google.protobuf.Any")
        .ok_or(Failure::Malformed)?;
    p.any(&any, bytes, depth, &mut out)?;
    Ok(out)
}

struct Printer<'a> {
    set: &'a DescriptorSet,
    enums_as_ints: bool,
}

const WELL_KNOWN: &[&str] = &[
    "google.protobuf.Timestamp",
    "google.protobuf.Duration",
    "google.protobuf.BoolValue",
    "google.protobuf.Int32Value",
    "google.protobuf.UInt32Value",
    "google.protobuf.Int64Value",
    "google.protobuf.UInt64Value",
    "google.protobuf.StringValue",
    "google.protobuf.BytesValue",
    "google.protobuf.FloatValue",
    "google.protobuf.DoubleValue",
    "google.protobuf.Any",
];

const REFUSED: &[&str] = &[
    "google.protobuf.Struct",
    "google.protobuf.Value",
    "google.protobuf.ListValue",
    "google.protobuf.FieldMask",
];

impl Printer<'_> {
    fn any(
        &self,
        any: &MessageDescriptor,
        bytes: &[u8],
        depth: u32,
        out: &mut String,
    ) -> Result<(), Failure> {
        let p = collect(any, bytes, depth)?;
        let url = string_of(&p, 1);
        let value = match p.get(2) {
            Some(Slot::One(Raw::Len(b))) => *b,
            _ => &[],
        };
        if url.is_empty() && value.is_empty() {
            out.push_str("{}");
            return Ok(());
        }
        let (_, name) = url.rsplit_once('/').ok_or(Failure::Malformed)?;
        let inner = self
            .set
            .message_by_full_name(name)
            .ok_or(Failure::Malformed)?;
        if REFUSED.contains(&inner.full_name()) {
            return Err(Failure::Refused(unsupported(format!(
                "from_protobuf: an Any holding {} printed as JSON",
                inner.full_name()
            ))));
        }
        out.push_str("{\"@type\":");
        string(&url, out);
        if inner.full_name() == "google.protobuf.Any" {
            // An Any holding an Any prints the inner one under "value" (protobuf-java's JsonFormat).
            out.push_str(",\"value\":");
            self.any(&inner, value, depth + 1, out)?;
            out.push('}');
            return Ok(());
        }
        let parsed = collect(&inner, value, depth + 1)?;
        if WELL_KNOWN.contains(&inner.full_name()) {
            out.push_str(",\"value\":");
            self.well_known(&inner, &parsed, depth + 1, out)?;
        } else {
            self.fields(&inner, &parsed, depth + 1, true, out)?;
        }
        out.push('}');
        Ok(())
    }

    /// The fields of a message as `"name":value` pairs; `leading_comma` when something precedes them.
    fn fields(
        &self,
        msg: &MessageDescriptor,
        p: &Parsed<'_>,
        depth: u32,
        mut leading_comma: bool,
        out: &mut String,
    ) -> Result<(), Failure> {
        let mut fields: Vec<FieldDescriptor> = msg.fields().collect();
        fields.sort_by_key(FieldDescriptor::number);
        for fd in fields {
            let Some(slot) = p.get(fd.number()) else {
                continue;
            };
            let items: Vec<Raw<'_>> = match slot {
                Slot::One(r) => {
                    if !has_presence(&fd) && zero(&fd, *r) {
                        continue;
                    }
                    vec![*r]
                }
                Slot::Many(rs) => rs.clone(),
            };
            if items.is_empty() {
                continue;
            }
            if leading_comma {
                out.push(',');
            }
            leading_comma = true;
            string(fd.name(), out);
            out.push(':');
            if fd.is_map() {
                self.map(&fd, &items, depth, out)?;
            } else if fd.is_list() {
                out.push('[');
                for (i, r) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    self.value(&fd, *r, depth, out)?;
                }
                out.push(']');
            } else if let Slot::Many(rs) = slot {
                // A singular message field: the pieces merge.
                let joined: Vec<u8> = rs
                    .iter()
                    .flat_map(|r| match r {
                        Raw::Len(b) => b.to_vec(),
                        _ => Vec::new(),
                    })
                    .collect();
                self.value(&fd, Raw::Len(&joined), depth, out)?;
            } else {
                self.value(&fd, items[0], depth, out)?;
            }
        }
        Ok(())
    }

    fn map(
        &self,
        fd: &FieldDescriptor,
        items: &[Raw<'_>],
        depth: u32,
        out: &mut String,
    ) -> Result<(), Failure> {
        let Kind::Message(entry) = fd.kind() else {
            return Err(Failure::Malformed);
        };
        let Some((kf, vf)) = entry.map_entry_fields() else {
            return Err(Failure::Malformed);
        };
        out.push('{');
        for (i, raw) in items.iter().enumerate() {
            let Raw::Len(b) = raw else {
                return Err(Failure::Malformed);
            };
            let e = collect(&entry, b, depth + 1)?;
            if i > 0 {
                out.push(',');
            }
            let mut key = String::new();
            self.scalar_or_default(&kf, &e, depth, &mut key)?;
            if key.starts_with('"') {
                out.push_str(&key);
            } else {
                string(&key, out);
            }
            out.push(':');
            self.scalar_or_default(&vf, &e, depth, out)?;
        }
        out.push('}');
        Ok(())
    }

    fn scalar_or_default(
        &self,
        fd: &FieldDescriptor,
        e: &Parsed<'_>,
        depth: u32,
        out: &mut String,
    ) -> Result<(), Failure> {
        match e.get(fd.number()) {
            Some(Slot::One(r)) => self.value(fd, *r, depth, out),
            Some(Slot::Many(rs)) => {
                let joined: Vec<u8> = rs
                    .iter()
                    .flat_map(|r| match r {
                        Raw::Len(b) => b.to_vec(),
                        _ => Vec::new(),
                    })
                    .collect();
                self.value(fd, Raw::Len(&joined), depth, out)
            }
            None => match fd.kind() {
                Kind::Message(_) => self.value(fd, Raw::Len(&[]), depth, out),
                Kind::String | Kind::Bytes => self.value(fd, Raw::Len(&[]), depth, out),
                Kind::Float | Kind::Fixed32 | Kind::Sfixed32 => {
                    self.value(fd, Raw::Fixed32(0), depth, out)
                }
                Kind::Double | Kind::Fixed64 | Kind::Sfixed64 => {
                    self.value(fd, Raw::Fixed64(0), depth, out)
                }
                _ => self.value(fd, Raw::Varint(0), depth, out),
            },
        }
    }

    /// One value of `fd`.
    fn value(
        &self,
        fd: &FieldDescriptor,
        raw: Raw<'_>,
        depth: u32,
        out: &mut String,
    ) -> Result<(), Failure> {
        match (fd.kind(), raw) {
            (Kind::Int32, Raw::Varint(v)) => out.push_str(&(v as i32).to_string()),
            (Kind::Sint32, Raw::Varint(v)) => out.push_str(&zigzag32(v).to_string()),
            (Kind::Uint32, Raw::Varint(v)) => out.push_str(&(v as u32).to_string()),
            (Kind::Fixed32, Raw::Fixed32(v)) => out.push_str(&v.to_string()),
            (Kind::Sfixed32, Raw::Fixed32(v)) => out.push_str(&(v as i32).to_string()),
            (Kind::Int64, Raw::Varint(v)) => quoted(&(v as i64).to_string(), out),
            (Kind::Sint64, Raw::Varint(v)) => quoted(&zigzag64(v).to_string(), out),
            (Kind::Uint64, Raw::Varint(v)) | (Kind::Fixed64, Raw::Fixed64(v)) => {
                quoted(&v.to_string(), out);
            }
            (Kind::Sfixed64, Raw::Fixed64(v)) => quoted(&(v as i64).to_string(), out),
            (Kind::Float, Raw::Fixed32(v)) => {
                let f = f32::from_bits(v);
                if f.is_finite() {
                    out.push_str(&krishiv_kernels::text::java_float_string32(f));
                } else {
                    quoted(&non_finite(f64::from(f)), out);
                }
            }
            (Kind::Double, Raw::Fixed64(v)) => {
                let f = f64::from_bits(v);
                if f.is_finite() {
                    out.push_str(&krishiv_kernels::text::java_double_string(f));
                } else {
                    quoted(&non_finite(f), out);
                }
            }
            (Kind::Bool, Raw::Varint(v)) => out.push_str(if v != 0 { "true" } else { "false" }),
            (Kind::String, Raw::Len(b)) => string(&String::from_utf8_lossy(b), out),
            (Kind::Bytes, Raw::Len(b)) => quoted(&base64(b), out),
            (Kind::Enum(e), Raw::Varint(v)) => {
                let n = v as i32;
                if e.name() == "NullValue"
                    && e.get_value(0).is_some_and(|v| v.name() == "NULL_VALUE")
                {
                    // JsonFormat prints this enum as JSON null whatever the options.
                    out.push_str("null");
                } else if self.enums_as_ints {
                    out.push_str(&n.to_string());
                } else {
                    match e.get_value(n) {
                        Some(ev) => string(ev.name(), out),
                        // JsonFormat prints an unknown value as its number.
                        None => out.push_str(&n.to_string()),
                    }
                }
            }
            (Kind::Message(m), Raw::Len(b)) => {
                if REFUSED.contains(&m.full_name()) {
                    return Err(Failure::Refused(unsupported(format!(
                        "from_protobuf: {} printed as JSON",
                        m.full_name()
                    ))));
                }
                if m.full_name() == "google.protobuf.Any" {
                    return self.any(&m, b, depth + 1, out);
                }
                let p = collect(&m, b, depth + 1)?;
                if WELL_KNOWN.contains(&m.full_name()) {
                    self.well_known(&m, &p, depth + 1, out)?;
                } else {
                    out.push('{');
                    self.fields(&m, &p, depth + 1, false, out)?;
                    out.push('}');
                }
            }
            _ => return Err(Failure::Malformed),
        }
        Ok(())
    }

    /// The JSON of a well-known type other than `Any` (which has its own form).
    fn well_known(
        &self,
        m: &MessageDescriptor,
        p: &Parsed<'_>,
        depth: u32,
        out: &mut String,
    ) -> Result<(), Failure> {
        match m.full_name() {
            "google.protobuf.Timestamp" => timestamp(int_of(p, 1), int_of(p, 2) as i32, out),
            "google.protobuf.Duration" => duration(int_of(p, 1), int_of(p, 2) as i32, out),
            "google.protobuf.Empty" => {
                out.push_str("{}");
                Ok(())
            }
            "google.protobuf.Any" => Err(Failure::Malformed),
            _ => {
                // A wrapper: its value, zero when absent.
                let fd = m.get_field(1).ok_or(Failure::Malformed)?;
                let raw = match p.get(1) {
                    Some(Slot::One(r)) => *r,
                    _ => match fd.kind() {
                        Kind::Float | Kind::Fixed32 | Kind::Sfixed32 => Raw::Fixed32(0),
                        Kind::Double | Kind::Fixed64 | Kind::Sfixed64 => Raw::Fixed64(0),
                        Kind::String | Kind::Bytes => Raw::Len(&[]),
                        _ => Raw::Varint(0),
                    },
                };
                self.value(&fd, raw, depth, out)
            }
        }
    }
}

fn has_presence(fd: &FieldDescriptor) -> bool {
    fd.supports_presence()
}

fn zero(fd: &FieldDescriptor, raw: Raw<'_>) -> bool {
    match (fd.kind(), raw) {
        (_, Raw::Varint(v)) => {
            v == 0
                || (matches!(
                    fd.kind(),
                    Kind::Int32 | Kind::Uint32 | Kind::Sint32 | Kind::Enum(_)
                ) && v as u32 == 0)
        }
        (_, Raw::Fixed32(v)) => v == 0,
        (_, Raw::Fixed64(v)) => v == 0,
        (_, Raw::Len(b)) => b.is_empty(),
    }
}

fn int_of(p: &Parsed<'_>, number: u32) -> i64 {
    match p.get(number) {
        Some(Slot::One(Raw::Varint(v))) => *v as i64,
        _ => 0,
    }
}

fn string_of(p: &Parsed<'_>, number: u32) -> String {
    match p.get(number) {
        Some(Slot::One(Raw::Len(b))) => String::from_utf8_lossy(b).into_owned(),
        _ => String::new(),
    }
}

fn zigzag32(v: u64) -> i32 {
    let n = v as u32;
    ((n >> 1) as i32) ^ -((n & 1) as i32)
}

fn zigzag64(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

fn non_finite(f: f64) -> String {
    if f.is_nan() {
        "NaN".to_owned()
    } else if f > 0.0 {
        "Infinity".to_owned()
    } else {
        "-Infinity".to_owned()
    }
}

fn quoted(s: &str, out: &mut String) {
    out.push('"');
    out.push_str(s);
    out.push('"');
}

/// A JSON string literal, escaped as protobuf-java does.
fn string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || (0x7f..=0x9f).contains(&(c as u32)) => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for chunk in b.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |a, (i, x)| a | (u32::from(*x) << (16 - 8 * i)));
        for i in 0..4 {
            if i <= chunk.len() {
                s.push(char::from(T[((n >> (18 - 6 * i)) & 63) as usize]));
            } else {
                s.push('=');
            }
        }
    }
    s
}

/// The fraction protobuf-java prints: 3, 6 or 9 digits.
fn fraction(nanos: u32, out: &mut String) {
    if nanos == 0 {
        return;
    }
    if nanos.is_multiple_of(1_000_000) {
        out.push_str(&format!(".{:03}", nanos / 1_000_000));
    } else if nanos.is_multiple_of(1_000) {
        out.push_str(&format!(".{:06}", nanos / 1_000));
    } else {
        out.push_str(&format!(".{nanos:09}"));
    }
}

/// `Timestamps.toString`: `0001-01-01T00:00:00Z` to `9999-12-31T23:59:59.999999999Z`.
fn timestamp(seconds: i64, nanos: i32, out: &mut String) -> Result<(), Failure> {
    if !(-62_135_596_800..=253_402_300_799).contains(&seconds)
        || !(0..1_000_000_000).contains(&nanos)
    {
        return Err(Failure::Malformed);
    }
    let days = seconds.div_euclid(86_400);
    let rem = seconds.rem_euclid(86_400);
    // Days since 1970-01-01 to a civil date (Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    out.push('"');
    out.push_str(&format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    ));
    fraction(nanos as u32, out);
    out.push_str("Z\"");
    Ok(())
}

/// `Durations.toString`.
fn duration(seconds: i64, nanos: i32, out: &mut String) -> Result<(), Failure> {
    let signs_agree = (seconds >= 0 && nanos >= 0) || (seconds <= 0 && nanos <= 0);
    if !signs_agree
        || seconds.unsigned_abs() > 315_576_000_000
        || nanos.unsigned_abs() > 999_999_999
    {
        return Err(Failure::Malformed);
    }
    out.push('"');
    if seconds < 0 || nanos < 0 {
        out.push('-');
    }
    out.push_str(&seconds.unsigned_abs().to_string());
    fraction(nanos.unsigned_abs(), out);
    out.push_str("s\"");
    Ok(())
}

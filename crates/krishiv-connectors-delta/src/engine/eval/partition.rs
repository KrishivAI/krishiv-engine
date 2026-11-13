// SPDX-License-Identifier: Apache-2.0
//! `MapToStruct`: an add action's `partitionValues` (a string map) as a typed struct, with
//! Delta's partition-value encoding (port of the kernel's `evaluate_map_to_struct` and
//! `parse_partition_scalar`, `delta_kernel` 0.30.1, Apache-2.0).
//!
//! - A missing key or a NULL value is NULL; a NULL map is a NULL struct (so a non-nullable
//!   partition column of a non-add row, under `COALESCE`, does not fail).
//! - An empty string is the empty string or empty bytes for a string or binary column and NULL
//!   for every other type (Spark's non-ANSI cast of the map).
//! - `date` is `YYYY-MM-DD`; `timestamp_ntz` is Arrow's timestamp grammar read without a zone;
//!   `timestamp` honours an offset in the value, and an offset-less value is read in the reader
//!   zone of `MapToStructOptions` (UTC by default). The reader zone may be `UTC` or a fixed
//!   `+HH:MM`/`-HH:MM` offset; a named IANA zone is unsupported here (the kernel's scan code never
//!   sets one, and chrono-tz is banned by ADR 0028).
//! - A `timestamp` value may end in a zone name, spaced or glued to the time, as the kernel's
//!   Arrow grammar reads it (`12:30:45 UTC`, `12:30:45UTC`); the name is then the value's zone,
//!   and a value with an offset as well (`12:30:45+02:00 UTC`) does not parse. The names of UTC
//!   (`UTC`, `GMT`, `UCT`, `Zulu`, `Universal`, `Greenwich`, `GMT0`, `GMT+0`, `GMT-0`, each also
//!   under `Etc/`) are read as UTC.
//! - **A difference from the kernel:** any other zone name (`CET`,
//!   `America/New_York`) is refused as the kernel's `Unsupported` (raised as
//!   `UNSUPPORTED_FEATURE.TABLE_OPERATION`, naming the zone), where the kernel resolves the
//!   name with chrono-tz. Spark and Delta writers do not write such values (they write no zone
//!   or a `Z`/offset); Krishiv has no zone database (no chrono-tz, no jiff tz data).
//! - Other types use the kernel's `PrimitiveType::parse_scalar` (decimals with exponents,
//!   booleans case-insensitively, intervals).
//! - A value that does not parse is the kernel's `ParseError` (a broken table), not NULL.
//! - Duplicate keys: the last one wins (undefined in the kernel's contract).
//!
//! Values are appended to typed builders straight from the map's string buffer: no per-row
//! allocation for strings or binaries.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    ArrayBuilder, BinaryBuilder, Date32Builder, StringBuilder, TimestampMicrosecondBuilder,
    make_builder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::Date32Type;
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_cast::parse::{Parser, string_to_datetime};
use arrow_schema::Fields;
use chrono::{FixedOffset, TimeZone, Utc};
use delta_kernel::expressions::MapToStructOptions;
use delta_kernel::schema::{DataType, PrimitiveType, StructType};
use delta_kernel::{KernelError, KernelResult};

use super::arrow_err;
use crate::engine::schema::{append_scalar, to_arrow_schema};

/// The zone an offset-less `timestamp` partition value is read in.
#[derive(Debug, Clone, Copy)]
enum ReaderZone {
    Utc,
    Fixed(FixedOffset),
}

impl ReaderZone {
    fn from_options(options: &MapToStructOptions) -> KernelResult<Self> {
        let Some(zone) = options.timestamp_timezone() else {
            return Ok(Self::Utc);
        };
        if matches!(zone, "UTC" | "Etc/UTC" | "Z") {
            return Ok(Self::Utc);
        }
        if zone.starts_with(['+', '-']) {
            return fixed_offset(zone).map(Self::Fixed).ok_or_else(|| {
                KernelError::generic(format!("Invalid timestamp timezone: {zone}"))
            });
        }
        Err(KernelError::unsupported(format!(
            "the named reader time zone {zone} for partition timestamps"
        )))
    }

    fn parse(self, raw: &str) -> Option<i64> {
        let parsed = match self {
            Self::Utc => micros(&Utc, raw),
            Self::Fixed(z) => micros(&z, raw),
        };
        // A UTC name needs no zone database.
        parsed.or_else(|| match zone_suffix(raw)? {
            (local, zone) if is_utc_name(zone) => micros(&Utc, local),
            _ => None,
        })
    }
}

fn micros<T: TimeZone>(zone: &T, raw: &str) -> Option<i64> {
    string_to_datetime(zone, raw)
        .ok()
        .map(|t| t.timestamp_micros())
}

/// A `+HH:MM`/`-HH:MM` offset within ±18:00, as the kernel accepts.
fn fixed_offset(value: &str) -> Option<FixedOffset> {
    let sign = match value.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let (h, m) = value.get(1..)?.split_once(':')?;
    let two = |s: &str| -> Option<i32> {
        (s.len() == 2 && s.bytes().all(|b| b.is_ascii_digit()))
            .then(|| s.parse().ok())
            .flatten()
    };
    let (h, m) = (two(h)?, two(m)?);
    if h > 18 || m > 59 || (h == 18 && m != 0) {
        return None;
    }
    FixedOffset::east_opt(sign * (h * 3_600 + m * 60))
}

/// The names of UTC in the zone database (`Etc/` forms included).
fn is_utc_name(zone: &str) -> bool {
    const NAMES: &[&str] = &[
        "UTC",
        "GMT",
        "UCT",
        "Zulu",
        "Universal",
        "Greenwich",
        "GMT0",
        "GMT+0",
        "GMT-0",
    ];
    let bare = zone.strip_prefix("Etc/").unwrap_or(zone);
    NAMES.contains(&bare)
}

/// The zone database's names without a `/` (tzdata's `backward` links and the old System V
/// zones), so a trailing word that is no zone (`PM`) is a parse error, as in the kernel.
const TOP_LEVEL_ZONES: &[&str] = &[
    "CET",
    "CST6CDT",
    "Cuba",
    "EET",
    "EST",
    "EST5EDT",
    "Egypt",
    "Eire",
    "Factory",
    "GB",
    "GB-Eire",
    "GMT",
    "GMT+0",
    "GMT-0",
    "GMT0",
    "Greenwich",
    "HST",
    "Hongkong",
    "Iceland",
    "Iran",
    "Israel",
    "Jamaica",
    "Japan",
    "Kwajalein",
    "Libya",
    "MET",
    "MST",
    "MST7MDT",
    "NZ",
    "NZ-CHAT",
    "Navajo",
    "PRC",
    "PST8PDT",
    "Poland",
    "Portugal",
    "ROC",
    "ROK",
    "Singapore",
    "Turkey",
    "UCT",
    "UTC",
    "Universal",
    "W-SU",
    "WET",
    "Zulu",
];

/// The value `raw` split into its local timestamp and a trailing zone name, as Arrow's grammar
/// reads a zone after the time (spaced or glued): `None` unless the local part is a timestamp
/// with a time and no offset or `Z` of its own, and the rest is a name (letters first; letters,
/// digits, `/`, `_`, `-`, `+`) with a `/` or one of [`TOP_LEVEL_ZONES`].
fn zone_suffix(raw: &str) -> Option<(&str, &str)> {
    let raw = raw.trim_end();
    // The date is the first ten characters; the time starts after the separator.
    let at = raw
        .char_indices()
        .skip(11)
        .find(|(_, c)| c.is_ascii_alphabetic())
        .map(|(i, _)| i)?;
    let (local, zone) = (raw[..at].trim_end(), &raw[at..]);
    let has_offset = local
        .get(11..)
        .is_some_and(|time| time.contains(['+', '-', 'Z', 'z']));
    let is_name = (zone.contains('/') || TOP_LEVEL_ZONES.contains(&zone))
        && zone
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '_' | '-' | '+'));
    (is_name && !has_offset && micros(&Utc, local).is_some()).then_some((local, zone))
}

/// Why the `timestamp` partition value `raw` did not parse: a trailing zone name other than
/// UTC's on an offset-less local timestamp is `Unsupported` (see the module docs), anything
/// else the kernel's `ParseError`.
fn named_zone_or_parse_error(raw: &str, prim: &PrimitiveType) -> KernelError {
    if let Some((_, zone)) = zone_suffix(raw) {
        return KernelError::unsupported(format!(
            "the time zone name {zone} in the timestamp partition value '{raw}': Krishiv reads \
             partition timestamps with no zone, `Z`, a numeric offset or a name of UTC"
        ));
    }
    parse_error(raw, prim)
}

fn parse_error(raw: &str, prim: &PrimitiveType) -> KernelError {
    KernelError::ParseError(raw.to_owned(), DataType::Primitive(prim.clone()))
}

fn wrong_builder(prim: &PrimitiveType) -> KernelError {
    KernelError::generic(format!("no Arrow builder for a {prim} partition column"))
}

macro_rules! builder {
    ($b:expr, $t:ty, $prim:expr) => {
        $b.as_any_mut()
            .downcast_mut::<$t>()
            .ok_or_else(|| wrong_builder($prim))?
    };
}

/// Appends the partition value `raw` (`None`: null) of type `prim` to `b`.
fn append(
    b: &mut dyn ArrayBuilder,
    prim: &PrimitiveType,
    raw: Option<&str>,
    zone: ReaderZone,
) -> KernelResult<()> {
    let data_type = DataType::Primitive(prim.clone());
    let Some(raw) = raw else {
        return append_scalar(&delta_kernel::expressions::Scalar::Null(data_type), b, 1);
    };
    match prim {
        PrimitiveType::String => builder!(b, StringBuilder, prim).append_value(raw),
        PrimitiveType::Binary => builder!(b, BinaryBuilder, prim).append_value(raw.as_bytes()),
        _ if raw.is_empty() => {
            return append_scalar(&delta_kernel::expressions::Scalar::Null(data_type), b, 1);
        }
        PrimitiveType::Date => {
            let days = Date32Type::parse(raw).ok_or_else(|| parse_error(raw, prim))?;
            builder!(b, Date32Builder, prim).append_value(days);
        }
        PrimitiveType::Timestamp => {
            let Some(v) = zone.parse(raw) else {
                return Err(named_zone_or_parse_error(raw, prim));
            };
            builder!(b, TimestampMicrosecondBuilder, prim).append_value(v);
        }
        PrimitiveType::TimestampNtz => {
            let v = micros(&Utc, raw).ok_or_else(|| parse_error(raw, prim))?;
            builder!(b, TimestampMicrosecondBuilder, prim).append_value(v);
        }
        _ => append_scalar(&prim.parse_scalar(raw)?, b, 1)?,
    }
    Ok(())
}

/// Evaluates `MapToStruct(map)` into `schema` (see the module docs).
///
/// # Errors
/// A non-string map, a non-primitive target field, an unsupported reader zone, or a value that
/// does not parse.
pub(crate) fn map_to_struct(
    map: &ArrayRef,
    schema: &StructType,
    options: &MapToStructOptions,
) -> KernelResult<ArrayRef> {
    let zone = ReaderZone::from_options(options)?;
    let map = map
        .as_map_opt()
        .ok_or_else(|| KernelError::generic("MapToStruct requires a MapArray as input"))?;
    let keys = map
        .keys()
        .as_string_opt::<i32>()
        .ok_or_else(|| KernelError::generic("MapToStruct requires maps with string keys"))?;
    let values = map
        .values()
        .as_string_opt::<i32>()
        .ok_or_else(|| KernelError::generic("MapToStruct requires maps with string values"))?;
    let arrow = to_arrow_schema(schema)?;
    let mut prims = Vec::with_capacity(schema.num_fields());
    let mut builders = Vec::with_capacity(schema.num_fields());
    for (f, a) in schema.fields().zip(arrow.fields()) {
        let DataType::Primitive(p) = f.data_type() else {
            return Err(KernelError::generic(format!(
                "MapToStruct only supports primitive target types, got {:?}",
                f.data_type()
            )));
        };
        prims.push(p);
        builders.push(make_builder(a.data_type(), map.len()));
    }
    // Field index by name, built once per batch.
    let index: HashMap<&str, usize> = schema
        .fields()
        .enumerate()
        .map(|(i, f)| (f.name().as_str(), i))
        .collect();
    // The entry each field last matched; an entry before the row's first belongs to an earlier
    // row (offsets only grow), so nothing is reset between rows.
    let mut matched: Vec<Option<usize>> = vec![None; prims.len()];
    let offsets = map.value_offsets();
    for row in 0..map.len() {
        let start = usize::try_from(offsets[row]).map_err(|_| arrow_offset())?;
        let end = usize::try_from(offsets[row + 1]).map_err(|_| arrow_offset())?;
        if map.is_valid(row) {
            for entry in start..end {
                let key = keys.value(entry);
                if let Some(&i) = index.get(key) {
                    matched[i] = Some(entry);
                }
            }
        }
        for (i, b) in builders.iter_mut().enumerate() {
            let raw = matched[i]
                .filter(|e| *e >= start && *e < end && values.is_valid(*e))
                .map(|e| values.value(e));
            append(b.as_mut(), prims[i], raw, zone)?;
        }
    }
    let columns: Vec<ArrayRef> = builders.iter_mut().map(ArrayBuilder::finish).collect();
    let fields: Fields = arrow.fields().clone();
    if columns.is_empty() {
        return Ok(Arc::new(StructArray::new_empty_fields(
            map.len(),
            map.nulls().cloned(),
        )));
    }
    Ok(Arc::new(
        StructArray::try_new(fields, columns, map.nulls().cloned()).map_err(arrow_err)?,
    ))
}

fn arrow_offset() -> KernelError {
    KernelError::generic("a negative map offset")
}

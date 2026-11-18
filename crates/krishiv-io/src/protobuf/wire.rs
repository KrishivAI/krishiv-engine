// SPDX-License-Identifier: Apache-2.0
//! The protobuf wire format as `DynamicMessage.parseFrom` reads it, under Spark's reading of the result.
//!
//! [`collect`] splits one message into its fields by number: the last value of a singular scalar wins, the values of a repeated
//! field (packed or not) and the pieces of a singular message field (which merge) accumulate, and a oneof keeps its last member.
//! A known field number with another wire type, and an enum number a proto2 (closed) enum does not define, are what protobuf-java
//! keeps as unknown fields: [`Parsed::mismatch`] records them (Spark fails the top-level message on one, and ignores them in a
//! nested message). Truncated input, a required field that is missing, a proto3 string that is not UTF-8 and nesting past 100
//! levels are [`Malformed`].

use std::collections::BTreeMap;

use super::desc::{FieldDescriptor, Kind, MessageDescriptor};

/// protobuf-java's recursion limit for nested messages.
pub(crate) const MAX_DEPTH: u32 = 100;

/// The input is not a valid message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Malformed;

/// One value as it is on the wire.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Raw<'a> {
    /// A varint.
    Varint(u64),
    /// Four little-endian bytes.
    Fixed32(u32),
    /// Eight little-endian bytes.
    Fixed64(u64),
    /// Length-delimited bytes, or the body of a group.
    Len(&'a [u8]),
}

/// The values collected for one field number.
#[derive(Debug)]
pub(crate) enum Slot<'a> {
    /// A singular scalar: the last value.
    One(Raw<'a>),
    /// A repeated field, or the pieces of a singular message field.
    Many(Vec<Raw<'a>>),
}

/// One message, split by field number.
#[derive(Debug, Default)]
pub(crate) struct Parsed<'a> {
    slots: BTreeMap<u32, Slot<'a>>,
    /// A known field arrived with another wire type, or as an enum number its closed enum lacks.
    pub(crate) mismatch: bool,
}

impl<'a> Parsed<'a> {
    /// The values of field `number`.
    pub(crate) fn get(&self, number: u32) -> Option<&Slot<'a>> {
        self.slots.get(&number)
    }
}

/// A varint at `*pos`.
fn varint(buf: &[u8], pos: &mut usize) -> Result<u64, Malformed> {
    let mut v = 0u64;
    for i in 0..10 {
        let b = *buf.get(*pos).ok_or(Malformed)?;
        *pos += 1;
        v |= u64::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(Malformed)
}

fn take<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], Malformed> {
    let end = pos.checked_add(n).ok_or(Malformed)?;
    let out = buf.get(*pos..end).ok_or(Malformed)?;
    *pos = end;
    Ok(out)
}

fn le32(b: &[u8]) -> Result<u32, Malformed> {
    Ok(u32::from_le_bytes(b.try_into().map_err(|_| Malformed)?))
}

fn le64(b: &[u8]) -> Result<u64, Malformed> {
    Ok(u64::from_le_bytes(b.try_into().map_err(|_| Malformed)?))
}

/// Skips the group opened with field `number`; returns its body (without the end tag).
fn group_body<'a>(
    buf: &'a [u8],
    pos: &mut usize,
    number: u32,
    depth: u32,
) -> Result<&'a [u8], Malformed> {
    if depth > MAX_DEPTH {
        return Err(Malformed);
    }
    let start = *pos;
    loop {
        let before = *pos;
        let tag = varint(buf, pos)?;
        let (n, wt) = ((tag >> 3) as u32, (tag & 7) as u8);
        match wt {
            0 => {
                varint(buf, pos)?;
            }
            1 => {
                take(buf, pos, 8)?;
            }
            2 => {
                let len = usize::try_from(varint(buf, pos)?).map_err(|_| Malformed)?;
                take(buf, pos, len)?;
            }
            3 => {
                group_body(buf, pos, n, depth + 1)?;
            }
            4 => {
                return if n == number {
                    buf.get(start..before).ok_or(Malformed)
                } else {
                    Err(Malformed)
                };
            }
            5 => {
                take(buf, pos, 4)?;
            }
            _ => return Err(Malformed),
        }
    }
}

/// The wire type a field's values have (packed repeated scalars arrive as length-delimited).
fn wire_type(kind: &Kind, group: bool) -> u8 {
    match kind {
        Kind::Int32
        | Kind::Int64
        | Kind::Uint32
        | Kind::Uint64
        | Kind::Sint32
        | Kind::Sint64
        | Kind::Bool
        | Kind::Enum(_) => 0,
        Kind::Fixed64 | Kind::Sfixed64 | Kind::Double => 1,
        Kind::Fixed32 | Kind::Sfixed32 | Kind::Float => 5,
        Kind::String | Kind::Bytes => 2,
        Kind::Message(_) => {
            if group {
                3
            } else {
                2
            }
        }
    }
}

fn packable(kind: &Kind) -> bool {
    !matches!(kind, Kind::String | Kind::Bytes | Kind::Message(_))
}

/// Whether `number` is a value of the closed enum of `fd`.
fn closed_enum_lacks(fd: &FieldDescriptor, value: u64) -> bool {
    match fd.kind() {
        Kind::Enum(e) if e.is_closed() => e.get_value(value as i32).is_none(),
        _ => false,
    }
}

/// The `(field number, value)` pairs of a message, in wire order (a group's value is its body); no descriptor needed.
pub(crate) fn tagged_values(buf: &[u8]) -> Result<Vec<(u32, Raw<'_>)>, Malformed> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < buf.len() {
        let tag = varint(buf, &mut pos)?;
        let number = u32::try_from(tag >> 3).map_err(|_| Malformed)?;
        if number == 0 {
            return Err(Malformed);
        }
        let raw = match tag & 7 {
            0 => Raw::Varint(varint(buf, &mut pos)?),
            1 => Raw::Fixed64(le64(take(buf, &mut pos, 8)?)?),
            2 => {
                let len = usize::try_from(varint(buf, &mut pos)?).map_err(|_| Malformed)?;
                Raw::Len(take(buf, &mut pos, len)?)
            }
            3 => Raw::Len(group_body(buf, &mut pos, number, 0)?),
            5 => Raw::Fixed32(le32(take(buf, &mut pos, 4)?)?),
            _ => return Err(Malformed),
        };
        out.push((number, raw));
    }
    Ok(out)
}

/// Splits `buf`, a message of type `msg`, by field number (see the module docs).
pub(crate) fn collect<'a>(
    msg: &MessageDescriptor,
    buf: &'a [u8],
    depth: u32,
) -> Result<Parsed<'a>, Malformed> {
    if depth > MAX_DEPTH {
        return Err(Malformed);
    }
    let mut out = Parsed::default();
    let mut pos = 0usize;
    while pos < buf.len() {
        let tag = varint(buf, &mut pos)?;
        let number = u32::try_from(tag >> 3).map_err(|_| Malformed)?;
        let wt = (tag & 7) as u8;
        if number == 0 {
            return Err(Malformed);
        }
        let known = msg.get_field(number);
        let expected = known
            .as_ref()
            .map(|fd| wire_type(&fd.kind(), fd.is_group()));
        let packed = known
            .as_ref()
            .is_some_and(|fd| fd.is_list() && packable(&fd.kind()) && wt == 2);
        let value: Option<Raw<'a>> = match wt {
            0 => Some(Raw::Varint(varint(buf, &mut pos)?)),
            1 => Some(Raw::Fixed64(le64(take(buf, &mut pos, 8)?)?)),
            2 => {
                let len = usize::try_from(varint(buf, &mut pos)?).map_err(|_| Malformed)?;
                Some(Raw::Len(take(buf, &mut pos, len)?))
            }
            3 => Some(Raw::Len(group_body(buf, &mut pos, number, depth)?)),
            4 => return Err(Malformed),
            5 => Some(Raw::Fixed32(le32(take(buf, &mut pos, 4)?)?)),
            _ => return Err(Malformed),
        };
        let (Some(fd), Some(value)) = (known, value) else {
            continue;
        };
        if packed {
            let Raw::Len(body) = value else {
                return Err(Malformed);
            };
            let element = expected.unwrap_or(0);
            let mut p = 0usize;
            while p < body.len() {
                let raw = match element {
                    0 => Raw::Varint(varint(body, &mut p)?),
                    1 => Raw::Fixed64(le64(take(body, &mut p, 8)?)?),
                    _ => Raw::Fixed32(le32(take(body, &mut p, 4)?)?),
                };
                store(&mut out, &fd, raw);
            }
        } else if Some(wt) == expected {
            if let (Kind::String, Raw::Len(b)) = (fd.kind(), value)
                && !fd.is_proto2()
                && std::str::from_utf8(b).is_err()
            {
                return Err(Malformed);
            }
            if let (Kind::Message(_), Raw::Len(b)) = (fd.kind(), value) {
                // Spark parses each embedded message whole before projecting it: one that is not a well-formed message
                // fails the row even when merging its pieces, or leaving its fields out, would hide it.
                tagged_values(b)?;
            }
            store(&mut out, &fd, value);
        } else {
            out.mismatch = true;
        }
    }
    for fd in msg.fields() {
        if fd.is_required() && !out.slots.contains_key(&fd.number()) {
            return Err(Malformed);
        }
    }
    Ok(out)
}

/// Records one value of `fd`.
fn store<'a>(out: &mut Parsed<'a>, fd: &FieldDescriptor, raw: Raw<'a>) {
    if let Raw::Varint(v) = raw
        && closed_enum_lacks(fd, v)
    {
        out.mismatch = true;
        return;
    }
    if let Some(oneof) = fd.containing_oneof()
        && !oneof.is_synthetic()
    {
        for other in oneof.fields() {
            if other.number() != fd.number() {
                out.slots.remove(&other.number());
            }
        }
    }
    let many = fd.is_list() || fd.is_map() || matches!(fd.kind(), Kind::Message(_));
    if many {
        match out
            .slots
            .entry(fd.number())
            .or_insert_with(|| Slot::Many(Vec::new()))
        {
            Slot::Many(v) => v.push(raw),
            Slot::One(_) => {}
        }
    } else {
        out.slots.insert(fd.number(), Slot::One(raw));
    }
}

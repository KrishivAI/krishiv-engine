// SPDX-License-Identifier: Apache-2.0
//! Decoding of Connect messages from bytes, with limits that hold against
//! hostile input.
//!
//! `prost` recurses once per nested message and, as built here (the
//! `no-recursion-limit` feature), never stops: its fixed limit of 100 rejects
//! ordinary PySpark plans (a loop of `withColumn` calls nests two messages
//! per call). So every decode goes through this module, which first walks
//! the bytes with an explicit stack (never recursing) and a table of what
//! each field of each message holds (`proto/<version>/layout.pb.rs`,
//! generated from the protos). The walk is linear in the input and rejects:
//!
//! - input larger than [`MAX_PLAN_BYTES`];
//! - more than [`MAX_MESSAGES`] nested messages, and input that would decode
//!   to more than [`MAX_DECODED_BYTES`], because a decoded message takes far
//!   more memory than its few bytes of wire form (an empty `Expression` is
//!   two bytes on the wire and about 300 decoded, an empty string two and
//!   24, a packed integer one and 8);
//! - nesting deeper than [`MAX_NESTING`] messages;
//! - a field the protocol version does not define, which a newer client may
//!   send: prost would skip it silently, and a plan read without a field
//!   that changes its meaning could give a wrong answer
//!   (`KRISHIV_UNSUPPORTED_FEATURE` naming the message and field number);
//! - group wire types (`proto3` has none, and `prost` skips an unknown group
//!   recursively);
//! - lengths and varints that run past their message.
//!
//! Only then does `prost` decode, on a thread with a big stack when the
//! message is deeper than a few levels (`stack`). A server must decode with
//! these functions, not with the codec of its gRPC stack: clippy's
//! `disallowed_methods` (`clippy.toml`) rejects any other call of prost's
//! decoding methods in the workspace, because the feature applies to every
//! crate linked with this one.

use std::ops::Deref;

use krishiv_common::error::{
    ConnectInvalidPlanCannotParse, ConnectInvalidPlanPlanSizeLargerThanMax,
    FailedToParseTooComplex, KrishivError, KrishivUnsupportedFeature,
};
use prost::Message;

use crate::proto::v4_2::{Expression, Plan, Relation};
use crate::proto::{Field, Layout, Scrub, layout};
use crate::stack::{INLINE_LEVELS, on_deep_stack};

/// The deepest message nesting accepted, counted below the decoded message:
/// a `Relation` inside a `Plan` is at depth 1. It is Spark's default for
/// `spark.connect.grpc.marshallerRecursionLimit`, so a plan Spark accepts is
/// accepted here. A relation costs two levels (the relation and its
/// operator), as does an expression, so this allows about 500 chained
/// operators or nested expressions.
pub const MAX_NESTING: usize = 1024;

/// The most bytes of a message accepted: Spark's default maximum inbound
/// gRPC message size (`spark.connect.grpc.maxInboundMessageSize`).
pub const MAX_PLAN_BYTES: usize = 128 * 1024 * 1024;

/// The most messages (relations, expressions, literals, options, ...) a
/// decoded plan may hold.
pub const MAX_MESSAGES: usize = 2_000_000;

/// The most memory a decoded message may take, as the walk estimates it
/// from the sizes of the Rust types: every message, string, byte string and
/// repeated scalar counted at its decoded size (an upper bound, except that
/// growing vectors may allocate up to twice their contents). A plan of
/// [`MAX_PLAN_BYTES`] of Arrow data in a local relation fits; two-byte
/// empty messages or strings filling it do not.
pub const MAX_DECODED_BYTES: usize = 512 * 1024 * 1024;

/// What the walk counts for a string, byte string or message from outside
/// the package, besides its bytes: its header (`google.protobuf.Any` is two
/// of them).
const LEAF_SIZE: usize = 2 * size_of::<String>();

/// What the walk counts for a scalar sent unpacked: the widest element.
const SCALAR_SIZE: usize = 8;

/// A decoded message. It dereferences to the message. When it is dropped,
/// its option values, paths and `Any` payloads are overwritten with zeros
/// first ([`Scrub`], ADR 0038); when the message is deeply nested, that and
/// the drop run on a thread with a big stack, because dropping nested boxes
/// recurses like decoding does.
pub struct Decoded<M: Message + Default + Send + Scrub> {
    message: M,
    deep: bool,
}

/// Prints only the size: a decoded plan can hold literals and SQL text that
/// should not reach a log.
impl<M: Message + Default + Send + Scrub> std::fmt::Debug for Decoded<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoded")
            .field("encoded_len", &self.message.encoded_len())
            .field("deep", &self.deep)
            .finish_non_exhaustive()
    }
}

impl<M: Message + Default + Send + Scrub> Deref for Decoded<M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.message
    }
}

impl<M: Message + Default + Send + Scrub> Drop for Decoded<M> {
    fn drop(&mut self) {
        let mut message = std::mem::take(&mut self.message);
        let scrub_and_drop = move || {
            message.scrub();
            drop(message);
        };
        if self.deep {
            // If the thread cannot start, the message drops on this stack,
            // unscrubbed.
            let _ = on_deep_stack(scrub_and_drop);
        } else {
            scrub_and_drop();
        }
    }
}

/// Decodes a [`Plan`].
///
/// # Errors
/// `CONNECT_INVALID_PLAN.PLAN_SIZE_LARGER_THAN_MAX` for input over the size,
/// message or decoded-size limit, `FAILED_TO_PARSE_TOO_COMPLEX` for nesting
/// deeper than [`MAX_NESTING`], `KRISHIV_UNSUPPORTED_FEATURE` for a field
/// Spark Connect 4.2 does not define, `CONNECT_INVALID_PLAN.CANNOT_PARSE` for
/// malformed bytes.
pub fn decode_plan(bytes: &[u8]) -> Result<Decoded<Plan>, KrishivError> {
    decode(layout::PLAN, bytes)
}

/// Decodes a [`Relation`]; see [`decode_plan`] for the errors.
///
/// # Errors
/// As [`decode_plan`].
pub fn decode_relation(bytes: &[u8]) -> Result<Decoded<Relation>, KrishivError> {
    decode(layout::RELATION, bytes)
}

/// Decodes an [`Expression`]; see [`decode_plan`] for the errors.
///
/// # Errors
/// As [`decode_plan`].
pub fn decode_expression(bytes: &[u8]) -> Result<Decoded<Expression>, KrishivError> {
    decode(layout::EXPRESSION, bytes)
}

fn decode<M: Message + Default + Send + Scrub>(
    root: u16,
    bytes: &[u8],
) -> Result<Decoded<M>, KrishivError> {
    let deep = check(root, bytes)? > INLINE_LEVELS;
    // The one call of prost's decoder: the walk above bounded it.
    #[expect(
        clippy::disallowed_methods,
        reason = "the bounded decoder itself; `check` walked the bytes first"
    )]
    let decode = || M::decode(bytes);
    let decoded = if deep {
        on_deep_stack(decode)?
    } else {
        decode()
    };
    decoded
        .map(|message| Decoded { message, deep })
        .map_err(|e| malformed(&e.to_string()))
}

/// Why a walk of the bytes stopped.
#[derive(Debug, PartialEq, Eq)]
enum Reject {
    TooBig(usize),
    TooManyMessages,
    TooLarge,
    TooDeep,
    /// A field number message `message` (an id) does not define.
    UnknownField {
        message: u16,
        number: u32,
    },
    Malformed(&'static str),
}

impl From<Reject> for KrishivError {
    fn from(r: Reject) -> Self {
        match r {
            Reject::TooBig(size) => ConnectInvalidPlanPlanSizeLargerThanMax {
                plan_size: size.to_string(),
                max_plan_size: MAX_PLAN_BYTES.to_string(),
            }
            .into(),
            Reject::TooManyMessages => ConnectInvalidPlanPlanSizeLargerThanMax {
                plan_size: format!("more than {MAX_MESSAGES} messages"),
                max_plan_size: format!("{MAX_MESSAGES} messages"),
            }
            .into(),
            Reject::TooLarge => ConnectInvalidPlanPlanSizeLargerThanMax {
                plan_size: format!("more than {MAX_DECODED_BYTES} bytes decoded"),
                max_plan_size: format!("{MAX_DECODED_BYTES} bytes decoded"),
            }
            .into(),
            Reject::TooDeep => FailedToParseTooComplex {}.into(),
            Reject::UnknownField { message, number } => KrishivUnsupportedFeature {
                feature: format!(
                    "field {number} of spark.connect.{}, which Spark Connect 4.2 does not \
                     define (sent by a newer client?)",
                    layout::MESSAGES
                        .get(usize::from(message))
                        .map_or("?", |m| m.name)
                ),
            }
            .into(),
            Reject::Malformed(why) => malformed(why),
        }
    }
}

/// The error for a plan with more than [`MAX_MESSAGES`] messages.
pub(crate) fn too_many_messages() -> KrishivError {
    Reject::TooManyMessages.into()
}

fn malformed(why: &str) -> KrishivError {
    ConnectInvalidPlanCannotParse {
        error_msg: why.to_owned(),
    }
    .into()
}

/// Walks `bytes`; returns the deepest nesting.
fn check(root: u16, bytes: &[u8]) -> Result<usize, KrishivError> {
    scan(
        &layout::MESSAGES,
        root,
        bytes,
        &Limits {
            bytes: MAX_PLAN_BYTES,
            messages: MAX_MESSAGES,
            decoded: MAX_DECODED_BYTES,
            nesting: MAX_NESTING,
        },
    )
    .map_err(KrishivError::from)
}

/// The limits [`scan`] enforces; parameters so that tests can use small ones.
struct Limits {
    bytes: usize,
    messages: usize,
    decoded: usize,
    nesting: usize,
}

/// A message being walked: its id and where its bytes end.
struct Frame {
    message: u16,
    end: usize,
}

/// Walks `buf`, a message of id `root`, and every message nested in it,
/// without recursing, and returns the deepest nesting (0 for a message
/// without nested messages). `table[id]` describes message `id`.
fn scan(table: &[Layout], root: u16, buf: &[u8], limits: &Limits) -> Result<usize, Reject> {
    if buf.len() > limits.bytes {
        return Err(Reject::TooBig(buf.len()));
    }
    let size_of = |id: u16| table.get(usize::from(id)).map_or(0, |m| m.size);
    let mut stack = vec![Frame {
        message: root,
        end: buf.len(),
    }];
    let mut messages = 1_usize;
    let mut decoded = size_of(root);
    let mut deepest = 0_usize;
    let mut pos = 0_usize;
    while let Some(&Frame { message, end }) = stack.last() {
        if pos == end {
            stack.pop();
            continue;
        }
        let field = &buf[..end];
        let key = varint(field, &mut pos)?;
        let number =
            u32::try_from(key >> 3).map_err(|_| Reject::Malformed("invalid field number"))?;
        if number == 0 {
            return Err(Reject::Malformed("invalid field number"));
        }
        let wire_type = key & 7;
        match wire_type {
            0..=2 | 5 => {}
            3 | 4 => return Err(Reject::Malformed("group wire types are not supported")),
            _ => return Err(Reject::Malformed("invalid wire type")),
        }
        let kind = table
            .get(usize::from(message))
            .and_then(|m| {
                m.fields
                    .binary_search_by_key(&number, |&(n, _)| n)
                    .ok()
                    .map(|i| m.fields[i].1)
            })
            .ok_or(Reject::UnknownField { message, number })?;
        let cost = match wire_type {
            0 => {
                varint(field, &mut pos)?;
                SCALAR_SIZE
            }
            1 => {
                pos = skip(field, pos, 8)?;
                SCALAR_SIZE
            }
            5 => {
                pos = skip(field, pos, 4)?;
                SCALAR_SIZE
            }
            _ => {
                let len = varint(field, &mut pos)?;
                let len = usize::try_from(len).map_err(|_| Reject::Malformed("invalid length"))?;
                let body_end = skip(field, pos, len)?;
                match kind {
                    Field::Message(child) => {
                        // The root is at depth 0; the child is one deeper.
                        if stack.len() > limits.nesting {
                            return Err(Reject::TooDeep);
                        }
                        messages += 1;
                        if messages > limits.messages {
                            return Err(Reject::TooManyMessages);
                        }
                        stack.push(Frame {
                            message: child,
                            end: body_end,
                        });
                        deepest = deepest.max(stack.len() - 1);
                        size_of(child)
                    }
                    // A packed repeated scalar: each varint is at least one
                    // byte.
                    Field::Varint(width) => {
                        pos = body_end;
                        LEAF_SIZE.saturating_add(len.saturating_mul(usize::from(width)))
                    }
                    Field::External => {
                        flat(&buf[pos..body_end])?;
                        pos = body_end;
                        LEAF_SIZE.saturating_add(len)
                    }
                    Field::Bytes | Field::Fixed => {
                        pos = body_end;
                        LEAF_SIZE.saturating_add(len)
                    }
                }
            }
        };
        decoded = decoded.saturating_add(cost);
        if decoded > limits.decoded {
            return Err(Reject::TooLarge);
        }
    }
    Ok(deepest)
}

/// Checks the wire format of a message whose fields hold no messages
/// (`google.protobuf.Any`): no groups and no overrunning lengths. Nothing
/// is entered, so it cannot recurse.
fn flat(body: &[u8]) -> Result<(), Reject> {
    let mut pos = 0;
    while pos < body.len() {
        let key = varint(body, &mut pos)?;
        match key & 7 {
            0 => {
                varint(body, &mut pos)?;
            }
            1 => pos = skip(body, pos, 8)?,
            5 => pos = skip(body, pos, 4)?,
            2 => {
                let len = varint(body, &mut pos)?;
                let len = usize::try_from(len).map_err(|_| Reject::Malformed("invalid length"))?;
                pos = skip(body, pos, len)?;
            }
            _ => return Err(Reject::Malformed("group or invalid wire type in an Any")),
        }
    }
    Ok(())
}

/// The position `n` bytes after `pos`, which must lie within `buf`.
fn skip(buf: &[u8], pos: usize, n: usize) -> Result<usize, Reject> {
    pos.checked_add(n)
        .filter(|&end| end <= buf.len())
        .ok_or(Reject::Malformed(
            "a field runs past the end of its message",
        ))
}

/// Reads a varint of at most 10 bytes at `*pos` and advances past it.
fn varint(buf: &[u8], pos: &mut usize) -> Result<u64, Reject> {
    let mut value = 0_u64;
    for shift in (0..70).step_by(7) {
        let byte = *buf.get(*pos).ok_or(Reject::Malformed("truncated varint"))?;
        *pos += 1;
        // The tenth byte may only carry the top bit of a 64-bit value.
        if shift == 63 && byte > 1 {
            return Err(Reject::Malformed("varint overflows 64 bits"));
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(Reject::Malformed("varint is longer than 10 bytes"))
}

#[cfg(test)]
mod tests;

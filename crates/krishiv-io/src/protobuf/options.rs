// SPDX-License-Identifier: Apache-2.0
//! The options of `from_protobuf` (Spark's `ProtobufOptions`): keys are case-insensitive, values parse as Scala's `toBoolean` and
//! `toInt`; unknown keys are ignored. `to_protobuf` never reads its options.

use std::collections::BTreeMap;

use krishiv_common::error::{KrishivError, StdsInvalidOptionValueWithMessage};

/// What a malformed message does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseMode {
    /// Raise (the default).
    FailFast,
    /// NULL. Also what a mode text Spark does not know means.
    Permissive,
    /// Not supported by `from_protobuf`: raises `PARSE_MODE_UNSUPPORTED` when a message is malformed.
    DropMalformed,
}

/// The options of `from_protobuf`.
#[derive(Debug, Clone)]
pub struct ProtoOptions {
    /// `mode`.
    pub mode: ParseMode,
    /// `recursive.fields.max.depth` (default -1: recursion is an error).
    pub recursive_depth: i32,
    /// `convert.any.fields.to.json`.
    pub any_to_json: bool,
    /// `emit.default.values`.
    pub emit_defaults: bool,
    /// `enums.as.ints`.
    pub enums_as_ints: bool,
    /// `upcast.unsigned.ints`.
    pub upcast_unsigned: bool,
    /// `unwrap.primitive.wrapper.types`.
    pub unwrap_wrappers: bool,
    /// `retain.empty.message.types`.
    pub retain_empty: bool,
}

impl Default for ProtoOptions {
    fn default() -> Self {
        Self {
            mode: ParseMode::FailFast,
            recursive_depth: -1,
            any_to_json: false,
            emit_defaults: false,
            enums_as_ints: false,
            upcast_unsigned: false,
            unwrap_wrappers: false,
            retain_empty: false,
        }
    }
}

fn invalid(name: &str, value: &str, ty: &str) -> KrishivError {
    StdsInvalidOptionValueWithMessage {
        option_name: name.to_owned(),
        message: format!("Cannot cast value '{value}' to {ty}."),
    }
    .into()
}

fn boolean(opts: &BTreeMap<String, String>, name: &str) -> Result<bool, KrishivError> {
    match opts.get(name) {
        None => Ok(false),
        Some(v) if v.eq_ignore_ascii_case("true") => Ok(true),
        Some(v) if v.eq_ignore_ascii_case("false") => Ok(false),
        Some(v) => Err(invalid(name, v, "Boolean")),
    }
}

impl ProtoOptions {
    /// Reads the options; `entries` keys match case-insensitively.
    ///
    /// # Errors
    /// `STDS_INVALID_OPTION_VALUE.WITH_MESSAGE` for a value that is not a boolean (or an int for the depth).
    pub fn parse<'a>(
        entries: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, KrishivError> {
        let opts: BTreeMap<String, String> = entries
            .into_iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v.to_owned()))
            .collect();
        let mode = match opts.get("mode").map(|m| m.to_ascii_uppercase()) {
            None => ParseMode::FailFast,
            Some(m) if m == "FAILFAST" => ParseMode::FailFast,
            Some(m) if m == "DROPMALFORMED" => ParseMode::DropMalformed,
            Some(_) => ParseMode::Permissive,
        };
        // Spark reads them in this order, so the first bad value wins.
        let recursive_depth = match opts.get("recursive.fields.max.depth") {
            None => -1,
            Some(v) => {
                parse_int(v).ok_or_else(|| invalid("recursive.fields.max.depth", v, "Int"))?
            }
        };
        Ok(Self {
            mode,
            recursive_depth,
            any_to_json: boolean(&opts, "convert.any.fields.to.json")?,
            emit_defaults: boolean(&opts, "emit.default.values")?,
            enums_as_ints: boolean(&opts, "enums.as.ints")?,
            upcast_unsigned: boolean(&opts, "upcast.unsigned.ints")?,
            unwrap_wrappers: boolean(&opts, "unwrap.primitive.wrapper.types")?,
            retain_empty: boolean(&opts, "retain.empty.message.types")?,
        })
    }
}

/// Java's `Integer.parseInt`: an optional sign and decimal digits, no spaces.
fn parse_int(v: &str) -> Option<i32> {
    let digits = v.strip_prefix(['+', '-']).unwrap_or(v);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    v.strip_prefix('+').unwrap_or(v).parse().ok()
}

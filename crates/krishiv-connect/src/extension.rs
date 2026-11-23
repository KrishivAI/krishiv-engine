// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.v1` extensions of Spark Connect (P3-T015, ADR 0003 item 3, ADR 0023).
//!
//! A Krishiv client puts an extension message in a `google.protobuf.Any` slot
//! (`Relation.extension`, `Expression.extension`, `Command.extension`). [`resolve`] reads the slot
//! once: a registered `krishiv.v1` message becomes an [`Extension`]; anything else, including an
//! extension of a newer client this server does not know, is `KRISHIV_UNSUPPORTED_FEATURE` naming
//! the message and the capability, never a decode failure. The messages are `proto/krishiv/v1/krishiv.proto`
//! (a checked-in snapshot, additive only) written out as `prost` types. They are flat (no nested
//! message), so the payload goes through `wire::flat_message` before prost reads it: prost
//! recurses once per nesting level of an unknown group (the crate builds it without a recursion
//! limit), so a group is refused first. A registered message must stay flat to be decoded this way.
//!
//! A stock client never meets a `krishiv.v1` message: no response carries one unless the request did.
//! A Krishiv client learns the capability set from `system.capabilities` or by reading the
//! [`CAPABILITIES_KEY`] with `Config`.

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use prost_types::Any;

/// The protobuf package of every extension message.
pub const PACKAGE: &str = "krishiv.v1";

/// What an `Any` of this package starts with.
pub const TYPE_URL_PREFIX: &str = "type.googleapis.com/krishiv.v1.";

/// The read-only `Config` key a Krishiv client reads to learn the capabilities (comma separated).
pub const CAPABILITIES_KEY: &str = "krishiv.v1.capabilities";

/// The worked example (`Echo` in `krishiv.proto`). Its `Debug` shows the length of the text only.
#[derive(Clone, PartialEq, prost::Message)]
#[prost(skip_debug)]
pub struct Echo {
    /// The text echoed back.
    #[prost(string, tag = "1")]
    pub message: String,
}

impl std::fmt::Debug for Echo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Echo")
            .field("message_len", &self.message.len())
            .finish()
    }
}

/// `PersistPythonFunction` (`krishiv.proto`): keep (or, with `drop`, remove) the tenant's stored Python UDF `name`.
#[derive(Clone, PartialEq, Eq, prost::Message)]
pub struct PersistPythonFunction {
    /// The function's name, as registered with `register_function`.
    #[prost(string, tag = "1")]
    pub name: String,
    /// Remove the stored function instead of storing it.
    #[prost(bool, tag = "2")]
    pub drop: bool,
}

/// The slot of a Spark Connect message an extension sits in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    /// `Relation.extension`.
    Relation,
    /// `Expression.extension`.
    Expression,
    /// `Command.extension`.
    Command,
}

impl Slot {
    const fn word(self) -> &'static str {
        match self {
            Self::Relation => "relation",
            Self::Expression => "expression",
            Self::Command => "command",
        }
    }
}

/// A registered extension, decoded.
#[derive(Debug, Clone, PartialEq)]
pub enum Extension {
    /// [`Echo`].
    Echo(Echo),
    /// [`PersistPythonFunction`].
    PersistPythonFunction(PersistPythonFunction),
}

/// One registered message: its name, the capability that offers it and the slots it may sit in.
struct Registration {
    message: &'static str,
    capability: &'static str,
    slots: &'static [Slot],
}

/// Every extension this build offers. A feature's own task adds its row (and its message to
/// `krishiv.proto`); none is added here beyond the worked example.
static REGISTRY: &[Registration] = &[
    Registration {
        message: "Echo",
        capability: "echo",
        slots: &[Slot::Relation, Slot::Expression],
    },
    Registration {
        message: "PersistPythonFunction",
        capability: "persistpythonfunction",
        slots: &[Slot::Command],
    },
];

/// The capabilities this build offers, sorted.
#[must_use]
pub fn capabilities() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = REGISTRY.iter().map(|r| r.capability).collect();
    names.sort_unstable();
    names
}

/// [`capabilities`], comma separated: the value of [`CAPABILITIES_KEY`] and of the
/// `system.capabilities` row.
#[must_use]
pub fn capabilities_value() -> String {
    capabilities().join(",")
}

/// A name taken from the wire, safe to put in an error: printable ASCII, bounded.
fn shown(text: &str) -> String {
    text.chars()
        .take(96)
        .map(|c| if c.is_ascii_graphic() { c } else { '?' })
        .collect()
}

fn unsupported(text: String) -> KrishivError {
    KrishivUnsupportedFeature { feature: text }.into()
}

/// Reads the extension in `any`, for `slot`.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a message outside `krishiv.v1`, a `krishiv.v1` message this
/// server does not offer (the error names its capability), or a registered one in a slot it does not
/// sit in; `CONNECT_INVALID_PLAN.ASSERTION_FAILURE` when a registered message's bytes do not decode.
pub fn resolve(any: &Any, slot: Slot) -> Result<Extension, KrishivError> {
    let Some(name) = any.type_url.strip_prefix(TYPE_URL_PREFIX) else {
        return Err(unsupported(format!(
            "the {} extension `{}`: only `{PACKAGE}` extensions are accepted",
            slot.word(),
            shown(&any.type_url)
        )));
    };
    let Some(row) = REGISTRY.iter().find(|r| r.message == name) else {
        // The capability is the message's name in lower case, as a newer client names it.
        return Err(unsupported(format!(
            "the {PACKAGE} extension `{}` (capability `{}` is not offered by this server; \
             see system.capabilities)",
            shown(name),
            shown(&name.to_ascii_lowercase())
        )));
    };
    if !row.slots.contains(&slot) {
        return Err(unsupported(format!(
            "the {PACKAGE} extension `{}` in a {} (capability `{}` offers it elsewhere)",
            row.message,
            slot.word(),
            row.capability
        )));
    }
    decode(row, &any.value)
}

fn decode(row: &Registration, bytes: &[u8]) -> Result<Extension, KrishivError> {
    match row.message {
        "Echo" => crate::wire::decode_echo(bytes).map(Extension::Echo),
        "PersistPythonFunction" => {
            crate::wire::decode_persist_python_function(bytes).map(Extension::PersistPythonFunction)
        }
        other => Err(unsupported(format!("the {PACKAGE} extension `{other}`"))),
    }
}

#[cfg(test)]
mod tests;

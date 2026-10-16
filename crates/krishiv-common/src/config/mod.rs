//! Typed runtime configuration registry (ADR 0027).
//!
//! Every behaviour flag is a [`ConfigEntry`] in [`REGISTRY`], with a type,
//! default, docs, scope and the version it was added in. Keys are
//! `krishiv.*`; experimental features are `krishiv.experimental.*`. Spark
//! keys are accepted through each entry's `spark_alias`.
//!
//! [`Config`] holds validated values and has no global state:
//! - a registered key or Spark alias is parsed and stored under its
//!   `krishiv.*` key; an invalid value is a `KRISHIV_INVALID_CONFIG_VALUE`
//!   error;
//! - an unknown `krishiv.*` key is a `KRISHIV_UNKNOWN_CONFIG` error;
//! - any other unknown key (e.g. `spark.*`) is kept verbatim and reported as
//!   a [`ConfigWarning`], never an error.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use crate::error::{KrishivError, KrishivInvalidConfigValue, KrishivUnknownConfig};

mod registry;

pub use registry::*;

/// Where a config entry may be set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Per session; may change at runtime.
    Session,
    /// Per cluster; fixed at startup.
    Cluster,
}

impl Scope {
    fn as_str(self) -> &'static str {
        match self {
            Scope::Session => "session",
            Scope::Cluster => "cluster",
        }
    }
}

/// The type of a config value and how its string form is parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    /// `true` / `false`, ASCII case-insensitive, surrounding whitespace ignored.
    Bool,
    /// A signed 64-bit integer.
    Int,
    /// A byte count: a number with an optional binary unit suffix
    /// `b`, `k`/`kb`, `m`/`mb`, `g`/`gb`, `t`/`tb`, `p`/`pb` (as Spark);
    /// no suffix means bytes.
    Size,
    /// A duration: a number with an optional suffix `us`, `ms`, `s`,
    /// `m`/`min`, `h`, `d` (as Spark); no suffix means seconds.
    Duration,
    /// Any string, stored verbatim.
    String,
    /// One of the listed values, matched ASCII case-insensitively and
    /// stored in the listed spelling.
    Enum(&'static [&'static str]),
}

impl ValueType {
    /// Parses `raw`, returning a human-readable reason on failure.
    pub(crate) fn parse(&self, raw: &str) -> Result<ConfigValue, String> {
        let s = raw.trim();
        match self {
            ValueType::Bool => {
                if s.eq_ignore_ascii_case("true") {
                    Ok(ConfigValue::Bool(true))
                } else if s.eq_ignore_ascii_case("false") {
                    Ok(ConfigValue::Bool(false))
                } else {
                    Err("expected `true` or `false`".to_owned())
                }
            }
            ValueType::Int => s
                .parse()
                .map(ConfigValue::Int)
                .map_err(|_| "expected an integer".to_owned()),
            ValueType::Size => parse_with_unit(s, SIZE_UNITS, "b")
                .map(ConfigValue::Size)
                .ok_or_else(|| "expected a byte size such as `128m` or `1g`".to_owned()),
            ValueType::Duration => parse_with_unit(s, DURATION_UNITS_US, "s")
                .map(|us| ConfigValue::Duration(Duration::from_micros(us)))
                .ok_or_else(|| "expected a duration such as `500ms` or `120s`".to_owned()),
            ValueType::String => Ok(ConfigValue::String(Cow::Owned(raw.to_owned()))),
            ValueType::Enum(allowed) => allowed
                .iter()
                .find(|a| a.eq_ignore_ascii_case(s))
                .map(|a| ConfigValue::String(Cow::Borrowed(*a)))
                .ok_or_else(|| format!("expected one of {}", allowed.join(", "))),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::Bool => f.write_str("bool"),
            ValueType::Int => f.write_str("int"),
            ValueType::Size => f.write_str("size"),
            ValueType::Duration => f.write_str("duration"),
            ValueType::String => f.write_str("string"),
            ValueType::Enum(allowed) => {
                let list: Vec<String> = allowed.iter().map(|a| format!("`{a}`")).collect();
                write!(f, "enum ({})", list.join(", "))
            }
        }
    }
}

/// Unit suffixes with their multiplier (binary, as Spark's `byteStringAs`).
const SIZE_UNITS: &[(&str, u64)] = &[
    ("b", 1),
    ("k", 1 << 10),
    ("kb", 1 << 10),
    ("m", 1 << 20),
    ("mb", 1 << 20),
    ("g", 1 << 30),
    ("gb", 1 << 30),
    ("t", 1 << 40),
    ("tb", 1 << 40),
    ("p", 1 << 50),
    ("pb", 1 << 50),
];

/// Duration suffixes with their length in microseconds.
const DURATION_UNITS_US: &[(&str, u64)] = &[
    ("us", 1),
    ("ms", 1_000),
    ("s", 1_000_000),
    ("m", 60_000_000),
    ("min", 60_000_000),
    ("h", 3_600_000_000),
    ("d", 86_400_000_000),
];

/// Parses `<digits><unit>?` (unit ASCII case-insensitive), checking overflow.
fn parse_with_unit(s: &str, units: &[(&str, u64)], default_unit: &str) -> Option<u64> {
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (digits, unit) = s.split_at(split);
    let unit = if unit.is_empty() { default_unit } else { unit };
    let mult = units
        .iter()
        .find(|(u, _)| u.eq_ignore_ascii_case(unit))
        .map(|(_, m)| *m)?;
    digits.parse::<u64>().ok()?.checked_mul(mult)
}

/// A typed config value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigValue {
    /// A [`ValueType::Bool`] value.
    Bool(bool),
    /// A [`ValueType::Int`] value.
    Int(i64),
    /// A [`ValueType::Size`] value, in bytes.
    Size(u64),
    /// A [`ValueType::Duration`] value.
    Duration(Duration),
    /// A [`ValueType::String`] or [`ValueType::Enum`] value.
    String(Cow<'static, str>),
}

impl ConfigValue {
    /// The value if it is a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The value if it is an int.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ConfigValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// The value in bytes if it is a size.
    pub fn as_size(&self) -> Option<u64> {
        match self {
            ConfigValue::Size(n) => Some(*n),
            _ => None,
        }
    }

    /// The value if it is a duration.
    pub fn as_duration(&self) -> Option<Duration> {
        match self {
            ConfigValue::Duration(d) => Some(*d),
            _ => None,
        }
    }

    /// The value if it is a string or enum.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Formats the value in a form its [`ValueType`] parses back.
impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValue::Bool(b) => write!(f, "{b}"),
            ConfigValue::Int(i) => write!(f, "{i}"),
            ConfigValue::Size(n) => {
                let unit = [
                    ("PB", 1u64 << 50),
                    ("TB", 1 << 40),
                    ("GB", 1 << 30),
                    ("MB", 1 << 20),
                    ("KB", 1 << 10),
                ]
                .into_iter()
                .find(|(_, m)| *n != 0 && n % m == 0);
                match unit {
                    Some((u, m)) => write!(f, "{}{u}", n / m),
                    None => write!(f, "{n}B"),
                }
            }
            ConfigValue::Duration(d) => {
                let us = d.as_micros();
                let unit = [
                    ("d", 86_400_000_000u128),
                    ("h", 3_600_000_000),
                    ("min", 60_000_000),
                    ("s", 1_000_000),
                    ("ms", 1_000),
                ]
                .into_iter()
                .find(|(_, m)| us != 0 && us % m == 0);
                match unit {
                    Some((u, m)) => write!(f, "{}{u}", us / m),
                    None => write!(f, "{us}us"),
                }
            }
            ConfigValue::String(s) => f.write_str(s),
        }
    }
}

/// A registered configuration entry.
#[derive(Debug)]
pub struct ConfigEntry {
    /// Canonical key, `krishiv.*`.
    pub key: &'static str,
    /// Value type.
    pub value_type: ValueType,
    /// Default value; `None` means unset (the doc says what then applies).
    pub default: Option<ConfigValue>,
    /// One-paragraph description.
    pub doc: &'static str,
    /// Session or cluster scope.
    pub scope: Scope,
    /// Krishiv version that added the entry.
    pub version_added: &'static str,
    /// Spark key accepted as an alias, if any.
    pub spark_alias: Option<&'static str>,
}

impl ConfigEntry {
    /// Parses `raw` as this entry's type. `key` is the key as the user
    /// wrote it, used in the error message.
    pub fn parse(&self, key: &str, raw: &str) -> Result<ConfigValue, KrishivError> {
        self.value_type.parse(raw).map_err(|reason| {
            KrishivInvalidConfigValue {
                value: format!("'{raw}'"),
                key: format!("\"{key}\""),
                reason,
            }
            .into()
        })
    }
}

/// Finds the entry for a `krishiv.*` key or a Spark alias.
pub fn lookup(key: &str) -> Option<&'static ConfigEntry> {
    REGISTRY
        .iter()
        .copied()
        .find(|e| e.key == key || e.spark_alias == Some(key))
}

/// A non-fatal problem found while setting config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigWarning {
    /// The key is not registered; its value was kept but has no effect.
    UnknownKey {
        /// The key as given.
        key: String,
    },
}

impl fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigWarning::UnknownKey { key } => {
                write!(
                    f,
                    "config {key} is not recognised by Krishiv and has no effect"
                )
            }
        }
    }
}

/// A set of validated config values; unset entries fall back to defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Config {
    values: BTreeMap<&'static str, ConfigValue>,
    unregistered: BTreeMap<String, String>,
}

impl Config {
    /// An empty config: every entry has its default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a config from key/value pairs, collecting warnings. Stops at
    /// the first error.
    pub fn from_pairs<I, K, V>(pairs: I) -> Result<(Self, Vec<ConfigWarning>), KrishivError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut config = Self::new();
        let mut warnings = Vec::new();
        for (k, v) in pairs {
            warnings.extend(config.set(k.as_ref(), v.as_ref())?);
        }
        Ok((config, warnings))
    }

    /// Sets `key` (a `krishiv.*` key, a Spark alias or another key) to
    /// `value`. On error the config is unchanged.
    pub fn set(&mut self, key: &str, value: &str) -> Result<Option<ConfigWarning>, KrishivError> {
        if let Some(entry) = lookup(key) {
            let v = entry.parse(key, value)?;
            self.values.insert(entry.key, v);
            Ok(None)
        } else if key.starts_with("krishiv.") {
            Err(KrishivUnknownConfig {
                key: format!("\"{key}\""),
            }
            .into())
        } else {
            self.unregistered.insert(key.to_owned(), value.to_owned());
            Ok(Some(ConfigWarning::UnknownKey {
                key: key.to_owned(),
            }))
        }
    }

    /// The value of `entry`: the set value, else its default.
    pub fn get<'a>(&'a self, entry: &'a ConfigEntry) -> Option<&'a ConfigValue> {
        self.values.get(entry.key).or(entry.default.as_ref())
    }

    /// The raw value of an unregistered key (e.g. an unknown `spark.*` key).
    pub fn get_unregistered(&self, key: &str) -> Option<&str> {
        self.unregistered.get(key).map(String::as_str)
    }
}

/// Renders the config reference as Markdown (`cargo xtask docs-config`).
pub fn docs_markdown() -> String {
    let mut entries: Vec<&ConfigEntry> = REGISTRY.to_vec();
    entries.sort_by_key(|e| e.key);
    let mut out = String::from(
        "# Krishiv configuration reference\n\n\
         <!-- @generated by `cargo xtask docs-config`; do not edit. -->\n\n\
         Keys under `krishiv.experimental.` are experimental and off by default. \
         Spark aliases are accepted in place of the Krishiv key; other unknown \
         `spark.*` keys produce a warning, unknown `krishiv.*` keys an error.\n\n\
         | Key | Type | Default | Scope | Since | Spark alias | Description |\n\
         |---|---|---|---|---|---|---|\n",
    );
    for e in entries {
        let default = e
            .default
            .as_ref()
            .map_or_else(|| "(none)".to_owned(), |d| format!("`{d}`"));
        let alias = e.spark_alias.map(|a| format!("`{a}`")).unwrap_or_default();
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} | {} |\n",
            e.key,
            e.value_type,
            default,
            e.scope.as_str(),
            e.version_added,
            alias,
            e.doc.replace('|', "\\|"),
        ));
    }
    out
}

#[cfg(test)]
mod tests;

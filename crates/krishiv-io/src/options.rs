// SPDX-License-Identifier: Apache-2.0
//! Typed data source options (ADR 0038 values, Spark names).
//!
//! A format declares its options once in a static [`OptionSchema`]: Spark's
//! option names (matched ASCII case-insensitively, as Spark's
//! `CaseInsensitiveStringMap`), types, defaults, whether the value is a
//! secret, and the session conf a missing option falls back to
//! (`spark.sql.parquet.*`). [`OptionSchema::parse`] turns the user's
//! options into [`FormatOptions`] once, at planning.
//!
//! **Unknown options** are accepted, as in Spark, with a warning that names
//! the closest known option ("did you mean"); the ignored keys are kept for
//! `EXPLAIN`. With `krishiv.io.strictOptions=true` they are an error.
//! Options handled by the file-source framework itself
//! ([`FILE_SOURCE_OPTIONS`]) are never reported. Object-store configuration
//! (`fs.*`, resolved by the store layer from the same option map) is not a
//! format option either, but it is kept for `EXPLAIN`: non-secret values
//! print as given, secrets print redacted, and it is never an error in
//! strict mode.
//!
//! **Secrets** stay [`SecretValue`]s: `Debug` prints them redacted, and the
//! plaintext is reached only through [`FormatOptions::resolve_secret`]
//! inside `read_metadata`, `open` or `writer`.

use std::collections::BTreeMap;
use std::fmt;

use krishiv_common::config::{Config, IO_STRICT_OPTIONS};
use krishiv_common::error::{KrishivInvalidFormatOptionValue, KrishivUnknownFormatOption};
use krishiv_common::secret::{OptionValue, SecretString, SecretValue};

use crate::error::{Result, unsupported};
use crate::io::SecretResolver;
use crate::io::stores::is_store_option;

/// Options every file source accepts, handled by the framework (listing,
/// partition discovery, corrupt/missing file handling), never by a format.
pub const FILE_SOURCE_OPTIONS: &[&str] = &[
    "path",
    "paths",
    "basePath",
    "pathGlobFilter",
    "recursiveFileLookup",
    "modifiedBefore",
    "modifiedAfter",
    "ignoreCorruptFiles",
    "ignoreMissingFiles",
    "ignoreInvalidPartitionPaths",
    "maxRecordsPerFile",
    "timeZone",
];

/// The type of an option value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionKind {
    /// `true` / `false`, case-insensitive.
    Bool,
    /// An integer within `min..=max`.
    Int {
        /// Smallest value.
        min: i64,
        /// Largest value.
        max: i64,
    },
    /// Any string.
    String,
    /// One of the listed values, case-insensitive, stored in the listed spelling.
    Enum(&'static [&'static str]),
    /// A secret (an encryption key, a password): never printed.
    Secret,
}

/// The default of an option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionDefault {
    /// No default: the option is unset unless given.
    None,
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A string or enum value.
    Str(&'static str),
}

/// One option of a format.
#[derive(Debug, Clone, Copy)]
pub struct OptionSpec {
    /// Spark's name, e.g. `mergeSchema`.
    pub name: &'static str,
    /// The value type.
    pub kind: OptionKind,
    /// The default when neither the option nor its session conf is set.
    pub default: OptionDefault,
    /// The session conf used when the option is not given, e.g.
    /// `spark.sql.parquet.mergeSchema`.
    pub conf_fallback: Option<&'static str>,
    /// Whether the value changes how file metadata decodes (a footer
    /// decryption key): its fingerprint joins the metadata cache key.
    pub metadata_key: bool,
    /// One-line description.
    pub doc: &'static str,
}

/// The options of one format.
#[derive(Debug)]
pub struct OptionSchema {
    /// The format's name, for messages.
    pub format: &'static str,
    /// The options.
    pub options: &'static [OptionSpec],
}

/// A non-fatal problem found while parsing options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionWarning {
    /// The option is not known to the format; it is ignored.
    Unknown {
        /// The key as given.
        key: String,
        /// The closest known option, if one is close.
        suggestion: Option<&'static str>,
    },
}

impl fmt::Display for OptionWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { key, suggestion } => {
                write!(f, "option {key} is not known and is ignored")?;
                if let Some(s) = suggestion {
                    write!(f, "; did you mean {s}?")?;
                }
                Ok(())
            }
        }
    }
}

/// A parsed option value.
#[derive(Clone, PartialEq, Eq)]
enum Value {
    Bool(bool),
    Int(i64),
    Str(String),
    Secret(SecretValue),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(b) => b.fmt(f),
            Self::Int(i) => i.fmt(f),
            Self::Str(s) => s.fmt(f),
            // `SecretValue`'s Display never prints plaintext.
            Self::Secret(s) => write!(f, "{s}"),
        }
    }
}

/// The parsed options of one scan or write: typed, defaults applied.
#[derive(Clone)]
pub struct FormatOptions {
    schema: &'static OptionSchema,
    values: BTreeMap<&'static str, Value>,
    ignored: Vec<String>,
}

impl fmt::Debug for FormatOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FormatOptions")
            .field("format", &self.schema.format)
            .field("values", &self.values)
            .field("ignored", &self.ignored)
            .finish()
    }
}

impl OptionSchema {
    /// The spec of `name` (case-insensitive).
    #[must_use]
    pub fn spec(&self, name: &str) -> Option<&'static OptionSpec> {
        self.options
            .iter()
            .find(|o| o.name.eq_ignore_ascii_case(name))
    }

    /// Parses `user` options against this schema, with `session` for conf
    /// fallbacks and `krishiv.io.strictOptions`.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` for a value of the wrong type;
    /// `KRISHIV_UNKNOWN_FORMAT_OPTION` for an unknown option in strict mode.
    pub fn parse(
        &'static self,
        user: &BTreeMap<String, OptionValue>,
        session: &Config,
    ) -> Result<(FormatOptions, Vec<OptionWarning>)> {
        let strict = session
            .get(&IO_STRICT_OPTIONS)
            .and_then(krishiv_common::config::ConfigValue::as_bool)
            .unwrap_or(false);
        let mut values = BTreeMap::new();
        let mut ignored = Vec::new();
        let mut warnings = Vec::new();
        for (key, value) in user {
            if let Some(spec) = self.spec(key) {
                values.insert(spec.name, self.typed(spec, key, value)?);
            } else if is_store_option(key) {
                // Object-store configuration (`fs.s3a.*`, `fs.gs.*`,
                // `fs.azure.*`): resolved by the store layer from the same
                // option map, never a format option and never an error,
                // even in strict mode. Kept for `EXPLAIN` as `key=value`:
                // `OptionValue`'s `Display` prints plain values as given
                // and secrets redacted, so only non-secrets are visible.
                ignored.push(format!("{key}={value}"));
            } else if !FILE_SOURCE_OPTIONS
                .iter()
                .any(|o| o.eq_ignore_ascii_case(key))
            {
                let suggestion = self.closest(key);
                if strict {
                    return Err(KrishivUnknownFormatOption {
                        option: format!("\"{key}\""),
                        format: format!("\"{}\"", self.format),
                        suggestion: suggestion
                            .map(|s| format!(" Did you mean \"{s}\"?"))
                            .unwrap_or_default(),
                    }
                    .into());
                }
                let w = OptionWarning::Unknown {
                    key: key.clone(),
                    suggestion,
                };
                tracing::warn!(format = self.format, "{w}");
                warnings.push(w);
                ignored.push(key.clone());
            }
        }
        for spec in self.options {
            if values.contains_key(spec.name) {
                continue;
            }
            let fallback = spec.conf_fallback.and_then(|k| conf_value(session, k));
            if let Some(raw) = fallback {
                let v = OptionValue::from_user(spec.name, raw);
                values.insert(spec.name, self.typed(spec, spec.name, &v)?);
            } else if let Some(v) = default_value(spec.default) {
                values.insert(spec.name, v);
            }
        }
        Ok((
            FormatOptions {
                schema: self,
                values,
                ignored,
            },
            warnings,
        ))
    }

    fn typed(&self, spec: &OptionSpec, key: &str, value: &OptionValue) -> Result<Value> {
        if spec.kind == OptionKind::Secret {
            return Ok(Value::Secret(match value {
                OptionValue::Secret(s) => s.clone(),
                // A secret option given as a plain value or URL is still a
                // secret: never keep its text printable.
                other => SecretValue::Inline(SecretString::new(match other {
                    OptionValue::Url(u) => u.expose_secret().expose_secret().to_owned(),
                    OptionValue::Plain(p) => p.clone(),
                    OptionValue::Secret(_) => String::new(),
                })),
            }));
        }
        let Some(raw) = value.as_plain() else {
            return Err(self.invalid(key, "<secret>", "a secret is not allowed here"));
        };
        let s = raw.trim();
        match spec.kind {
            OptionKind::Bool => {
                if s.eq_ignore_ascii_case("true") {
                    Ok(Value::Bool(true))
                } else if s.eq_ignore_ascii_case("false") {
                    Ok(Value::Bool(false))
                } else {
                    Err(self.invalid(key, raw, "expected `true` or `false`"))
                }
            }
            OptionKind::Int { min, max } => s
                .parse::<i64>()
                .ok()
                .filter(|v| (min..=max).contains(v))
                .map(Value::Int)
                .ok_or_else(|| {
                    self.invalid(key, raw, &format!("expected an integer in {min}..={max}"))
                }),
            OptionKind::String => Ok(Value::Str(raw.to_owned())),
            OptionKind::Enum(allowed) => allowed
                .iter()
                .find(|a| a.eq_ignore_ascii_case(s))
                .map(|a| Value::Str((*a).to_owned()))
                .ok_or_else(|| {
                    self.invalid(key, raw, &format!("expected one of {}", allowed.join(", ")))
                }),
            OptionKind::Secret => Err(self.invalid(key, raw, "unreachable")),
        }
    }

    fn invalid(&self, key: &str, value: &str, reason: &str) -> krishiv_common::error::KrishivError {
        KrishivInvalidFormatOptionValue {
            value: format!("'{value}'"),
            option: format!("\"{key}\""),
            format: format!("\"{}\"", self.format),
            reason: reason.to_owned(),
        }
        .into()
    }

    /// The known option closest to `key` (case-insensitive edit distance at
    /// most 2, or a third of the key's length).
    fn closest(&self, key: &str) -> Option<&'static str> {
        let key = key.to_ascii_lowercase();
        self.options
            .iter()
            .map(|o| o.name)
            .chain(FILE_SOURCE_OPTIONS.iter().copied())
            .map(|n| (edit_distance(&key, &n.to_ascii_lowercase()), n))
            .filter(|(d, _)| *d <= 2.max(key.len() / 3))
            .min_by_key(|(d, _)| *d)
            .map(|(_, n)| n)
    }
}

/// The session value of `key`: a registered key (or Spark alias) through the
/// registry, any other key as given.
fn conf_value(session: &Config, key: &str) -> Option<String> {
    match krishiv_common::config::lookup(key) {
        Some(entry) => session.get(entry).map(ToString::to_string),
        None => session.get_unregistered(key).map(str::to_owned),
    }
}

fn default_value(d: OptionDefault) -> Option<Value> {
    match d {
        OptionDefault::None => None,
        OptionDefault::Bool(b) => Some(Value::Bool(b)),
        OptionDefault::Int(i) => Some(Value::Int(i)),
        OptionDefault::Str(s) => Some(Value::Str(s.to_owned())),
    }
}

/// Levenshtein distance over bytes.
fn edit_distance(a: &str, b: &str) -> usize {
    let b = b.as_bytes();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.bytes().enumerate() {
        let mut cur = Vec::with_capacity(b.len() + 1);
        cur.push(i + 1);
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != *cb);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

impl FormatOptions {
    /// The options of `schema` with every default applied (no user options,
    /// no session conf).
    #[must_use]
    pub fn defaults(schema: &'static OptionSchema) -> Self {
        let values = schema
            .options
            .iter()
            .filter_map(|o| default_value(o.default).map(|v| (o.name, v)))
            .collect();
        Self {
            schema,
            values,
            ignored: Vec::new(),
        }
    }

    /// The schema these options were parsed with.
    #[must_use]
    pub fn schema(&self) -> &'static OptionSchema {
        self.schema
    }

    /// A boolean option (`None` when unset or of another type).
    #[must_use]
    pub fn bool(&self, name: &str) -> Option<bool> {
        match self.values.get(self.schema.spec(name)?.name)? {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// An integer option.
    #[must_use]
    pub fn int(&self, name: &str) -> Option<i64> {
        match self.values.get(self.schema.spec(name)?.name)? {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// A string or enum option.
    #[must_use]
    pub fn str(&self, name: &str) -> Option<&str> {
        match self.values.get(self.schema.spec(name)?.name)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Whether the secret option `name` is set.
    #[must_use]
    pub fn has_secret(&self, name: &str) -> bool {
        self.schema
            .spec(name)
            .is_some_and(|s| matches!(self.values.get(s.name), Some(Value::Secret(_))))
    }

    /// The plaintext of the secret option `name`, resolving a
    /// `secret('scope', 'key')` reference through `secrets`. Call only
    /// inside `read_metadata`, `open` or `writer`, and drop the result
    /// with the decoder or encoder.
    ///
    /// # Errors
    /// What the resolver returns for a reference it cannot resolve.
    pub async fn resolve_secret(
        &self,
        name: &str,
        secrets: &dyn SecretResolver,
    ) -> Result<Option<SecretString>> {
        let Some(spec) = self.schema.spec(name) else {
            return Ok(None);
        };
        match self.values.get(spec.name) {
            Some(Value::Secret(SecretValue::Inline(s))) => Ok(Some(s.clone())),
            Some(Value::Secret(SecretValue::Reference { scope, key })) => {
                secrets.resolve(scope, key).await.map(Some)
            }
            _ => Ok(None),
        }
    }

    /// The fingerprint of every option that changes how metadata decodes
    /// ([`OptionSpec::metadata_key`]), or `None` when none is set: joined to
    /// the metadata cache key so another key never hits an entry, without
    /// storing plaintext.
    ///
    /// # Errors
    /// What the resolver returns for a reference it cannot resolve.
    pub async fn metadata_fingerprint(
        &self,
        secrets: &dyn SecretResolver,
    ) -> Result<Option<[u8; 16]>> {
        let mut material = String::new();
        let mut any = false;
        for spec in self.schema.options.iter().filter(|o| o.metadata_key) {
            let text = match self.values.get(spec.name) {
                None => continue,
                Some(Value::Secret(_)) => match self.resolve_secret(spec.name, secrets).await? {
                    Some(s) => s,
                    None => continue,
                },
                Some(other) => SecretString::new(format!("{other:?}")),
            };
            any = true;
            // Only the value's keyed fingerprint joins the material, so no
            // plaintext copy is left in an unzeroed buffer.
            material.push_str(&format!(
                "{}:{}:{:?};",
                spec.name.len(),
                spec.name,
                text.fingerprint()
            ));
        }
        Ok(any.then(|| SecretString::new(material).fingerprint()))
    }

    /// Options given but not known to the format (shown by `EXPLAIN`),
    /// including object-store configuration (`fs.*`) kept as `key=value`
    /// with secrets redacted.
    #[must_use]
    pub fn ignored(&self) -> &[String] {
        &self.ignored
    }
}

/// A [`SecretResolver`] that resolves nothing: secret references need the
/// `SecretProvider` of ADR 0036 R12 (P4).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NoSecrets;

impl SecretResolver for NoSecrets {
    fn resolve<'a>(
        &'a self,
        _scope: &'a str,
        _key: &'a str,
    ) -> futures::future::BoxFuture<'a, Result<SecretString>> {
        Box::pin(async {
            Err(unsupported(
                "secret('scope', 'key') references in file options",
            ))
        })
    }
}

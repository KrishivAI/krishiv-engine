// SPDX-License-Identifier: Apache-2.0
//! Secret references in server settings (ADR 0038): `env:NAME` or `file:/path`, never a value
//! in a configuration dump. A `file:` secret readable by group or others refuses startup
//! (threat model T-A3).

use std::path::PathBuf;

use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue};
use krishiv_common::secret::SecretString;

/// Where a secret comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SecretRef {
    /// `env:NAME`: the environment variable `NAME`.
    Env(String),
    /// `file:/path`: the file's contents, one trailing line break removed.
    File(PathBuf),
    /// A value given inline. Only Spark's `spark.connect.authenticate.token` takes one, so a
    /// Spark configuration works unchanged.
    Inline(SecretString),
}

impl SecretRef {
    /// Parses the setting `key = raw`. Without `allow_inline` a value that is not a reference
    /// is refused (its text is never echoed).
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for an empty reference or an inline value where none is
    /// allowed.
    pub fn parse(key: &str, raw: &str, allow_inline: bool) -> Result<Self, KrishivError> {
        if let Some(name) = raw.strip_prefix("env:") {
            if name.is_empty() {
                return Err(invalid(
                    key,
                    "env:",
                    "an environment variable name is required",
                ));
            }
            return Ok(Self::Env(name.to_owned()));
        }
        if let Some(path) = raw.strip_prefix("file:") {
            if path.is_empty() {
                return Err(invalid(key, "file:", "a file path is required"));
            }
            return Ok(Self::File(PathBuf::from(path)));
        }
        if allow_inline {
            return Ok(Self::Inline(SecretString::new(raw)));
        }
        Err(invalid(
            key,
            krishiv_common::secret::REDACTED,
            "a secret must be a reference, env:NAME or file:/path (ADR 0038)",
        ))
    }

    /// The secret.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` when the variable is unset, the file cannot be read, or
    /// the file is readable by group or others.
    pub fn resolve(&self, key: &str) -> Result<SecretString, KrishivError> {
        match self {
            Self::Env(name) => std::env::var(name).map(SecretString::new).map_err(|_| {
                invalid(
                    key,
                    &format!("env:{name}"),
                    "the environment variable is not set",
                )
            }),
            Self::File(path) => {
                use std::io::Read as _;
                let shown = format!("file:{}", path.display());
                let unreadable = |e: std::io::Error| {
                    invalid(key, &shown, &format!("cannot be read: {}", e.kind()))
                };
                // The path is checked before it is opened, so a FIFO or device is refused
                // instead of blocking startup in `open`; the handle is checked again, so the file
                // checked is the file read.
                check_private(key, &shown, &std::fs::metadata(path).map_err(unreadable)?)?;
                let file = std::fs::File::open(path).map_err(unreadable)?;
                check_private(key, &shown, &file.metadata().map_err(unreadable)?)?;
                // One byte past the token cap is enough to refuse an oversized secret.
                let cap = super::tokens::MAX_SECRET_BYTES as u64 + 2;
                let mut text = String::new();
                file.take(cap)
                    .read_to_string(&mut text)
                    .map_err(unreadable)?;
                if text.ends_with('\n') {
                    text.pop();
                    if text.ends_with('\r') {
                        text.pop();
                    }
                }
                Ok(SecretString::new(text))
            }
            Self::Inline(secret) => Ok(secret.clone()),
        }
    }

    /// Whether this is an inline value.
    #[must_use]
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline(_))
    }
}

/// Refuses a secret file that group or others may read.
#[cfg(unix)]
fn check_private(key: &str, shown: &str, meta: &std::fs::Metadata) -> Result<(), KrishivError> {
    use std::os::unix::fs::PermissionsExt;
    if !meta.is_file() {
        return Err(invalid(key, shown, "not a regular file"));
    }
    if meta.permissions().mode() & 0o077 != 0 {
        return Err(invalid(
            key,
            shown,
            "the file is readable by group or others; make it mode 0600 or 0400",
        ));
    }
    Ok(())
}

/// Other platforms have no mode bits to check.
#[cfg(not(unix))]
fn check_private(_key: &str, _shown: &str, _meta: &std::fs::Metadata) -> Result<(), KrishivError> {
    Ok(())
}

/// `KRISHIV_INVALID_CONFIG_VALUE` for `key`; `shown` must never be a secret.
pub(crate) fn invalid(key: &str, shown: &str, reason: &str) -> KrishivError {
    KrishivInvalidConfigValue {
        value: format!("'{shown}'"),
        key: format!("\"{key}\""),
        reason: reason.to_owned(),
    }
    .into()
}

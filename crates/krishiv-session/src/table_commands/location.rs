// SPDX-License-Identifier: Apache-2.0
//! The location of a namespace, qualified as Spark qualifies it.

/// Spark's `CatalogUtils.makeQualifiedDBObjectPath`: a location without a scheme is resolved against the
/// warehouse directory (an absolute path stays itself) and qualified as a `file:` URI. Every location,
/// whatever its scheme, is then normalised as Hadoop's `Path` does: no empty, `.` or trailing segment, a
/// `..` removes the segment before it and is kept when there is none; the scheme and authority stay as
/// written (Spark 4.2.0, `new Path(..)`: `s3a://b/x//y/` is `s3a://b/x/y`, `file:///a/b//c/` is
/// `file:/a/b/c`, `/../x` is `file:/../x`; the root slash of `s3a://b/` stays).
/// No other trailing slash remains. An empty location is `INVALID_EMPTY_LOCATION`. A scheme whose
/// file system Spark cannot open (`s3a`, `hdfs` without the connector) is refused by Spark itself; here it is
/// only normalised, which is not checked against a real Spark run with that file system.
pub(super) fn qualified_namespace_location(
    location: &str,
    warehouse: impl FnOnce() -> Result<std::path::PathBuf, krishiv_common::error::KrishivError>,
) -> Result<String, krishiv_common::error::KrishivError> {
    if location.is_empty() {
        return Err(krishiv_common::error::InvalidEmptyLocation {
            location: location.to_owned(),
        }
        .into());
    }
    // On Windows a schemeless location is a local path with `\` as `/` (Hadoop's `normalizePath`), and a
    // drive (`C:\x`, `\\?\C:\x`) is absolute, never a scheme `C`.
    let local = windows_local(location)?;
    let location = local.as_deref().unwrap_or(location);
    let scheme = location.split_once(':').filter(|(s, _)| {
        s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    });
    let (prefix, path) = match scheme {
        Some((s, rest)) => match rest.strip_prefix("//") {
            Some(after) => {
                let (authority, path) = after.split_at(after.find('/').unwrap_or(after.len()));
                if authority.is_empty() {
                    (format!("{s}:"), path.to_owned())
                } else {
                    (format!("{s}://{authority}"), path.to_owned())
                }
            }
            None => (format!("{s}:"), rest.to_owned()),
        },
        // Only a relative location needs the warehouse (and may create its directory).
        None if location.starts_with('/') => ("file:".to_owned(), location.to_owned()),
        None => {
            let joined = warehouse()?.join(location).display().to_string();
            // `Path::join` puts a `\` before the location on Windows; the URI's separator is `/`.
            let joined = windows_local(&joined)?.unwrap_or(joined);
            ("file:".to_owned(), joined)
        }
    };
    let mut parts: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            // Java's `URI.normalize` keeps a `..` that has nothing left to remove: `/../x` is
            // `file:/../x` (Spark 4.2.0).
            ".." => {
                if parts.last().is_some_and(|p| *p != "..") {
                    parts.pop();
                } else {
                    parts.push("..");
                }
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() && prefix.contains("//") {
        // Hadoop's `Path` keeps the root slash of `s3a://b/` and `s3a://b//` (Spark 4.2.0 JVM) and
        // adds none to `s3a://b`.
        return Ok(if path.is_empty() {
            prefix
        } else {
            format!("{prefix}/")
        });
    }
    Ok(format!("{prefix}/{}", parts.join("/")))
}

/// On Windows, a location without a URL scheme in the form the locator reads: separators `/`, a drive
/// (`C:\x`, `C:/x`, verbatim `\\?\C:\x`) led by `/` (`/C:/x`); `None` for a URL and on every other
/// platform.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a UNC location (`\\srv\share\x`, `\\?\UNC\..`) and a
/// drive-relative one (`C:x`): neither is placed in the warehouse or on the current drive.
fn windows_local(text: &str) -> Result<Option<String>, krishiv_common::error::KrishivError> {
    if cfg!(windows) {
        windows_form(text)
    } else {
        Ok(None)
    }
}

/// [`windows_local`]'s rule on any platform (tested everywhere).
pub(super) fn windows_form(
    text: &str,
) -> Result<Option<String>, krishiv_common::error::KrishivError> {
    let url = text.split_once(':').is_some_and(|(s, _)| {
        s.len() > 1
            && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    });
    if url {
        return Ok(None);
    }
    let slashed = text.replace('\\', "/");
    let led = slashed.strip_prefix("//?/").unwrap_or(&slashed);
    let refuse = |what: &str| {
        krishiv_common::error::KrishivError::from(
            krishiv_common::error::KrishivUnsupportedFeature {
                feature: format!("{what} namespace locations"),
            },
        )
    };
    if led.starts_with("//") || (slashed.starts_with("//?/") && led.starts_with("UNC/")) {
        return Err(refuse("UNC"));
    }
    let b = led.as_bytes();
    if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        if b.len() > 2 && b[2] != b'/' {
            return Err(refuse("drive-relative"));
        }
        return Ok(Some(format!("/{led}")));
    }
    Ok(Some(slashed))
}

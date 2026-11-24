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
    // On Windows a drive path (`C:\x`, `\\?\C:\x`) is an absolute local path, never a scheme `C`.
    let drive = drive_led(location);
    let location = drive.as_deref().unwrap_or(location);
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
            let joined = if cfg!(windows) {
                joined.replace('\\', "/")
            } else {
                joined
            };
            ("file:".to_owned(), drive_led(&joined).unwrap_or(joined))
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

/// On Windows, a drive-led local path (`C:\x`, `C:/x`, verbatim `\\?\C:\x`) in the form the
/// locator reads, `/C:/x` (separators `/`); `None` for anything else and on every other platform.
fn drive_led(text: &str) -> Option<String> {
    if !cfg!(windows) {
        return None;
    }
    let slashed = text.replace('\\', "/");
    let led = slashed.strip_prefix("//?/").unwrap_or(&slashed);
    let b = led.as_bytes();
    let is_drive = b.len() >= 2
        && b[0].is_ascii_alphabetic()
        && b[1] == b':'
        && (b.len() == 2 || b[2] == b'/');
    is_drive.then(|| format!("/{led}"))
}

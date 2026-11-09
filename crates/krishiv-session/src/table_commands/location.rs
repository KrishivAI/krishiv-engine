// SPDX-License-Identifier: Apache-2.0
//! The location of a namespace, qualified as Spark qualifies it.

/// Spark's `CatalogUtils.makeQualifiedDBObjectPath`: a location without a scheme is resolved against the
/// warehouse directory (an absolute path stays itself) and qualified as a `file:` URI. Every location,
/// whatever its scheme, is then normalised as Hadoop's `Path` does: no empty, `.` or trailing segment, a
/// `..` removes the segment before it and is kept when there is none; the scheme and authority stay as
/// written (Spark 4.2.0, `new Path(..)`: `s3a://b/x//y/` is `s3a://b/x/y`, `file:///a/b//c/` is
/// `file:/a/b/c`, `/../x` is `file:/../x`). An empty location is `INVALID_EMPTY_LOCATION`. A scheme whose
/// file system Spark cannot open (`s3a`, `hdfs` without the connector) is refused by Spark itself; here it is
/// only normalised, which is not checked against a real Spark run with that file system.
pub(super) fn qualified_namespace_location(
    location: &str,
    warehouse: &std::path::Path,
) -> Result<String, krishiv_common::error::KrishivError> {
    if location.is_empty() {
        return Err(krishiv_common::error::InvalidEmptyLocation {
            location: location.to_owned(),
        }
        .into());
    }
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
        None => (
            "file:".to_owned(),
            warehouse.join(location).display().to_string(),
        ),
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

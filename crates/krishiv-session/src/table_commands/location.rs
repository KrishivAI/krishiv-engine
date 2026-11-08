// SPDX-License-Identifier: Apache-2.0
//! The location of a namespace, qualified as Spark qualifies it.

/// Spark's `CatalogUtils.makeQualifiedDBObjectPath` on the local file system: a location with a URI scheme
/// other than `file` stays as written; any other is resolved against the warehouse directory (an absolute
/// path stays itself) and qualified as a `file:` URI. A local path is normalised as Hadoop's `Path` does:
/// no empty, `.` or `..` segment, no trailing slash (Spark 4.2.0: `rel/dir//x/` and `a/../b` are
/// `<warehouse>/rel/dir/x` and `<warehouse>/b`, `file:/abs2//w/` is `file:/abs2/w`, `/../x` is `file:/../x`, an empty location is `INVALID_EMPTY_LOCATION`).
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
    let path = match scheme {
        Some((s, rest)) if s.eq_ignore_ascii_case("file") && !rest.starts_with("//") => {
            rest.to_owned()
        }
        Some(_) => return Ok(location.to_owned()),
        None => warehouse.join(location).display().to_string(),
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
    Ok(format!("file:/{}", parts.join("/")))
}

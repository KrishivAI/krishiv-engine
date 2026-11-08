// SPDX-License-Identifier: Apache-2.0
//! The location of a namespace, qualified as Spark qualifies it.

/// Spark's `CatalogUtils.makeQualifiedDBObjectPath` on the local file system: a location with a URI scheme
/// other than `file` stays as written; any other is resolved against the warehouse directory (an absolute
/// path stays itself) and qualified as a `file:` URI. A local path is normalised as Hadoop's `Path` does:
/// no empty, `.` or `..` segment, no trailing slash (Spark 4.2.0: `rel/dir//x/` and `a/../b` are
/// `<warehouse>/rel/dir/x` and `<warehouse>/b`, `file:/abs2//w/` is `file:/abs2/w`).
pub(super) fn qualified_namespace_location(location: &str, warehouse: &std::path::Path) -> String {
    let scheme = location.split_once(':').filter(|(s, _)| {
        s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    });
    let path = match scheme {
        Some((s, rest)) if s.eq_ignore_ascii_case("file") && !rest.starts_with("//") => {
            rest.to_owned()
        }
        Some(_) => return location.to_owned(),
        None => warehouse.join(location).display().to_string(),
    };
    let mut parts: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    format!("file:/{}", parts.join("/"))
}

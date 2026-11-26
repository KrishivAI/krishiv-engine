// SPDX-License-Identifier: Apache-2.0
//! A local path as the `file:` URL the engine prints for it.

use std::path::Path;

/// `path` as the `file:` URL the engine prints for it (`file:/tmp/a`; `file:/C:/Users/a` on
/// Windows, as Hadoop qualifies a drive path), with `/` separators, so it can stand in a SQL
/// string literal (a `\` there is an escape) or an expected text.
pub(crate) fn file_url(path: &Path) -> String {
    let p = path.display().to_string().replace('\\', "/");
    if p.starts_with('/') {
        format!("file:{p}")
    } else {
        format!("file:/{p}")
    }
}

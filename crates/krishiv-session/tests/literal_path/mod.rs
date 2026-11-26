// SPDX-License-Identifier: Apache-2.0
//! A local path as the text of a SQL string literal.

use std::path::Path;

/// `path` with `/` separators, for a SQL string literal (`LOCATION '...'`): a Windows path's `\`
/// would be read as an escape (`C:\Users` becomes `C:Users`, a path with the scheme `C`).
/// Unix paths are unchanged.
pub(crate) fn sql_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "/")
}

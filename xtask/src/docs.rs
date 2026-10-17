// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask docs-config` / `docs-errors`: Markdown reference docs
//! generated from the krishiv-common registries.

use std::io::Write;
use std::path::Path;

/// A generated reference document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc {
    /// The config registry.
    Config,
    /// The error-class registry.
    Errors,
}

impl Doc {
    /// Maps an xtask subcommand to its document.
    pub fn from_command(cmd: &str) -> Option<Doc> {
        match cmd {
            "docs-config" => Some(Doc::Config),
            "docs-errors" => Some(Doc::Errors),
            _ => None,
        }
    }

    /// Renders the document.
    pub fn markdown(self) -> String {
        match self {
            Doc::Config => krishiv_common::config::docs_markdown(),
            Doc::Errors => krishiv_common::error::docs_markdown(),
        }
    }
}

/// Writes `markdown` to `path` (creating parent directories), or to `stdout`
/// when `path` is `None`.
pub fn emit(markdown: &str, path: Option<&Path>, stdout: &mut impl Write) -> Result<(), String> {
    match path {
        Some(path) => {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
            }
            std::fs::write(path, markdown)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))
        }
        None => stdout
            .write_all(markdown.as_bytes())
            .map_err(|e| format!("cannot write to stdout: {e}")),
    }
}

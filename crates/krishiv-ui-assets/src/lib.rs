//! Embedded web UI assets (layer L6).
//!
//! The assets in `assets/` are the committed output of `ui/` (`pnpm build`,
//! copied by `ui/scripts/sync-assets.sh`), so building Rust never needs Node.

#![forbid(unsafe_code)]

use std::borrow::Cow;

#[derive(rust_embed::RustEmbed)]
#[folder = "assets/"]
struct Assets;

/// An embedded file and its content type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    /// File contents.
    pub data: Cow<'static, [u8]>,
    /// MIME type derived from the file extension.
    pub content_type: &'static str,
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit_once('.').map(|(_, ext)| ext) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

/// Looks up an embedded file by its path relative to the asset root
/// (`""` is `index.html`).
#[must_use]
pub fn get(path: &str) -> Option<Asset> {
    let path = path.trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    Assets::get(path).map(|f| Asset {
        data: f.data,
        content_type: content_type(path),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_is_html() {
        let a = get("").map(|a| a.content_type);
        assert_eq!(a, Some("text/html; charset=utf-8"));
        assert!(get("/index.html").is_some());
    }

    #[test]
    fn missing_is_none() {
        assert!(get("nope.js").is_none());
    }
}

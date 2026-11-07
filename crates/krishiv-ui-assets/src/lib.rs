// SPDX-License-Identifier: Apache-2.0
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

    fn all() -> Vec<(String, Vec<u8>)> {
        Assets::iter()
            .filter_map(|p| Assets::get(&p).map(|f| (p.to_string(), f.data.into_owned())))
            .collect()
    }

    /// Spec 09 section 5: the bundle stays within 3 MB brotli. The raw size is an upper bound of that, so a
    /// bundle under it passes without a compressor in the build.
    #[test]
    fn the_bundle_is_within_the_size_budget() {
        let total: usize = all().iter().map(|(_, d)| d.len()).sum();
        assert!(total <= 3 * 1024 * 1024, "{total} bytes of embedded assets");
    }

    /// The server sends `script-src 'self'; style-src 'self'`: the page must not need inline script or style,
    /// nor code built from strings.
    #[test]
    fn the_page_works_under_a_strict_content_security_policy() {
        let files = all();
        let index = files
            .iter()
            .find(|(p, _)| p == "index.html")
            .map(|(_, d)| String::from_utf8_lossy(d).into_owned());
        let index = index.unwrap_or_default();
        assert!(index.contains("src=\"./assets/"), "{index}");
        assert!(!index.contains("<style"), "inline style: {index}");
        assert!(!index.contains(" style="), "style attribute: {index}");
        for tag in index.split("<script").skip(1) {
            let tag = tag.split('>').next().unwrap_or("");
            assert!(tag.contains("src="), "an inline script: <script{tag}>");
        }
        for (path, data) in &files {
            if std::path::Path::new(path)
                .extension()
                .is_some_and(|e| e == "js")
            {
                let js = String::from_utf8_lossy(data);
                assert!(!js.contains("eval("), "{path} calls eval");
                assert!(
                    !js.contains("new Function("),
                    "{path} builds code from a string"
                );
            }
        }
    }
}

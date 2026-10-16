//! Message-template placeholders, matching Spark's `<([a-zA-Z0-9_-]+)>`.
//!
//! Shared with `build.rs` (via `#[path]`) so that the parameters the build
//! script declares and the ones the renderer substitutes cannot disagree.

/// Returns `Some((name, end))` if a placeholder starts at byte `start`
/// (which must point at `<`); `end` is the index just past the closing `>`.
fn placeholder_at(template: &str, start: usize) -> Option<(&str, usize)> {
    let rest = template.get(start + 1..)?;
    let len = rest
        .bytes()
        .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
        .count();
    if len > 0 && rest.as_bytes().get(len) == Some(&b'>') {
        Some((rest.get(..len)?, start + len + 2))
    } else {
        None
    }
}

/// Unique placeholder names in order of first appearance.
#[allow(dead_code)] // used by build.rs and tests only
pub fn placeholders(template: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0;
    while let Some(off) = template.get(i..).and_then(|s| s.find('<')) {
        let start = i + off;
        match placeholder_at(template, start) {
            Some((name, end)) => {
                if !out.contains(&name) {
                    out.push(name);
                }
                i = end;
            }
            None => i = start + 1,
        }
    }
    out
}

/// Substitutes `<name>` placeholders in one left-to-right pass. Values are
/// inserted verbatim (never re-scanned); unknown placeholders are kept.
#[allow(dead_code)] // unused by build.rs, which includes this file
pub fn render(template: &str, params: &[(&str, String)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut i = 0;
    while let Some(off) = template.get(i..).and_then(|s| s.find('<')) {
        let start = i + off;
        out.push_str(template.get(i..start).unwrap_or_default());
        let value = placeholder_at(template, start).and_then(|(name, end)| {
            params
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| (v, end))
        });
        match value {
            Some((v, end)) => {
                out.push_str(v);
                i = end;
            }
            None => {
                out.push('<');
                i = start + 1;
            }
        }
    }
    out.push_str(template.get(i..).unwrap_or_default());
    out
}

// SPDX-License-Identifier: Apache-2.0
//! `OPTIONS (...)` and `TBLPROPERTIES (...)` clauses as text pairs.
use std::collections::BTreeMap;

use krishiv_sql::ast::TableOption;

/// `OPTIONS (...)` / `TBLPROPERTIES (...)` as text pairs, keys lower-cased
/// for options as Spark does not (kept as written).
pub(super) fn options_of(options: &[TableOption], original: &str) -> BTreeMap<String, String> {
    options
        .iter()
        .map(|o| {
            let key = o
                .key
                .parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
                .join(".");
            let value = o
                .value
                .as_ref()
                .map(|v| literal_text(original.get(v.span.start..v.span.end).unwrap_or_default()))
                .unwrap_or_default();
            (key, value)
        })
        .collect()
}

/// A literal's text without its quotes.
fn literal_text(fragment: &str) -> String {
    let f = fragment.trim();
    for q in ['\'', '"'] {
        if f.len() >= 2 && f.starts_with(q) && f.ends_with(q) {
            return f[1..f.len() - 1].to_owned();
        }
    }
    f.to_owned()
}

#[cfg(test)]
mod tests {
    use super::literal_text;

    #[test]
    fn literal_text_strips_one_pair_of_quotes() {
        assert_eq!(literal_text("'a b'"), "a b");
        assert_eq!(literal_text("\"x\""), "x");
        assert_eq!(literal_text(" true "), "true");
        assert_eq!(literal_text("'"), "'");
    }
}

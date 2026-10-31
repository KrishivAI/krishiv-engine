// SPDX-License-Identifier: Apache-2.0
//! The element list of `SET PATH` and `krishiv.sql.defaultPath`: parsing the
//! configuration text, and rendering a path entry as `current_path()` shows it.

/// One element of a `SET PATH` list.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PathElement {
    /// `catalog.schema` (or deeper): the parts, unquoted.
    Schema(Vec<String>),
    /// `CURRENT_SCHEMA` or `CURRENT_DATABASE`.
    CurrentSchema {
        /// Written `CURRENT_DATABASE`.
        database: bool,
    },
    /// `DEFAULT_PATH`.
    DefaultPath,
    /// `SYSTEM_PATH`.
    SystemPath,
    /// `PATH`: the live path so far (not allowed in the configuration).
    Path,
}

/// A name part as `current_path()` writes it: plain identifiers as they are,
/// anything else in backquotes.
#[must_use]
pub fn render_part(part: &str) -> String {
    if !part.is_empty() && part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        part.to_owned()
    } else {
        format!("`{}`", part.replace('`', "``"))
    }
}

/// A schema entry as `current_path()` writes it.
#[must_use]
pub fn render(parts: &[String]) -> String {
    parts
        .iter()
        .map(|p| render_part(p))
        .collect::<Vec<_>>()
        .join(".")
}

/// Splits a dotted name written with optional backquotes; `None` when it is
/// malformed.
fn name_parts(text: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut chars = text.chars().peekable();
    loop {
        let mut part = String::new();
        if chars.peek() == Some(&'`') {
            chars.next();
            loop {
                match chars.next()? {
                    '`' if chars.peek() == Some(&'`') => {
                        chars.next();
                        part.push('`');
                    }
                    '`' => break,
                    c => part.push(c),
                }
            }
        } else {
            while let Some(&c) = chars.peek() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    part.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            if part.is_empty() {
                return None;
            }
        }
        parts.push(part);
        match chars.next() {
            None => return Some(parts),
            Some('.') => {}
            Some(_) => return None,
        }
    }
}

/// Parses a comma-separated element list (the grammar of `SET PATH`, `PATH`
/// excluded). `None` for text that is not such a list; an empty list for
/// blank text.
#[must_use]
pub fn parse_conf(text: &str) -> Option<Vec<PathElement>> {
    if text.trim().is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        // An element ends at the first comma outside backquotes.
        let mut quoted = false;
        let mut end = rest.len();
        for (i, c) in rest.char_indices() {
            match c {
                '`' => quoted = !quoted,
                ',' if !quoted => {
                    end = i;
                    break;
                }
                _ => {}
            }
        }
        let item = rest[..end].trim();
        out.push(match item.to_ascii_uppercase().as_str() {
            "DEFAULT_PATH" => PathElement::DefaultPath,
            "SYSTEM_PATH" => PathElement::SystemPath,
            "CURRENT_SCHEMA" => PathElement::CurrentSchema { database: false },
            "CURRENT_DATABASE" => PathElement::CurrentSchema { database: true },
            _ => {
                let parts = name_parts(item)?;
                if parts.len() < 2 {
                    return None;
                }
                PathElement::Schema(parts)
            }
        });
        if end == rest.len() {
            return Some(out);
        }
        rest = &rest[end + 1..];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(parts: &[&str]) -> PathElement {
        PathElement::Schema(parts.iter().map(|p| (*p).to_owned()).collect())
    }

    #[test]
    fn a_conf_list_parses() {
        assert_eq!(
            parse_conf("system.session, system.builtin, current_schema"),
            Some(vec![
                schema(&["system", "session"]),
                schema(&["system", "builtin"]),
                PathElement::CurrentSchema { database: false },
            ])
        );
        assert_eq!(parse_conf(""), Some(Vec::new()));
        assert_eq!(
            parse_conf("spark_catalog.`a.b`"),
            Some(vec![schema(&["spark_catalog", "a.b"])])
        );
    }

    #[test]
    fn bad_conf_lists_are_refused() {
        assert_eq!(parse_conf("this is not a path"), None);
        assert_eq!(parse_conf("PATH, system.builtin"), None);
        assert_eq!(parse_conf("one_part"), None);
    }

    #[test]
    fn entries_render_with_quoting() {
        assert_eq!(
            render(&["spark_catalog".into(), "a.b".into()]),
            "spark_catalog.`a.b`"
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
//! Spark's `parseMultipartIdentifier`, for `IDENTIFIER('a.b')` under the legacy clause.

/// Spark's `parseMultipartIdentifier`: dot-separated parts, each bare or in
/// backticks (a doubled backtick is one). `None` for an empty or malformed
/// string, which stays an unresolved call.
pub(super) fn multipart_identifier(text: &str) -> Option<Vec<String>> {
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
                if !(c.is_ascii_alphanumeric() || c == '_') {
                    break;
                }
                part.push(c);
                chars.next();
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

#[cfg(test)]
mod identifier_tests {
    use super::multipart_identifier;

    #[test]
    fn multipart_identifiers_split_like_spark() {
        let parts = |t| multipart_identifier(t);
        assert_eq!(parts("a"), Some(vec!["a".to_owned()]));
        assert_eq!(parts("t.c1"), Some(vec!["t".to_owned(), "c1".to_owned()]));
        assert_eq!(
            parts("`a.b`.c"),
            Some(vec!["a.b".to_owned(), "c".to_owned()])
        );
        assert_eq!(parts("`a``b`"), Some(vec!["a`b".to_owned()]));
        for bad in ["", ".a", "a.", "a..b", "`a", "`a`b", "c 1", "a-b"] {
            assert_eq!(parts(bad), None, "{bad}");
        }
    }
}

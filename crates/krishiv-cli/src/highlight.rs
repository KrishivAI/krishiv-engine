// SPDX-License-Identifier: Apache-2.0
//! ANSI highlighting of SQL for the REPL.
//!
//! Keywords, literals, identifiers and comments each get a color; an
//! unbalanced quote colors the rest of the line as a literal instead of
//! breaking the prompt. The highlighter is a small scanner, not the SQL
//! parser, so it never fails: [`highlight`] returns plain text unchanged
//! when colors cannot be used.

use crate::complete::is_keyword_word;

/// ANSI reset.
const RESET: &str = "\x1b[0m";
/// Bold blue: keywords.
const KEYWORD: &str = "\x1b[1;34m";
/// Green: string literals.
const LITERAL: &str = "\x1b[32m";
/// Yellow: numbers.
const NUMBER: &str = "\x1b[33m";
/// Dim: comments.
const COMMENT: &str = "\x1b[2;37m";
/// Cyan: quoted identifiers.
const IDENT: &str = "\x1b[36m";

/// Whether ANSI colors may be emitted on stdout.
pub(crate) fn colors_enabled() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

/// `sql` with keywords, literals, identifiers and comments colored. An
/// unbalanced quote or block comment runs to the end of the input: the
/// prompt stays usable.
pub(crate) fn highlight(sql: &str) -> String {
    highlight_with(sql, true)
}

fn highlight_with(sql: &str, colors: bool) -> String {
    if !colors {
        return sql.to_owned();
    }
    let mut out = String::with_capacity(sql.len() + 32);
    let bytes = sql.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'-' && bytes.get(i + 1) == Some(&b'-') {
            let end = sql[i..].find('\n').map_or(sql.len(), |e| i + e);
            push(&mut out, COMMENT, &sql[i..end]);
            i = end;
        } else if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let end = skip_block_comment(sql, i);
            push(&mut out, COMMENT, &sql[i..end]);
            i = end;
        } else if b == b'\'' {
            let end = skip_quoted(sql, i, '\'');
            push(&mut out, LITERAL, &sql[i..end]);
            i = end;
        } else if b == b'"' || b == b'`' {
            let end = skip_quoted(sql, i, char::from(b));
            push(&mut out, IDENT, &sql[i..end]);
            i = end;
        } else if b.is_ascii_digit()
            || (b == b'.' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit))
        {
            let mut end = i;
            while end < bytes.len()
                && (bytes[end].is_ascii_alphanumeric()
                    || bytes[end] == b'.'
                    || bytes[end] == b'_'
                    || bytes[end] == b'e'
                    || bytes[end] == b'E'
                    || ((bytes[end] == b'+' || bytes[end] == b'-')
                        && end > i
                        && (bytes[end - 1] == b'e' || bytes[end - 1] == b'E')))
            {
                end += 1;
            }
            push(&mut out, NUMBER, &sql[i..end]);
            i = end;
        } else if (b.is_ascii_alphabetic() || b == b'_')
            && (i == 0 || !is_word_char(sql[..i].chars().next_back()))
        {
            let mut end = i + 1;
            while end < bytes.len() && is_word_byte(bytes[end]) {
                end += 1;
            }
            let word = &sql[i..end];
            if is_keyword_word(word) {
                push(&mut out, KEYWORD, word);
            } else {
                out.push_str(word);
            }
            i = end;
        } else if let Some(c) = sql[i..].chars().next() {
            // Any other character (including multibyte text) passes
            // through whole, so highlighting never corrupts UTF-8.
            out.push(c);
            i += c.len_utf8();
        } else {
            break;
        }
    }
    out
}

/// `text` wrapped in `color`..[`RESET`].
fn push(out: &mut String, color: &str, text: &str) {
    out.push_str(color);
    out.push_str(text);
    out.push_str(RESET);
}

/// Whether `c` continues a word.
fn is_word_char(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '$')
}

/// Whether `b` continues an ASCII word.
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// End of the block comment starting at `start` (nested, as Spark
/// lexes); the input end when it never closes.
fn skip_block_comment(sql: &str, start: usize) -> usize {
    let bytes = sql.as_bytes();
    let mut i = start + 2;
    let mut depth = 1_usize;
    while i < bytes.len() {
        if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
            depth += 1;
            i += 2;
        } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
            depth -= 1;
            i += 2;
            if depth == 0 {
                // Floor at the boundary: indexes advanced ASCII-only.
                return sql.floor_char_boundary(i);
            }
        } else {
            i += 1;
        }
    }
    sql.len()
}

/// End of the quoted literal starting at `start`: a quote closes unless
/// immediately followed by another quote (`''` escapes inside single
/// quotes); the input end when it never closes, so an unbalanced quote
/// cannot break the prompt.
fn skip_quoted(sql: &str, start: usize, quote: char) -> usize {
    let rest = &sql[start..];
    let mut it = rest.char_indices();
    it.next(); // the opening quote
    let mut prev_was_quote = false;
    for (off, c) in it {
        if c == quote {
            if prev_was_quote {
                prev_was_quote = false;
                continue;
            }
            let after = start + off + quote.len_utf8();
            let doubled = sql[after..].starts_with(quote);
            if quote == '\'' && doubled {
                prev_was_quote = true;
                continue;
            }
            return sql.floor_char_boundary(after);
        }
        prev_was_quote = false;
    }
    sql.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_literals_and_comments_are_colored() {
        let got = highlight_with("SELECT 1, 'a' -- tail", true);
        assert!(got.contains("\x1b[1;34mSELECT\x1b[0m"), "{got:?}");
        assert!(got.contains("\x1b[33m1\x1b[0m"), "{got:?}");
        assert!(got.contains("\x1b[32m'a'\x1b[0m"), "{got:?}");
        assert!(got.contains("\x1b[2;37m-- tail\x1b[0m"), "{got:?}");
    }

    #[test]
    fn quoted_identifiers_are_colored() {
        let got = highlight_with("select `c`, \"d\" from t", true);
        assert!(got.contains("\x1b[36m`c`\x1b[0m"), "{got:?}");
        assert!(got.contains("\x1b[36m\"d\"\x1b[0m"), "{got:?}");
        // `select` lower case still matches keywords; `from` is a keyword
        // too, while the table alias `t` stays plain.
        assert!(got.contains("\x1b[1;34mselect\x1b[0m"), "{got:?}");
        assert!(got.ends_with("\x1b[0m t"), "{got:?}");
    }

    #[test]
    fn unbalanced_quotes_do_not_break_the_prompt() {
        for open in [
            "select 'abc",
            "select \"abc",
            "select `abc",
            "select /* abc",
        ] {
            let got = highlight_with(open, true);
            // Colored to the end, then usable again: no panic, valid UTF-8.
            assert!(got.len() >= open.len(), "{got:?}");
            let again = highlight_with(&format!("{open}\nselect 1"), true);
            assert!(again.contains("\x1b[1;34mselect\x1b[0m"), "{again:?}");
        }
    }

    #[test]
    fn escaped_quotes_stay_inside_the_literal() {
        let got = highlight_with("select 'it''s', 2", true);
        assert!(got.contains("\x1b[32m'it''s'\x1b[0m"), "{got:?}");
        assert!(got.contains("\x1b[33m2\x1b[0m"), "{got:?}");
    }

    #[test]
    fn plain_passthrough_and_non_keywords() {
        assert_eq!(highlight_with("select 1", false), "select 1");
        let got = highlight_with("select year from t", true);
        // `year` is not a reserved word: no color.
        assert!(!got.contains("\x1b[1;34myear\x1b[0m"), "{got:?}");
    }

    #[test]
    fn multibyte_text_passes_through_whole() {
        let got = highlight_with("select 'héllo', naïveté", true);
        assert!(got.contains("'héllo'"), "{got:?}");
        assert!(got.contains("naïveté"), "{got:?}");
        // Stripping the codes yields the input back.
        let mut plain = got;
        for code in [RESET, KEYWORD, LITERAL, NUMBER, COMMENT, IDENT] {
            plain = plain.replace(code, "");
        }
        assert_eq!(plain, "select 'héllo', naïveté");
    }
}

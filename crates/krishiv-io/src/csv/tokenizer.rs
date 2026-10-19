// SPDX-License-Identifier: Apache-2.0
//! The CSV tokenizer: one record of text into values, with the rules of
//! the Univocity parser Spark uses (quotes, escapes, `charToEscapeQuoteEscaping`,
//! white-space trimming, unescaped quotes), checked against records Spark 4.2
//! parsed (`tests/fixtures/spark-4.2.0/csv.json`).

use super::options::Syntax;

/// What happens to a quote inside a quoted value that is followed by other
/// text (`unescapedQuoteHandling`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unescaped {
    StopAtClosingQuote,
    BackToDelimiter,
    StopAtDelimiter,
    SkipValue,
    RaiseError,
}

/// Where a record ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Eol {
    /// The end of the text (the lines were split already).
    EndOfText,
    /// `\n`, `\r` or `\r\n` outside quotes.
    Default,
    /// This separator outside quotes.
    Custom(String),
}

/// A record that cannot be tokenized at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenizeError {
    /// More columns than `maxColumns`.
    TooManyColumns,
    /// A value longer than `maxCharsPerColumn`.
    ValueTooLong,
    /// `unescapedQuoteHandling` is `RAISE_ERROR` and a quote was unescaped.
    UnescapedQuote,
}

/// A record of a multi-line text: its raw text, its values (or why it has
/// none) and the position after it.
pub(crate) type Record<'t> = (&'t str, Result<Vec<Token>, TokenizeError>, usize);

/// One value: `None` is Univocity's null (an empty unquoted value).
pub(crate) type Token = Option<String>;

pub(crate) struct Tokenizer {
    s: Syntax,
}

/// Position in the text.
struct Cursor<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn rest(&self) -> &'a str {
        &self.text[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }
}

impl Tokenizer {
    pub(crate) fn new(s: Syntax) -> Self {
        Self { s }
    }

    fn at_sep(&self, c: &Cursor<'_>) -> bool {
        c.rest().starts_with(self.s.sep.as_str())
    }

    /// The length of the record terminator at the cursor (0: none).
    fn eol_len(&self, c: &Cursor<'_>) -> usize {
        match &self.s.eol {
            Eol::EndOfText => 0,
            Eol::Default => match c.rest().as_bytes() {
                [b'\r', b'\n', ..] => 2,
                [b'\r' | b'\n', ..] => 1,
                _ => 0,
            },
            Eol::Custom(sep) => {
                if c.rest().starts_with(sep.as_str()) {
                    sep.len()
                } else {
                    0
                }
            }
        }
    }

    fn at_end(&self, c: &Cursor<'_>) -> bool {
        c.rest().is_empty() || self.eol_len(c) > 0
    }

    fn check_len(&self, value: &str) -> Result<(), TokenizeError> {
        match self.s.max_chars {
            Some(max) if value.chars().count() > max => Err(TokenizeError::ValueTooLong),
            _ => Ok(()),
        }
    }

    fn push(&self, out: &mut Vec<Token>, t: Token) -> Result<(), TokenizeError> {
        if let Some(v) = &t {
            self.check_len(v)?;
        }
        if out.len() >= self.s.max_columns {
            return Err(TokenizeError::TooManyColumns);
        }
        out.push(t);
        Ok(())
    }

    /// Skips white space (`<= ' '`) before a value, but not a separator or
    /// quote that is itself white space.
    fn skip_white(&self, c: &mut Cursor<'_>) {
        while let Some(ch) = c.peek() {
            if ch > ' ' || self.at_sep(c) || Some(ch) == self.s.quote || self.eol_len(c) > 0 {
                break;
            }
            c.bump();
        }
    }

    /// The next record of a multi-line `text` from `pos`, skipping empty and
    /// comment lines (a line of white space is a record): its raw text, its
    /// values (or why it has none) and the position after it. `None` at the
    /// end of the text.
    pub(crate) fn next_record<'t>(
        &self,
        text: &'t str,
        mut pos: usize,
        comment: Option<char>,
    ) -> Option<Record<'t>> {
        loop {
            let rest = text.get(pos..).filter(|r| !r.is_empty())?;
            let mut c = Cursor { text, pos };
            let empty = self.eol_len(&c);
            if empty > 0 || comment.is_some_and(|ch| rest.starts_with(ch)) {
                while !self.at_eol(&c) {
                    c.bump();
                }
                pos = c.pos + self.eol_len(&c);
                continue;
            }
            let mut tokens = Vec::new();
            let (res, end) = match self.record(text, pos, &mut tokens) {
                Ok(end) => (Ok(tokens), end),
                Err(e) => (Err(e), text.len()),
            };
            let raw = &text[pos..end];
            let raw = match &self.s.eol {
                Eol::Custom(sep) => raw.strip_suffix(sep.as_str()).unwrap_or(raw),
                _ => raw.trim_end_matches(['\r', '\n']),
            };
            return Some((raw, res, end));
        }
    }

    /// The record at `text[pos..]`: its values appended to `out`; returns
    /// the position after the record and its terminator.
    ///
    /// # Errors
    /// [`TokenizeError`] for a record the settings cannot represent.
    pub(crate) fn record(
        &self,
        text: &str,
        pos: usize,
        out: &mut Vec<Token>,
    ) -> Result<usize, TokenizeError> {
        let mut c = Cursor { text, pos };
        loop {
            if self.s.ltrim {
                self.skip_white(&mut c);
            }
            if self.at_sep(&c) || self.at_end(&c) {
                self.push(out, None)?;
            } else if Some(c.peek().unwrap_or('\0')) == self.s.quote && self.s.quote.is_some() {
                let t = self.quoted(&mut c, out)?;
                self.push(out, t)?;
            } else {
                let t = self.unquoted(&mut c);
                self.push(out, t)?;
            }
            if self.at_sep(&c) {
                c.pos += self.s.sep.len();
                if self.at_end(&c) {
                    self.push(out, None)?;
                    break;
                }
            } else {
                break;
            }
        }
        let n = self.eol_len(&c);
        Ok(c.pos + n)
    }

    /// With a separator of several characters Univocity doubles a quote that
    /// ends the last value of a record.
    fn double_closing_quote(&self, c: &Cursor<'_>, v: &mut String) {
        if self.s.sep.chars().nth(1).is_some()
            && !self.at_sep(c)
            && let Some(q) = self.s.quote
            && v.ends_with(q)
        {
            v.push(q);
        }
    }

    /// An unquoted value: up to the next separator.
    fn unquoted(&self, c: &mut Cursor<'_>) -> Token {
        let mut v = String::new();
        while !self.at_sep(c) && !self.at_end(c) {
            let Some(ch) = c.bump() else { break };
            v.push(ch);
        }
        self.double_closing_quote(c, &mut v);
        if self.s.rtrim {
            let t = v.trim_end_matches(|ch: char| ch <= ' ').len();
            v.truncate(t);
        }
        if v.is_empty() { None } else { Some(v) }
    }

    /// Whether a record ends at the cursor (or the text does).
    fn at_eol(&self, c: &Cursor<'_>) -> bool {
        self.eol_len(c) > 0 || c.rest().is_empty()
    }

    /// Consumes the rest of a value, up to the next separator or the end of
    /// the record, appending to `out`.
    fn rest_of_value(&self, c: &mut Cursor<'_>, out: &mut String) {
        while !self.at_sep(c) && !self.at_eol(c) {
            let Some(ch) = c.bump() else { break };
            out.push(ch);
        }
    }

    /// The rest of a value up to the next separator, with the escape
    /// character applied (`\\` is one backslash, `\"` a quote). A quote
    /// that is not escaped and ends the text is dropped, as the closing
    /// quote of the value.
    fn escaped_rest(&self, c: &mut Cursor<'_>, out: &mut String) {
        let q = self.s.quote.unwrap_or('"');
        let mut plain_quote_last = false;
        while !self.at_sep(c) && !self.at_eol(c) {
            let Some(ch) = c.bump() else { break };
            plain_quote_last = false;
            if Some(ch) == self.s.escape && (self.at_sep(c) || self.at_eol(c)) {
                // An escape character before the separator escapes nothing.
            } else if Some(ch) == self.s.escape
                && let Some(n) = c.peek()
                && (n == q || Some(n) == self.s.escape_escape)
            {
                c.bump();
                out.push(n);
            } else {
                plain_quote_last = ch == q;
                out.push(ch);
            }
        }
        if plain_quote_last {
            out.pop();
        }
    }

    /// Skips to the next separator or the end of the record.
    fn skip_value(&self, c: &mut Cursor<'_>) {
        while !self.at_sep(c) && !self.at_eol(c) {
            c.bump();
        }
    }

    /// A quoted value; the cursor is on the opening quote. The state machine
    /// is Univocity's: `prev` is the special character (quote or escape)
    /// just consumed, and decides what the next character means.
    fn quoted(&self, c: &mut Cursor<'_>, out: &mut Vec<Token>) -> Result<Token, TokenizeError> {
        // Univocity reads a plain quoted value that ends at a separator
        // with a fast path; only the general path drops a lone
        // `charToEscapeQuoteEscaping` character.
        if self.s.escape.is_none()
            || self
                .s
                .escape_escape
                .is_some_and(|ee| Some(ee) != self.s.escape && Some(ee) != self.s.quote)
        {
            let mut probe = Cursor {
                text: c.text,
                pos: c.pos,
            };
            let mut scratch = Vec::new();
            if let Ok((token, true)) = self.quoted_machine(&mut probe, &mut scratch, false)
                && scratch.is_empty()
            {
                c.pos = probe.pos;
                return Ok(token);
            }
        }
        self.quoted_machine(c, out, true).map(|r| r.0)
    }

    /// The quoted-value state machine; `slow` is the general path, which drops a lone
    /// `charToEscapeQuoteEscaping` character and, without an escape
    /// character, reads a quote at the start of the value as an escaped one. Also tells whether
    /// the value was closed by a quote followed by a separator.
    fn quoted_machine(
        &self,
        c: &mut Cursor<'_>,
        out: &mut Vec<Token>,
        slow: bool,
    ) -> Result<(Token, bool), TokenizeError> {
        let q = self.s.quote.unwrap_or('"');
        let esc = self.s.escape;
        let esc_esc = self.s.escape_escape;
        let is_esc = |ch: char| Some(ch) == esc;
        c.bump();
        let mut v = String::new();
        let mut prev: Option<char> = None;
        let mut began = false;
        let mut unescaped = false;
        let mut skipped = false;
        loop {
            let ch = c.peek();
            // A quote followed by white space, a separator or the end closes the value.
            if prev == Some(q) && (ch.is_none_or(|x| x <= ' ') || self.at_sep(c) || self.at_eol(c))
            {
                break;
            }
            let Some(ch) = ch else {
                // The text ends inside the quotes.
                if !began && v.is_empty() {
                    v.push(q);
                }
                self.check_len(&v)?;
                return Ok(((!skipped).then_some(v), false));
            };
            began = true;
            if ch != q && !is_esc(ch) {
                if prev == Some(q) {
                    match self.s.unescaped {
                        Unescaped::StopAtDelimiter => {
                            return Ok((self.stop_at_delimiter(c, &v, "")?, false));
                        }
                        Unescaped::StopAtClosingQuote => {
                            v.push(q);
                            prev = None;
                            unescaped = true;
                        }
                        Unescaped::BackToDelimiter => {
                            // Back to the last separator: the text before it is a value.
                            self.split_at_last_sep(&mut v, out)?;
                            prev = None;
                            unescaped = true;
                        }
                        Unescaped::SkipValue => {
                            self.skip_value(c);
                            return Ok((None, false));
                        }
                        Unescaped::RaiseError => return Err(TokenizeError::UnescapedQuote),
                    }
                }
                if prev.is_some() && prev == esc {
                    v.push(prev.unwrap_or_default());
                }
                // Plain characters up to the next quote or escape.
                // (A lone `charToEscapeQuoteEscaping` character is swallowed.)
                while let Some(x) = c.peek() {
                    if x == q || is_esc(x) || (slow && Some(x) == esc_esc) {
                        break;
                    }
                    if x == '\r' && self.s.eol == Eol::Default {
                        // Line ends inside quotes are normalized to `\n`.
                        v.push('\n');
                        c.bump();
                        if c.peek() == Some('\n') {
                            c.bump();
                        }
                        continue;
                    }
                    v.push(x);
                    c.bump();
                }
                prev = c.bump();
                continue;
            }
            // `ch` is a quote or the escape character.
            let mut seen = Some(ch);
            if is_esc(ch) && prev.is_some() && prev == esc_esc {
                v.extend(esc);
                seen = None;
            } else if prev == esc && (esc.is_some() || slow) {
                // (Without an escape character `prev` is NUL at the start of the
                // value, which equals it, as in Univocity.)
                if ch == q {
                    v.push(q);
                    seen = None;
                } else {
                    v.extend(prev);
                }
            } else if ch == q && prev == Some(q) {
                v.push(q);
            } else if prev == Some(q) {
                // The escape character right after a quote: Univocity
                // appends the quote, reads the value up to the next
                // separator (escapes applied), drops that separator and
                // goes on in the quoted value.
                match self.s.unescaped {
                    Unescaped::SkipValue => {
                        // The value is dropped, but the quoted value goes on.
                        self.skip_value(c);
                        if self.at_sep(c) {
                            c.pos += self.s.sep.len();
                        }
                        skipped = true;
                        prev = None;
                        began = true;
                        continue;
                    }
                    Unescaped::RaiseError => return Err(TokenizeError::UnescapedQuote),
                    _ => {
                        unescaped = true;
                        v.push(q);
                        v.push(ch);
                        c.bump();
                        self.escaped_rest(c, &mut v);
                        if self.at_sep(c) {
                            c.pos += self.s.sep.len();
                        }
                        prev = None;
                        began = true;
                        continue;
                    }
                }
            }
            c.bump();
            prev = seen;
            // After an unescaped quote a separator ends the value again.
            if unescaped && (self.at_sep(c) || self.at_eol(c)) {
                break;
            }
        }
        // Closed by a quote: white space may follow before the separator.
        let mut white = String::new();
        while let Some(x) = c.peek() {
            if x > ' ' || self.at_sep(c) || self.at_eol(c) {
                break;
            }
            white.push(x);
            c.bump();
        }
        if white.is_empty() || self.at_sep(c) || self.at_eol(c) {
            self.check_len(&v)?;
            if skipped {
                return Ok((None, false));
            }
            let clean = white.is_empty() && !unescaped && self.at_sep(c);
            let token = Some(if v.is_empty() {
                self.s.empty_value.clone()
            } else {
                v
            });
            return Ok((token, clean));
        }
        let token = match self.s.unescaped {
            Unescaped::StopAtDelimiter | Unescaped::BackToDelimiter => {
                self.stop_at_delimiter(c, &v, &white)?
            }
            Unescaped::StopAtClosingQuote => {
                v.push(q);
                v.push_str(&white);
                self.quoted_rest(c, v)?
            }
            Unescaped::SkipValue => {
                self.skip_value(c);
                None
            }
            Unescaped::RaiseError => return Err(TokenizeError::UnescapedQuote),
        };
        Ok((token, false))
    }

    /// `BACK_TO_DELIMITER`: the separators of `v` after its last quote split
    /// it into values; `v` keeps the text after the last one.
    fn split_at_last_sep(&self, v: &mut String, out: &mut Vec<Token>) -> Result<(), TokenizeError> {
        let sep = self.s.sep.as_str();
        let from = self
            .s
            .quote
            .and_then(|q| v.rfind(q))
            .map_or(0, |i| i + self.s.quote.map_or(0, char::len_utf8));
        if !v[from..].contains(sep) {
            return Ok(());
        }
        let text = std::mem::take(v);
        let (head, tail) = text.split_at(from);
        let mut parts: Vec<&str> = tail.split(sep).collect();
        let rest = parts.pop().unwrap_or_default();
        for (i, part) in parts.iter().enumerate() {
            let value = if i == 0 {
                format!("{head}{part}")
            } else {
                (*part).to_owned()
            };
            self.push(out, if value.is_empty() { None } else { Some(value) })?;
        }
        *v = rest.to_owned();
        Ok(())
    }

    /// `STOP_AT_DELIMITER`: the value is the text from its opening quote to
    /// the next separator, with the content so far unescaped.
    fn stop_at_delimiter(
        &self,
        c: &mut Cursor<'_>,
        content: &str,
        white: &str,
    ) -> Result<Token, TokenizeError> {
        let q = self.s.quote.unwrap_or('"');
        let mut v = String::with_capacity(content.len() + 2);
        v.push(q);
        v.push_str(content);
        v.push(q);
        v.push_str(white);
        self.rest_of_value(c, &mut v);
        if !white.is_empty() {
            self.double_closing_quote(c, &mut v);
        }
        if self.s.rtrim {
            let t = v.trim_end_matches(|ch: char| ch <= ' ').len();
            v.truncate(t);
        }
        self.check_len(&v)?;
        Ok(Some(v))
    }

    /// `STOP_AT_CLOSING_QUOTE` after white space: the value goes on as a
    /// quoted one until a quote followed by a separator, or the end.
    fn quoted_rest(&self, c: &mut Cursor<'_>, mut v: String) -> Result<Token, TokenizeError> {
        let q = self.s.quote.unwrap_or('"');
        while let Some(x) = c.bump() {
            if x == q && (self.at_sep(c) || self.at_eol(c)) {
                break;
            }
            v.push(x);
        }
        self.check_len(&v)?;
        Ok(Some(v))
    }
}

// SPDX-License-Identifier: Apache-2.0
//! A strict JSON lexer with Spark's `JSONOptions` leniency flags.
//!
//! [`Lexer`] reads one JSON text: objects, arrays, strings (with the
//! escapes the flags allow), numbers (with the leading-zero rule the flags
//! choose), and the `true`/`false`/`null` literals. Whitespace is skipped;
//! comments are skipped only with `allowComments`. Anything else is
//! [`Error::Malformed`], which every caller maps to a SQL `NULL` (Spark's
//! `PERMISSIVE` mode). Nesting deeper than [`MAX_JSON_DEPTH`] (Jackson's
//! default) is malformed too, so every recursive walk below stays bounded.
//!
//! All readers work straight off the lexer: values bound for Arrow are
//! never boxed into a document tree first, and navigation only materializes
//! the matched subtree.

/// The deepest JSON nesting the parser accepts (Jackson's default limit).
pub(crate) const MAX_JSON_DEPTH: usize = 1000;

/// What the lexer allows beyond strict JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LexerFlags {
    /// `allowComments`.
    pub allow_comments: bool,
    /// `allowSingleQuotes`.
    pub allow_single_quotes: bool,
    /// `allowNumericLeadingZeros`.
    pub allow_numeric_leading_zeros: bool,
    /// `allowBackslashEscapingAnyCharacter`.
    pub allow_backslash_escaping_any: bool,
    /// `allowUnquotedFieldNames`.
    pub allow_unquoted_field_names: bool,
    /// `allowUnquotedControlChars`.
    pub allow_unquoted_control_chars: bool,
    /// `allowNonNumericNumbers` (Spark's default): the `NaN`, `Infinity`
    /// and `INF` spellings, signed, as floating-point numbers.
    pub allow_non_numeric_numbers: bool,
}

impl LexerFlags {
    /// Strict JSON.
    pub(crate) const fn strict() -> Self {
        Self {
            allow_comments: false,
            allow_single_quotes: false,
            allow_numeric_leading_zeros: false,
            allow_backslash_escaping_any: false,
            allow_unquoted_field_names: false,
            allow_unquoted_control_chars: false,
            allow_non_numeric_numbers: false,
        }
    }
}

/// Why a text is not the JSON the caller wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Error {
    /// Malformed JSON (what every caller maps to a SQL `NULL`).
    Malformed,
    /// Nesting past [`MAX_JSON_DEPTH`].
    TooDeep,
}

/// A JSON string with its quotes, validated but not unescaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RawStr<'a> {
    /// The string with its quotes (single quotes kept as written).
    pub raw: &'a str,
    /// The quote character.
    pub quote: u8,
}

impl<'a> RawStr<'a> {
    /// The string's value, unescaped; `None` for a bad escape or a lone
    /// surrogate (Jackson rejects those too). With
    /// `allowBackslashEscapingAnyCharacter` any other escaped character is
    /// the character itself (as `skip_escape` accepts it), and an escaped
    /// single quote is a quote in any quoting.
    pub(crate) fn unescape(&self, allow_backslash_escaping_any: bool) -> Option<String> {
        unescape(self.raw, self.quote, allow_backslash_escaping_any)
    }
}

/// The lexer over one JSON text.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lexer<'a> {
    /// The whole text (for sub-lexers over value spans).
    text: &'a str,
    bytes: &'a [u8],
    /// The next unread byte.
    pub pos: usize,
    flags: LexerFlags,
}

/// One token: punctuation, or a borrowed literal, string or number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Token<'a> {
    /// `{`.
    OpenBrace,
    /// `}`.
    CloseBrace,
    /// `[`.
    OpenBracket,
    /// `]`.
    CloseBracket,
    /// `,`.
    Comma,
    /// `:`.
    Colon,
    /// A string with its quotes.
    Str(RawStr<'a>),
    /// A number as written, and whether it is floating-point.
    Num(&'a str, bool),
    /// `true` or `false`.
    Bool(bool),
    /// `null`.
    Null,
}

impl<'a> Lexer<'a> {
    /// Lexes `text` with `flags`.
    pub(crate) const fn new(text: &'a str, flags: LexerFlags) -> Self {
        Self {
            text,
            bytes: text.as_bytes(),
            pos: 0,
            flags,
        }
    }

    /// The whole text this lexer reads.
    pub(crate) const fn text(&self) -> &'a str {
        self.text
    }

    /// The value of a quoted `raw`, with this lexer's escaping rules.
    pub(crate) fn unescape(&self, raw: RawStr<'a>) -> Option<String> {
        raw.unescape(self.flags.allow_backslash_escaping_any)
    }

    /// The unread bytes.
    fn rest(&self) -> &'a [u8] {
        self.bytes.get(self.pos..).unwrap_or_default()
    }

    /// Skips whitespace and (with `allowComments`) comments.
    pub(crate) fn skip_trivia(&mut self) -> Result<(), Error> {
        loop {
            let mut i = self.pos;
            while let Some(&b) = self.bytes.get(i) {
                if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                    i = i.saturating_add(1);
                } else {
                    break;
                }
            }
            self.pos = i;
            if !self.flags.allow_comments || self.rest().first() != Some(&b'/') {
                return Ok(());
            }
            let second = self.rest().get(1).copied().unwrap_or(0);
            if second == b'/' {
                self.pos = self.pos.saturating_add(2);
                while let Some(&b) = self.bytes.get(self.pos) {
                    if matches!(b, b'\n' | b'\r') {
                        break;
                    }
                    self.pos = self.pos.saturating_add(1);
                }
            } else if second == b'*' {
                self.pos = self.pos.saturating_add(2);
                loop {
                    match self.bytes.get(self.pos).copied() {
                        None => return Err(Error::Malformed),
                        Some(b'/') if self.bytes.get(self.pos.saturating_sub(1)) == Some(&b'*') => {
                            self.pos = self.pos.saturating_add(1);
                            break;
                        }
                        Some(_) => self.pos = self.pos.saturating_add(1),
                    }
                }
            } else {
                return Err(Error::Malformed);
            }
        }
    }

    /// The next token, with leading trivia skipped.
    pub(crate) fn next(&mut self) -> Result<Option<Token<'a>>, Error> {
        self.skip_trivia()?;
        let Some(&b) = self.bytes.get(self.pos) else {
            return Ok(None);
        };
        // Single-byte punctuation advances here; the sub-parsers (strings,
        // numbers, literals) position the cursor themselves.
        let bump = |this: &mut Self| {
            this.pos = this.pos.saturating_add(1);
        };
        match b {
            b'{' => {
                bump(self);
                Ok(Some(Token::OpenBrace))
            }
            b'}' => {
                bump(self);
                Ok(Some(Token::CloseBrace))
            }
            b'[' => {
                bump(self);
                Ok(Some(Token::OpenBracket))
            }
            b']' => {
                bump(self);
                Ok(Some(Token::CloseBracket))
            }
            b',' => {
                bump(self);
                Ok(Some(Token::Comma))
            }
            b':' => {
                bump(self);
                Ok(Some(Token::Colon))
            }
            b'"' => Ok(Some(Token::Str(self.string(b'"')?))),
            b'\'' if self.flags.allow_single_quotes => Ok(Some(Token::Str(self.string(b'\'')?))),
            b'-' => {
                if let Some(token) = self.non_numeric_if_allowed()? {
                    return Ok(Some(token));
                }
                Ok(Some(self.number()?))
            }
            b'+' | b'N' | b'I' => match self.non_numeric_if_allowed()? {
                Some(token) => Ok(Some(token)),
                None => Err(Error::Malformed),
            },
            b'0'..=b'9' => Ok(Some(self.number()?)),
            b't' => Ok(Some(self.literal("true", Token::Bool(true))?)),
            b'f' => Ok(Some(self.literal("false", Token::Bool(false))?)),
            b'n' => Ok(Some(self.literal("null", Token::Null)?)),
            _ => Err(Error::Malformed),
        }
    }

    /// Peeks at the next token without consuming it.
    pub(crate) fn peek(&mut self) -> Result<Option<Token<'a>>, Error> {
        let pos = self.pos;
        let token = self.next()?;
        self.pos = pos;
        Ok(token)
    }

    /// Whether the next token is `token` (consumed when it is).
    pub(crate) fn eat(&mut self, token: Token<'a>) -> Result<bool, Error> {
        let pos = self.pos;
        let hit = match self.next()? {
            Some(t) => token_kind(t) == token_kind(token),
            None => false,
        };
        if !hit {
            self.pos = pos;
        }
        Ok(hit)
    }

    /// Whether the next non-trivia byte is `b` (consumed when it is). Unlike
    /// [`Self::eat`], this never lexes a token, so it stays `Ok(false)` when
    /// an unquoted field name comes next (which is not a token).
    pub(crate) fn eat_byte(&mut self, b: u8) -> Result<bool, Error> {
        self.skip_trivia()?;
        if self.bytes.get(self.pos).copied() == Some(b) {
            self.pos = self.pos.saturating_add(1);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// A `true`/`false`/`null` literal: the keyword must end there.
    fn literal(&mut self, word: &str, token: Token<'a>) -> Result<Token<'a>, Error> {
        let end = self.pos.saturating_add(word.len());
        if self.bytes.get(self.pos..end) != Some(word.as_bytes()) {
            return Err(Error::Malformed);
        }
        if self
            .bytes
            .get(end)
            .copied()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(Error::Malformed);
        }
        self.pos = end;
        Ok(token)
    }

    /// A number at the cursor: validated, with its float flag.
    fn number(&mut self) -> Result<Token<'a>, Error> {
        let start = self.pos;
        let mut i = start;
        let byte = |i: usize| self.bytes.get(i).copied().unwrap_or(0);
        if byte(i) == b'-' {
            i = i.saturating_add(1);
        }
        let int_start = i;
        while byte(i).is_ascii_digit() {
            i = i.saturating_add(1);
        }
        if i == int_start {
            return Err(Error::Malformed);
        }
        let digits = self.bytes.get(int_start..i).unwrap_or_default();
        if !self.flags.allow_numeric_leading_zeros
            && digits.len() > 1
            && digits.first() == Some(&b'0')
        {
            return Err(Error::Malformed);
        }
        let mut is_float = false;
        if byte(i) == b'.' {
            is_float = true;
            i = i.saturating_add(1);
            let frac = i;
            while byte(i).is_ascii_digit() {
                i = i.saturating_add(1);
            }
            if i == frac {
                return Err(Error::Malformed);
            }
        }
        if matches!(byte(i), b'e' | b'E') {
            is_float = true;
            i = i.saturating_add(1);
            if matches!(byte(i), b'+' | b'-') {
                i = i.saturating_add(1);
            }
            let exp = i;
            while byte(i).is_ascii_digit() {
                i = i.saturating_add(1);
            }
            if i == exp {
                return Err(Error::Malformed);
            }
        }
        let text = self.bytes.get(start..i).unwrap_or_default();
        let text = core::str::from_utf8(text).map_err(|_| Error::Malformed)?;
        self.pos = i;
        Ok(Token::Num(text, is_float))
    }

    /// A non-numeric number (`NaN`, `Infinity` or `INF`, signed) at the
    /// cursor when the flag allows it: the token, or `None` when the
    /// cursor is not on one.
    fn non_numeric_if_allowed(&mut self) -> Result<Option<Token<'a>>, Error> {
        if self.flags.allow_non_numeric_numbers {
            self.non_numeric()
        } else {
            Ok(None)
        }
    }

    /// A non-numeric number (`NaN`, `Infinity` or `INF`, signed) at the
    /// cursor: the token, or `None` when the cursor is not on one. Longer
    /// spellings win (`+Infinity` before `+INF`), and the word must end
    /// there, as a literal's must.
    fn non_numeric(&mut self) -> Result<Option<Token<'a>>, Error> {
        for word in [
            "+Infinity",
            "-Infinity",
            "+INF",
            "-INF",
            "Infinity",
            "NaN",
            "INF",
        ] {
            let end = self.pos.saturating_add(word.len());
            if self.bytes.get(self.pos..end) != Some(word.as_bytes()) {
                continue;
            }
            if self
                .bytes
                .get(end)
                .copied()
                .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
            {
                return Err(Error::Malformed);
            }
            let text = self.bytes.get(self.pos..end).unwrap_or_default();
            let text = core::str::from_utf8(text).map_err(|_| Error::Malformed)?;
            self.pos = end;
            return Ok(Some(Token::Num(text, true)));
        }
        Ok(None)
    }

    /// A string at the cursor (the cursor is on `quote`): validated, with
    /// its quotes.
    fn string(&mut self, quote: u8) -> Result<RawStr<'a>, Error> {
        let start = self.pos;
        let mut i = start.saturating_add(1);
        loop {
            match self.bytes.get(i).copied() {
                None => return Err(Error::Malformed),
                Some(b) if b == quote => {
                    let end = i.saturating_add(1);
                    let raw = self.bytes.get(start..end).unwrap_or_default();
                    let raw = core::str::from_utf8(raw).map_err(|_| Error::Malformed)?;
                    check_escapes(raw, quote, self.flags)?;
                    self.pos = end;
                    return Ok(RawStr { raw, quote });
                }
                Some(b'\\') => {
                    i = skip_escape(self.bytes, i, quote, self.flags)?;
                }
                Some(b) if b < 0x20 && !self.flags.allow_unquoted_control_chars => {
                    return Err(Error::Malformed);
                }
                Some(_) => i = i.saturating_add(1),
            }
        }
    }

    /// The raw text of one object key (quoted or bare); the caller compares
    /// or hashes it.
    pub(crate) fn raw_key(&mut self) -> Result<RawKey<'a>, Error> {
        self.skip_trivia()?;
        let quote = self.bytes.get(self.pos).copied().unwrap_or(0);
        if quote == b'"' || (quote == b'\'' && self.flags.allow_single_quotes) {
            return Ok(RawKey::Quoted(self.string(quote)?));
        }
        if self.flags.allow_unquoted_field_names {
            let start = self.pos;
            self.skip_bare_word()?;
            let word = self.bytes.get(start..self.pos).unwrap_or_default();
            let word = core::str::from_utf8(word).map_err(|_| Error::Malformed)?;
            return Ok(RawKey::Bare(word));
        }
        Err(Error::Malformed)
    }

    /// Skips a bare (unquoted) field name.
    fn skip_bare_word(&mut self) -> Result<(), Error> {
        let mut len = 0_usize;
        while let Some(&b) = self.bytes.get(self.pos) {
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'$' || b >= 0x80 {
                self.pos = self.pos.saturating_add(1);
                len = len.saturating_add(1);
            } else {
                break;
            }
        }
        if len == 0 {
            Err(Error::Malformed)
        } else {
            Ok(())
        }
    }

    /// Skips one JSON value; the cursor is before its first token. A
    /// structural token (`}`, `]`, `,`, `:`) is not a value. Nesting past
    /// [`MAX_JSON_DEPTH`] is [`Error::TooDeep`].
    pub(crate) fn skip_value(&mut self) -> Result<(), Error> {
        self.skip_value_at(0)
    }

    /// [`Self::skip_value`] with the current nesting depth.
    pub(crate) fn skip_value_at(&mut self, depth: usize) -> Result<(), Error> {
        if depth > MAX_JSON_DEPTH {
            return Err(Error::TooDeep);
        }
        match self.next()? {
            Some(Token::OpenBrace | Token::OpenBracket) => self.skip_container(depth),
            // Only scalars skip as one token; a structural token is not
            // a value (`[1, }]` is malformed, not a two-element array).
            Some(Token::Str(_) | Token::Num(_, _) | Token::Bool(_) | Token::Null) => Ok(()),
            None | Some(_) => Err(Error::Malformed),
        }
    }

    /// The rest of an object or array whose opening token was just read, at nesting `depth`. Its
    /// own function so that [`Self::skip_value_at`]'s frame stays small: the walk recurses up to
    /// [`MAX_JSON_DEPTH`] deep and must fit a 2 MiB thread stack, also in sanitizer and debug builds.
    #[inline(never)]
    fn skip_container(&mut self, depth: usize) -> Result<(), Error> {
        let brace = self.bytes.get(self.pos.saturating_sub(1)).copied() == Some(b'{');
        loop {
            self.skip_trivia()?;
            let b = self.bytes.get(self.pos).copied();
            if brace && b == Some(b'}') || !brace && b == Some(b']') {
                self.pos = self.pos.saturating_add(1);
                return Ok(());
            }
            if brace {
                self.skip_key()?;
                self.skip_trivia()?;
                if self.bytes.get(self.pos).copied() != Some(b':') {
                    return Err(Error::Malformed);
                }
                self.pos = self.pos.saturating_add(1);
            }
            self.skip_value_at(depth.saturating_add(1))?;
            self.skip_trivia()?;
            match self.bytes.get(self.pos).copied() {
                Some(b',') => self.pos = self.pos.saturating_add(1),
                Some(b) if brace && b == b'}' || !brace && b == b']' => {
                    self.pos = self.pos.saturating_add(1);
                    return Ok(());
                }
                _ => return Err(Error::Malformed),
            }
        }
    }

    /// Skips one object key (a string, or a bare word with
    /// `allowUnquotedFieldNames`).
    pub(crate) fn skip_key(&mut self) -> Result<(), Error> {
        match self.raw_key()? {
            RawKey::Quoted(_) | RawKey::Bare(_) => Ok(()),
        }
    }
}

/// One object key: quoted (with its quotes) or bare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RawKey<'a> {
    /// A quoted key.
    Quoted(RawStr<'a>),
    /// A bare key (only with `allowUnquotedFieldNames`).
    Bare(&'a str),
}

/// The punctuation kind of a token.
fn token_kind(token: Token<'_>) -> u8 {
    match token {
        Token::OpenBrace => 0,
        Token::CloseBrace => 1,
        Token::OpenBracket => 2,
        Token::CloseBracket => 3,
        Token::Comma => 4,
        Token::Colon => 5,
        Token::Str(_) => 6,
        Token::Num(_, _) => 7,
        Token::Bool(_) => 8,
        Token::Null => 9,
    }
}

/// Validates the escapes of `raw` (with `quote`) without unescaping.
fn check_escapes(raw: &str, quote: u8, flags: LexerFlags) -> Result<(), Error> {
    let bytes = raw.as_bytes();
    let mut i = 1_usize;
    let end = bytes.len().saturating_sub(1);
    while i < end {
        match bytes.get(i).copied().unwrap_or(0) {
            b'\\' => {
                i = skip_escape(bytes, i, quote, flags)?;
            }
            _ => i = i.saturating_add(1),
        }
    }
    Ok(())
}

/// The index after the escape at `i` (which is a backslash).
fn skip_escape(bytes: &[u8], i: usize, quote: u8, flags: LexerFlags) -> Result<usize, Error> {
    let esc = bytes.get(i.saturating_add(1)).copied().unwrap_or(0);
    let simple = matches!(esc, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't')
        || (esc == b'\'' && quote == b'\'');
    if simple {
        return Ok(i.saturating_add(2));
    }
    if esc == b'u' {
        for k in 0..4_usize {
            if !bytes
                .get(i.saturating_add(2).saturating_add(k))
                .copied()
                .unwrap_or(0)
                .is_ascii_hexdigit()
            {
                return Err(Error::Malformed);
            }
        }
        return Ok(i.saturating_add(6));
    }
    if flags.allow_backslash_escaping_any && esc != 0 {
        return Ok(i.saturating_add(2));
    }
    Err(Error::Malformed)
}

/// Whether `raw` is a non-numeric number spelling (`NaN`, `Infinity` or
/// `INF`, signed), which the lexer only produces with
/// `allowNonNumericNumbers`.
#[must_use]
pub(crate) fn is_non_numeric_number(raw: &str) -> bool {
    matches!(
        raw,
        "NaN" | "INF" | "+INF" | "-INF" | "Infinity" | "+Infinity" | "-Infinity"
    )
}

/// The value of a non-numeric number spelling (`None` for anything else).
#[must_use]
pub(crate) fn non_numeric_value(raw: &str) -> Option<f64> {
    Some(match raw {
        "NaN" => f64::NAN,
        "INF" | "+INF" | "Infinity" | "+Infinity" => f64::INFINITY,
        "-INF" | "-Infinity" => f64::NEG_INFINITY,
        _ => return None,
    })
}

/// Unescapes `raw` (with `quote`); `None` for a bad escape or a lone
/// surrogate. With `allow_backslash_escaping_any` any other escaped
/// character is the character itself.
fn unescape(raw: &str, quote: u8, allow_backslash_escaping_any: bool) -> Option<String> {
    let bytes = raw.as_bytes();
    let body = bytes.get(1..bytes.len().saturating_sub(1))?;
    if !body.contains(&b'\\') {
        return core::str::from_utf8(body).ok().map(str::to_owned);
    }
    let text = core::str::from_utf8(body).ok()?;
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next()? {
            '"' => out.push('"'),
            '\\' => out.push('\\'),
            '/' => out.push('/'),
            'b' => out.push('\u{8}'),
            'f' => out.push('\u{c}'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            '\'' if quote == b'\'' || allow_backslash_escaping_any => out.push('\''),
            'u' => {
                let hex: String = chars.by_ref().take(4).collect();
                if hex.len() != 4 {
                    return None;
                }
                let unit = u32::from_str_radix(&hex, 16).ok()?;
                let high = u16::try_from(unit).ok()?;
                if (0xD800..0xDC00).contains(&high) {
                    // A surrogate pair.
                    if chars.next()? != '\\' || chars.next()? != 'u' {
                        return None;
                    }
                    let hex: String = chars.by_ref().take(4).collect();
                    if hex.len() != 4 {
                        return None;
                    }
                    let low = u16::try_from(u32::from_str_radix(&hex, 16).ok()?).ok()?;
                    if !(0xDC00..0xE000).contains(&low) {
                        return None;
                    }
                    let scalar = 0x10000_u32
                        .checked_add(u32::from(high).checked_sub(0xD800)?.checked_mul(0x400)?)?
                        .checked_add(u32::from(low).checked_sub(0xDC00)?)?;
                    out.push(char::from_u32(scalar)?);
                } else if (0xDC00..0xE000).contains(&high) {
                    return None;
                } else {
                    out.push(char::from_u32(u32::from(high))?);
                }
            }
            other if allow_backslash_escaping_any => out.push(other),
            _ => return None,
        }
    }
    Some(out)
}

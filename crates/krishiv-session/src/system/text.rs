// SPDX-License-Identifier: Apache-2.0
//! How system.queries shows a statement and its error message: the cut, and the masking of literals, quoted values and bare numbers.

/// Most bytes of a statement `system.queries` keeps
/// (`query_text_truncated` says when it cut).
pub(super) const TEXT_BYTES: usize = 4096;

/// How `system.queries.query_text` shows a statement: its most bytes
/// (`krishiv.system.queryText.maxBytes`) and whether its literals are masked
/// (`krishiv.system.queryText.literals = mask`). Read from the session's
/// configuration when a statement starts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TextPolicy {
    pub(super) max_bytes: usize,
    pub(super) mask: bool,
}

impl Default for TextPolicy {
    fn default() -> Self {
        Self {
            max_bytes: TEXT_BYTES,
            mask: false,
        }
    }
}

impl TextPolicy {
    /// The policy `config` sets.
    pub(crate) fn from_config(config: &krishiv_common::config::Config) -> Self {
        use krishiv_common::config::{
            ConfigValue, SYSTEM_QUERY_TEXT_LITERALS, SYSTEM_QUERY_TEXT_MAX_BYTES,
        };
        Self {
            max_bytes: config
                .get(&SYSTEM_QUERY_TEXT_MAX_BYTES)
                .and_then(ConfigValue::as_int)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(TEXT_BYTES),
            mask: config
                .get(&SYSTEM_QUERY_TEXT_LITERALS)
                .and_then(ConfigValue::as_str)
                == Some("mask"),
        }
    }

    /// `text` as the table shows it, and whether it was cut.
    pub(super) fn apply(self, text: &str) -> (String, bool) {
        let masked;
        let text = if self.mask {
            masked = krishiv_sql::mask_literals(text);
            masked.as_str()
        } else {
            text
        };
        self.cut(text)
    }

    /// An error message as the table shows it: English, not SQL, so the
    /// SQL tokenizer never reads it (an apostrophe or `--` in prose would
    /// hide the rest). Credentials are redacted; under `mask` every closed
    /// `'...'` or backquoted run (where Spark quotes values) becomes `?`.
    #[cfg(test)]
    pub(super) fn apply_message(self, message: &str) -> String {
        self.apply_message_with(message, &[])
    }

    /// [`apply_message`](Self::apply_message), and under `mask` the error parameters that carry a raw value
    /// (a malformed record, printed without quotes) replaced by `?` wherever the message shows them.
    pub(super) fn apply_message_with(
        self,
        message: &str,
        parameters: &[(&'static str, String)],
    ) -> String {
        let message = if self.mask {
            // Not the SQLSTATE at the end: a short value must not garble it.
            let (body, tail) = message
                .rfind("SQLSTATE:")
                .map_or((message, ""), |at| message.split_at(at));
            let mut masked = body.to_owned();
            for (name, value) in parameters {
                if !is_structural(name) && value.chars().count() >= 3 {
                    masked = masked.replace(value.as_str(), "?");
                }
            }
            masked.push_str(tail);
            std::borrow::Cow::Owned(masked)
        } else {
            std::borrow::Cow::Borrowed(message)
        };
        let redacted = krishiv_common::redact::redact(&message);
        let shown = if self.mask {
            mask_numbers(&mask_backquoted(&mask_quoted(&redacted)))
        } else {
            redacted.into_owned()
        };
        self.cut(&shown).0
    }

    /// `text` cut to `max_bytes` on a character boundary.
    pub(super) fn cut(self, text: &str) -> (String, bool) {
        if text.len() <= self.max_bytes {
            return (text.to_owned(), false);
        }
        let mut end = self.max_bytes;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        (text[..end].to_owned(), true)
    }
}

/// For each position, the index of the quote that closes a run starting there: a quote before a non-word
/// character, a backslash escaping the character after it (`\'` inside a value is not the end). Built right to
/// left, so every answer is one lookup and the whole table is linear.
fn closing_quotes(chars: &[(usize, char)], word: impl Fn(char) -> bool) -> Vec<Option<usize>> {
    let n = chars.len();
    let mut next = vec![None; n + 2];
    for j in (0..n).rev() {
        next[j] = match chars[j].1 {
            '\\' => next[j + 2],
            '\'' if chars.get(j + 1).is_none_or(|c| !word(c.1)) => Some(j),
            _ => next[j + 1],
        };
    }
    next
}

/// `message` with each quoted `'...'` run replaced by `'?'`. A quote opens
/// only after a non-word character and closes only before one, so the
/// apostrophe of `can't` is prose; an unclosed quote is left as it is.
pub(super) fn mask_quoted(message: &str) -> String {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let chars: Vec<(usize, char)> = message.char_indices().collect();
    let closers = closing_quotes(&chars, word);
    let mut out = String::with_capacity(message.len());
    let mut copied = 0;
    let mut i = 0;
    while i < chars.len() {
        let (at, c) = chars[i];
        let opens = c == '\'' && (i == 0 || !word(chars[i - 1].1));
        if opens && let Some(j) = closers[i + 1] {
            out.push_str(&message[copied..at]);
            out.push_str("'?'");
            copied = chars.get(j + 1).map_or(message.len(), |n| n.0);
            i = j + 1;
            continue;
        }
        i += 1;
    }
    out.push_str(&message[copied..]);
    out
}

/// Whether an error parameter names part of the statement (an object, a type, a function, a position) rather
/// than carrying a value that came from the data or a literal. Every other parameter is masked: a list of
/// the value-bearing ones would miss the next one Spark prints without quotes.
fn is_structural(name: &str) -> bool {
    const SUFFIXES: &[&str] = &[
        "Name", "Names", "Type", "Types", "Class", "Id", "Index", "Size", "Count", "Num", "Mode",
        "Conf", "Config", "Function", "Operator", "Clause", "Status", "Version", "Level", "Kind",
        "Scope", "Path",
    ];
    SUFFIXES.iter().any(|s| name.ends_with(s)) || name == "sqlState"
}

/// The state of a JSON string after `c`: an unescaped double quote opens or closes it, a backslash inside it
/// escapes the next character.
fn step_json((open, escaped): (bool, bool), c: char) -> (bool, bool) {
    match c {
        _ if escaped => (open, false),
        '\\' if open => (open, true),
        '"' => (!open, false),
        _ => (open, escaped),
    }
}

/// `message` with each backquoted run that is not a plain name replaced by `` `?` ``: some messages print an
/// offending value (a variant rendered as JSON, say) between backquotes, while names (`db`.`t`) stay.
///
/// A backquote opens after a non-word character and closes before one, so a backquote inside a rendered value
/// (`"x`secret"`) is part of the value; one before whitespace is not told apart by that rule, so while the span
/// has an open JSON string it runs on to the next closing backquote. One pass: each character is read once.
pub(super) fn mask_backquoted(message: &str) -> String {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let chars: Vec<(usize, char)> = message.char_indices().collect();
    let closers: Vec<usize> = (0..chars.len())
        .filter(|&k| chars[k].1 == '`' && chars.get(k + 1).is_none_or(|n| !word(n.1)))
        .collect();
    let mut out = String::with_capacity(message.len());
    let mut copied = 0;
    let mut i = 0;
    while i < chars.len() {
        let opens = chars[i].1 == '`' && (i == 0 || !word(chars[i - 1].1));
        let first = closers.partition_point(|&k| k <= i);
        if !opens || first == closers.len() {
            if opens {
                // No closing backquote is left, so no later one opens a span either.
                break;
            }
            i += 1;
            continue;
        }
        let (mut at, mut state, mut plain) = (first, (false, false), true);
        let mut from = i + 1;
        loop {
            let end = closers[at];
            for &(_, c) in &chars[from..end] {
                state = step_json(state, c);
                plain &= c.is_alphanumeric() || matches!(c, '_' | '.' | '$' | '-');
            }
            from = end;
            if !state.0 || at + 1 == closers.len() {
                break;
            }
            at += 1;
        }
        let j = closers[at];
        if !plain || state.0 {
            out.push_str(&message[copied..chars[i].0]);
            out.push_str("`?`");
            copied = chars.get(j + 1).map_or(message.len(), |n| n.0);
        }
        i = j + 1;
    }
    out.push_str(&message[copied..]);
    out
}

/// `message` with each bare number (`123456789`, `1.5`) replaced by `?`: a cast overflow and its like print
/// the offending value unquoted. A number inside a word (`42P01`, `v2`) and the SQLSTATE at the end stay.
pub(super) fn mask_numbers(message: &str) -> String {
    let (body, tail) = match message.rfind("SQLSTATE:") {
        Some(at) => message.split_at(at),
        None => (message, ""),
    };
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::with_capacity(message.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() && (i == 0 || !word(chars[i - 1])) {
            let mut j = i;
            while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                j += 1;
            }
            // A trailing dot is the sentence's.
            while j > i && chars[j - 1] == '.' {
                j -= 1;
            }
            // An exponent: `1e400`, `1.5E-3`.
            if matches!(chars.get(j), Some('e' | 'E')) {
                let mut k = j + 1;
                if matches!(chars.get(k), Some('+' | '-')) {
                    k += 1;
                }
                if chars.get(k).is_some_and(char::is_ascii_digit) {
                    while chars.get(k).is_some_and(char::is_ascii_digit) {
                        k += 1;
                    }
                    j = k;
                }
            }
            if chars.get(j).is_none_or(|n| !word(*n)) {
                out.push('?');
                i = j;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out.push_str(tail);
    out
}

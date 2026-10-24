// SPDX-License-Identifier: Apache-2.0
//! `java.util.regex` pattern syntax to `fancy_regex` syntax.

use std::fmt::Write;

/// A translated pattern.
pub(super) struct Translated {
    /// The pattern in `fancy_regex` syntax.
    pub(super) text: String,
    /// Whether the source has a `$` whose meaning depends on a final line
    /// terminator (so [`translate`] has a second form for those inputs).
    pub(super) has_dollar: bool,
}

/// Java's line terminators other than `\n` and `\r`, as regex escapes.
const OTHER_TERMINATORS: &str = r"\x{85}\x{2028}\x{2029}";

/// The lookahead for `$` without the multiline flag: the end of the input,
/// or a final line terminator and then the end.
const DOLLAR: &str = r"(?=(?:\r\n|[\n\r\x{85}\x{2028}\x{2029}])?\z)";

/// The same under `(?d)` (UNIX_LINES), where only `\n` is a line terminator.
const DOLLAR_UNIX: &str = r"(?=\n?\z)";
const DOLLAR_MULTILINE_UNIX: &str = r"(?:\z|(?=\n))";
const CARET_MULTILINE_UNIX: &str = r"(?:\A|(?<=\n)(?!\z))";

/// `$` with the multiline flag: before any line terminator, or at the end.
const DOLLAR_MULTILINE: &str =
    r"(?:\z|(?=\r\n)|(?=[\x{85}\x{2028}\x{2029}])|(?<!\r)(?=\n)|(?=\r(?!\n)))";

/// `^` with the multiline flag: at the start, or after a line terminator
/// that is not the end of the input.
const CARET_MULTILINE: &str = r"(?:\A|(?<=[\n\x{85}\x{2028}\x{2029}])(?!\z)|(?<=\r)(?![\n])(?!\z))";

/// A pattern longer than this (in characters) is rejected as invalid:
/// the engines translate to an NFA whose build is superlinear, so
/// untrusted patterns could blow up memory. Java itself has no limit; this
/// is a deliberate divergence, raised as `INVALID_PARAMETER_VALUE.PATTERN`.
pub(super) const MAX_PATTERN_LEN: usize = 64 * 1024;

/// A counted repetition bound above this is rejected as invalid (same
/// error class and reason as [`MAX_PATTERN_LEN`]).
pub(super) const MAX_REPETITION: u32 = 1_000_000;

/// POSIX classes of `\p{Name}` (ASCII only, as in Java).
const POSIX: &[(&str, &str)] = &[
    ("Lower", "a-z"),
    ("Upper", "A-Z"),
    ("ASCII", r"\x00-\x7F"),
    ("Alpha", "a-zA-Z"),
    ("Digit", "0-9"),
    ("Alnum", "a-zA-Z0-9"),
    ("Punct", r##"!-/:-@\[-`{-~"##),
    ("Graph", "!-~"),
    ("Print", " -~"),
    ("Blank", r" \t"),
    ("Cntrl", r"\x00-\x1F\x7F"),
    ("XDigit", "0-9a-fA-F"),
    ("Space", r" \t\n\x0B\f\r"),
];

/// Scoped flags of a group.
#[derive(Clone, Copy, Default)]
struct Flags {
    dotall: bool,
    multiline: bool,
    /// `(?d)`: only `\n` is a line terminator.
    unix_lines: bool,
    /// `(?x)`: whitespace and `#` comments in the pattern are ignored.
    comments: bool,
    /// `(?i)`: case-insensitive; ASCII only unless `unicode_case`.
    ignore_case: bool,
    /// `(?u)`: case folding beyond ASCII.
    unicode_case: bool,
}

impl Flags {
    /// Whether `fancy_regex` folds case itself (Unicode folding); ASCII-only
    /// folding is spelled out by the translator.
    fn native_case(self) -> bool {
        self.ignore_case && self.unicode_case
    }

    fn ascii_case(self) -> bool {
        self.ignore_case && !self.unicode_case
    }
}

/// Translates the Java pattern `src`. With `terminated`, a `$` also
/// matches before a final line terminator (the form for inputs that end
/// with one; without it `$` is `\z`).
pub(super) fn translate(src: &str, terminated: bool) -> Result<Translated, String> {
    if src.chars().count() > MAX_PATTERN_LEN {
        return Err("pattern too large".to_owned());
    }
    let chars: Vec<char> = src.chars().collect();
    let mut t = Translator {
        chars: &chars,
        at: 0,
        out: String::with_capacity(src.len() + 8),
        terminated,
        has_dollar: false,
        stack: vec![Flags::default()],
        quantified: false,
        lazy_stack: false,
        atom_start: 0,
        atom_stack: Vec::new(),
        groups: 0,
    };
    t.run()?;
    Ok(Translated {
        text: t.out,
        has_dollar: t.has_dollar,
    })
}

struct Translator<'a> {
    chars: &'a [char],
    at: usize,
    out: String,
    terminated: bool,
    has_dollar: bool,
    stack: Vec<Flags>,
    /// The previous token was a quantifier.
    quantified: bool,
    /// The previous token was a lazy `?` (a second one dangles, as in Java).
    lazy_stack: bool,
    /// The byte offset in [`Self::out`] where the current atom starts: what
    /// a following possessive `+` wraps in an atomic group.
    atom_start: usize,
    /// The atom starts of enclosing groups and classes, innermost last;
    /// the closer restores its own.
    atom_stack: Vec<usize>,
    /// Capturing groups opened so far (Java reads back-reference digits
    /// against this count).
    groups: usize,
}

impl Translator<'_> {
    fn peek(&self, n: usize) -> Option<char> {
        self.chars.get(self.at + n).copied()
    }

    /// `$` without the multiline flag (and `\Z`).
    fn dollar(&mut self) {
        self.has_dollar = true;
        let unix = self.flags().unix_lines;
        self.out.push_str(match (self.terminated, unix) {
            (false, _) => r"\z",
            (true, false) => DOLLAR,
            (true, true) => DOLLAR_UNIX,
        });
    }

    fn flags(&self) -> Flags {
        self.stack.last().copied().unwrap_or_default()
    }

    /// Notes the start of a new atom: a following possessive `+` wraps
    /// everything from here in an atomic group.
    fn atom(&mut self) {
        self.atom_start = self.out.len();
        self.lazy_stack = false;
    }

    /// Wraps the current atom and its quantifier (`X*`, `X+`, `X?`,
    /// `X{n,m}`) in an atomic group: Java's possessive quantifier, which the
    /// engines spell `(?>X*)`. The wrap is complete, so a further
    /// quantifier dangles (rejected when it arrives, as in Java); `atom`
    /// stays at the end, so the check sees nothing to quantify.
    fn possessive(&mut self) {
        let atom = self.out.split_off(self.atom_start);
        self.out.push_str("(?>");
        self.out.push_str(&atom);
        self.out.push(')');
        self.atom_start = self.out.len();
        self.quantified = true;
        self.lazy_stack = false;
    }

    /// Under `(?x)`: skips whitespace and `#` comments, which Java's parser
    /// never shows to the pattern. The comment runs to a line terminator;
    /// like Java, `\u0085`, `\u2028` and `\u2029` end it but are then read
    /// as pattern characters.
    fn skip_ignorable(&mut self) {
        if !self.flags().comments {
            return;
        }
        while let Some(c) = self.peek(0) {
            if matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r') {
                self.at += 1;
            } else if c == '#' {
                let unix = self.flags().unix_lines;
                while let Some(d) = self.peek(0) {
                    let ends = if unix {
                        d == '\n'
                    } else {
                        matches!(d, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}')
                    };
                    if ends {
                        break;
                    }
                    self.at += 1;
                }
            } else {
                break;
            }
        }
    }

    fn run(&mut self) -> Result<(), String> {
        loop {
            self.skip_ignorable();
            let Some(c) = self.peek(0) else { break };
            let was_quantified = std::mem::take(&mut self.quantified);
            self.at += 1;
            match c {
                '\\' => {
                    self.atom();
                    self.escape(false)?;
                }
                '[' => {
                    self.atom();
                    self.class()?;
                }
                '(' => {
                    self.atom();
                    self.group()?;
                }
                ')' => {
                    if self.stack.len() > 1 {
                        self.stack.pop();
                    }
                    self.out.push(')');
                    // The closed group is one atom (or a literal `)` with
                    // no group open).
                    if let Some(start) = self.atom_stack.pop() {
                        self.atom_start = start;
                    } else {
                        self.atom();
                    }
                    self.quantified = false;
                    self.lazy_stack = false;
                }
                '.' => {
                    self.atom();
                    if self.flags().dotall {
                        self.out.push_str(r"[\s\S]");
                    } else if self.flags().unix_lines {
                        self.out.push_str(r"[^\n]");
                    } else {
                        let _ = write!(self.out, r"[^\n\r{OTHER_TERMINATORS}]");
                    }
                }
                '$' => {
                    self.atom();
                    if self.flags().multiline {
                        let unix = self.flags().unix_lines;
                        self.out.push_str(if unix {
                            DOLLAR_MULTILINE_UNIX
                        } else {
                            DOLLAR_MULTILINE
                        });
                    } else {
                        self.dollar();
                    }
                }
                '^' => {
                    self.atom();
                    if self.flags().multiline {
                        let unix = self.flags().unix_lines;
                        self.out.push_str(if unix {
                            CARET_MULTILINE_UNIX
                        } else {
                            CARET_MULTILINE
                        });
                    } else {
                        self.out.push_str(r"\A");
                    }
                }
                '*' | '?' | '+' if was_quantified => {
                    if self.out.len() == self.atom_start {
                        // Nothing to quantify: at the start, after `|`
                        // or `(`, or right after a possessive wrap.
                        return Err(format!("dangling meta character {c}"));
                    }
                    match c {
                        // A possessive quantifier: the atom and its
                        // quantifier become an atomic group.
                        '+' => self.possessive(),
                        '?' if self.lazy_stack => {
                            return Err("dangling meta character ?".to_owned());
                        }
                        // A lazy quantifier.
                        '?' => {
                            self.out.push('?');
                            self.quantified = true;
                            self.lazy_stack = true;
                        }
                        // A stacked `*` dangles, as in Java.
                        _ => return Err("dangling meta character *".to_owned()),
                    }
                }
                '*' | '+' | '?' => {
                    if self.out.len() == self.atom_start {
                        // Nothing to quantify: at the start or after `|`
                        // (a quantifier after `(` is a literal the engines
                        // reject at build, as Java does).
                        return Err(format!("dangling meta character {c}"));
                    }
                    self.out.push(c);
                    self.quantified = true;
                }
                '|' => {
                    self.out.push('|');
                    self.quantified = false;
                    self.lazy_stack = false;
                    self.atom_start = self.out.len();
                }
                '{' => {
                    if was_quantified {
                        // `a*{2}`, `a{2}{3}`, `a*+{2}`: a repetition of a
                        // quantifier dangles, as in Java.
                        return Err("dangling meta character {".to_owned());
                    }
                    // Java rejects a brace that is not `{n}`, `{n,}` or `{n,m}`.
                    let close = self.chars[self.at..]
                        .iter()
                        .position(|&d| d == '}')
                        .ok_or("illegal repetition")?;
                    let body: String = self.chars[self.at..self.at + close].iter().collect();
                    let (low, high) = body.split_once(',').unwrap_or((&body, &body));
                    let digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
                    if low.is_empty() || !digits(low) || !digits(high) {
                        return Err("illegal repetition".to_owned());
                    }
                    // Java reads the bounds as `int`s and needs `low <= high`.
                    let bound = |s: &str| s.parse::<i32>().map_err(|_| "illegal repetition range");
                    let high_bound = if high.is_empty() {
                        i32::MAX
                    } else {
                        bound(high)?
                    };
                    let low_bound = bound(low)?;
                    if low_bound > high_bound {
                        return Err("illegal repetition range".to_owned());
                    }
                    // The engines expand (or loop) counted repetitions;
                    // bound them so untrusted patterns cannot blow up
                    // memory or time.
                    if low_bound > MAX_REPETITION as i32
                        || (!high.is_empty() && high_bound > MAX_REPETITION as i32)
                    {
                        return Err("repetition too large".to_owned());
                    }
                    self.out.push('{');
                    self.out.push_str(&body);
                    self.out.push('}');
                    self.at += close + 1;
                    self.quantified = true;
                    self.lazy_stack = false;
                }
                c => self.literal(c),
            }
        }
        Ok(())
    }

    /// A group opening: plain, non-capturing, named, look-around, atomic or
    /// with inline flags.
    fn group(&mut self) -> Result<(), String> {
        let mut flags = self.flags();
        if self.peek(0) != Some('?') {
            self.groups += 1;
            self.stack.push(flags);
            self.open_paren();
            return Ok(());
        }
        let rest: String = self.chars[self.at..].iter().take(3).collect();
        if rest.starts_with("?<") && !rest.starts_with("?<=") && !rest.starts_with("?<!") {
            // `(?<name>`: the name is letters and digits in Java.
            self.groups += 1;
            self.stack.push(flags);
            // The name is copied as written (case folding must not touch
            // it). Java allows ASCII letters and digits, starting with a letter.
            let mut end = self.at + 2;
            while self.chars.get(end).is_some_and(char::is_ascii_alphanumeric) {
                end += 1;
            }
            let named = self.chars[self.at + 2..end]
                .first()
                .is_some_and(char::is_ascii_alphabetic);
            if !named || self.chars.get(end) != Some(&'>') {
                return Err("illegal group name".to_owned());
            }
            self.open_paren();
            self.out.extend(&self.chars[self.at..=end]);
            self.at = end + 1;
            return Ok(());
        }
        if rest.starts_with("?:")
            || rest.starts_with("?=")
            || rest.starts_with("?!")
            || rest.starts_with("?>")
            || rest.starts_with("?<=")
            || rest.starts_with("?<!")
        {
            self.stack.push(flags);
            self.open_paren();
            return Ok(());
        }
        // `(?flags)` or `(?flags:`.
        let before = flags;
        let mut j = self.at + 1;
        let mut on = true;
        while let Some(&c) = self.chars.get(j) {
            match c {
                '-' => on = false,
                'i' => flags.ignore_case = on,
                'u' => flags.unicode_case = on,
                'x' => flags.comments = on,
                'm' => flags.multiline = on,
                's' => flags.dotall = on,
                'd' => flags.unix_lines = on,
                ')' | ':' => break,
                _ => return Err(format!("unsupported flag {c}")),
            }
            j += 1;
        }
        let close = *self.chars.get(j).ok_or("unclosed group")?;
        self.at = j + 1;
        // Only Unicode case folding is left to `fancy_regex`; the other
        // flags (`m`, `s`, `d`, `x`, ASCII `i`) are expanded here.
        let spec = match (before.native_case(), flags.native_case()) {
            (false, true) => "i",
            (true, false) => "-i",
            _ => "",
        };
        if close == ':' {
            self.stack.push(flags);
            self.atom_stack.push(self.out.len());
            self.atom_start = self.out.len();
            let _ = write!(self.out, "(?{spec}:");
        } else {
            // `(?flags)` applies to the rest of the enclosing group.
            if let Some(top) = self.stack.last_mut() {
                *top = flags;
            }
            if !spec.is_empty() {
                let _ = write!(self.out, "(?{spec})");
            }
        }
        Ok(())
    }

    /// Pushes a group-open `(`: the closed group reads as one atom, so the
    /// paren position is saved for the `)` to restore (group content must
    /// not move the current atom).
    fn open_paren(&mut self) {
        self.atom_stack.push(self.out.len());
        self.out.push('(');
        self.atom_start = self.out.len() - 1;
    }

    /// A literal character outside a class.
    fn literal(&mut self, c: char) {
        self.atom();
        self.fold_literal(c, false);
    }

    /// `c` as a literal; an ASCII letter under `(?i)` (without `u`) matches
    /// both cases and nothing else (Rust's `(?i)` would also match `\u212A`
    /// for `k`).
    fn fold_literal(&mut self, c: char, in_class: bool) {
        if c.is_ascii_alphabetic() && self.flags().ascii_case() {
            let (lower, upper) = (c.to_ascii_lowercase(), c.to_ascii_uppercase());
            let _ = if in_class {
                write!(self.out, "{lower}{upper}")
            } else {
                write!(self.out, "[{lower}{upper}]")
            };
        } else {
            push_literal(&mut self.out, c, in_class);
        }
    }

    /// The other-case counterparts of the ASCII letters in `lo..=hi`, for a
    /// class range under ASCII `(?i)`.
    fn fold_range(&mut self, lo: char, hi: char) {
        for (first, last, to_upper) in [('a', 'z', true), ('A', 'Z', false)] {
            let (l, h) = (lo.max(first), hi.min(last));
            if l <= h {
                let fold = |c: char| {
                    if to_upper {
                        c.to_ascii_uppercase()
                    } else {
                        c.to_ascii_lowercase()
                    }
                };
                let _ = write!(self.out, "{}-{}", fold(l), fold(h));
            }
        }
    }

    /// A backslash escape (the backslash is consumed). In a class the
    /// result is a class member.
    fn escape(&mut self, in_class: bool) -> Result<(), String> {
        let c = self.peek(0).ok_or("a trailing backslash")?;
        self.at += 1;
        match c {
            'Q' => {
                // Quoted until `\E` or the end.
                while let Some(q) = self.peek(0) {
                    self.at += 1;
                    if q == '\\' && self.peek(0) == Some('E') {
                        self.at += 1;
                        break;
                    }
                    self.fold_literal(q, in_class);
                }
            }
            'E' => {}
            'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'h' | 'H' | 'v' | 'V' => {
                let body = match c.to_ascii_lowercase() {
                    'd' => "0-9",
                    'w' => "a-zA-Z0-9_",
                    's' => r" \t\n\x0B\f\r",
                    'h' => r" \t\x{A0}\x{1680}\x{180e}\x{2000}-\x{200a}\x{202f}\x{205f}\x{3000}",
                    _ => r"\n\x0B\f\r\x{85}\x{2028}\x{2029}",
                };
                let negated = c.is_ascii_uppercase();
                let _ = write!(self.out, "[{}{body}]", if negated { "^" } else { "" });
            }
            'R' if !in_class => {
                let _ = write!(self.out, r"(?:\r\n|[\n\x0B\f\r{OTHER_TERMINATORS}])");
            }
            // Anchors and word boundaries keep their spelling.
            'A' | 'z' | 'b' | 'B' if !in_class => {
                self.out.push('\\');
                self.out.push(c);
            }
            '1'..='9' if !in_class => self.back_reference(c),
            'Z' if !in_class => self.dollar(),
            't' | 'n' | 'r' | 'f' => {
                self.out.push('\\');
                self.out.push(c);
            }
            'a' => self.out.push_str(r"\x07"),
            'e' => self.out.push_str(r"\x1B"),
            'x' | 'u' => self.hex(c, in_class)?,
            '0' => {
                // `\0n`, `\0nn`, `\0mnn`: an octal value.
                let mut value = 0_u32;
                let mut digits = 0;
                while digits < 3 {
                    let Some(d) = self.peek(0).and_then(|d| d.to_digit(8)) else {
                        break;
                    };
                    if digits == 2 && value > 0o37 {
                        break;
                    }
                    value = value * 8 + d;
                    self.at += 1;
                    digits += 1;
                }
                if digits == 0 {
                    return Err("illegal octal escape".to_owned());
                }
                let _ = write!(self.out, r"\x{{{value:X}}}");
            }
            'k' if !in_class => {
                // `\k<name>`: a named back-reference (compared
                // case-insensitively under `(?i)`, see `back_reference`).
                if self.peek(0) != Some('<') {
                    return Err("illegal named back-reference".to_owned());
                }
                self.out.push_str(if self.flags().ignore_case {
                    r"(?i:\k"
                } else {
                    r"(?:\k"
                });
                while let Some(d) = self.peek(0) {
                    self.at += 1;
                    self.out.push(d);
                    if d == '>' {
                        break;
                    }
                }
                self.out.push(')');
            }
            'p' | 'P' => self.property(c)?,
            c if c.is_ascii_alphanumeric() => {
                return Err(format!("illegal escape \\{c}"));
            }
            c => push_literal(&mut self.out, c, in_class),
        }
        Ok(())
    }

    /// `\n`: Java takes further digits only while they name a group opened
    /// so far, and a reference to a group that is not open (or does not
    /// exist) compiles but never matches.
    fn back_reference(&mut self, first: char) {
        let mut n = first.to_digit(10).map_or(0, |d| d as usize);
        while let Some(d) = self.peek(0).and_then(|d| d.to_digit(10)) {
            let wider = n * 10 + d as usize;
            if wider > self.groups {
                break;
            }
            n = wider;
            self.at += 1;
        }
        if n > self.groups {
            self.out.push_str("(?!)");
        } else if self.flags().ignore_case {
            // Java compares case-insensitively here (ASCII, or Unicode with
            // `u`); the engine folds Unicode.
            let _ = write!(self.out, r"(?i:\{n})");
        } else {
            let _ = write!(self.out, r"(?:\{n})");
        }
    }

    /// `\xhh`, `\x{h..h}` or `\uhhhh`.
    fn hex(&mut self, kind: char, in_class: bool) -> Result<(), String> {
        let digits: String = if self.peek(0) == Some('{') && kind == 'x' {
            self.at += 1;
            let mut s = String::new();
            loop {
                let d = self.peek(0).ok_or("unclosed hex escape")?;
                self.at += 1;
                if d == '}' {
                    break;
                }
                s.push(d);
            }
            s
        } else {
            let n = if kind == 'x' { 2 } else { 4 };
            let s: String = self.chars[self.at..].iter().take(n).collect();
            if s.chars().count() != n {
                return Err("illegal hex escape".to_owned());
            }
            self.at += n;
            s
        };
        let value = u32::from_str_radix(&digits, 16).map_err(|e| e.to_string())?;
        let c = char::from_u32(value).ok_or("illegal code point")?;
        if c.is_ascii_alphabetic() && self.flags().ascii_case() {
            self.fold_literal(c, in_class);
        } else {
            let _ = write!(self.out, r"\x{{{value:X}}}");
        }
        Ok(())
    }

    /// `\p{Name}`, `\P{Name}`, `\pL`.
    fn property(&mut self, c: char) -> Result<(), String> {
        let name: String = if self.peek(0) == Some('{') {
            self.at += 1;
            let mut s = String::new();
            loop {
                let d = self.peek(0).ok_or("unclosed property")?;
                self.at += 1;
                if d == '}' {
                    break;
                }
                s.push(d);
            }
            s
        } else {
            let d = self.peek(0).ok_or("missing property")?;
            self.at += 1;
            d.to_string()
        };
        let negated = c == 'P';
        if let Some((_, body)) = POSIX.iter().find(|(n, _)| *n == name) {
            let _ = write!(self.out, "[{}{body}]", if negated { "^" } else { "" });
            return Ok(());
        }
        let name = match name.as_str() {
            "javaLowerCase" => "Ll".to_owned(),
            "javaUpperCase" => "Lu".to_owned(),
            "javaWhitespace" => "White_Space".to_owned(),
            "javaAlphabetic" | "IsAlphabetic" => "Alphabetic".to_owned(),
            n => n.strip_prefix("Is").unwrap_or(n).to_owned(),
        };
        let _ = write!(self.out, r"\{c}{{{name}}}");
        Ok(())
    }

    /// A bracketed class (the `[` is consumed), possibly nested. The
    /// closed class reads as one atom, so the bracket position is saved
    /// for the `]` to restore.
    fn class(&mut self) -> Result<(), String> {
        self.atom_stack.push(self.out.len());
        self.atom_start = self.out.len();
        self.out.push('[');
        if self.peek(0) == Some('^') {
            self.at += 1;
            self.out.push('^');
        }
        let mut first = true;
        loop {
            self.skip_ignorable();
            let c = self.peek(0).ok_or("unclosed character class")?;
            self.at += 1;
            match c {
                ']' if !first => {
                    self.out.push(']');
                    if let Some(start) = self.atom_stack.pop() {
                        self.atom_start = start;
                    }
                    return Ok(());
                }
                '[' => self.class()?,
                '\\' => self.escape(true)?,
                '&' if self.peek(0) == Some('&') => {
                    self.at += 1;
                    self.out.push_str("&&");
                }
                '&' | '~' => {
                    self.out.push('\\');
                    self.out.push(c);
                }
                '-' => self.out.push('-'),
                c => {
                    let range_end = match (self.peek(0), self.peek(1)) {
                        (Some('-'), Some(h)) if !"]\\[&".contains(h) => Some(h),
                        _ => None,
                    };
                    if let (Some(hi), true) = (range_end, self.flags().ascii_case()) {
                        // `a-z` under ASCII `(?i)` also takes `A-Z`.
                        self.at += 2;
                        push_literal(&mut self.out, c, true);
                        self.out.push('-');
                        push_literal(&mut self.out, hi, true);
                        self.fold_range(c, hi);
                    } else {
                        self.fold_literal(c, true);
                    }
                }
            }
            first = false;
        }
    }
}

/// `c` as a literal, escaped when it is a metacharacter or not printable.
fn push_literal(out: &mut String, c: char, in_class: bool) {
    const META: &str = r"\.+*?()|[]{}^$#&-~";
    if META.contains(c) && !(in_class && c == '-') {
        out.push('\\');
        out.push(c);
    } else if c.is_ascii_graphic() || (!c.is_ascii() && !c.is_control() && !c.is_whitespace()) {
        out.push(c);
    } else {
        let _ = write!(out, r"\x{{{:X}}}", u32::from(c));
    }
}

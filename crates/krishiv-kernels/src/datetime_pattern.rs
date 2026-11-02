// SPDX-License-Identifier: Apache-2.0
//! Spark datetime patterns (`yyyy-MM-dd'T'HH:mm:ss.SSSXXX`): the pattern
//! language of `dateFormat`/`timestampFormat` options, `date_format` and
//! `to_timestamp`, with Spark 3+ semantics (`java.time` formatters as Spark
//! builds them, `spark.sql.legacy.timeParserPolicy=CORRECTED`).
//!
//! - **Compiling** ([`DateTimePattern::new`]) applies Spark's rules before
//!   `java.time`'s: `y` means the proleptic year (`u`) unless the pattern
//!   has an era (`G`); week-based letters, `u` and over-long runs are
//!   rejected; `E`, `F`, `Q` and `q` cannot parse. A rejected pattern fails
//!   with Spark's class: `INCONSISTENT_BEHAVIOR_CROSS_VERSION.*` when the
//!   pre-3.0 formatter would have accepted it, else
//!   `INVALID_DATETIME_PATTERN.*`.
//! - **Formatting** prints fields as `java.time` does: sign and padding
//!   rules per letter count, English month/day/era/AM-PM texts, offsets
//!   (`X`, `x`, `Z`, `O`) and zone ids/names (`VV`, `z`) for UTC and fixed
//!   offsets.
//! - **Parsing** is strict and case-insensitive, consumes the whole input,
//!   reserves the widths of fixed-width numbers that follow a variable-width
//!   one without a separator (`yyyyMMdd`), reads `S`-runs as 1 to n fraction
//!   digits, and fills missing fields as Spark does (year 1970, January,
//!   day 1, midnight). A string that does not match is `None`, never an
//!   error.
//!
//! Only English (`en-US`) texts are known. Zone ids in strings resolve
//! against the bundled time-zone database; short `z`/`zzzz` display names
//! of region zones need CLDR data and fail with
//! `KRISHIV_UNSUPPORTED_FEATURE`. Checked against Spark 4.2 by the
//! `datetime_pattern_differential` test.
#![expect(
    clippy::arithmetic_side_effects,
    reason = "field values are bounded by the pattern's digit limits and checked ranges; \
              conversions to microseconds use checked arithmetic"
)]

mod format;
pub mod legacy;
pub(crate) mod parse;
mod zone_names;

use krishiv_common::error::{
    InconsistentBehaviorCrossVersionDatetimePatternRecognition,
    InconsistentBehaviorCrossVersionDatetimeWeekBasedPattern,
    InvalidDatetimePatternIllegalCharacter, InvalidDatetimePatternLength,
    InvalidDatetimePatternWithSuggestion, KrishivError, KrishivUnsupportedFeature,
};

use crate::cast::TimeZoneRules;

const DOC_ROOT: &str = "https://spark.apache.org/docs/latest";
const POLICY_CONF: &str = "\"spark.sql.legacy.timeParserPolicy\"";

/// What a pattern is compiled for: Spark allows fewer letters for parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// `date_format`, writing files.
    Format,
    /// `to_timestamp`, reading files.
    Parse,
}

/// A zone a timestamp is shown in: its id (for `VV` and `z`) and its rules.
#[derive(Debug, Clone, Copy)]
pub struct Zone<'a> {
    /// The zone id as Spark normalises it (`UTC`, `+05:30`).
    pub id: &'a str,
    /// Its offsets.
    pub rules: &'a dyn TimeZoneRules,
}

/// A numeric field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    /// The proleptic year (`u`, or `y` without an era).
    Year,
    /// The year of era (`y` with `G`).
    YearOfEra,
    Month,
    Day,
    DayOfYear,
    Quarter,
    /// `F`: the aligned week of the month.
    AlignedWeekOfMonth,
    /// `H`: 0..=23.
    HourOfDay,
    /// `k`: 1..=24.
    ClockHourOfDay,
    /// `K`: 0..=11.
    HourOfAmPm,
    /// `h`: 1..=12.
    ClockHourOfAmPm,
    Minute,
    Second,
}

/// `java.time`'s `SignStyle` (the ones patterns produce).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sign {
    /// `-` for negatives only.
    Normal,
    /// `+` when the value has more digits than the minimum width.
    ExceedsPad,
    /// No sign.
    NotNegative,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Number {
    field: Field,
    min: usize,
    max: usize,
    sign: Sign,
    /// Widths of the fixed-width numbers right after this one, reserved
    /// when parsing (`java.time`'s adjacent value parsing); `None` when a
    /// later variable-width number made this one fixed.
    subsequent: Option<usize>,
}

/// A text field and whether it is the full form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Text {
    Era,
    Month,
    AmPm,
    DayOfWeek,
    Quarter,
}

/// An offset printed as `+HH`, `+HH:MM`, ...: whether `:` separates the
/// parts, whether minutes and seconds are required, optional or absent,
/// and the text of a zero offset.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OffsetStyle {
    colon: bool,
    minutes: Part,
    seconds: Part,
    zero: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    Absent,
    Optional,
    Required,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Item {
    Literal(String),
    Number(Number),
    /// `yy`: two digits, base 2000.
    Reduced(Field),
    /// `S`-run: `min..=max` fraction digits.
    Fraction {
        min: usize,
        max: usize,
    },
    Text {
        text: Text,
        full: bool,
    },
    Offset(OffsetStyle),
    /// `O` (short) and `OOOO`/`ZZZZ` (full): `GMT+5:30`, `GMT+05:30`.
    Localized {
        full: bool,
    },
    /// `VV`.
    ZoneId,
    /// `z` (short) and `zzzz` (full).
    ZoneName {
        full: bool,
    },
    Optional(Vec<Item>),
}

/// A compiled Spark datetime pattern (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DateTimePattern {
    pattern: String,
    items: Vec<Item>,
    /// Set under `spark.sql.legacy.timeParserPolicy=LEGACY`: every method
    /// then answers from the `SimpleDateFormat`/`FastDateFormat` pattern.
    legacy: Option<Box<legacy::LegacyPattern>>,
}

/// Why a pattern does not compile, before mapping to Spark's class.
enum Reject {
    WeekBased(char),
    IllegalChar(char),
    Length,
    /// `java.time` rejects it.
    Invalid,
    /// Valid in Spark, not implemented here.
    Unsupported(String),
}

impl DateTimePattern {
    /// Compiles `pattern` for `purpose`.
    ///
    /// # Errors
    /// Spark's error for a pattern Spark rejects;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for letters Krishiv does not implement
    /// (`v`, `g`, non-English locales are not reachable here).
    pub fn new(pattern: &str, purpose: Purpose) -> Result<Self, KrishivError> {
        match compile(pattern, purpose) {
            Ok(items) => Ok(Self {
                pattern: pattern.to_owned(),
                items,
                legacy: None,
            }),
            Err(Reject::Unsupported(what)) => Err(KrishivUnsupportedFeature {
                feature: format!("the datetime pattern letter {what}"),
            }
            .into()),
            Err(reject) => Err(spark_error(pattern, &reject)),
        }
    }

    /// [`Self::new`], or the legacy `SimpleDateFormat`/`FastDateFormat`
    /// pattern when `legacy` (`timeParserPolicy=LEGACY`).
    ///
    /// # Errors
    /// As the chosen pattern kind.
    pub fn new_with(pattern: &str, purpose: Purpose, legacy: bool) -> Result<Self, KrishivError> {
        if !legacy {
            return Self::new(pattern, purpose);
        }
        Ok(Self {
            pattern: pattern.to_owned(),
            items: Vec::new(),
            legacy: Some(Box::new(legacy::LegacyPattern::new(pattern)?)),
        })
    }

    /// The pattern text.
    #[must_use]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Whether the pattern prints or reads a zone or offset.
    #[must_use]
    pub fn has_zone(&self) -> bool {
        if let Some(l) = &self.legacy {
            return l.has_zone();
        }
        fn any(items: &[Item]) -> bool {
            items.iter().any(|i| match i {
                Item::Offset(_) | Item::Localized { .. } | Item::ZoneId | Item::ZoneName { .. } => {
                    true
                }
                Item::Optional(inner) => any(inner),
                _ => false,
            })
        }
        any(&self.items)
    }

    /// Whether the pattern has a time-of-day field.
    #[must_use]
    pub fn has_time(&self) -> bool {
        if self.legacy.is_some() {
            return true;
        }
        fn any(items: &[Item]) -> bool {
            items.iter().any(|i| match i {
                Item::Number(n) => matches!(
                    n.field,
                    Field::HourOfDay
                        | Field::ClockHourOfDay
                        | Field::HourOfAmPm
                        | Field::ClockHourOfAmPm
                        | Field::Minute
                        | Field::Second
                ),
                Item::Fraction { .. }
                | Item::Text {
                    text: Text::AmPm, ..
                } => true,
                Item::Optional(inner) => any(inner),
                _ => false,
            })
        }
        any(&self.items)
    }

    /// `micros` (an instant) shown in `zone`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` when the local time overflows (an
    /// instant within an offset of the `i64` range) or the pattern prints
    /// the name of a zone that needs the time-zone database.
    pub fn format_timestamp(&self, micros: i64, zone: Zone<'_>) -> Result<String, KrishivError> {
        if let Some(l) = &self.legacy {
            return l.format_timestamp(micros, zone);
        }
        let offset = zone.rules.offset_at_utc(micros.div_euclid(1_000_000));
        let local = micros
            .checked_add(i64::from(offset) * 1_000_000)
            .ok_or_else(|| unsupported("a timestamp past the year range in a pattern"))?;
        format::format(&self.items, local, Some((offset, zone.id)), true)
    }

    /// `local` (a `TIMESTAMP_NTZ`, microseconds of the local date-time).
    ///
    /// # Errors
    /// A pattern with a zone or offset (a local date-time has none).
    pub fn format_local(&self, local: i64) -> Result<String, KrishivError> {
        if let Some(l) = &self.legacy {
            return l.format_local(local);
        }
        format::format(&self.items, local, None, true)
    }

    /// `days` (a `DATE`).
    ///
    /// # Errors
    /// A pattern with time-of-day or zone fields (a date has none).
    pub fn format_date(&self, days: i32) -> Result<String, KrishivError> {
        if let Some(l) = &self.legacy {
            return l.format_date(days);
        }
        format::format_days(&self.items, i64::from(days))
    }

    /// `s` as an instant: its zone or offset, else `session`. `Ok(None)`
    /// when `s` does not match, or names a zone id the database cannot
    /// resolve (Spark's lenient readers read it as no timestamp at all).
    ///
    /// # Errors
    /// Never for the zone itself; `INVALID_TIMEZONE` is the scalar zone
    /// arguments' error, not the readers'.
    pub fn parse_timestamp(
        &self,
        s: &str,
        session: &dyn TimeZoneRules,
    ) -> Result<Option<i64>, KrishivError> {
        if let Some(l) = &self.legacy {
            return l.parse_timestamp(s, session);
        }
        let Some(p) = parse::parse(&self.items, s) else {
            return Ok(None);
        };
        let Some(local) = p.local_micros() else {
            return Ok(None);
        };
        match p.zone {
            parse::ParsedZone::None => Ok(crate::cast::datetime::local_to_utc(local, session)),
            parse::ParsedZone::Offset(o) => Ok(local.checked_sub(i64::from(o) * 1_000_000)),
            parse::ParsedZone::Region(id) => {
                let Ok(zone) = crate::datetime::resolve_zone(&id) else {
                    return Ok(None);
                };
                Ok(crate::cast::datetime::local_to_utc(local, &zone))
            }
        }
    }

    /// `s` as a local date-time (`TIMESTAMP_NTZ`). A zone or offset in `s`
    /// is ignored when `allow_zone`, else the string does not match. An
    /// unresolvable zone id does not match either.
    pub fn parse_local(&self, s: &str, allow_zone: bool) -> Result<Option<i64>, KrishivError> {
        if let Some(l) = &self.legacy {
            return l.parse_local(s);
        }
        let Some(p) = parse::parse(&self.items, s) else {
            return Ok(None);
        };
        match &p.zone {
            parse::ParsedZone::None => {}
            _ if !allow_zone => return Ok(None),
            parse::ParsedZone::Offset(_) => {}
            parse::ParsedZone::Region(id) => {
                if crate::datetime::resolve_zone(id).is_err() {
                    return Ok(None);
                }
            }
        }
        Ok(p.local_micros())
    }

    /// `s` as a date (days since the epoch), as Spark's CSV and JSON
    /// readers parse dates: time and zone fields are read but ignored. An
    /// unresolvable zone id does not match either.
    pub fn parse_date(&self, s: &str) -> Result<Option<i32>, KrishivError> {
        if let Some(l) = &self.legacy {
            return l.parse_date(s);
        }
        let Some(p) = parse::parse(&self.items, s) else {
            return Ok(None);
        };
        if let parse::ParsedZone::Region(id) = &p.zone
            && crate::datetime::resolve_zone(id).is_err()
        {
            return Ok(None);
        }
        Ok(p.days().and_then(|d| i32::try_from(d).ok()))
    }

    /// `s` with Java's failure detail: the resolved local date-time and
    /// zone, or how the match or resolution failed. The batch parsers of
    /// [`crate::datetime::format`] report it as Spark does.
    pub(crate) fn parse_detailed(&self, s: &str) -> Result<parse::Resolved, parse::Fail> {
        parse::parse_detailed(&self.items, s)
    }
}

fn unsupported(what: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: what.to_owned(),
    }
    .into()
}

fn spark_error(pattern: &str, reject: &Reject) -> KrishivError {
    if legacy_accepts(pattern) {
        return InconsistentBehaviorCrossVersionDatetimePatternRecognition {
            pattern: format!("'{pattern}'"),
            config: POLICY_CONF.to_owned(),
            docroot: DOC_ROOT.to_owned(),
        }
        .into();
    }
    match *reject {
        Reject::WeekBased(c) => InconsistentBehaviorCrossVersionDatetimeWeekBasedPattern {
            c: format!("'{c}'"),
        }
        .into(),
        Reject::IllegalChar(c) => InvalidDatetimePatternIllegalCharacter {
            pattern: format!("'{pattern}'"),
            c: format!("'{c}'"),
        }
        .into(),
        Reject::Length => InvalidDatetimePatternLength {
            pattern: format!("'{pattern}'"),
        }
        .into(),
        Reject::Invalid | Reject::Unsupported(_) => InvalidDatetimePatternWithSuggestion {
            pattern: format!("'{pattern}'"),
            docroot: DOC_ROOT.to_owned(),
        }
        .into(),
    }
}

/// Whether Java's `SimpleDateFormat` (Spark's legacy formatter) accepts
/// `pattern`: it decides between Spark's two error classes.
fn legacy_accepts(pattern: &str) -> bool {
    const LETTERS: &str = "GyYMLwWDdFEuaHkKhmsSzZX";
    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' {
            // A quoted run; `''` is a quote.
            i += 1;
            loop {
                match chars.get(i) {
                    None => return false,
                    Some('\'') if chars.get(i + 1) == Some(&'\'') => i += 2,
                    Some('\'') => break,
                    Some(_) => i += 1,
                }
            }
            i += 1;
        } else if c.is_ascii_alphabetic() {
            let run = chars[i..].iter().take_while(|&&x| x == c).count();
            if !LETTERS.contains(c) || (c == 'X' && run > 3) {
                return false;
            }
            i += run;
        } else {
            i += 1;
        }
    }
    true
}

/// `java.time`'s `appendPattern` with Spark's rewrites and checks.
fn compile(pattern: &str, purpose: Purpose) -> Result<Vec<Item>, Reject> {
    let chars: Vec<char> = pattern.chars().collect();
    let era = has_unquoted(&chars, 'G');
    // Spark's checks run on the unquoted text before java.time sees it.
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' {
            i = skip_quoted(&chars, i).ok_or(Reject::Invalid)?;
            continue;
        }
        match c {
            'Y' | 'W' | 'w' | 'u' | 'e' | 'c' => return Err(Reject::WeekBased(c)),
            'A' | 'n' | 'N' | 'p' | 'B' => return Err(Reject::IllegalChar(c)),
            // `E` and `F` print but cannot parse; so do `Q` and `q`.
            'Q' | 'q' | 'E' | 'F' if purpose == Purpose::Parse => {
                return Err(Reject::IllegalChar(c));
            }
            _ => {}
        }
        i += 1;
    }
    let mut stack: Vec<Vec<Item>> = vec![Vec::new()];
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let item = if c == '\'' {
            let end = skip_quoted(&chars, i).ok_or(Reject::Invalid)?;
            let inner: String = chars[i + 1..end - 1].iter().collect();
            i = end;
            Item::Literal(if inner.is_empty() {
                "'".to_owned()
            } else {
                inner.replace("''", "'")
            })
        } else if c == '[' {
            stack.push(Vec::new());
            i += 1;
            continue;
        } else if c == ']' {
            if stack.len() < 2 {
                return Err(Reject::Invalid);
            }
            let inner = stack.pop().unwrap_or_default();
            i += 1;
            Item::Optional(inner)
        } else if matches!(c, '{' | '}' | '#') {
            return Err(Reject::Invalid);
        } else if c.is_ascii_alphabetic() {
            let run = chars[i..].iter().take_while(|&&x| x == c).count();
            i += run;
            letter(c, run, era, purpose)?
        } else {
            i += 1;
            Item::Literal(c.to_string())
        };
        push(stack.last_mut().ok_or(Reject::Invalid)?, item);
    }
    // Unclosed optional sections end with the pattern.
    while stack.len() > 1 {
        let inner = stack.pop().unwrap_or_default();
        push(
            stack.last_mut().ok_or(Reject::Invalid)?,
            Item::Optional(inner),
        );
    }
    let items = stack.pop().unwrap_or_default();
    Ok(adjacent(items))
}

fn push(items: &mut Vec<Item>, item: Item) {
    // Adjacent literals merge (one match, one print).
    if let (Some(Item::Literal(prev)), Item::Literal(next)) = (items.last_mut(), &item) {
        prev.push_str(next);
        return;
    }
    items.push(item);
}

/// `java.time`'s adjacent value parsing: a variable-width number followed
/// directly by fixed-width numbers reserves their widths.
fn adjacent(items: Vec<Item>) -> Vec<Item> {
    let mut out: Vec<Item> = Vec::with_capacity(items.len());
    let mut active: Option<usize> = None;
    for item in items {
        match item {
            Item::Number(n) => {
                let fixed = n.min == n.max && n.sign == Sign::NotNegative;
                if let Some(a) = active
                    && let Some(Item::Number(base)) = out.get_mut(a)
                {
                    if fixed {
                        base.subsequent = base.subsequent.map(|w| w + n.max);
                        out.push(Item::Number(n));
                        continue;
                    }
                    base.subsequent = None;
                }
                active = Some(out.len());
                out.push(Item::Number(n));
            }
            Item::Reduced(f) => {
                // A two-digit reduced value is fixed width (NOT_NEGATIVE).
                if let Some(a) = active
                    && let Some(Item::Number(base)) = out.get_mut(a)
                {
                    base.subsequent = base.subsequent.map(|w| w + 2);
                    out.push(Item::Reduced(f));
                    continue;
                }
                active = Some(out.len());
                out.push(Item::Reduced(f));
            }
            Item::Optional(inner) => {
                active = None;
                out.push(Item::Optional(adjacent(inner)));
            }
            other => {
                active = None;
                out.push(other);
            }
        }
    }
    out
}

fn has_unquoted(chars: &[char], letter: char) -> bool {
    let mut quoted = false;
    chars.iter().any(|&c| {
        if c == '\'' {
            quoted = !quoted;
        }
        !quoted && c == letter
    })
}

/// The index after the quoted run starting at `start`, or `None` if it is
/// not closed.
fn skip_quoted(chars: &[char], start: usize) -> Option<usize> {
    let mut i = start + 1;
    loop {
        match chars.get(i)? {
            '\'' if chars.get(i + 1) == Some(&'\'') && i > start + 1 => i += 2,
            '\'' => return Some(i + 1),
            _ => i += 1,
        }
    }
}

fn number(field: Field, min: usize, max: usize, sign: Sign) -> Item {
    Item::Number(Number {
        field,
        min,
        max,
        sign,
        subsequent: Some(0),
    })
}

fn offset(colon: bool, minutes: Part, seconds: Part, zero: &'static str) -> Item {
    Item::Offset(OffsetStyle {
        colon,
        minutes,
        seconds,
        zero,
    })
}

/// One run of a letter, as `java.time`'s `appendPattern` maps it.
fn letter(c: char, n: usize, era: bool, purpose: Purpose) -> Result<Item, Reject> {
    use Field as F;
    use Part::{Absent, Optional, Required};
    let simple = |field| match n {
        1 => Ok(number(field, 1, 19, Sign::Normal)),
        2 => Ok(number(field, 2, 2, Sign::NotNegative)),
        _ => Err(Reject::Invalid),
    };
    let _ = purpose;
    Ok(match c {
        'G' => match n {
            1..=3 => Item::Text {
                text: Text::Era,
                full: false,
            },
            4 => Item::Text {
                text: Text::Era,
                full: true,
            },
            _ => return Err(Reject::Invalid),
        },
        'y' => {
            let field = if era { F::YearOfEra } else { F::Year };
            match n {
                1 => number(field, 1, 19, Sign::Normal),
                2 => Item::Reduced(field),
                3 => number(field, 3, 19, Sign::Normal),
                4..=6 => number(field, n, 19, Sign::ExceedsPad),
                _ => return Err(Reject::Invalid),
            }
        }
        'M' | 'L' => match n {
            1 => number(F::Month, 1, 19, Sign::Normal),
            2 => number(F::Month, 2, 2, Sign::NotNegative),
            3 | 4 => Item::Text {
                text: Text::Month,
                full: n == 4,
            },
            _ => return Err(Reject::Invalid),
        },
        'd' => simple(F::Day)?,
        'D' => match n {
            1 => number(F::DayOfYear, 1, 19, Sign::Normal),
            2 => number(F::DayOfYear, 2, 3, Sign::NotNegative),
            3 => number(F::DayOfYear, 3, 3, Sign::NotNegative),
            _ => return Err(Reject::Invalid),
        },
        'Q' | 'q' => match n {
            1 => number(F::Quarter, 1, 19, Sign::Normal),
            2 => number(F::Quarter, 2, 2, Sign::NotNegative),
            3 | 4 => Item::Text {
                text: Text::Quarter,
                full: n == 4,
            },
            _ => return Err(Reject::Length),
        },
        'E' => match n {
            1..=3 => Item::Text {
                text: Text::DayOfWeek,
                full: false,
            },
            4 => Item::Text {
                text: Text::DayOfWeek,
                full: true,
            },
            _ => return Err(Reject::Invalid),
        },
        'F' if n == 1 => number(F::AlignedWeekOfMonth, 1, 19, Sign::Normal),
        'a' if n == 1 => Item::Text {
            text: Text::AmPm,
            full: false,
        },
        'H' => simple(F::HourOfDay)?,
        'k' => simple(F::ClockHourOfDay)?,
        'K' => simple(F::HourOfAmPm)?,
        'h' => simple(F::ClockHourOfAmPm)?,
        'm' => simple(F::Minute)?,
        's' => simple(F::Second)?,
        'S' if n <= 9 => Item::Fraction { min: n, max: n },
        'X' | 'x' => {
            let zero = |z: &'static str, nz: &'static str| if c == 'X' { z } else { nz };
            match n {
                1 => offset(false, Optional, Absent, zero("Z", "+00")),
                2 => offset(false, Required, Absent, zero("Z", "+0000")),
                3 => offset(true, Required, Absent, zero("Z", "+00:00")),
                4 => offset(false, Required, Optional, zero("Z", "+0000")),
                5 => offset(true, Required, Optional, zero("Z", "+00:00")),
                _ => return Err(Reject::Invalid),
            }
        }
        'Z' => match n {
            1..=3 => offset(false, Required, Absent, "+0000"),
            4 => Item::Localized { full: true },
            5 => offset(true, Required, Optional, "Z"),
            _ => return Err(Reject::Invalid),
        },
        'O' => match n {
            1 => Item::Localized { full: false },
            4 => Item::Localized { full: true },
            _ => return Err(Reject::Invalid),
        },
        'V' if n == 2 => Item::ZoneId,
        'z' => match n {
            1..=3 => Item::ZoneName { full: false },
            4 => Item::ZoneName { full: true },
            _ => return Err(Reject::Invalid),
        },
        'v' | 'g' => return Err(Reject::Unsupported(format!("'{c}'"))),
        _ => return Err(Reject::Invalid),
    })
}

/// The offset in seconds of a zone id Spark accepts without the time-zone
/// database: `UTC`, `GMT`, `UT`, `Z`, `Etc/UTC`, `Etc/GMT`, and fixed
/// offsets (`+05:30`, `-8`, `UTC+01:00`, `GMT-5`). `None` for region ids
/// and malformed ids.
#[must_use]
pub fn fixed_zone_offset(id: &str) -> Option<i32> {
    match id {
        "UTC" | "GMT" | "UT" | "Z" | "Etc/UTC" | "Etc/GMT" | "Etc/UCT" | "UCT"
        | "Etc/Universal" | "Etc/Zulu" | "Universal" | "Zulu" | "Etc/GMT0" | "Etc/GMT+0"
        | "Etc/GMT-0" | "Etc/Greenwich" | "GMT0" | "Greenwich" => return Some(0),
        _ => {}
    }
    let rest = ["UTC", "GMT", "UT"]
        .iter()
        .find_map(|p| id.strip_prefix(p))
        .unwrap_or(id);
    parse::offset_id(rest)
}

#[cfg(test)]
mod tests;

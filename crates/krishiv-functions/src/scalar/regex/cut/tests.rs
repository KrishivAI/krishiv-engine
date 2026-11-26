// SPDX-License-Identifier: Apache-2.0
//! The cut against the kernels' own capture search.
//!
//! [`agrees`] runs inside every cut search of a test build, so each regex test of the crate
//! that reaches a cut pattern is a differential test too.

use arrow_array::Array as _;
use arrow_array::cast::AsArray;

use super::super::{Kept, compiled_on_thread};
use super::{Cut, Found, PROBE, delegated, unquote};
use crate::scalar::tests::{ints, run, strs};

/// Every group span of one match.
type Spans = Vec<Option<(usize, usize)>>;

/// Panics unless `found` is what the engine's capture search of `plain` finds in
/// `text[from..]`: the same match or none, and the same span for every group.
pub(in super::super) fn agrees(
    plain: &fancy_regex::Regex,
    text: &str,
    from: usize,
    found: Option<&Found<'_>>,
) {
    let want: Option<Spans> = plain.captures_from_pos(text, from).unwrap().map(|caps| {
        (0..caps.len())
            .map(|g| caps.get(g).map(|m| (m.start(), m.end())))
            .collect()
    });
    let got: Option<Spans> = found.map(|f| (0..plain.captures_len()).map(|g| f.span(g)).collect());
    assert_eq!(
        got,
        want,
        "cut against the engine: pattern {:?}, text {text:?}, from {from}",
        plain.as_str()
    );
}

/// The cut of the Java pattern `pattern`, as a kernel gets it.
fn cut_of(pattern: &str) -> Option<Cut> {
    Kept::new(pattern, "`regexp_extract`").unwrap().cut
}

/// The spans of the match of `cut` in `text[from..]`.
fn spans(cut: &Cut, text: &str, from: usize) -> Option<Spans> {
    cut.search(text, from)
        .map(|f| (0..=cut.groups()).map(|g| f.span(g)).collect())
}

const DOMAIN: &str = r"^https?://(?:www\.)?([^/]+)/.*$";

#[test]
fn the_domain_pattern_is_cut_after_its_group() {
    let cut = cut_of(DOMAIN).unwrap();
    assert_eq!(cut.groups(), 1);
    assert!(cut.start_anchored);
    let text = "http://www.example.com/a/b?c=d";
    assert_eq!(
        spans(&cut, text, 0),
        Some(vec![Some((0, text.len())), Some((11, 22))])
    );
    assert_eq!(cut.probed(), (1, 0));
    // No path: the head matches, the tail does not, and the full search finds no match.
    assert_eq!(spans(&cut, "http://example.com", 0), None);
    assert_eq!(cut.probed(), (2, 1));
    // The head does not match at the start: no match, without the full search.
    assert_eq!(spans(&cut, "ftp://example.com/x", 0), None);
    assert_eq!(cut.probed(), (2, 1));
    // A pattern anchored at the start never matches from further in.
    assert_eq!(spans(&cut, text, 1), None);
    assert_eq!(spans(&cut, "", 0), None);
}

#[test]
fn a_later_choice_of_the_head_falls_back_to_the_full_search() {
    let cut = cut_of("(a|ab)c.*").unwrap();
    assert!(!cut.start_anchored);
    // The head alone stops after `a`, where `c` does not follow.
    assert_eq!(
        spans(&cut, "xabcd", 0),
        Some(vec![Some((1, 5)), Some((1, 3))])
    );
    assert_eq!(cut.probed(), (1, 1));
    assert_eq!(
        spans(&cut, "xacd", 0),
        Some(vec![Some((1, 4)), Some((1, 2))])
    );
    assert_eq!(cut.probed(), (2, 1));
    assert_eq!(spans(&cut, "xabcd", 2), None);
    // The second match of a text, and a group that took no part in the match.
    let cut = cut_of("(?:(a)|b)c.*?;").unwrap();
    assert_eq!(spans(&cut, "ac;bc;", 3), Some(vec![Some((3, 6)), None]));
    assert_eq!(
        spans(&cut, "ac;bc;", 0),
        Some(vec![Some((0, 3)), Some((0, 1))])
    );
}

#[test]
fn a_two_byte_character_on_both_sides_of_the_cut() {
    let cut = cut_of("(é+)é.*").unwrap();
    assert_eq!(
        spans(&cut, "xééé!", 0),
        Some(vec![Some((1, 8)), Some((1, 5))])
    );
    assert_eq!(
        spans(&cut, "xééé!", 3),
        Some(vec![Some((3, 8)), Some((3, 5))])
    );
    assert_eq!(spans(&cut, "xééé!", 5), None);
}

#[test]
fn patterns_without_a_cut() {
    for pattern in [
        // No group, or nothing after the last one.
        "abc.*",
        "a(b).*(c)",
        // The tail matches a bounded length: little to save.
        "(a|ab)c",
        "(a)b{1,3}",
        // The top level is not a concatenation.
        "(a)b.*|c",
        "((a)b.*)",
        // Look-around, back-references and atomic groups: the engine runs these itself.
        "(a)b.*(?=c).*",
        r"(a)\1.*",
        "(a)b*+.*",
        // `$` under the multiline flag is a look-ahead.
        "(?m)(a)b.*$.*",
    ] {
        assert!(cut_of(pattern).is_none(), "{pattern}");
    }
    // A part past the size limit.
    assert!(Cut::new("(a)[^a]{60000}.*").is_none());
    assert!(Cut::new("(a)[^a]{6}.*").is_some());
    for pattern in ["(a)b.*", "^(a)(?<n>b)c*", r"\Q(\E(a)[^)]*\)+", "(?i)(a)b.*"] {
        assert!(cut_of(pattern).is_some(), "{pattern}");
    }
}

#[test]
fn the_engines_own_pattern_is_read_from_its_debug_print() {
    let regex = |text: &str| fancy_regex::Regex::new(text).unwrap();
    // A pattern the engine delegates whole, and one it runs itself.
    assert_eq!(delegated(&regex("(ab|)*c*")), Some("(ab|)*c*".to_owned()));
    assert_eq!(delegated(&regex(r"(a)\1")), None);
    assert_eq!(
        unquote(r#""a\"b\\c\n\r\t\0\'\u{85}é""#),
        Some("a\"b\\c\n\r\t\0'\u{85}é".to_owned())
    );
    for print in [
        "a",
        "\"a",
        "a\"",
        r#""\q""#,
        r#""\u{85""#,
        r#""\u85}""#,
        r#""\u{zz}""#,
        r#""a\""#,
    ] {
        assert_eq!(unquote(print), None, "{print}");
    }
}

#[test]
fn a_cut_that_mostly_falls_back_is_dropped() {
    // The head takes every `a`, so the tail never finds its own.
    let cut = cut_of("(a+)a.*").unwrap();
    for n in 0..PROBE - 1 {
        assert_eq!(
            spans(&cut, "aaa", 0),
            Some(vec![Some((0, 3)), Some((0, 2))])
        );
        assert!(cut.is_on(), "{n}");
    }
    assert!(spans(&cut, "aaa", 0).is_some());
    assert!(!cut.is_on());
    assert_eq!(cut.probed(), (PROBE, PROBE));
    // A quarter of the probe falling back is still fine.
    let cut = cut_of("(a|ab)c.*").unwrap();
    for n in 0..PROBE * 2 {
        let text = if n % 4 == 0 { "abc" } else { "ac" };
        assert!(spans(&cut, text, 0).is_some());
    }
    assert!(cut.is_on());
    assert_eq!(cut.probed(), (PROBE, PROBE / 4));
    cut.rearm();
    assert_eq!(cut.probed(), (0, 0));
}

#[test]
fn the_kernels_give_the_engine_results_with_the_cut_dropped() {
    let kept = Kept::new("(a+)(b?)a.*", "`regexp_extract`").unwrap();
    assert!(kept.cut.is_some());
    let group = |text: &str| {
        let found = kept.captures_at(text, 0).unwrap();
        found.map(|g| {
            (
                g.whole(),
                g.get(1).map(str::to_owned),
                g.get(2).map(str::to_owned),
            )
        })
    };
    let want = Some((Some((1, 6)), Some("aa".to_owned()), Some(String::new())));
    for _ in 0..=PROBE {
        assert_eq!(group("xaaabc"), want);
    }
    assert!(!kept.cut.as_ref().unwrap().is_on());
    assert_eq!(group("xaaabc"), want);
    assert_eq!(group("xa"), None);
}

/// One-row `regexp_replace` and `regexp_extract` through the bound kernels.
fn replaced(text: &str, pattern: &str, rep: &str, position: i32) -> Option<String> {
    let out = run(
        "regexp_replace",
        &[
            &strs(&[Some(text)]),
            &strs(&[Some(pattern)]),
            &strs(&[Some(rep)]),
            &ints(&[Some(position)]),
        ],
    );
    let out = out.as_string::<i32>();
    out.is_valid(0).then(|| out.value(0).to_owned())
}

fn extracted(text: &str, pattern: &str, idx: i32) -> Option<String> {
    let out = run(
        "regexp_extract",
        &[
            &strs(&[Some(text)]),
            &strs(&[Some(pattern)]),
            &ints(&[Some(idx)]),
        ],
    );
    let out = out.as_string::<i32>();
    out.is_valid(0).then(|| out.value(0).to_owned())
}

#[test]
fn the_kernels_use_the_cut() {
    assert!(
        compiled_on_thread(DOMAIN, "`regexp_replace`")
            .unwrap()
            .cut
            .is_some()
    );
    let some = |s: &str| Some(s.to_owned());
    assert_eq!(
        replaced("https://www.example.com/a/b", DOMAIN, "$1", 1),
        some("example.com")
    );
    assert_eq!(
        replaced("https://example.com", DOMAIN, "$1", 1),
        some("https://example.com")
    );
    // An input ending in a line terminator takes the other variant of `$`, without the cut.
    assert_eq!(
        replaced("http://example.com/a\n", DOMAIN, "$1", 1),
        some("example.com\n")
    );
    assert_eq!(
        extracted("http://www.example.com/a", DOMAIN, 1),
        some("example.com")
    );
    assert_eq!(
        extracted("http://www.example.com/a", DOMAIN, 0),
        some("http://www.example.com/a")
    );
    assert_eq!(extracted("example.com/a", DOMAIN, 1), some(""));
    // Several matches in a row, an empty one included, and a `position` past the first.
    assert_eq!(
        replaced("k1=v1;k2=v2;k3", "(k[0-9])=?[^;]*", "<$1>", 1),
        some("<k1>;<k2>;<k3>")
    );
    assert_eq!(
        replaced("k1=v1;k2=v2;k3", "(k[0-9])=?[^;]*", "<$1>", 2),
        some("k1=v1;<k2>;<k3>")
    );
    assert_eq!(replaced("ab", "(x?)[^b]*", "[$1]", 1), some("[][]b[]"));
    assert_eq!(replaced("éaé", "(é?)[^é]*", "[$1]", 2), some("é[][é][]"));
    // A group that took no part is empty in the replacement and in `regexp_extract`.
    assert_eq!(replaced("bc;", "(?:(a)|b)c.*", "<$1>", 1), some("<>"));
    assert_eq!(extracted("bc;", "(?:(a)|b)c.*", 1), some(""));
    // A pattern anchored at the start, searched from a later `position`: the text from
    // there on is what the pattern sees.
    assert_eq!(replaced("xab", "^(a)b.*", "<$1>", 2), some("x<a>"));
    assert_eq!(replaced("xab", "^(a)b.*", "<$1>", 3), some("xab"));
}

/// A xorshift generator: the fuzz is the same on every run.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % n as u64).unwrap()
    }
}

/// Appends a random Java pattern piece: alternations, groups (nested and repeated), greedy
/// and lazy repetition, anchors, a word boundary, an empty branch and a two-byte character.
fn piece(r: &mut Rng, depth: u32, out: &mut String) {
    const ATOMS: &[&str] = &[
        "a", "b", "c", "ab", "bc", ".", "[ab]", "[^a]", r"\b", "é", "$", "^", r"\z", "",
    ];
    const REPEATS: &[&str] = &["*", "+", "?", "*?", "+?", "??", "{1,2}"];
    match if depth == 0 { 0 } else { r.below(8) } {
        0 | 1 => out.push_str(ATOMS[r.below(ATOMS.len())]),
        2 => {
            for _ in 0..=r.below(3) {
                piece(r, depth - 1, out);
            }
        }
        3 => {
            out.push_str("(?:");
            piece(r, depth - 1, out);
            out.push('|');
            piece(r, depth - 1, out);
            out.push(')');
        }
        4 => {
            out.push('(');
            piece(r, depth - 1, out);
            out.push(')');
        }
        5 => {
            out.push_str("(?:");
            piece(r, depth - 1, out);
            out.push(')');
            out.push_str(REPEATS[r.below(REPEATS.len())]);
        }
        6 => {
            out.push('(');
            piece(r, depth - 1, out);
            out.push('|');
            piece(r, depth - 1, out);
            out.push(')');
            out.push_str(["", "*", "?", "+?"][r.below(4)]);
        }
        _ => {
            piece(r, depth - 1, out);
            out.push_str(".*");
        }
    }
}

/// Random Java patterns through the translator, each cut pattern searched from every
/// character of random short texts: [`agrees`] compares every search with the engine's.
#[test]
fn random_patterns_agree_with_the_engine() {
    const ALPHABET: &[&str] = &["a", "b", "c", "é", "\n", "ab"];
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut cut_patterns, mut matched, mut fell) = (0_u32, 0_u32, 0_u32);
    for _ in 0..1500 {
        let mut pattern = String::new();
        if r.below(3) == 0 {
            pattern.push('^');
        }
        for _ in 0..2 + r.below(3) {
            piece(&mut r, 3, &mut pattern);
        }
        // The generator also writes what Java refuses (a dangling `*`).
        let Ok(kept) = Kept::new(&pattern, "`regexp_extract`") else {
            continue;
        };
        let Some(cut) = &kept.cut else { continue };
        cut_patterns += 1;
        for _ in 0..24 {
            let mut text = String::new();
            for _ in 0..r.below(9) {
                text.push_str(ALPHABET[r.below(ALPHABET.len())]);
            }
            cut.rearm();
            let mut from = 0;
            loop {
                if kept.captures_at(&text, from).unwrap().is_some() {
                    matched += 1;
                }
                match text[from..].chars().next() {
                    Some(c) => from += c.len_utf8(),
                    None => break,
                }
            }
            fell += cut.probed().1;
        }
    }
    // The run has to reach all three outcomes, or it proves nothing.
    let seen = format!("{cut_patterns} cut patterns, {matched} matches, {fell} fallbacks");
    assert!(
        cut_patterns > 100 && matched > 2_000 && fell > 400,
        "{seen}"
    );
}

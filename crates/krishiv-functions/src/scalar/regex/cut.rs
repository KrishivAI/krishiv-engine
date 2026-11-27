// SPDX-License-Identifier: Apache-2.0
//! Capture groups without walking a pattern's capture-free tail (perf C-1b).
//!
//! A capture search runs a backtracking or simulating engine over every byte of the match,
//! at roughly ten times the cost a byte of the lazy DFA that only finds where a match is.
//! A pattern like `^https?://(?:www\.)?([^/]+)/.*$` has its last group early: the tail
//! `/.*$` holds no group, yet the capture engine walks it too.
//!
//! [`Cut`] writes the pattern as a concatenation `head tail`, where `tail` is the longest
//! suffix of the top-level concatenation without a group, and searches in three steps:
//! 1. the start `s` of the match: 0 for a pattern that can only match at the start of the
//!    text, otherwise the start the DFA finds;
//! 2. a capture search of `head` anchored at `s`, ending at `p`;
//! 3. an anchored DFA search of `tail` at `p`. When it matches, ending at `e`, the answer is
//!    the whole match `(s, e)` with `head`'s groups.
//!
//! When step 3 finds nothing, the whole pattern is searched for captures as before.
//!
//! This is exact under leftmost-first semantics: among the matches starting at `s`, the
//! winner is the first in backtracking order, and for a concatenation that order tries
//! `head`'s alternatives before `tail`'s. So the winner is `head`'s first path that ends
//! where `tail` matches at all, followed by `tail`'s first path from there. Step 2 returns
//! `head`'s first path; if `tail` matches at its end, it is the winner's. If not, the winner
//! uses a later path of `head`, which step 2 cannot name, and the full search answers.
//! Example of the fallback: `(a|ab)c.*` on `abc`, where `head` alone stops after `a`.
//!
//! All searches run on the whole text with a span, never on a slice, so an assertion at the
//! cut (`\b`, `\z`) sees the bytes on both sides.
//!
//! The parts are compiled from the pattern text the kernels' engine, `fancy_regex`, prints
//! for the pattern it delegates to `regex_automata` ([`delegated`]), parsed once with the
//! default parser. The engine builds its own automaton from its `to_hir` translation of the
//! pattern, which upstream defines (and tests) to equal that parse, so the cut and the engine
//! search equal patterns. An upgrade of `fancy_regex` can drift in two places: the printed
//! text and `to_hir`. Either one fails safe (no cut, or a test of this module fails). A
//! pattern the engine runs itself (look-around, back-references, atomic groups) has no cut.
//! In tests every cut search is compared with the engine's.
//!
//! Whether the cut pays is probed on [`PROBE`] searches. A cut the probe drops rests for
//! [`REST`] searches and is then probed again: the pattern outlives the query on its thread,
//! and one sample must not decide for later data. The rest is counted in searches, not
//! batches, so a pattern that always falls back pays the probe on a fixed small share of its
//! rows whatever the batch size.

use std::cell::{Cell, Ref, RefCell};
use std::fmt;

use regex_automata::meta::{Config, Regex};
use regex_automata::util::captures::Captures;
use regex_automata::{Anchored, Input};
use regex_syntax::ParserBuilder;
use regex_syntax::hir::{Hir, HirKind, Look};

#[cfg(test)]
pub(super) mod tests;

/// Searches that reached step 3 before the cut decides whether it pays.
const PROBE: u32 = 256;

/// The cut is dropped when more than one in this many probed searches fell back.
const FALLBACK_SHARE: u32 = 4;

/// Searches the engine answers after the probe dropped the cut, before it is probed again.
const REST: u32 = 64 * PROBE;

/// The most heap one part's automaton may take; a pattern over it gets no cut. The three
/// parts stay on the thread with the compiled pattern and are not charged to the memory pool.
const PART_LIMIT: usize = 1 << 20;

/// A pattern split after its last capture group.
pub(super) struct Cut {
    /// The whole pattern: finds the start of a match, and answers when the cut cannot.
    full: Regex,
    /// The pattern up to and including its last group.
    head: Regex,
    /// The rest: no group, and it can match text of any length.
    tail: Regex,
    /// Whether the pattern can only match at the start of the text.
    start_anchored: bool,
    /// Capture groups, the whole match not counted.
    groups: usize,
    /// The capture slots of the two capture searches, reused across rows.
    slots: RefCell<Slots>,
    /// Searches that reached step 3, counted up to [`PROBE`].
    tried: Cell<u32>,
    /// How many of those fell back to the full search.
    fell: Cell<u32>,
    /// Set while the cut rests: the last probe found that it does not pay for the input.
    off: Cell<bool>,
    /// Searches the engine answered since the cut was dropped, counted up to [`REST`].
    rested: Cell<u32>,
}

/// Capture slots for `head` and for the whole pattern.
struct Slots {
    head: Captures,
    full: Captures,
}

/// The slots are lent to a [`Found`] that is still alive: the caller searches with the
/// engine instead.
#[derive(Debug)]
pub(super) struct Busy;

/// One match: the whole match and where the groups are. It borrows the cut's capture
/// slots, so a search while it is alive answers [`Busy`].
pub(super) struct Found<'c> {
    whole: (usize, usize),
    groups: Ref<'c, Captures>,
}

impl Found<'_> {
    /// The byte span of group `group` (0 is the whole match), `None` for a group that took
    /// no part in the match.
    pub(super) fn span(&self, group: usize) -> Option<(usize, usize)> {
        if group == 0 {
            return Some(self.whole);
        }
        self.groups.get_group(group).map(|s| (s.start, s.end))
    }
}

/// The pattern `fancy_regex` hands to `regex_automata` for `regex`, when it delegates the
/// pattern whole and as written (no group added for the match bounds).
///
/// `fancy_regex` does not compile the text it is given: it parses it with its own parser,
/// rewrites the tree (nested repetitions, for one) and builds the automaton from the
/// result, so a cut built from the original text can place a group differently from the
/// engine (`(ab|\b)*` after `abab`). The rewritten pattern is only reachable through the
/// crate's debug print, which this reads; any other shape of that print means no cut, and
/// the tests pin the shape and compare every cut search with the engine's.
pub(super) fn delegated(regex: &fancy_regex::Regex) -> Option<String> {
    struct Printed<'a>(&'a fancy_regex::Regex);
    impl fmt::Debug for Printed<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.debug_print(f)
        }
    }
    let printed = format!("{:?}", Printed(regex));
    let quoted = printed
        .strip_prefix("wrapped Regex ")?
        .strip_suffix(", explicit_capture_group_0: false")?;
    unquote(quoted)
}

/// The string a `{:?}` print of it reads as, `None` for anything else.
fn unquote(quoted: &str) -> Option<String> {
    let inner = quoted.strip_prefix('"')?.strip_suffix('"')?;
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        out.push(match chars.next()? {
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            '0' => '\0',
            c @ ('\\' | '"' | '\'') => c,
            'u' => {
                let rest = chars.as_str().strip_prefix('{')?;
                let (hex, after) = rest.split_once('}')?;
                chars = after.chars();
                char::from_u32(u32::from_str_radix(hex, 16).ok()?)?
            }
            _ => return None,
        });
    }
    Some(out)
}

impl Cut {
    /// The cut of the pattern `text`, as [`delegated`] returns it; `None` when it has none
    /// worth building: the top level is not a concatenation, there is no group or nothing
    /// after the last one, the tail can only match a bounded length (little to save), or a
    /// part is too large.
    pub(super) fn new(text: &str) -> Option<Self> {
        // The defaults (Unicode on, matches are valid UTF-8) are the ones `fancy_regex`
        // compiles a pattern it delegates whole with.
        let hir = ParserBuilder::new().build().parse(text).ok()?;
        let HirKind::Concat(parts) = hir.kind() else {
            return None;
        };
        let last = parts
            .iter()
            .rposition(|h| h.properties().explicit_captures_len() > 0)?;
        let (head, tail) = parts.split_at(last + 1);
        if tail.is_empty() {
            return None;
        }
        let head = Hir::concat(head.to_vec());
        let tail = Hir::concat(tail.to_vec());
        if tail.properties().maximum_len().is_some() {
            return None;
        }
        let build = |hir: &Hir, prefilter: bool| {
            Regex::builder()
                .configure(
                    Config::new()
                        .nfa_size_limit(Some(PART_LIMIT))
                        .auto_prefilter(prefilter),
                )
                .build_from_hir(hir)
                .ok()
        };
        let full = build(&hir, true)?;
        // The parts are only searched anchored, where a prefilter is never used.
        let head = build(&head, false)?;
        let tail = build(&tail, false)?;
        let slots = Slots {
            head: head.create_captures(),
            full: full.create_captures(),
        };
        Some(Self {
            start_anchored: hir.properties().look_set_prefix().contains(Look::Start),
            groups: hir.properties().explicit_captures_len(),
            full,
            head,
            tail,
            slots: RefCell::new(slots),
            tried: Cell::new(0),
            fell: Cell::new(0),
            off: Cell::new(false),
            rested: Cell::new(0),
        })
    }

    /// Capture groups of the pattern, the whole match not counted.
    pub(super) fn groups(&self) -> usize {
        self.groups
    }

    /// Whether the cut is still in use: it is dropped for a pattern whose first choice of
    /// `head` usually fails (`(a+)a.*`), where each row would pay steps 2 and 3 on top of
    /// the full search (tests; the kernels ask [`Cut::take`]).
    #[cfg(test)]
    pub(super) fn is_on(&self) -> bool {
        !self.off.get()
    }

    /// Whether the search a kernel is about to make uses the cut. While the cut rests this
    /// counts the search, which the engine answers; after [`REST`] of them the probe starts
    /// again.
    pub(super) fn take(&self) -> bool {
        if !self.off.get() {
            return true;
        }
        let rested = self.rested.get() + 1;
        if rested >= REST {
            self.rearm();
        } else {
            self.rested.set(rested);
        }
        false
    }

    /// Counts one search that reached step 3; `fell` when the tail did not match there.
    fn probe(&self, fell: bool) {
        let tried = self.tried.get();
        if tried >= PROBE {
            return;
        }
        self.tried.set(tried + 1);
        if fell {
            self.fell.set(self.fell.get() + 1);
        }
        if tried + 1 == PROBE && self.fell.get() * FALLBACK_SHARE > PROBE {
            self.off.set(true);
        }
    }

    /// The leftmost-first match of the pattern in `text[from..]` with its groups, as the
    /// capture search of the whole pattern finds it. `from` is at most `text.len()`.
    ///
    /// # Errors
    /// [`Busy`] while a [`Found`] of an earlier search is alive.
    pub(super) fn search(&self, text: &str, from: usize) -> Result<Option<Found<'_>>, Busy> {
        let end = text.len();
        let mut slots = self.slots.try_borrow_mut().map_err(|_| Busy)?;
        let start = if self.start_anchored {
            if from != 0 {
                return Ok(None);
            }
            0
        } else {
            match self.full.search(&Input::new(text).span(from..end)) {
                Some(found) => found.start(),
                None => return Ok(None),
            }
        };
        let at_start = Input::new(text).span(start..end).anchored(Anchored::Yes);
        self.head.search_captures(&at_start, &mut slots.head);
        match slots.head.get_match() {
            Some(head) => {
                let rest = Input::new(text)
                    .span(head.end()..end)
                    .anchored(Anchored::Yes);
                let tail = self.tail.search(&rest);
                self.probe(tail.is_none());
                if let Some(tail) = tail {
                    drop(slots);
                    return Ok(Some(Found {
                        whole: (start, tail.end()),
                        groups: Ref::map(self.slots.borrow(), |s| &s.head),
                    }));
                }
            }
            // The head does not match at 0, so the pattern does not match at all.
            None if self.start_anchored => return Ok(None),
            None => {}
        }
        self.full
            .search_captures(&Input::new(text).span(from..end), &mut slots.full);
        let Some(whole) = slots.full.get_match() else {
            return Ok(None);
        };
        let whole = (whole.start(), whole.end());
        drop(slots);
        Ok(Some(Found {
            whole,
            groups: Ref::map(self.slots.borrow(), |s| &s.full),
        }))
    }

    /// Probed searches and how many of them fell back (tests).
    #[cfg(test)]
    pub(super) fn probed(&self) -> (u32, u32) {
        (self.tried.get(), self.fell.get())
    }

    /// Starts the probe again with the cut on.
    pub(super) fn rearm(&self) {
        self.tried.set(0);
        self.fell.set(0);
        self.rested.set(0);
        self.off.set(false);
    }
}

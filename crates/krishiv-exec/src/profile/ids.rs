// SPDX-License-Identifier: Apache-2.0
//! Dense expression ids for plan text: the planner's high ids follow the plan's own.

use std::fmt::Write as _;

/// The ids the planner and the common-subexpression pass give their own columns start at 2^62 (the planner) and 2^32 (the common-subexpression pass), far above the
/// plan's; in the text they follow the plan's largest id in order of first
/// appearance (`sum(amount#1)#4`, as Spark prints `sum(amount#1)#12L`),
/// never `#4611686018427387904` (PY-008).
pub fn dense_ids(text: &str) -> String {
    dense_ids_all(&[text]).remove(0)
}

/// [`dense_ids`] over several texts (one plan node's detail each) with one
/// numbering for all of them, so no separator can mis-split a node's text.
pub fn dense_ids_all(texts: &[&str]) -> Vec<String> {
    const HIGH: u64 = 1 << 32;
    // `#<digits>` outside single-quoted literals (`Literal::String` prints
    // `'..'` with `\` escapes), as (digits start, digits end, value).
    let ids = |text: &str| -> Vec<(usize, usize, u64)> {
        let b = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        let mut quoted = false;
        while i < b.len() {
            match b[i] {
                b'\\' if quoted => i += 2,
                b'\'' => {
                    quoted = !quoted;
                    i += 1;
                }
                b'#' if !quoted && b.get(i + 1).is_some_and(u8::is_ascii_digit) => {
                    let end = (i + 1..b.len())
                        .find(|&j| !b[j].is_ascii_digit())
                        .unwrap_or(b.len());
                    if let Ok(n) = text[i + 1..end].parse::<u64>() {
                        out.push((i + 1, end, n));
                    }
                    i = end;
                }
                _ => i += 1,
            }
        }
        out
    };
    let found: Vec<_> = texts.iter().map(|t| ids(t)).collect();
    let all = || found.iter().flatten();
    if !all().any(|&(_, _, n)| n >= HIGH) {
        return texts.iter().map(|&t| t.to_owned()).collect();
    }
    let base = all()
        .map(|&(_, _, n)| n)
        .filter(|&n| n < HIGH)
        .max()
        .map_or(0, |m| m.saturating_add(1));
    let mut mapped: Vec<u64> = Vec::new();
    texts
        .iter()
        .zip(&found)
        .map(|(text, found)| {
            let mut out = String::with_capacity(text.len());
            let mut at = 0;
            for &(start, end, n) in found {
                if n < HIGH {
                    continue;
                }
                let rank = mapped.iter().position(|&m| m == n).unwrap_or_else(|| {
                    mapped.push(n);
                    mapped.len() - 1
                });
                out.push_str(&text[at..start]);
                let _ = write!(out, "{}", base.saturating_add(rank as u64));
                at = end;
            }
            out.push_str(&text[at..]);
            out
        })
        .collect()
}

#[cfg(test)]
mod dense_ids_tests {
    use super::{dense_ids, dense_ids_all};

    #[test]
    fn planner_ids_follow_the_plans_own() {
        let text = "HashAggregate(sum(amount#1)#4611686018427387904 AS revenue#2)\nx#4611686018427387905 y#4611686018427387904 z#7";
        assert_eq!(
            dense_ids(text),
            "HashAggregate(sum(amount#1)#8 AS revenue#2)\nx#9 y#8 z#7"
        );
        assert_eq!(dense_ids("a#1 b#22"), "a#1 b#22");
        // the common-subexpression pass starts at 2^32
        assert_eq!(dense_ids("a#1 b#4294967296 c#4294967297"), "a#1 b#2 c#3");
    }

    #[test]
    fn a_quoted_literal_is_left_as_written() {
        assert_eq!(
            dense_ids("s = '#4294967296' AND x#4294967296 AND t = 'it\\'s #4294967297'"),
            "s = '#4294967296' AND x#0 AND t = 'it\\'s #4294967297'"
        );
    }

    #[test]
    fn one_numbering_across_texts_whatever_they_contain() {
        let a = "a#4294967296 \u{1} b#3";
        assert_eq!(
            dense_ids_all(&[a, "c#4294967297 a#4294967296"]),
            ["a#4 \u{1} b#3", "c#5 a#4"]
        );
    }
}

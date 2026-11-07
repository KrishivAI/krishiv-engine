// SPDX-License-Identifier: Apache-2.0
//! Spark's `ParserUtils.checkDuplicateKeys`: which repeated key of a key-value list Spark reports.
//!
//! Spark groups the keys with Scala 2.13's `groupBy` (an `immutable.HashMap`), keeps the groups of
//! more than one and throws for the first in the map's iteration order. That order is the CHAMP
//! trie's: at each level the entries alone in their 5-bit hash fragment come first, by fragment,
//! then the sub-tries, by fragment. The hash is `Hashing.improve(key.hashCode)`. Checked against
//! Spark 4.2.0 on 403 random key lists (`research/spark/partition-spec-duplicate-key.md`).

/// Java's `String.hashCode` (UTF-16 code units), as an unsigned word.
fn java_hash(s: &str) -> u32 {
    s.encode_utf16()
        .fold(0u32, |h, c| h.wrapping_mul(31).wrapping_add(u32::from(c)))
}

/// Scala 2.13's `scala.collection.Hashing.improve`, which `immutable.HashMap` applies to `key.##`.
fn improve(h: u32) -> u32 {
    let mut h = h.wrapping_add(!(h << 9));
    h ^= h >> 14;
    h = h.wrapping_add(h << 4);
    h ^ (h >> 10)
}

/// The first of `keys` (distinct) in a CHAMP trie's iteration order, from the level at `shift`.
/// Keys whose 32-bit hashes are equal sit in a collision node in insertion order; the first listed
/// stands in for that order.
fn champ_first(keys: Vec<(&str, u32)>, shift: u32) -> Option<&str> {
    if keys.len() <= 1 || shift >= 32 {
        return keys.first().map(|(k, _)| *k);
    }
    let fragment = |h: u32| (h >> shift) & 31;
    let mut counts = [0usize; 32];
    for (_, h) in &keys {
        counts[fragment(*h) as usize] += 1;
    }
    if let Some((k, _)) = keys
        .iter()
        .filter(|(_, h)| counts[fragment(*h) as usize] == 1)
        .min_by_key(|(_, h)| fragment(*h))
    {
        return Some(k);
    }
    let lowest = keys.iter().map(|(_, h)| fragment(*h)).min()?;
    let below = keys
        .into_iter()
        .filter(|(_, h)| fragment(*h) == lowest)
        .collect();
    champ_first(below, shift + 5)
}

/// The repeated key Spark's `checkDuplicateKeys` reports for `keys` (already lower-cased when
/// Spark lower-cases them), or `None` when no key repeats.
pub(super) fn first_duplicate_key<'a>(keys: &[&'a str]) -> Option<&'a str> {
    let mut repeated: Vec<&str> = Vec::new();
    for (i, k) in keys.iter().enumerate() {
        if keys[..i].contains(k) && !repeated.contains(k) {
            repeated.push(k);
        }
    }
    champ_first(
        repeated
            .into_iter()
            .map(|k| (k, improve(java_hash(k))))
            .collect(),
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::first_duplicate_key;

    /// Cases from the Spark 4.2.0 run (`ALTER TABLE nosuch ADD PARTITION (...)`, keys as written).
    #[test]
    fn spark_reports_the_first_repeated_key_in_hash_map_order() {
        let cases: &[(&[&str], Option<&str>)] = &[
            (&["p", "q"], None),
            (&["p", "p"], Some("p")),
            (&["a", "a", "b", "b"], Some("a")),
            (&["b", "b", "a", "a"], Some("a")),
            (&["z", "z", "p.q", "p.q"], Some("p.q")),
            (
                &[
                    "yoo", "je", "sgf", "d", "je", "put", "d", "yoo", "sgf", "r", "r",
                ],
                Some("sgf"),
            ),
            (&["ayo", "u", "q", "u", "os", "os", "ayo"], Some("u")),
            (
                &[
                    "ydm", "q", "j", "ca", "ydm", "q", "ky", "p", "agg", "ky", "ac", "ac", "ca",
                    "agg",
                ],
                Some("agg"),
            ),
            (&["o", "fv", "fv", "goq"], Some("fv")),
            (&["i", "i", "jag", "ng", "jag"], Some("i")),
            (
                &["pjq", "hx", "q", "cnb", "min", "pjq", "d", "d"],
                Some("pjq"),
            ),
            (
                &["hb", "f", "ecl", "lq", "vx", "f", "o", "lq", "o", "hb"],
                Some("hb"),
            ),
            (
                &["x", "ezk", "ke", "rl", "c", "rl", "c", "ezk", "bxx", "x"],
                Some("x"),
            ),
            (
                &[
                    "ooo", "cqy", "xk", "i", "xk", "c", "u", "ooo", "xw", "c", "i",
                ],
                Some("xk"),
            ),
            (
                &["hk", "hk", "vi", "y", "vo", "g", "zz", "y", "vi", "u"],
                Some("y"),
            ),
        ];
        for (keys, want) in cases {
            assert_eq!(first_duplicate_key(keys), *want, "{keys:?}");
        }
    }
}

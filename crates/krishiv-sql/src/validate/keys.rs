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

/// `keys` (distinct) in a CHAMP trie's iteration order, from the level at `shift`, appended to
/// `out`. Keys whose 32-bit hashes are equal sit in a collision node in insertion order, which is
/// the order they come in.
fn champ_order<'a>(keys: Vec<(&'a str, u32)>, shift: u32, out: &mut Vec<&'a str>) {
    if keys.len() <= 1 || shift >= 32 {
        out.extend(keys.into_iter().map(|(k, _)| k));
        return;
    }
    let fragment = |h: u32| ((h >> shift) & 31) as usize;
    let mut groups: [Vec<(&str, u32)>; 32] = Default::default();
    for (k, h) in keys {
        groups[fragment(h)].push((k, h));
    }
    // A node's own entries first, by fragment, then its sub-nodes, by fragment.
    out.extend(groups.iter().filter(|g| g.len() == 1).map(|g| g[0].0));
    for g in groups {
        if g.len() > 1 {
            champ_order(g, shift + 5, out);
        }
    }
}

/// `keys` (distinct) in the iteration order of an `immutable.HashMap` built from them in this order.
fn hash_map_order<'a>(keys: &[&'a str]) -> Vec<&'a str> {
    let mut out = Vec::with_capacity(keys.len());
    champ_order(
        keys.iter().map(|k| (*k, improve(java_hash(k)))).collect(),
        0,
        &mut out,
    );
    out
}

/// The repeated key Spark's `checkDuplicateKeys` reports for `keys` (already lower-cased when
/// Spark lower-cases them), or `None` when no key repeats. The repeated keys enter the map in the
/// order they are first written (Spark 4.2.0: `c0, an, an, c0`, whose hash codes are equal, gives `c0`).
pub(super) fn first_duplicate_key<'a>(keys: &[&'a str]) -> Option<&'a str> {
    let mut repeated: Vec<&str> = Vec::new();
    for (i, k) in keys.iter().enumerate() {
        if !keys[..i].contains(k) && keys[i + 1..].contains(k) {
            repeated.push(k);
        }
    }
    hash_map_order(&repeated).first().copied()
}

/// The iteration order of Scala's `Seq.toMap` over `keys` (distinct): the written order up to four
/// keys (`Map1`..`Map4`), the [`hash_map_order`] from five.
pub(super) fn to_map_order<'a>(keys: &[&'a str]) -> Vec<&'a str> {
    if keys.len() <= 4 {
        keys.to_vec()
    } else {
        hash_map_order(keys)
    }
}

/// The keys of `keys` (distinct, as written) for which `pick` holds, in the order Spark's
/// `visitPropertyKeyValues` lists them: `collect` over the property map, so the picked keys keep
/// their place in [`to_map_order`] of all the keys.
pub(super) fn collected_order<'a>(keys: &[&'a str], pick: impl Fn(&str) -> bool) -> Vec<&'a str> {
    to_map_order(keys).into_iter().filter(|k| pick(k)).collect()
}

/// The keys of `keys` (distinct, as written) for which `pick` holds, in the order Spark's
/// `visitPropertyKeys` lists them: `filter` over the property map, then its keys. A map of up to
/// four keys filters in written order; a larger `HashMap` filters into a new trie of the picked
/// keys alone, whose order is that trie's own, not their order in the whole map (Spark 4.2.0:
/// `UNSET TBLPROPERTIES (jee=1, hb=1, b=1, jbd, bg, cfi)` lists `[hb,b,jee]`, `SET` with the
/// same keys missing values `[b,jee,hb]`).
pub(super) fn filtered_order<'a>(keys: &[&'a str], pick: impl Fn(&str) -> bool) -> Vec<&'a str> {
    let picked: Vec<&str> = keys.iter().copied().filter(|k| pick(k)).collect();
    if keys.len() <= 4 {
        picked
    } else {
        hash_map_order(&picked)
    }
}

#[cfg(test)]
mod tests {
    use super::{collected_order, filtered_order, first_duplicate_key, to_map_order};

    /// Spark 4.2.0 `ALTER TABLE ... SET` / `UNSET TBLPROPERTIES` with these keys missing (or
    /// holding) a value (`research/spark/partition-spec-duplicate-key/probe9.py`).
    #[test]
    fn bad_property_keys_are_listed_in_map_order() {
        type Case<'a> = (&'a [&'a str], &'a [&'a str], &'a [&'a str], &'a [&'a str]);
        let cases: &[Case<'_>] = &[
            (
                &["jee", "hb", "b", "jbd", "bg", "cfi"],
                &["hb", "jee", "b"],
                &["b", "jee", "hb"],
                &["hb", "b", "jee"],
            ),
            (
                &["ehf", "cge", "geg", "d", "gag", "bjf"],
                &["ehf", "d", "gag"],
                &["ehf", "d", "gag"],
                &["ehf", "gag", "d"],
            ),
            (
                &["i", "fj", "d", "jg", "jc"],
                &["jg", "d", "jc"],
                &["jg", "d", "jc"],
                &["jc", "jg", "d"],
            ),
            (
                &["dd", "c", "h", "d", "abd", "a"],
                &["abd", "a", "d"],
                &["a", "abd", "d"],
                &["d", "a", "abd"],
            ),
            (
                &["yoo", "sgf", "a", "b", "c"],
                &["yoo", "sgf"],
                &["sgf", "yoo"],
                &["sgf", "yoo"],
            ),
            (
                &["yoo", "sgf", "a"],
                &["yoo", "sgf"],
                &["yoo", "sgf"],
                &["yoo", "sgf"],
            ),
        ];
        for (keys, bad, set, unset) in cases {
            let pick = |k: &str| bad.contains(&k);
            assert_eq!(collected_order(keys, pick), *set, "SET {keys:?}");
            assert_eq!(filtered_order(keys, pick), *unset, "UNSET {keys:?}");
        }
    }

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

    /// Spark 4.2.0: `DESC t PARTITION (a=1, b=1, c=1, d=1, yoo, sgf)` names `sgf`.
    #[test]
    fn to_map_keeps_four_keys_in_order_and_hashes_five() {
        assert_eq!(to_map_order(&["yoo", "sgf", "a"]), ["yoo", "sgf", "a"]);
        let five = to_map_order(&["a", "b", "c", "d", "yoo", "sgf"]);
        let yoo = five.iter().position(|k| *k == "yoo");
        let sgf = five.iter().position(|k| *k == "sgf");
        assert!(sgf < yoo, "{five:?}");
    }
}

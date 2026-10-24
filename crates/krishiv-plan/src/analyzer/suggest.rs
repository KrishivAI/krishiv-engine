// SPDX-License-Identifier: Apache-2.0
//! The names proposed for a column the analyzer cannot resolve
//! (`UNRESOLVED_COLUMN.WITH_SUGGESTION`): up to five of the scope's columns,
//! ranked by edit distance with ties ordered by name.
//!
//! Only this class proposes names; the messages of the others, such as
//! `UNRESOLVED_ROUTINE` and `TABLE_OR_VIEW_NOT_FOUND`, carry none, so their
//! errors do not use this.

/// How many names Spark proposes at most.
const MAX_PROPOSALS: usize = 5;

/// Up to five of `candidates` (each a name as its parts) closest to
/// `target` (the name as written, parts joined by `.`), by edit distance
/// between the candidate's parts joined by `.` and `target`. Ties are
/// ordered by the candidate's parts, as Spark orders them, so the result
/// depends only on the names and not on their order in `candidates`.
pub(super) fn closest(target: &str, candidates: Vec<Vec<String>>) -> Vec<Vec<String>> {
    let mut ranked: Vec<(usize, Vec<String>)> = candidates
        .into_iter()
        .map(|parts| (levenshtein(&parts.join("."), target), parts))
        .collect();
    ranked.sort();
    ranked
        .into_iter()
        .take(MAX_PROPOSALS)
        .map(|(_, parts)| parts)
        .collect()
}

/// The edit distance (insertions, deletions and substitutions of
/// characters) between `a` and `b`.
fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(x != *y)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(parts: &[&[&str]]) -> Vec<Vec<String>> {
        parts
            .iter()
            .map(|p| p.iter().map(|s| (*s).to_owned()).collect())
            .collect()
    }

    fn flat(found: &[Vec<String>]) -> Vec<String> {
        found.iter().map(|p| p.join(".")).collect()
    }

    #[test]
    fn distance_counts_character_edits() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("é", "e"), 1);
    }

    #[test]
    fn a_typo_of_one_edit_ranks_first() {
        let found = closest("nme", names(&[&["id"], &["age"], &["name"], &["city"]]));
        assert_eq!(flat(&found)[0], "name");
        assert_eq!(levenshtein("name", "nme"), 1);
    }

    #[test]
    fn a_typo_of_two_edits_ranks_ahead_of_unrelated_names() {
        let found = closest("nxye", names(&[&["zzzzzz"], &["name"], &["id"]]));
        assert_eq!(levenshtein("name", "nxye"), 2);
        assert_eq!(flat(&found), ["name", "id", "zzzzzz"]);
    }

    #[test]
    fn no_candidate_proposes_nothing() {
        assert!(closest("x", Vec::new()).is_empty());
    }

    #[test]
    fn at_most_five_are_proposed() {
        let all = names(&[&["a"], &["b"], &["c"], &["d"], &["e"], &["f"], &["g"]]);
        assert_eq!(flat(&closest("x", all)), ["a", "b", "c", "d", "e"]);
    }

    #[test]
    fn ties_are_ordered_by_name_whatever_the_input_order() {
        let forward = closest("x", names(&[&["b"], &["a"], &["c"]]));
        let reverse = closest("x", names(&[&["c"], &["a"], &["b"]]));
        assert_eq!(flat(&forward), ["a", "b", "c"]);
        assert_eq!(forward, reverse);
        // Equal distance, different parts: compared part by part.
        let found = closest("t.x", names(&[&["u", "x"], &["t", "y"], &["s", "x"]]));
        assert_eq!(flat(&found), ["s.x", "t.y", "u.x"]);
    }

    #[test]
    fn qualified_candidates_are_compared_as_written() {
        let found = closest("t.nme", names(&[&["name"], &["t", "name"], &["u", "name"]]));
        assert_eq!(flat(&found), ["t.name", "u.name", "name"]);
    }
}

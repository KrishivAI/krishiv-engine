// SPDX-License-Identifier: Apache-2.0
//! Hadoop glob patterns for reads by path (P2-T082b, design §2.5), the syntax
//! Spark's `DataSource.checkAndGlobPathIfNecessary` accepts: `*` and `?` within
//! one path segment, `[abc]`, `[a-c]`, `[^a]`, `{x,y}` (which may hold `/`) and
//! `\` to escape the next character.

/// Whether `text` holds a glob metacharacter (an escaped one counts: Hadoop
/// treats any path with one as a pattern).
#[must_use]
pub fn has_glob(text: &str) -> bool {
    text.contains(['*', '?', '[', '{'])
}

/// A glob: its brace alternatives, each a list of path segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Glob {
    alternatives: Vec<Vec<String>>,
}

impl Glob {
    /// Parse `pattern`.
    ///
    /// # Errors
    /// A message naming the unbalanced brace or bracket.
    pub fn parse(pattern: &str) -> Result<Self, String> {
        let mut alternatives = Vec::new();
        for text in expand_braces(pattern)? {
            check_brackets(&text)?;
            let segments: Vec<String> = text
                .split('/')
                .filter(|s| !s.is_empty())
                .map(collapse_stars)
                .collect();
            // Matching recurses once per pattern character: a segment is bounded.
            if segments
                .iter()
                .any(|s| s.chars().count() > MAX_SEGMENT_CHARS)
            {
                return Err(format!(
                    "a segment of the pattern is longer than {MAX_SEGMENT_CHARS} characters"
                ));
            }
            alternatives.push(segments);
        }
        Ok(Self { alternatives })
    }

    /// The directories common to every alternative that hold no glob
    /// character: the place to start listing.
    #[must_use]
    pub fn literal_prefix(&self) -> Vec<String> {
        let mut prefix: Option<Vec<String>> = None;
        for alt in &self.alternatives {
            let lit: Vec<String> = alt
                .iter()
                .take_while(|s| !has_glob(s))
                .map(|s| unescape(s))
                .collect();
            prefix = Some(match prefix {
                None => lit,
                Some(p) => p
                    .into_iter()
                    .zip(lit)
                    .take_while(|(a, b)| a == b)
                    .map(|(a, _)| a)
                    .collect(),
            });
        }
        prefix.unwrap_or_default()
    }

    /// Whether the path made of `segments` matches an alternative entirely.
    #[must_use]
    pub fn matches(&self, segments: &[&str]) -> bool {
        self.alternatives.iter().any(|alt| {
            alt.len() == segments.len()
                && alt
                    .iter()
                    .zip(segments)
                    .all(|(pattern, segment)| match_segment(pattern, segment))
        })
    }
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            out.extend(chars.next());
        } else {
            out.push(c);
        }
    }
    out
}

/// Most patterns `{a,b}` groups may expand to: repeated groups multiply, and the expansion is eager.
const MAX_ALTERNATIVES: usize = 4096;

/// Longest pattern segment (after runs of `*` are collapsed): the matcher recurses once per character.
const MAX_SEGMENT_CHARS: usize = 1024;

/// `a***b` is `a*b` inside a segment: a run of unescaped stars matches what one does.
fn collapse_stars(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    let mut escaped = false;
    let mut last_star = false;
    for c in segment.chars() {
        if escaped {
            escaped = false;
            last_star = false;
            out.push(c);
        } else if c == '\\' {
            escaped = true;
            last_star = false;
            out.push(c);
        } else if c == '*' {
            if !last_star {
                out.push(c);
            }
            last_star = true;
        } else {
            last_star = false;
            out.push(c);
        }
    }
    out
}

/// Most `{` a pattern may hold: the expansion recurses once per group, so a
/// pattern of thousands of single-alternative groups (`{}{}{}...`), which
/// never reaches `MAX_ALTERNATIVES`, would exhaust the stack.
const MAX_BRACE_GROUPS: usize = 64;

/// `a{b,c{d,e}}f` → `abf`, `acdf`, `acef`.
fn expand_braces(pattern: &str) -> Result<Vec<String>, String> {
    let mut opens = 0usize;
    let mut escaped = false;
    for c in pattern.chars() {
        match c {
            _ if escaped => escaped = false,
            '\\' => escaped = true,
            '{' => opens += 1,
            _ => {}
        }
    }
    if opens > MAX_BRACE_GROUPS {
        return Err(format!(
            "more than {MAX_BRACE_GROUPS} brace groups in the pattern"
        ));
    }
    expand_groups(pattern)
}

fn expand_groups(pattern: &str) -> Result<Vec<String>, String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut open = None;
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => i += 1,
            '{' => {
                open = Some(i);
                break;
            }
            _ => {}
        }
        i += 1;
    }
    let Some(start) = open else {
        return Ok(vec![pattern.to_owned()]);
    };
    let (mut depth, mut end, mut cuts) = (0usize, None, vec![start]);
    let mut j = start;
    while j < chars.len() {
        match chars[j] {
            '\\' => j += 1,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(j);
                    break;
                }
            }
            ',' if depth == 1 => cuts.push(j),
            _ => {}
        }
        j += 1;
    }
    let end = end.ok_or_else(|| format!("unclosed group near index {start}"))?;
    cuts.push(end);
    let head: String = chars[..start].iter().collect();
    let tail: String = chars[end + 1..].iter().collect();
    let mut out = Vec::new();
    for pair in cuts.windows(2) {
        let body: String = chars[pair[0] + 1..pair[1]].iter().collect();
        out.extend(expand_groups(&format!("{head}{body}{tail}"))?);
        if out.len() > MAX_ALTERNATIVES {
            return Err(format!(
                "more than {MAX_ALTERNATIVES} alternatives in the braces of the pattern"
            ));
        }
    }
    Ok(out)
}

fn check_brackets(text: &str) -> Result<(), String> {
    let mut chars = text.chars().enumerate();
    let mut open = None;
    while let Some((i, c)) = chars.next() {
        match c {
            '\\' => {
                chars.next();
            }
            '[' if open.is_none() => open = Some(i),
            ']' => open = None,
            _ => {}
        }
    }
    open.map_or(Ok(()), |i| {
        Err(format!("unclosed character class near index {i}"))
    })
}

/// One segment, no braces.
fn match_segment(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    match_at(&p, &t, &mut std::collections::HashSet::new())
}

/// Whether `p` matches all of `t`. `failed` holds the (pattern left, text left) states already known not to
/// match, so a pattern with many stars costs at most pattern length times text length steps instead of
/// trying every split of the text among the stars.
fn match_at(
    p: &[char],
    t: &[char],
    failed: &mut std::collections::HashSet<(usize, usize)>,
) -> bool {
    if failed.contains(&(p.len(), t.len())) {
        return false;
    }
    let hit = match p.split_first() {
        None => t.is_empty(),
        Some((&c, rest)) => match c {
            '*' => (0..=t.len()).any(|n| match_at(rest, &t[n..], failed)),
            '?' => !t.is_empty() && match_at(rest, &t[1..], failed),
            '\\' => match rest.split_first() {
                Some((&lit, rest)) => t.first() == Some(&lit) && match_at(rest, &t[1..], failed),
                None => false,
            },
            '[' => match t.first() {
                Some(&ch) => match class(rest, ch) {
                    Some((hit, after)) => hit && match_at(after, &t[1..], failed),
                    None => false,
                },
                None => false,
            },
            lit => t.first() == Some(&lit) && match_at(rest, &t[1..], failed),
        },
    };
    if !hit {
        failed.insert((p.len(), t.len()));
    }
    hit
}

/// Whether `ch` is in the class that starts after `[`, and what follows `]`.
fn class(p: &[char], ch: char) -> Option<(bool, &[char])> {
    let (negated, mut i) = match p.first() {
        Some('^' | '!') => (true, 1),
        _ => (false, 0),
    };
    let mut hit = false;
    let mut first = true;
    while i < p.len() {
        let mut lo = p[i];
        if lo == ']' && !first {
            return Some((hit != negated, &p[i + 1..]));
        }
        first = false;
        if lo == '\\' {
            i += 1;
            lo = *p.get(i)?;
        }
        if p.get(i + 1) == Some(&'-') && p.get(i + 2).is_some_and(|&c| c != ']') {
            let hi = p[i + 2];
            hit |= (lo..=hi).contains(&ch);
            i += 3;
        } else {
            hit |= lo == ch;
            i += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(pattern: &str, path: &str) -> bool {
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        Glob::parse(pattern).is_ok_and(|g| g.matches(&segments))
    }

    #[test]
    fn star_and_question_stay_inside_a_segment() {
        assert!(m("/a/*/c", "/a/b/c"));
        assert!(!m("/a/*/c", "/a/b/x/c"));
        assert!(m("/a/p?.csv", "/a/p1.csv"));
        assert!(!m("/a/p?.csv", "/a/p12.csv"));
        assert!(m("/a/*", "/a/.hidden"));
    }

    #[test]
    fn classes_and_ranges() {
        assert!(m("/d/[ab]1", "/d/a1"));
        assert!(!m("/d/[ab]1", "/d/c1"));
        assert!(m("/d/[a-c]", "/d/b"));
        assert!(m("/d/[^a-c]", "/d/z"));
        assert!(!m("/d/[^a-c]", "/d/b"));
        assert!(m("/d/[]]", "/d/]"));
    }

    #[test]
    fn braces_nest_and_cross_segments() {
        assert!(m("/d/{x,y}/f", "/d/y/f"));
        assert!(m("/d/{x,y/z}", "/d/y/z"));
        assert!(m("/d/a{b,c{d,e}}f", "/d/acef"));
        assert!(!m("/d/a{b,c{d,e}}f", "/d/acf"));
    }

    #[test]
    fn escapes_and_prefix() {
        assert!(m("/d/a\\*b", "/d/a*b"));
        assert!(!m("/d/a\\*b", "/d/axb"));
        let g = Glob::parse("/data/year=2024/m*/f").unwrap();
        assert_eq!(g.literal_prefix(), ["data", "year=2024"]);
        let g = Glob::parse("/data/{a,b}/x").unwrap();
        assert_eq!(g.literal_prefix(), ["data"]);
    }

    #[test]
    fn thousands_of_brace_groups_are_refused_not_recursed_into() {
        assert!(Glob::parse(&"{}".repeat(4000)).is_err());
        assert!(Glob::parse(&format!("/a/{}", "{x}".repeat(64))).is_ok());
        assert!(Glob::parse(&format!("/a/{}", "{x}".repeat(65))).is_err());
        // Escaped braces are literals and do not count.
        assert!(Glob::parse(&"\\{".repeat(400)).is_ok());
    }

    #[test]
    fn long_runs_of_stars_and_long_segments_do_not_recurse_deeply() {
        let g = Glob::parse(&"*".repeat(100_000)).unwrap();
        assert!(g.matches(&["x"]));
        assert!(
            !Glob::parse(&format!("{}a", "*".repeat(100_000)))
                .unwrap()
                .matches(&["x"])
        );
        assert!(Glob::parse(&"a?".repeat(2_000)).is_err());
        assert!(m("/d/a**b", "/d/axyb"));
        assert!(m("/d/a\\**b", "/d/a*xb"));
    }

    #[test]
    fn unbalanced_patterns_are_errors() {
        assert!(Glob::parse("/a/{b").is_err());
        assert!(Glob::parse("/a/[b").is_err());
    }
}

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
            alternatives.push(
                text.split('/')
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect(),
            );
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

/// `a{b,c{d,e}}f` → `abf`, `acdf`, `acef`.
fn expand_braces(pattern: &str) -> Result<Vec<String>, String> {
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
        out.extend(expand_braces(&format!("{head}{body}{tail}"))?);
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
    match_at(&p, &t)
}

fn match_at(p: &[char], t: &[char]) -> bool {
    let Some((&c, rest)) = p.split_first() else {
        return t.is_empty();
    };
    match c {
        '*' => (0..=t.len()).any(|n| match_at(rest, &t[n..])),
        '?' => !t.is_empty() && match_at(rest, &t[1..]),
        '\\' => match rest.split_first() {
            Some((&lit, rest)) => t.first() == Some(&lit) && match_at(rest, &t[1..]),
            None => false,
        },
        '[' => {
            let Some(&ch) = t.first() else { return false };
            let Some((hit, after)) = class(rest, ch) else {
                return false;
            };
            hit && match_at(after, &t[1..])
        }
        lit => t.first() == Some(&lit) && match_at(rest, &t[1..]),
    }
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
    fn unbalanced_patterns_are_errors() {
        assert!(Glob::parse("/a/{b").is_err());
        assert!(Glob::parse("/a/[b").is_err());
    }
}

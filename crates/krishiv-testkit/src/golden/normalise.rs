// SPDX-License-Identifier: Apache-2.0
//! Spark's output masking: an in-order port of `SQLQueryTestHelper.replaceNotIncludedMsg` and of
//! `normalizeTestResults` (Spark 4.2.0). The recorded golden files are already masked, so engine
//! rows go through the same rules before they are compared.
//!
//! Spark applies the function to every row string of a result (`hiveResultString(..)
//! .map(replaceNotIncludedMsg)`) and sorts afterwards, so callers must mask each row first.
//! Java's `\d`, `\w` and `\s` are ASCII-only and its `.` stops at any line terminator; the
//! patterns below spell that out. Spark's `clsName` (the suite class, which ends the warehouse
//! path) is the engine's warehouse directory here; the three rules that use it are skipped when
//! the engine has no warehouse directory.

use regex_lite::{NoExpand, Regex};

const MSG: &str = "[not included in comparison]";
/// Java `.`: anything but a line terminator.
const DOT: &str = "[^\\n\\r\\u{85}\\u{2028}\\u{2029}]";

/// The compiled rules for one warehouse directory.
#[derive(Debug)]
pub struct Masker {
    rules: Vec<(Regex, String)>,
}

impl Masker {
    /// Compile the rules; `warehouse` is the engine's warehouse directory (may be empty).
    pub fn new(warehouse: &str) -> Result<Self, regex_lite::Error> {
        let cls = regex_lite::escape(warehouse.strip_prefix("file:").unwrap_or(warehouse));
        let has_wh = !cls.is_empty();
        let key = |k: &str| format!("\"{k}\":\".*?\"").replace('.', DOT);
        let dot = |s: &str| s.replace('.', DOT);
        // (pattern, replacement), in the order of `replaceNotIncludedMsg`.
        let mut spec: Vec<(String, String)> = Vec::new();
        let mut add = |p: String, r: String| spec.push((p, r));
        add("#\\d+".into(), "#x".into());
        add("plan_id=\\d+".into(), "plan_id=x".into());
        add("uuid\\(Some\\(-?\\d+\\)\\)".into(), "uuid(Some(x))".into());
        if has_wh {
            add(
                dot(&format!("Location.*{cls}/")),
                format!("Location {MSG}/{{warehouse_dir}}/"),
            );
            add(
                format!("file:[^\\s,]*{cls}"),
                format!("file:{MSG}/{{warehouse_dir}}"),
            );
        }
        add(dot("Created By.*"), format!("Created By {MSG}"));
        add(dot("Created Time.*"), format!("Created Time {MSG}"));
        add(dot("Last Access.*"), format!("Last Access {MSG}"));
        add(dot("Owner[\\t ]+.*"), format!("Owner\t{MSG}"));
        add(
            "Partition Statistics\\t\\d+".into(),
            format!("Partition Statistics\t{MSG}"),
        );
        add("CTERelationDef \\d+,".into(), "CTERelationDef xxxx,".into());
        add("CTERelationRef \\d+,".into(), "CTERelationRef xxxx,".into());
        add("UnionLoop \\d+".into(), "UnionLoop xxxx".into());
        add("UnionLoopRef \\d+,".into(), "UnionLoopRef xxxx,".into());
        add("Loop id: \\d+".into(), "Loop id: xxxx".into());
        add("@\\w*,".into(), "@xxxxxxxx,".into());
        add("\\*\\(\\d+\\) ".into(), "*".into());
        if has_wh {
            add(
                dot(&format!("\"location\":.*?{cls}/")),
                format!("\"location\": \"{MSG}/{{warehouse_dir}}/"),
            );
        }
        let none = |k: &str| format!("\"{k} {MSG}\":\"None\"");
        add(key("created_by"), none("created_by"));
        add(key("created_time"), none("created_time"));
        add(
            "transient_lastDdlTime=\\d+".into(),
            format!("transient_lastDdlTime={MSG}"),
        );
        add(
            "\"transient_lastDdlTime\":\"\\d+\"".into(),
            none("transient_lastDdlTime"),
        );
        add(key("last_access"), none("last_access"));
        add(key("owner"), none("owner"));
        add(
            "\"partition_statistics\":\"\\d+\"".into(),
            none("partition_statistics"),
        );
        add("cterelationdef \\d+,".into(), "cterelationdef xxxx,".into());
        add("cterelationref \\d+,".into(), "cterelationref xxxx,".into());
        let rules = spec
            .into_iter()
            .map(|(p, r)| Regex::new(&p).map(|re| (re, r)))
            .collect::<Result<_, _>>()?;
        Ok(Self { rules })
    }

    /// `replaceNotIncludedMsg` over one row string.
    #[must_use]
    pub fn mask(&self, row: &str) -> String {
        let mut s = row.to_owned();
        for (re, rep) in &self.rules {
            s = re.replace_all(&s, NoExpand(rep)).into_owned();
        }
        s
    }
}

/// Java's `\s`: the ASCII whitespace `[ \t\n\x0B\f\r]`.
/// Java's `\\s`: ASCII whitespace only. Spark trims with it, so a trailing
/// U+00A0 or U+3000 is kept.
pub(crate) const JAVA_SPACE: [char; 6] = [' ', '\t', '\n', '\u{b}', '\u{c}', '\r'];

/// `replaceAll("\\s+$", "")`: Java's `$` also matches before a final line terminator, so a
/// trailing `U+0085`, `U+2028` or `U+2029` stays and the whitespace before it goes.
fn trim_trailing_space(s: &str) -> String {
    let (head, terminator) = match s.chars().next_back() {
        Some(c @ ('\u{85}' | '\u{2028}' | '\u{2029}')) => s.split_at(s.len() - c.len_utf8()),
        _ => (s, ""),
    };
    format!("{}{terminator}", head.trim_end_matches(JAVA_SPACE))
}

/// `SQLQueryTestHelper.normalizeTestResults`: drops every Python traceback (the line
/// `Traceback (most recent call last):` and the lines after it that start with a space, which
/// differ between machines) and the trailing whitespace. `SQLQueryTestSuite` applies it to the
/// recorded and the actual output alike before comparing them.
#[must_use]
pub fn normalize_test_results(output: &str) -> String {
    let mut traceback = false;
    // Java's `String.split` drops trailing empty fields before the filter runs.
    let mut lines: Vec<&str> = output.split('\n').collect();
    while lines.last() == Some(&"") {
        lines.pop();
    }
    let kept: Vec<&str> = lines
        .into_iter()
        .filter(|line| {
            if *line == "Traceback (most recent call last):" {
                traceback = true;
            } else if !line.starts_with(' ') {
                traceback = false;
            }
            !traceback
        })
        .collect();
    trim_trailing_space(&kept.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;

    /// Each pair is (raw engine text, the masked line as it appears in the vendored goldens).
    #[test]
    fn each_rule_produces_the_golden_form() -> R {
        let m = Masker::new("/tmp/wh/SQLQueryTestSuite")?;
        let cases = [
            ("Project [a#123L, b#4]", "Project [a#xL, b#x]"),
            (
                "Exchange hashpartitioning(a#1, 5), ENSURE_REQUIREMENTS, [plan_id=77]",
                "Exchange hashpartitioning(a#x, 5), ENSURE_REQUIREMENTS, [plan_id=x]",
            ),
            ("uuid(Some(-5)) AS __qid#9", "uuid(Some(x)) AS __qid#x"),
            (
                "Location file:/tmp/wh/SQLQueryTestSuite/explain_temp1",
                "Location [not included in comparison]/{warehouse_dir}/explain_temp1",
            ),
            (
                "InMemoryFileIndex(file:/tmp/wh/SQLQueryTestSuite/explain_temp1)",
                "InMemoryFileIndex(file:[not included in comparison]/{warehouse_dir}/explain_temp1)",
            ),
            (
                "Created By\tSpark 4.2.0",
                "Created By [not included in comparison]",
            ),
            (
                "Created Time\tMon Oct 17 10:00:00 UTC 2026",
                "Created Time [not included in comparison]",
            ),
            (
                "Last Access\tUNKNOWN",
                "Last Access [not included in comparison]",
            ),
            ("Owner\troot", "Owner\t[not included in comparison]"),
            (
                "Partition Statistics\t1234 bytes, 3 rows",
                "Partition Statistics\t[not included in comparison] bytes, 3 rows",
            ),
            ("CTERelationDef 12, false", "CTERelationDef xxxx, false"),
            ("CTERelationRef 12, true", "CTERelationRef xxxx, true"),
            ("+- UnionLoop 5", "+- UnionLoop xxxx"),
            ("UnionLoopRef 5, [a#1]", "UnionLoopRef xxxx, [a#x]"),
            ("Loop id: 77", "Loop id: xxxx"),
            ("Subquery@1f2e3d, x", "Subquery@xxxxxxxx, x"),
            ("*(2) Project [a#1]", "*Project [a#x]"),
            (
                "Partition Parameters\t[numFiles=1, transient_lastDdlTime=1700000000]",
                "Partition Parameters\t[numFiles=1, transient_lastDdlTime=[not included in comparison]]",
            ),
            ("cterelationdef 3, false", "cterelationdef xxxx, false"),
        ];
        for (raw, golden) in cases {
            assert_eq!(m.mask(raw), golden, "{raw}");
        }
        Ok(())
    }

    #[test]
    fn json_describe_keys() -> R {
        let m = Masker::new("/tmp/wh/SQLQueryTestSuite")?;
        let raw = r#"{"location":"file:/tmp/wh/SQLQueryTestSuite/t","created_time":"Mon","last_access":"UNKNOWN","created_by":"Spark 4.2.0","owner":"root","transient_lastDdlTime":"17","partition_statistics":"12"}"#;
        let golden = r#"{"location":"file:[not included in comparison]/{warehouse_dir}/t","created_time [not included in comparison]":"None","last_access [not included in comparison]":"None","created_by [not included in comparison]":"None","owner [not included in comparison]":"None","transient_lastDdlTime [not included in comparison]":"None","partition_statistics [not included in comparison]":"None"}"#;
        assert_eq!(m.mask(raw), golden);
        Ok(())
    }

    #[test]
    fn no_warehouse_skips_path_rules_and_dot_stops_at_newline() -> R {
        let m = Masker::new("")?;
        assert_eq!(m.mask("Location file:/x/y"), "Location file:/x/y");
        assert_eq!(
            m.mask("Created By a\nnext#1"),
            "Created By [not included in comparison]\nnext#x"
        );
        Ok(())
    }

    /// Masking a recorded (already masked) golden row changes nothing.
    #[test]
    fn vendored_goldens_are_fixed_points() -> R {
        use crate::golden::parse::Expected;
        use crate::golden::run::load_statements;
        use crate::golden::{DEFAULT_TAG, GoldenSet};
        let m = Masker::new("")?;
        let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
        for pair in &set.pairs {
            for s in load_statements(pair)? {
                if let Expected::Rows(out) = s.expected {
                    for line in out.lines() {
                        assert_eq!(m.mask(line), line, "{}", pair.name);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn tracebacks_are_stripped_up_to_the_first_unindented_line() {
        let raw = "before\nTraceback (most recent call last):\n  File \"udf.py\", line 3, in f\n    raise\n\
ValueError: bad\nafter\nTraceback (most recent call last):\n  File \"b.py\"\n";
        assert_eq!(
            normalize_test_results(raw),
            "before\nValueError: bad\nafter"
        );
        // Only the exact header starts a traceback; indented lines elsewhere are kept.
        let kept = "Traceback (most recent call last): \n  indented\n\ttab";
        assert_eq!(normalize_test_results(kept), kept);
        // An empty line ends a traceback.
        assert_eq!(
            normalize_test_results("Traceback (most recent call last):\n  x\n\ny"),
            "\ny"
        );
        assert_eq!(normalize_test_results(""), "");
        assert_eq!(
            normalize_test_results("Traceback (most recent call last):"),
            ""
        );
    }

    /// `\s+$` is ASCII-only and its `$` also matches before a final line terminator.
    #[test]
    fn trailing_whitespace_follows_java_regex() {
        assert_eq!(normalize_test_results("a \t\r\n\u{b}\u{c} "), "a");
        assert_eq!(normalize_test_results("a\u{a0}"), "a\u{a0}");
        assert_eq!(normalize_test_results("a\u{3000}\n"), "a\u{3000}");
        assert_eq!(normalize_test_results("a  \u{85}"), "a\u{85}");
        // Split drops the empty field after the final newline first.
        assert_eq!(normalize_test_results("a \u{85}\n"), "a\u{85}");
        assert_eq!(normalize_test_results("a \u{85}\n\n\n"), "a\u{85}");
        assert_eq!(normalize_test_results("a \u{2028}"), "a\u{2028}");
        assert_eq!(
            normalize_test_results("a\u{2029}\u{2029}"),
            "a\u{2029}\u{2029}"
        );
        assert_eq!(normalize_test_results("a \u{85} "), "a \u{85}");
        assert_eq!(normalize_test_results("\n  \n"), "");
    }

    #[test]
    fn normalising_a_recorded_golden_changes_nothing_but_trailing_space() -> R {
        use crate::golden::parse::Expected;
        use crate::golden::run::load_statements;
        use crate::golden::{DEFAULT_TAG, GoldenSet};
        let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
        for pair in &set.pairs {
            for s in load_statements(pair)? {
                if let Expected::Rows(out) = s.expected {
                    assert_eq!(
                        normalize_test_results(&out),
                        out.trim_end_matches(JAVA_SPACE),
                        "{}",
                        pair.name
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn trailing_non_ascii_whitespace_is_kept_like_java() {
        // Java `\s` is ASCII-only: Spark keeps a trailing NBSP / ideographic space.
        assert_eq!(normalize_test_results("a\u{a0}"), "a\u{a0}");
        assert_eq!(normalize_test_results("a\u{3000}\n"), "a\u{3000}");
        assert_eq!(normalize_test_results("a \t\n"), "a");
    }
}

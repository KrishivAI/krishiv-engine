// SPDX-License-Identifier: Apache-2.0
//! Classification rules for failing golden statements: the first rule that matches a failure
//! names its category in the known-failures baseline ([`super::known_failures`]).
//!
//! A rule is data in the baseline file:
//!
//! ```json
//! {"category": "string-functions", "family": ["string"], "detail": "not supported"}
//! ```
//!
//! Every condition present must hold:
//! * `command`: the statement is (not) a SQL command, i.e. does (not) start like a query;
//! * `depends`: the statement does (not) mention a name an earlier command of the file creates
//!   (`CREATE ... VIEW|TABLE|FUNCTION|VARIABLE name`, `DECLARE name`);
//! * `file`, `query`, `detail`: regular expressions (`regex-lite` syntax) searched in the golden
//!   file name, the statement text and the failure detail;
//! * `family`: the function the failure names (`The feature function f`, `UNRESOLVED_ROUTINE`)
//!   belongs to one of these Spark function groups (from the Spark function matrix).
//!
//! Rules only seed new entries: the baseline lists every statement with its category, and the
//! ratchet compares runs with that list, never with the rules.

use std::collections::{BTreeMap, BTreeSet};

use regex_lite::Regex;
use serde_json::Value;

use super::parse::Statement;

/// One classification rule.
#[derive(Debug, Clone)]
pub struct Rule {
    /// Category key the rule assigns.
    pub category: String,
    command: Option<bool>,
    depends: Option<bool>,
    file: Option<Regex>,
    query: Option<Regex>,
    detail: Option<Regex>,
    family: Option<BTreeSet<String>>,
    /// The rule as written, for round trips.
    source: Value,
}

impl PartialEq for Rule {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl Eq for Rule {}

impl Rule {
    /// Parse one rule.
    pub fn parse(v: &Value) -> Result<Self, String> {
        let obj = v.as_object().ok_or("a rule is not an object")?;
        for key in obj.keys() {
            if ![
                "category", "command", "depends", "file", "query", "detail", "family",
            ]
            .contains(&key.as_str())
            {
                return Err(format!("unknown rule key `{key}`"));
            }
        }
        let category = obj
            .get("category")
            .and_then(Value::as_str)
            .ok_or("a rule without `category`")?
            .to_owned();
        let flag = |k: &str| -> Result<Option<bool>, String> {
            obj.get(k)
                .map(|v| {
                    v.as_bool()
                        .ok_or(format!("rule `{category}`: `{k}` is not a boolean"))
                })
                .transpose()
        };
        let re = |k: &str| -> Result<Option<Regex>, String> {
            obj.get(k)
                .map(|v| {
                    let s = v
                        .as_str()
                        .ok_or(format!("rule `{category}`: `{k}` is not a string"))?;
                    Regex::new(s).map_err(|e| format!("rule `{category}`: `{k}`: {e}"))
                })
                .transpose()
        };
        let family = obj
            .get("family")
            .map(|v| {
                v.as_array()
                    .ok_or(format!("rule `{category}`: `family` is not an array"))?
                    .iter()
                    .map(|f| {
                        f.as_str()
                            .map(str::to_owned)
                            .ok_or(format!("rule `{category}`: a family is not a string"))
                    })
                    .collect::<Result<BTreeSet<_>, _>>()
            })
            .transpose()?;
        let rule = Self {
            command: flag("command")?,
            depends: flag("depends")?,
            file: re("file")?,
            query: re("query")?,
            detail: re("detail")?,
            family,
            source: v.clone(),
            category,
        };
        if rule.source.as_object().is_some_and(|o| o.len() == 1) {
            return Err(format!("rule `{}` has no condition", rule.category));
        }
        Ok(rule)
    }

    /// The rule as JSON.
    #[must_use]
    pub const fn to_json(&self) -> &Value {
        &self.source
    }

    fn matches(&self, case: &Failure<'_>, families: &BTreeMap<String, String>) -> bool {
        let re = |r: &Option<Regex>, s: &str| r.as_ref().is_none_or(|r| r.is_match(s));
        self.command.is_none_or(|c| c == case.command)
            && self.depends.is_none_or(|d| d == case.depends)
            && re(&self.file, case.file)
            && re(&self.query, case.query)
            && re(&self.detail, case.detail)
            && self.family.as_ref().is_none_or(|fams| {
                function_name(case.detail)
                    .and_then(|f| families.get(&f.to_ascii_lowercase()))
                    .is_some_and(|fam| fams.contains(fam))
            })
    }
}

/// A failing statement, as the rules see it.
#[derive(Debug, Clone, Copy)]
pub struct Failure<'a> {
    /// Golden file name.
    pub file: &'a str,
    /// Statement text.
    pub query: &'a str,
    /// Failure detail ([`super::run::Case::detail`]).
    pub detail: &'a str,
    /// The statement is a SQL command ([`is_command`]).
    pub command: bool,
    /// The statement mentions a name a command of the file creates ([`FileNames::mentioned_by`]).
    pub depends: bool,
}

/// The first rule of `rules` that matches `case`.
#[must_use]
pub fn classify<'r>(
    rules: &'r [Rule],
    case: &Failure<'_>,
    families: &BTreeMap<String, String>,
) -> Option<&'r str> {
    rules
        .iter()
        .find(|r| r.matches(case, families))
        .map(|r| r.category.as_str())
}

/// Whether `query` is a SQL command rather than a query: it does not start with `SELECT`,
/// `WITH`, `VALUES`, `FROM`, `TABLE`, `EXPLAIN` or `(` (and is not empty).
#[must_use]
pub fn is_command(query: &str) -> bool {
    let q = query.trim_start();
    if q.starts_with('(') {
        return false;
    }
    let word: String = q
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_ascii_uppercase();
    !word.is_empty()
        && !["SELECT", "WITH", "VALUES", "FROM", "TABLE", "EXPLAIN"].contains(&word.as_str())
}

/// The function a failure detail names: `The feature function f is not supported` or
/// `` UNRESOLVED_ROUTINE: Cannot resolve routine `f` ``.
#[must_use]
pub fn function_name(detail: &str) -> Option<&str> {
    if let Some(rest) = detail.split("The feature function ").nth(1) {
        return rest.split(" is not supported").next();
    }
    let rest = detail.split("Cannot resolve routine `").nth(1)?;
    rest.split('`').next()
}

/// The names the commands of one golden file create (lower case, last name part).
#[derive(Debug, Clone, Default)]
pub struct FileNames(BTreeSet<String>);

impl FileNames {
    /// Collect from the statements of a file.
    #[must_use]
    pub fn of(statements: &[Statement]) -> Self {
        let mut names = BTreeSet::new();
        for s in statements {
            // Words with their dots and backquotes, so a qualified name is one word.
            let words: Vec<&str> = s
                .query
                .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '`')))
                .filter(|w| !w.is_empty())
                .collect();
            let upper: Vec<String> = words.iter().map(|w| w.to_ascii_uppercase()).collect();
            let at = |i: usize| upper.get(i).map_or("", String::as_str);
            // `CREATE [OR REPLACE] [GLOBAL] [TEMP|TEMPORARY] VIEW|TABLE|FUNCTION|VARIABLE
            // [IF NOT EXISTS] name`, `DECLARE [OR REPLACE] [VARIABLE] name`.
            let mut i = match at(0) {
                "CREATE" | "DECLARE" => 1,
                _ => continue,
            };
            let declare = at(0) == "DECLARE";
            while matches!(at(i), "OR" | "REPLACE" | "GLOBAL" | "TEMP" | "TEMPORARY") {
                i += 1;
            }
            if matches!(at(i), "VIEW" | "TABLE" | "FUNCTION" | "VARIABLE") {
                i += 1;
            } else if !declare {
                continue;
            }
            if at(i) == "IF" && at(i + 1) == "NOT" && at(i + 2) == "EXISTS" {
                i += 3;
            }
            if let Some(last) = words
                .get(i)
                .and_then(|name| name.rsplit('.').next())
                .map(|n| n.trim_matches('`'))
                .filter(|n| !n.is_empty())
            {
                names.insert(last.to_ascii_lowercase());
            }
        }
        Self(names)
    }

    /// Whether `query` mentions one of the names.
    #[must_use]
    pub fn mentioned_by(&self, query: &str) -> bool {
        words(query)
            .iter()
            .any(|w| self.0.contains(&w.to_ascii_lowercase()))
    }
}

/// The identifier-like words of `s` (ASCII letters, digits, `_`; backquotes dropped), so a
/// qualified name gives its parts.
fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn stmt(q: &str) -> Statement {
        Statement {
            query: q.into(),
            schema: String::new(),
            expected: super::super::parse::Expected::Rows(String::new()),
        }
    }

    fn failure<'a>(query: &'a str, detail: &'a str) -> Failure<'a> {
        Failure {
            file: "group-by",
            query,
            detail,
            command: is_command(query),
            depends: false,
        }
    }

    #[test]
    fn commands_and_created_names() {
        for q in [
            "SELECT 1",
            " with t as (select 1) select * from t",
            "(SELECT 1)",
            "VALUES 1",
            "explain select 1",
            "FROM t SELECT a",
            "",
            "TABLE t",
        ] {
            assert!(!is_command(q), "{q}");
        }
        for q in [
            "CREATE TEMPORARY VIEW t AS SELECT 1",
            "set a=b",
            "DESC t",
            "insert into t values 1",
        ] {
            assert!(is_command(q), "{q}");
        }
        let names = FileNames::of(&[
            stmt("CREATE OR REPLACE TEMPORARY VIEW testData AS SELECT 1"),
            stmt("create table if not exists db.`t2` (a int)"),
            stmt("DECLARE OR REPLACE VARIABLE v INT"),
            stmt("declare w = 1"),
            stmt("CREATE DATABASE nope"),
            stmt("SELECT 1"),
        ]);
        assert!(names.mentioned_by("SELECT a FROM TESTDATA"));
        assert!(names.mentioned_by("select * from db.t2"));
        assert!(names.mentioned_by("SELECT v + w"));
        assert!(!names.mentioned_by("SELECT * FROM nope2"));
        assert!(!names.mentioned_by("SELECT * FROM db"));
    }

    #[test]
    fn function_names_from_details() {
        assert_eq!(
            function_name(
                "error KRISHIV_UNSUPPORTED_FEATURE: The feature function to_timestamp is not supported by Krishiv yet."
            ),
            Some("to_timestamp")
        );
        assert_eq!(
            function_name("error UNRESOLVED_ROUTINE: Cannot resolve routine `udf` on search path"),
            Some("udf")
        );
        assert_eq!(function_name("rows differ"), None);
    }

    #[test]
    fn first_matching_rule_wins() -> Result<(), String> {
        let rules = [
            Rule::parse(&json!({"category": "cmd", "command": true}))?,
            Rule::parse(&json!({"category": "dep", "depends": true}))?,
            Rule::parse(&json!({"category": "str", "family": ["string"]}))?,
            Rule::parse(
                &json!({"category": "win", "detail": "window functions", "file": "^group"}),
            )?,
            Rule::parse(&json!({"category": "cte", "query": "(?i)^with"}))?,
        ];
        let families = BTreeMap::from([("upper".to_owned(), "string".to_owned())]);
        let pick = |f: &Failure<'_>| classify(&rules, f, &families).map(str::to_owned);
        assert_eq!(
            pick(&failure("SET x=1", "parse failed")).as_deref(),
            Some("cmd")
        );
        let mut dep = failure("SELECT * FROM t", "error TABLE_OR_VIEW_NOT_FOUND");
        dep.depends = true;
        assert_eq!(pick(&dep).as_deref(), Some("dep"));
        assert_eq!(
            pick(&failure(
                "SELECT upper(a)",
                "The feature function upper is not supported"
            ))
            .as_deref(),
            Some("str")
        );
        assert_eq!(
            pick(&failure(
                "SELECT lower(a)",
                "The feature function lower is not supported"
            )),
            None
        );
        assert_eq!(
            pick(&failure(
                "SELECT rank() OVER ()",
                "The feature window functions is not"
            ))
            .as_deref(),
            Some("win")
        );
        assert_eq!(
            pick(&failure("WITH t AS (SELECT 1) SELECT 2", "x")).as_deref(),
            Some("cte")
        );
        Ok(())
    }

    #[test]
    fn bad_rules_are_rejected() {
        for (rule, msg) in [
            (json!(1), "not an object"),
            (json!({"command": true}), "without `category`"),
            (json!({"category": "a"}), "no condition"),
            (
                json!({"category": "a", "colour": "red"}),
                "unknown rule key",
            ),
            (json!({"category": "a", "command": "yes"}), "not a boolean"),
            (json!({"category": "a", "detail": "("}), "`detail`"),
            (json!({"category": "a", "detail": 1}), "not a string"),
            (json!({"category": "a", "family": "string"}), "not an array"),
            (json!({"category": "a", "family": [1]}), "not a string"),
        ] {
            let err = Rule::parse(&rule).map(drop).unwrap_err();
            assert!(err.contains(msg), "{msg}: {err}");
        }
        let r = Rule::parse(&json!({"category": "a", "query": "x"}));
        assert_eq!(
            r.map(|r| r.to_json().clone()),
            Ok(json!({"category": "a", "query": "x"}))
        );
    }
}

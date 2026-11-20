// SPDX-License-Identifier: Apache-2.0
//! Spark's documentation group of its math, string, datetime and misc functions
//! (`vendored/spark-function-groups.tsv`, derived from Spark 4.2.0's own docs dump
//! `vendored/spark-function-docs.jsonl`; a test keeps the two in step). Flight SQL's
//! `GetSqlInfo` lists functions by these groups (design P3-T101 §5).

/// The vendored groups: `name\tgroup`, one `#` header line.
const TSV: &str = include_str!("../../vendored/spark-function-groups.tsv");

/// A function group Spark's documentation uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FunctionGroup {
    /// `math_funcs`.
    Math,
    /// `string_funcs`.
    String,
    /// `datetime_funcs`.
    Datetime,
    /// `misc_funcs`.
    Misc,
}

impl FunctionGroup {
    fn of(name: &str) -> Option<Self> {
        match name {
            "math_funcs" => Some(Self::Math),
            "string_funcs" => Some(Self::String),
            "datetime_funcs" => Some(Self::Datetime),
            "misc_funcs" => Some(Self::Misc),
            _ => None,
        }
    }
}

/// Every function of the four groups, by name, as Spark's documentation groups them.
pub fn spark_function_groups() -> impl Iterator<Item = (&'static str, FunctionGroup)> {
    TSV.lines().filter(|l| !l.starts_with('#')).filter_map(|l| {
        let (name, group) = l.split_once('\t')?;
        Some((name, FunctionGroup::of(group)?))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCS: &str = include_str!("../../vendored/spark-function-docs.jsonl");

    /// A string field of one docs line (the dump writes `"key": "value"`, no escaped quotes in
    /// the names and groups read here).
    fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        let start = line.find(&format!("\"{key}\": \""))? + key.len() + 5;
        let len = line[start..].find('"')?;
        Some(&line[start..start + len])
    }

    #[test]
    fn groups_match_the_docs_dump() {
        let mut want: Vec<(&str, FunctionGroup)> = DOCS
            .lines()
            .filter(|l| field(l, "registry") == Some("function"))
            .filter_map(|l| Some((field(l, "name")?, FunctionGroup::of(field(l, "group")?)?)))
            .collect();
        want.sort_unstable_by_key(|(n, _)| *n);
        let got: Vec<_> = spark_function_groups().collect();
        assert_eq!(got, want, "regenerate vendored/spark-function-groups.tsv");
        assert!(
            got.iter()
                .any(|&(n, g)| n == "abs" && g == FunctionGroup::Math)
        );
        assert!(
            got.iter()
                .any(|&(n, g)| n == "upper" && g == FunctionGroup::String)
        );
        assert!(
            got.iter()
                .any(|&(n, g)| n == "date_add" && g == FunctionGroup::Datetime)
        );
    }
}

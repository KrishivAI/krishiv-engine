// SPDX-License-Identifier: Apache-2.0
//! `${name}` substitution in SQL text, before parsing (Spark's `VariableSubstitution`, on by default,
//! `spark.sql.variable.substitute`).
//!
//! `${name}` is the configuration value of `name`; `${hivevar:name}`, `${hiveconf:name}`, `${spark:name}`
//! and `${sparkconf:name}` look the same name up (also stored with their prefix, as `SET hivevar:name = v`
//! stores it); `${env:NAME}` and `${system:NAME}` read the process environment. Values are substituted
//! again (to a depth of 10); an unknown name stays as written. Only the form `${` + characters other than
//! `}`, `$` and space + `}` is a reference.

use std::borrow::Cow;

/// The nesting depth after which a reference stays as written (Spark's `ConfigReader` limit).
const MAX_DEPTH: usize = 10;

/// `text` with every known `${name}` replaced by its value from `lookup` (configuration by key).
pub(crate) fn substitute<'a>(
    text: &'a str,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Cow<'a, str> {
    if !text.contains("${") {
        return Cow::Borrowed(text);
    }
    Cow::Owned(go(text, lookup, 0))
}

fn go(text: &str, lookup: &dyn Fn(&str) -> Option<String>, depth: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("${") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 2..];
        let end = after.find(['}', '$', ' ']);
        match end {
            Some(i) if i > 0 && after[i..].starts_with('}') => {
                let name = &after[..i];
                match value(name, lookup).filter(|_| depth < MAX_DEPTH) {
                    Some(v) => out.push_str(&go(&v, lookup, depth + 1)),
                    None => out.push_str(&rest[at..at + 2 + i + 1]),
                }
                rest = &after[i + 1..];
            }
            _ => {
                out.push_str("${");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn value(name: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    if let Some((prefix, key)) = name.split_once(':') {
        return match prefix {
            "env" | "system" => std::env::var(key).ok(),
            "hivevar" | "hiveconf" | "spark" | "sparkconf" => lookup(key).or_else(|| lookup(name)),
            _ => lookup(name),
        };
    }
    lookup(name)
        .or_else(|| lookup(&format!("hivevar:{name}")))
        .or_else(|| lookup(&format!("hiveconf:{name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str) -> String {
        let lookup = |k: &str| match k {
            "a" => Some("1".to_owned()),
            "hivevar:b" => Some("${a}2".to_owned()),
            "loop" => Some("${loop}".to_owned()),
            _ => None,
        };
        substitute(text, &lookup).into_owned()
    }

    #[test]
    fn replaces_known_names_and_keeps_the_rest() {
        assert_eq!(run("SELECT ${a}, ${b}, ${hivevar:b}"), "SELECT 1, 12, 12");
        assert_eq!(
            run("${missing} ${} ${ a} $ {a} ${a"),
            "${missing} ${} ${ a} $ {a} ${a"
        );
        assert_eq!(run("no references"), "no references");
        assert_eq!(run("${hiveconf:a}${spark:a}"), "11");
    }

    #[test]
    fn a_self_reference_stops_at_the_depth_limit() {
        assert_eq!(run("${loop}"), "${loop}");
    }
}

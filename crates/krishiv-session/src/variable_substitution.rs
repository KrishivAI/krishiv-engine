// SPDX-License-Identifier: Apache-2.0
//! `${name}` substitution in SQL text, before parsing (Spark's `VariableSubstitution`, on by default,
//! `spark.sql.variable.substitute`).
//!
//! `${name}` is the configuration value of `name`; `${hivevar:name}`, `${hiveconf:name}`, `${spark:name}`
//! and `${sparkconf:name}` look the same name up (also stored with their prefix, as `SET hivevar:name = v`
//! stores it); `${env:NAME}` and `${system:NAME}` read the process environment, except in a served session
//! (`krishiv server`), where they stay as written unless `krishiv.server.sql.substituteEnvironment` is
//! on (a recorded difference from Spark: a client must not read the server's secrets). In a served session
//! a setting that `SET` and Connect `Config.get` hide (a secret-like key, or a value holding a credential,
//! `krishiv_common::redact::is_redacted_setting`) stays as written too. Values are substituted
//! again (to a depth of 10); an unknown name stays as written. Only the form `${` + characters other than
//! `}`, `$` and space + `}` is a reference.

use std::borrow::Cow;

/// The nesting depth after which a reference stays as written (Spark's `ConfigReader` limit).
const MAX_DEPTH: usize = 10;

impl crate::Session {
    /// `query` as this session's parser reads it: its `${name}` references replaced from the
    /// session's configuration, unless `spark.sql.variable.substitute` is false (Spark's
    /// `SparkSqlParser` substitutes before it parses). For callers that parse SQL text themselves
    /// (the golden harness); outside semver, like the rest of `unstable-ir`.
    #[cfg(feature = "unstable-ir")]
    #[must_use]
    pub fn substitute_variables(&self, query: &str) -> String {
        self.substituted(query).into_owned()
    }

    /// `query` with its `${name}` references replaced, unless `spark.sql.variable.substitute` is false.
    pub(crate) fn substituted<'q>(&self, query: &'q str) -> std::borrow::Cow<'q, str> {
        let config = self.state.conf.snapshot();
        let serving = self.serves_others();
        if crate::conf::get(&config, "spark.sql.variable.substitute")
            .is_none_or(|v| !v.trim().eq_ignore_ascii_case("false"))
        {
            crate::variable_substitution::substitute(
                query,
                &crate::variable_substitution::Sources {
                    lookup: &|k| {
                        crate::conf::get(&config, k).filter(|v| {
                            !serving || !krishiv_common::redact::is_redacted_setting(k, Some(v))
                        })
                    },
                    environment: self.reads_environment(&config),
                },
            )
        } else {
            std::borrow::Cow::Borrowed(query)
        }
    }

    /// Whether this session is a served one: its runtime (`DeclMode::Server`) serves callers other
    /// than the process owner, so `${name}` shows no setting `SET` and `Config.get` hide
    /// (`krishiv_common::redact::is_redacted_setting`), whatever `substituteEnvironment` says.
    fn serves_others(&self) -> bool {
        self.runtime().catalogs().mode() == crate::catalog_decl::DeclMode::Server
    }

    /// Whether `${env:..}` and `${system:..}` read the process environment: always outside a served
    /// session; in one only when the operator turned `krishiv.server.sql.substituteEnvironment` on at
    /// startup (a cluster key: no `SET` or session builder changes it).
    pub(crate) fn reads_environment(&self, config: &krishiv_common::config::Config) -> bool {
        !self.serves_others()
            || config
                .get(&krishiv_common::config::SERVER_SQL_SUBSTITUTE_ENVIRONMENT)
                .and_then(krishiv_common::config::ConfigValue::as_bool)
                .unwrap_or(false)
    }
}

/// Where a `${name}` reference's value comes from.
pub(crate) struct Sources<'l> {
    /// The configuration by key.
    pub(crate) lookup: &'l dyn Fn(&str) -> Option<String>,
    /// Whether `${env:..}` and `${system:..}` read the process environment; when not, they stay
    /// as written, like an unknown name.
    pub(crate) environment: bool,
}

/// `text` with every known `${name}` replaced by its value from `sources`.
pub(crate) fn substitute<'a>(text: &'a str, sources: &Sources<'_>) -> Cow<'a, str> {
    if !text.contains("${") {
        return Cow::Borrowed(text);
    }
    Cow::Owned(go(text, sources, 0))
}

fn go(text: &str, sources: &Sources<'_>, depth: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("${") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 2..];
        let end = after.find(['}', '$', ' ']);
        match end {
            Some(i) if i > 0 && after[i..].starts_with('}') => {
                let name = &after[..i];
                match value(name, sources).filter(|_| depth < MAX_DEPTH) {
                    Some(v) => out.push_str(&go(&v, sources, depth + 1)),
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

fn value(name: &str, sources: &Sources<'_>) -> Option<String> {
    let lookup = sources.lookup;
    if let Some((prefix, key)) = name.split_once(':') {
        return match prefix {
            "env" | "system" if !sources.environment => None,
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
        substitute(
            text,
            &Sources {
                lookup: &lookup,
                environment: true,
            },
        )
        .into_owned()
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
    fn without_the_environment_env_and_system_stay_as_written() {
        // `PATH` is set in every test environment; the configuration still substitutes.
        let lookup = |k: &str| (k == "a").then(|| "${env:PATH}".to_owned());
        let sources = |environment| Sources {
            lookup: &lookup,
            environment,
        };
        let text = "${env:PATH} ${system:PATH} ${a}";
        assert_eq!(
            substitute(text, &sources(false)),
            "${env:PATH} ${system:PATH} ${env:PATH}"
        );
        let read = substitute(text, &sources(true));
        assert!(!read.contains("${env:PATH}"), "{read}");
    }

    #[test]
    fn a_self_reference_stops_at_the_depth_limit() {
        assert_eq!(run("${loop}"), "${loop}");
    }
}

//! Runtime gate for `krishiv.experimental.*` config keys (ADR 0027).
//!
//! Experimental features are off by default and are enabled only by an
//! explicit `true` value in the configuration map handed in by the caller.
//! There is no global state.

use std::collections::HashMap;

/// Config key prefix for experimental features.
pub const PREFIX: &str = "krishiv.experimental.";

/// Returns whether the experimental feature `key` is enabled in `config`.
///
/// `key` may be given with or without the `krishiv.experimental.` prefix.
/// Only a value of `true` (ASCII case-insensitive, surrounding whitespace
/// ignored) enables a feature; missing keys and any other value are off.
pub fn is_enabled(config: &HashMap<String, String>, key: &str) -> bool {
    let full = if key.starts_with(PREFIX) {
        key.to_owned()
    } else {
        format!("{PREFIX}{key}")
    };
    config
        .get(&full)
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("true"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn off_by_default() {
        assert!(!is_enabled(&HashMap::new(), "ivm"));
    }

    #[test]
    fn enabled_with_short_or_full_key() {
        let c = cfg(&[("krishiv.experimental.ivm", "true")]);
        assert!(is_enabled(&c, "ivm"));
        assert!(is_enabled(&c, "krishiv.experimental.ivm"));
    }

    #[test]
    fn value_parsing_is_strict() {
        let c = cfg(&[
            ("krishiv.experimental.a", " TRUE "),
            ("krishiv.experimental.b", "false"),
            ("krishiv.experimental.c", "1"),
        ]);
        assert!(is_enabled(&c, "a"));
        assert!(!is_enabled(&c, "b"));
        assert!(!is_enabled(&c, "c"));
    }

    #[test]
    fn other_keys_do_not_enable() {
        let c = cfg(&[("krishiv.experimental.ivm", "true")]);
        assert!(!is_enabled(&c, "other"));
        assert!(!is_enabled(&cfg(&[("ivm", "true")]), "ivm"));
    }
}

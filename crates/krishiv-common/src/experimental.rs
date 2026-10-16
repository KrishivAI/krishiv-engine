//! Runtime gate for `krishiv.experimental.*` config keys (ADR 0027).
//!
//! Experimental features are registered [`crate::config`] entries of type
//! bool, off by default. Values are validated when the [`Config`] is built,
//! so only `true`/`false` (ASCII case-insensitive, trimmed) are accepted.
//! There is no global state.

use crate::config::{self, Config, ConfigValue};

/// Config key prefix for experimental features.
pub const PREFIX: &str = "krishiv.experimental.";

/// Returns whether the experimental feature `key` is enabled in `config`.
///
/// `key` may be given with or without the `krishiv.experimental.` prefix.
/// Unregistered features are never enabled.
pub fn is_enabled(config: &Config, key: &str) -> bool {
    let full = if key.starts_with(PREFIX) {
        key.to_owned()
    } else {
        format!("{PREFIX}{key}")
    };
    config::lookup(&full)
        .and_then(|entry| config.get(entry))
        .and_then(ConfigValue::as_bool)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::error::ErrorClass;

    fn cfg(pairs: &[(&str, &str)]) -> Config {
        Config::from_pairs(pairs.iter().copied()).unwrap().0
    }

    #[test]
    fn off_by_default() {
        assert!(!is_enabled(&Config::new(), "ivm"));
    }

    #[test]
    fn enabled_with_short_or_full_key() {
        let c = cfg(&[("krishiv.experimental.ivm", "true")]);
        assert!(is_enabled(&c, "ivm"));
        assert!(is_enabled(&c, "krishiv.experimental.ivm"));
    }

    #[test]
    fn value_parsing_is_strict() {
        assert!(is_enabled(
            &cfg(&[("krishiv.experimental.ivm", " TRUE ")]),
            "ivm"
        ));
        assert!(!is_enabled(
            &cfg(&[("krishiv.experimental.ivm", "false")]),
            "ivm"
        ));
        // Anything but true/false is rejected when the config is built.
        let e = Config::from_pairs([("krishiv.experimental.ivm", "1")]).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivInvalidConfigValue);
    }

    #[test]
    fn other_keys_do_not_enable() {
        let c = cfg(&[("krishiv.experimental.ivm", "true"), ("ivm", "true")]);
        assert!(!is_enabled(&c, "unalignedCheckpoints"));
        assert!(!is_enabled(&c, "notRegistered"));
        assert!(!is_enabled(&cfg(&[("ivm", "true")]), "ivm"));
    }
}

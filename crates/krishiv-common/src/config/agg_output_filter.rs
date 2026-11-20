// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.exec.agg.outputFilter` entry (kept out of `registry.rs`, which is at its size
//! cap).

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// A filter over a grouped aggregate's values (a `HAVING`) is also
/// evaluated by the aggregate before it decodes its groups' keys.
pub static EXEC_AGG_OUTPUT_FILTER: ConfigEntry = ConfigEntry {
    key: "krishiv.exec.agg.outputFilter",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Internal: a deterministic filter that reads only a grouped aggregate's values (a \
          HAVING such as `sum(x) > 300`, through plain column projections) is also evaluated \
          by the aggregate on each output batch's values before it decodes the keys, so only \
          the groups that pass are decoded and handed on. The filter above still runs; results \
          are the same either way.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

#[cfg(test)]
mod tests {
    use super::EXEC_AGG_OUTPUT_FILTER;
    use crate::config::{Config, ConfigValue};

    #[test]
    fn defaults_on_and_parses() {
        let mut c = Config::new();
        let get = |c: &Config| {
            c.get(&EXEC_AGG_OUTPUT_FILTER)
                .and_then(ConfigValue::as_bool)
        };
        assert_eq!(get(&c), Some(true));
        c.set("krishiv.exec.agg.outputFilter", "false").unwrap();
        assert_eq!(get(&c), Some(false));
        assert!(c.set("krishiv.exec.agg.outputFilter", "maybe").is_err());
    }
}

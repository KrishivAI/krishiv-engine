// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.optimizer.joinCost` entry (kept out of `registry.rs`, which is at its size cap).

use std::borrow::Cow;

use super::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// `krishiv.optimizer.joinCost`.
pub static JOIN_COST: ConfigEntry = ConfigEntry {
    key: "krishiv.optimizer.joinCost",
    value_type: ValueType::Enum(&["rows", "bytes"]),
    default: Some(ConfigValue::String(Cow::Borrowed("rows"))),
    doc: "What the join optimizer minimises when it orders a tree of inner joins, and which input \
          of each join it builds. `rows`: the rows of the intermediate results; the input with \
          fewer rows is built; when two orders cost the same rows (to a relative 1e-9), the one whose \
          builds hold fewer bytes wins. `bytes`: each join's result rows times 48 bytes plus the bytes of \
          its build input (rows times the estimated width of the columns still read, Spark's \
          default size per type); the input with fewer rows is built unless it holds more than \
          twice the other's bytes. Results are the same either way.",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: None,
    settable_by: SettableBy::User,
};

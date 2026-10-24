---
name: krishiv-add-config
description: Use when adding or changing a krishiv.* or krishiv.experimental.* config key, or a Spark config alias. Covers the registry entry, the sorted lookup index, tests and regenerating the config reference.
---

# Add a config key

Keys are typed entries in one registry: `crates/krishiv-common/src/config/registry.rs`. Behaviour flags are runtime
config, never Cargo features (ADR 0027). Model: the `krishiv.sql.optimizer.windowGroupLimitThreshold` entry
(`WINDOW_GROUP_LIMIT_THRESHOLD`) or the `krishiv.spill.*` entries.

## Checklist
1. **Entry.** A `pub static NAME: ConfigEntry` in `registry.rs` with `key`, `value_type` (`ValueType`), `default`,
   `doc` (user-facing; it becomes the reference text), `scope` (`Scope::Session` if it can change at runtime,
   `Scope::Cluster` if fixed at startup), `version_added`, `spark_alias` (`Some("spark....")` only when Spark has the
   same setting with the same meaning). Experimental flags: key under `krishiv.experimental.`, `Bool`, default `false`
   (a test enforces it).
2. **Register it twice.**
   - Add `&NAME` to `REGISTRY`.
   - Add `(key, &NAME)` to `INDEX`, and `(alias, &NAME)` as well when there is an alias. `INDEX` is sorted by
     (length, then name); a test (`the_lookup_index_is_sorted_and_complete`) fails if an entry is missing or out of
     order.
3. **Tests.** `crates/krishiv-common/src/config/tests.rs`: default, parse of valid values, rejection of invalid
   ones (model: `spill_entries_parse_and_default`); an alias case in `spark_aliases_map_to_krishiv_keys` if you
   added one. `registry_is_well_formed` already checks prefixes, duplicates and that defaults round-trip.
4. **Use it.** Read it with `Config::get(&NAME)` where the value is needed (session values reach queries through
   `crates/krishiv-session/src/conf.rs`). For experimental flags use `krishiv_common::experimental::is_enabled`. No
   process-global state.
5. **Reference docs.** `cargo xtask docs-config` prints the generated reference (stdout, or a path argument).
   Run it and read the row for the new key. The output is not committed in this repository.
6. **Behaviour changes** that Spark can observe also need goldens (skill `krishiv-spark-function`).

## Verify
`cargo nextest run -p krishiv-common --no-tests=pass`, `cargo xtask docs-config | grep <key>`, clippy `-D warnings`.

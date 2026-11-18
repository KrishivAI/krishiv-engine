// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.iceberg.*` write and commit keys (kept out of `tests.rs`, which is at its size cap).

use super::*;

/// PR-36a: the open-file cap of an Iceberg write task.
#[test]
fn iceberg_write_max_open_files_parses_and_defaults() {
    let mut c = Config::new();
    let get = |c: &Config| {
        c.get(&ICEBERG_WRITE_MAX_OPEN_FILES)
            .and_then(ConfigValue::as_int)
    };
    assert_eq!(get(&c), Some(100));
    c.set("krishiv.iceberg.write.maxOpenFiles", "7").unwrap();
    assert_eq!(get(&c), Some(7));
    assert!(c.set("krishiv.iceberg.write.maxOpenFiles", "0").is_err());
    assert!(c.set("krishiv.iceberg.write.maxOpenFiles", "-3").is_err());
    assert!(c.set("krishiv.iceberg.write.maxOpenFiles", "many").is_err());
    assert_eq!(ICEBERG_WRITE_MAX_OPEN_FILES.scope, Scope::Session);
}

/// PR-36b: the bound on the history an Iceberg commit validates.
#[test]
fn iceberg_commit_max_intervening_parses_and_defaults() {
    let mut c = Config::new();
    let get = |c: &Config| {
        c.get(&ICEBERG_COMMIT_MAX_INTERVENING)
            .and_then(ConfigValue::as_int)
    };
    assert_eq!(get(&c), Some(1000));
    c.set("krishiv.iceberg.commit.maxIntervening", "5").unwrap();
    assert_eq!(get(&c), Some(5));
    assert!(c.set("krishiv.iceberg.commit.maxIntervening", "0").is_err());
    assert!(
        c.set("krishiv.iceberg.commit.maxIntervening", "-1")
            .is_err()
    );
    assert!(
        c.set("krishiv.iceberg.commit.maxIntervening", "many")
            .is_err()
    );
    assert_eq!(ICEBERG_COMMIT_MAX_INTERVENING.scope, Scope::Session);
}

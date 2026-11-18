// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.exec.join.directoryLoad` setting (kept out of `tests.rs`, which is at its size cap).

use super::*;

#[test]
fn join_directory_load_defaults_to_auto_and_parses() {
    let mut c = Config::new();
    let get = |c: &Config| {
        c.get(&EXEC_JOIN_DIRECTORY_LOAD)
            .and_then(ConfigValue::as_str)
            .map(str::to_owned)
    };
    assert_eq!(get(&c).as_deref(), Some("auto"));
    for v in ["compact", "roomy", "auto"] {
        c.set("krishiv.exec.join.directoryLoad", v).unwrap();
        assert_eq!(get(&c).as_deref(), Some(v));
    }
    assert!(c.set("krishiv.exec.join.directoryLoad", "half").is_err());
}

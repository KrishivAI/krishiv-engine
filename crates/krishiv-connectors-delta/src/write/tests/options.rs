// SPDX-License-Identifier: Apache-2.0
//! Delta's write options the commit carries (P3-T021 PR-37e): `userMetadata` as
//! `commitInfo.userMetadata` on a create and on writes to an existing table, `mergeSchema` as
//! `operationParameters.canMergeSchema` (delta-spark 4.4.1 `WriteIntoDelta.scala:114-124`,
//! `DeltaOperations.scala:129-151`; probe `OPTIONS.json` `user_metadata`,
//! `merge_schema_true_same`).

use std::collections::BTreeMap;

use krishiv_exec::planner::WriteMode;
use serde_json::Value;

use super::super::NewDeltaTable;
use super::{Env, commit_info, modes_schema, rows};

#[test]
fn user_metadata_and_merge_schema_reach_the_commit() {
    let env = Env::new();
    let table = NewDeltaTable::new(
        env.engine.clone(),
        &modes_schema(),
        vec!["p".to_owned()],
        BTreeMap::new(),
        None,
    )
    .unwrap();
    env.write(
        &table,
        WriteMode::Append,
        rows(&[(1, "a", Some("x"))]),
        &[("userMetadata", "created")],
    )
    .unwrap();
    let info = commit_info(&env.log(0));
    assert_eq!(info["userMetadata"], Value::from("created"), "{info}");

    env.write(
        &env.open().unwrap(),
        WriteMode::Append,
        rows(&[(2, "b", Some("x"))]),
        // Option names are case insensitive, as `DeltaOptions` reads them.
        &[("USERMETADATA", "hello"), ("mergeSchema", "true")],
    )
    .unwrap();
    let info = commit_info(&env.log(1));
    assert_eq!(info["userMetadata"], Value::from("hello"), "{info}");
    assert_eq!(
        info["operationParameters"]["canMergeSchema"],
        Value::from("true"),
        "{info}"
    );

    // Without the options neither field is written, as delta-spark leaves them out.
    env.write(
        &env.open().unwrap(),
        WriteMode::Truncate,
        rows(&[(3, "c", Some("y"))]),
        &[("mergeSchema", "false")],
    )
    .unwrap();
    let info = commit_info(&env.log(2));
    assert!(info.get("userMetadata").is_none(), "{info}");
    assert!(
        info["operationParameters"].get("canMergeSchema").is_none(),
        "{info}"
    );
}

#[test]
fn an_empty_create_carries_its_user_metadata() {
    for (user_metadata, want) in [(Some("m"), Value::from("m")), (None, Value::Null)] {
        let env = Env::new();
        let table = NewDeltaTable::new(
            env.engine.clone(),
            &modes_schema(),
            vec!["p".to_owned()],
            BTreeMap::new(),
            None,
        )
        .unwrap();
        env.rt
            .block_on(table.create_empty("w", user_metadata))
            .unwrap();
        let info = commit_info(&env.log(0));
        assert_eq!(
            info.get("userMetadata").cloned().unwrap_or(Value::Null),
            want,
            "{info}"
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
//! P2-T082b acceptance: Spark 4.2.0 reads Krishiv's partitioned output identically. The fixture
//! (`tests/scripts/gen_partitioned_write_goldens.py`) holds, per case, the directories Spark's own writer makes and
//! what Spark's reader returns from Krishiv's output (checked at generation against Spark's read of its own output).
//! This test writes every case again and checks that Krishiv's directory names are Spark's writer's and that
//! Krishiv's own read gives the schema, nullability and rows Spark read. Spark itself runs only at generation; the
//! nightly `differential` job regenerates the fixture against Spark (`--check`), so a writer change Spark would
//! read differently fails there.

use krishiv_testkit::partitioned_writes::{
    ReadBack, cases_path, goldens_path, load, partition_dirs, read_back, session, write,
};
use serde_json::Value;

#[test]
fn spark_reads_krishiv_partitioned_writes() {
    let cases = load(&cases_path()).unwrap();
    let goldens: Value =
        serde_json::from_str(&std::fs::read_to_string(goldens_path()).unwrap()).unwrap();
    let goldens = goldens["cases"].as_array().unwrap();
    assert_eq!(
        goldens.len(),
        cases.len(),
        "regenerate the fixture: the cases changed"
    );
    let session = session().unwrap();
    let mut failures = Vec::new();
    for (case, golden) in cases.iter().zip(goldens) {
        assert_eq!(golden["id"], case.id.as_str());
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join(&case.id);
        write(&session, case, &dir).unwrap();
        let texts = |key: &str| -> Vec<String> {
            golden[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect()
        };
        let dirs = partition_dirs(&dir).unwrap();
        if dirs != texts("spark_partition_dirs") {
            failures.push(format!(
                "{}: Krishiv wrote {dirs:?}, Spark's writer {:?}",
                case.id,
                texts("spark_partition_dirs")
            ));
        }
        let spark = ReadBack {
            schema: golden["schema"].as_str().unwrap().to_owned(),
            nullable: golden["nullable"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect(),
            rows: texts("rows"),
        };
        let ours = read_back(&session, case, &dir).unwrap();
        if ours != spark {
            failures.push(format!(
                "{}: Krishiv read {ours:?}, Spark read {spark:?}",
                case.id
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

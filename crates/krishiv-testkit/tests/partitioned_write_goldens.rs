// SPDX-License-Identifier: Apache-2.0
//! P2-T082b acceptance: Spark 4.2.0 reads Krishiv's partitioned output identically. The fixture holds what a real
//! Spark read of each case's directory returned (`tests/scripts/gen_partitioned_write_goldens.py`); this test writes
//! every case again and checks that the partition directories are the ones Spark read and that Krishiv's own read of
//! them gives Spark's schema and rows. A layout change fails here first; regenerate the fixture with Spark then.

use krishiv_session::Session;
use krishiv_testkit::partitioned_writes::{
    ReadBack, cases_path, goldens_path, load, partition_dirs, read_back, write,
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
    let session = Session::new().unwrap();
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
        if dirs != texts("partition_dirs") {
            failures.push(format!(
                "{}: directories {dirs:?}, Spark read {:?}",
                case.id,
                texts("partition_dirs")
            ));
        }
        let spark = ReadBack {
            schema: golden["schema"].as_str().unwrap().to_owned(),
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

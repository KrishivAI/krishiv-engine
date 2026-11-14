// SPDX-License-Identifier: Apache-2.0
//! Every committed Spark function matrix loads and describes its own release (P3 PR-02): the matrices of the
//! supported releases are `tests/fixtures/spark-<version>/function-matrix/`, written by
//! `tests/scripts/gen_function_matrix.py` against that release's real server. A matrix that fails to load (a
//! count that disagrees with its manifest, a shard that is missing) or sits under another release's directory
//! fails here, so a hand edit or a copy between versions cannot pass.
#![forbid(unsafe_code)]

use krishiv_testkit::fn_matrix::FnMatrix;

#[test]
fn every_committed_matrix_loads_under_its_own_version() {
    let fixtures = krishiv_testkit::tests_dir().join("fixtures");
    let mut seen = Vec::new();
    for entry in std::fs::read_dir(&fixtures)
        .expect("fixtures dir")
        .flatten()
    {
        let dir = entry.path().join("function-matrix");
        if !dir.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let version = name
            .strip_prefix("spark-")
            .expect("spark-<version>")
            .to_owned();
        let matrix = FnMatrix::load(&dir).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            matrix.spark_version, version,
            "{name} holds another release's matrix"
        );
        assert!(!matrix.cases.is_empty(), "{name}: no cases");
        assert!(!matrix.functions.is_empty(), "{name}: no functions");
        seen.push(version);
    }
    seen.sort();
    // The releases of the 4.2/4.3 profiles are always there; the older ones join as their matrices are generated.
    for must in ["3.5.9", "4.2.0", "4.2.1", "4.3.0"] {
        assert!(
            seen.iter().any(|v| v == must),
            "no matrix for Spark {must}: {seen:?}"
        );
    }
}

#[test]
fn the_matrix_of_4_2_1_is_4_2_0s_plus_the_new_functions() {
    let tests = krishiv_testkit::tests_dir().join("fixtures");
    let v420 = FnMatrix::load(tests.join("spark-4.2.0/function-matrix")).expect("4.2.0");
    let v421 = FnMatrix::load(tests.join("spark-4.2.1/function-matrix")).expect("4.2.1");
    let names = |m: &FnMatrix| -> std::collections::BTreeSet<String> {
        m.functions.iter().map(|f| f.name.clone()).collect()
    };
    let (a, b) = (names(&v420), names(&v421));
    let removed: Vec<_> = a.difference(&b).collect();
    assert!(
        removed.is_empty(),
        "4.2.1 lost functions of 4.2.0: {removed:?}"
    );
}

// SPDX-License-Identifier: Apache-2.0
//! Golden tests: plans built by the real PySpark 4.2 Connect client (protobuf
//! JSON under `tests/fixtures/pyspark-4.2/`, written by
//! `tests/scripts/gen_connect_goldens.py` in the pinned Spark image) are
//! decoded, translated and snapshotted as the IR's pretty-printer shows them,
//! with the `plan_id` each relation carries. A plan outside the supported
//! subset snapshots its error.
//!
//! With `KRISHIV_FUZZ_SEED_DIR` set, the test also writes every fixture, in
//! binary, as a seed for the `connect_decode` fuzz target (the first byte
//! `0` picks `Plan`): `KRISHIV_FUZZ_SEED_DIR=fuzz/corpus/connect_decode cargo
//! test -p krishiv-connect --test goldens`.

use std::fmt::Write as _;
use std::path::PathBuf;

use krishiv_common::error::ErrorClass;
use krishiv_connect::translate::{TranslateConf, plan_from_bytes};
use krishiv_plan::{LogicalPlan, Plan};
use prost_reflect::{DescriptorPool, DynamicMessage};

/// The fixtures that must be rejected, and with which error class. Every
/// other fixture must translate.
const UNSUPPORTED: &[(&str, ErrorClass)] = &[
    ("alias_multi", ErrorClass::KrishivUnsupportedFeature),
    ("aggregate_pivot", ErrorClass::KrishivUnsupportedFeature),
    ("distinct", ErrorClass::KrishivUnsupportedFeature),
    (
        "filter_expression_string",
        ErrorClass::KrishivUnsupportedFeature,
    ),
    ("literals_array", ErrorClass::KrishivUnsupportedFeature),
    ("read_schema_string", ErrorClass::KrishivUnsupportedFeature),
    ("select_expr", ErrorClass::KrishivUnsupportedFeature),
    ("show_string", ErrorClass::KrishivUnsupportedFeature),
    ("to_df", ErrorClass::KrishivUnsupportedFeature),
    ("union", ErrorClass::KrishivUnsupportedFeature),
    (
        "user_defined_function_call",
        ErrorClass::KrishivUnsupportedFeature,
    ),
    ("with_column_renamed", ErrorClass::KrishivUnsupportedFeature),
];

#[cfg(test)]
mod support {
    use super::*;

    pub(crate) fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    /// The protobuf JSON of `json` as the bytes of a `spark.connect.Plan`.
    pub(crate) fn plan_bytes(pool: &DescriptorPool, json: &str) -> Vec<u8> {
        let descriptor = pool
            .get_message_by_name("spark.connect.Plan")
            .expect("the descriptor set has Plan");
        let mut de = serde_json::Deserializer::from_str(json);
        let message =
            DynamicMessage::deserialize(descriptor, &mut de).expect("valid protobuf JSON");
        de.end().expect("no trailing text");
        prost_reflect::prost::Message::encode_to_vec(&message)
    }

    /// Each relation's name and `plan_id`, indented by depth.
    pub(crate) fn origins(plan: &Plan) -> String {
        fn walk(p: &LogicalPlan, depth: usize, out: &mut String) {
            let name = format!("{:?}", p.node);
            let name = name.split(['(', ' ', '{']).next().unwrap_or_default();
            let id = p
                .origin
                .plan_id
                .map_or_else(|| "-".to_owned(), |i| i.to_string());
            writeln!(out, "{:indent$}{name} plan_id={id}", "", indent = depth * 2).unwrap();
            p.for_each_child(&mut |c| walk(c, depth + 1, out));
        }
        let mut out = String::new();
        walk(plan.root(), 0, &mut out);
        out
    }
}

use support::{fixtures, origins, plan_bytes};

#[test]
fn pyspark_plans_translate_to_the_ir() {
    let pool = DescriptorPool::decode(
        std::fs::read(fixtures().join("spark-connect-4.2.fds"))
            .expect("descriptor set")
            .as_slice(),
    )
    .expect("a valid descriptor set");
    let mut names: Vec<_> = std::fs::read_dir(fixtures().join("pyspark-4.2"))
        .expect("fixtures")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    names.sort();
    assert!(names.len() >= 70, "only {} fixtures", names.len());
    let mut translated = 0;
    for path in names {
        let name = path
            .file_stem()
            .and_then(|n| n.to_str())
            .expect("a name")
            .to_owned();
        let json = std::fs::read_to_string(&path).expect("readable");
        let bytes = plan_bytes(&pool, &json);
        if let Some(dir) = std::env::var_os("KRISHIV_FUZZ_SEED_DIR") {
            std::fs::create_dir_all(&dir).expect("seed directory");
            let seed = [&[0_u8][..], &bytes].concat();
            std::fs::write(PathBuf::from(dir).join(format!("{name}.bin")), seed).expect("seed");
        }
        let expected_error = UNSUPPORTED
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| *c);
        let text = match (
            plan_from_bytes(&bytes, &TranslateConf::default()),
            expected_error,
        ) {
            (Ok(plan), None) => {
                translated += 1;
                format!("{plan}-- plan ids --\n{}", origins(&plan))
            }
            (Err(e), Some(class)) => {
                assert_eq!(e.error_class(), class, "{name}: {e}");
                format!("{e}\n")
            }
            (Ok(plan), Some(_)) => panic!("{name} should be unsupported, but translated:\n{plan}"),
            (Err(e), None) => panic!("{name} should translate, but failed: {e}"),
        };
        insta::with_settings!({
            snapshot_path => "snapshots",
            prepend_module_to_snapshot => false,
            omit_expression => true,
        }, {
            insta::assert_snapshot!(name, text);
        });
    }
    assert!(translated >= 55, "only {translated} plans translate");
}

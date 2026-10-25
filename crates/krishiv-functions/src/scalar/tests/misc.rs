// SPDX-License-Identifier: Apache-2.0
//! The misc bindings: the session constants, the file placeholders and the
//! partition functions.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;

use super::{BindArg, BindOptions, bind, bind_scalar, class, strs};

fn strings(f: &Arc<dyn super::ScalarFunction>, rows: usize, partition: u32) -> Vec<String> {
    let out = f.invoke_in(&[], rows, partition).unwrap();
    out.as_string::<i32>()
        .iter()
        .map(|s| s.unwrap().to_owned())
        .collect()
}

#[test]
fn version_reports_the_kernels_release_line() {
    let f = bind("version", &[]);
    assert_eq!(f.return_type(), &DataType::Utf8);
    assert_eq!(strings(&f, 3, 0), vec!["4.2.0"; 3]);
    // The line follows the kernels' Spark version.
    let opts = BindOptions::new().with_kernel(
        krishiv_kernels::KernelCtx::new().with_spark_version(krishiv_kernels::SparkVersion::V4_0),
    );
    let f = super::bind_scalar("version", &[], &opts).unwrap();
    assert_eq!(strings(&f, 1, 0), vec!["4.0.0"]);
    // `typeof` never reaches execution: the analyzer rewrites it first.
    assert_eq!(
        class(
            &super::bind_scalar(
                "typeof",
                &[BindArg::Column(DataType::Int32)],
                &BindOptions::new()
            )
            .unwrap_err()
        ),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn current_database_and_catalog() {
    assert_eq!(
        strings(&bind("current_database", &[]), 2, 0),
        vec!["default"; 2]
    );
    assert_eq!(strings(&bind("current_schema", &[]), 1, 0), vec!["default"]);
    assert_eq!(
        strings(&bind("current_catalog", &[]), 1, 0),
        vec!["spark_catalog"]
    );
}

#[test]
fn input_file_functions_read_no_file() {
    assert_eq!(strings(&bind("input_file_name", &[]), 2, 0), vec![""; 2]);
    for name in ["input_file_block_start", "input_file_block_length"] {
        let f = bind(name, &[]);
        assert_eq!(f.return_type(), &DataType::Int64);
        let out = f.invoke_in(&[], 3, 0).unwrap();
        assert_eq!(
            out.as_primitive::<Int64Type>().values(),
            &[-1, -1, -1],
            "{name}"
        );
    }
}

#[test]
fn spark_partition_id_is_the_partition() {
    let f = bind("spark_partition_id", &[]);
    assert_eq!(f.return_type(), &DataType::Int32);
    for partition in [0, 1, 7] {
        let out = f.invoke_in(&[], 3, partition).unwrap();
        assert_eq!(
            out.as_primitive::<Int32Type>().values(),
            &[partition as i32; 3],
            "partition {partition}"
        );
    }
    // Without a partition the single partition reads 0.
    let out = f.invoke(&[], 2).unwrap();
    assert_eq!(out.as_primitive::<Int32Type>().values(), &[0, 0]);
}

#[test]
fn monotonically_increasing_id_counts_per_partition() {
    let f = bind("monotonically_increasing_id", &[]);
    assert_eq!(f.return_type(), &DataType::Int64);
    let ids = |rows: usize, partition: u32| {
        f.invoke_in(&[], rows, partition)
            .unwrap()
            .as_primitive::<Int64Type>()
            .values()
            .to_vec()
    };
    assert_eq!(ids(3, 0), vec![0, 1, 2]);
    // The next batch continues the partition's count.
    assert_eq!(ids(2, 0), vec![3, 4]);
    // Another partition counts from its own base.
    let base = 2_i64 << 33;
    assert_eq!(ids(2, 2), vec![base, base + 1]);
    assert_eq!(ids(1, 0), vec![5]);
}

#[test]
fn misc_functions_take_no_arguments() {
    for name in [
        "version",
        "current_database",
        "current_schema",
        "current_catalog",
        "input_file_name",
        "input_file_block_start",
        "input_file_block_length",
        "spark_partition_id",
        "monotonically_increasing_id",
    ] {
        assert_eq!(bind(name, &[]).name(), name);
        let arg = strs(&[Some("x")]);
        let err = bind_scalar(
            name,
            &[BindArg::Column(arg.data_type().clone())],
            &BindOptions::new(),
        )
        .unwrap_err();
        assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature, "{name}");
        // The error names the bad signature, not a missing function.
        assert!(err.to_string().contains(name), "{name}: {err}");
    }
}

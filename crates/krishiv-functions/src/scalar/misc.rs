// SPDX-License-Identifier: Apache-2.0
//! Misc scalar functions: `version`, `current_database` (`current_schema`),
//! `current_catalog`, `input_file_name` (and the block functions),
//! `monotonically_increasing_id` and `spark_partition_id`.
//!
//! Ground truth: Spark 4.2's `misc.scala` (`SparkVersion`, `CurrentDatabase`,
//! `CurrentCatalog`, `InputFileName`, ...), `MonotonicallyIncreasingID.scala`
//! and `SparkPartitionID.scala`, plus the `current_database_catalog.sql`
//! golden.
//!
//! `version` reports the kernels' Spark line (`4.2.0` for the default
//! build). The database and catalog are the engine's single namespace:
//! `default` in `spark_catalog` (see `krishiv-session`'s catalog), so they
//! bind as constants, with no session state. `input_file_name` and the
//! block functions read the file being scanned; the evaluator has no file
//! context yet, so they return what Spark returns with no file being read
//! (the empty string, `-1`).
//!
//! No golden pins the `version`, `input_file_*` or `uuid` values (no such
//! value queries in any `.out` file: `string-functions.sql` only pins
//! `length(uuid())` and that distinct calls differ), so those stand on the
//! documented behavior, unverified against a golden run.
//!
//! `monotonically_increasing_id` is `(partition << 33) + row_in_partition`
//! and `spark_partition_id` is the partition index, both from the
//! executing partition (see [`super::ScalarFunction::invoke_in`]); each
//! bound function counts its partition's rows in the partitioned body.

use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array, Int64Array, StringArray};
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::SparkVersion;

use super::args::{Inputs, internal, unsupported};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds. `typeof` is resolved by the analyzer
/// (a literal rewrite, like Spark's `RuntimeReplaceable`); its binding is
/// unreachable and reports that.
pub(super) const NAMES: &[&str] = &[
    "version",
    "current_database",
    "current_schema",
    "current_catalog",
    "current_path",
    "input_file_name",
    "input_file_block_start",
    "input_file_block_length",
    "monotonically_increasing_id",
    "spark_partition_id",
    "typeof",
];

/// The engine's single namespace and catalog (the only values
/// `current_database` / `current_catalog` can report).
const DATABASE: &str = "default";
const CATALOG: &str = "spark_catalog";

/// The release `version` reports for `kernel`.
fn release(kernel: SparkVersion) -> Result<&'static str, KrishivError> {
    match kernel {
        SparkVersion::V4_0 => Ok("4.0.0"),
        SparkVersion::V4_1 => Ok("4.1.0"),
        SparkVersion::V4_2 => Ok("4.2.0"),
        _ => Err(internal(
            "a Spark version with no release string".to_owned(),
        )),
    }
}

/// `version()`: the kernels' release line.
fn version(opts: &BindOptions) -> Result<Binding, KrishivError> {
    let version = Arc::<str>::from(release(opts.kernel.spark_version())?);
    Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        Ok(Arc::new(StringArray::from(vec![&*version; x.rows()])) as ArrayRef)
    }))
}

/// `current_database()` / `current_schema()` / `current_catalog()`.
fn catalog(name: &'static str) -> Result<Binding, KrishivError> {
    let value = Arc::<str>::from(match name {
        "current_catalog" => CATALOG,
        _ => DATABASE,
    });
    Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        Ok(Arc::new(StringArray::from(vec![&*value; x.rows()])) as ArrayRef)
    }))
}

/// `input_file_name()`: the empty string (no file is being read).
fn input_file_name() -> Binding {
    Binding::new(DataType::Utf8, |x: &Inputs<'_>| {
        Ok(Arc::new(StringArray::from(vec![""; x.rows()])) as ArrayRef)
    })
}

/// `input_file_block_start()` / `input_file_block_length()`: -1 (no file
/// is being read).
fn input_file_block() -> Binding {
    Binding::new(DataType::Int64, move |x: &Inputs<'_>| {
        Ok(Arc::new(Int64Array::from(vec![-1; x.rows()])) as ArrayRef)
    })
}

/// `spark_partition_id()`: the executing partition's index.
fn spark_partition_id() -> Binding {
    Binding::partitioned(DataType::Int32, |x: &Inputs<'_>, partition| {
        // Driver counts never approach `2^31` (one slot per worker thread).
        debug_assert!(i32::try_from(partition).is_ok());
        Ok(Arc::new(Int32Array::from(vec![partition as i32; x.rows()])) as ArrayRef)
    })
}

/// `monotonically_increasing_id()`: `(partition << 33) + row_in_partition`,
/// counted per partition in the partitioned body. The per-partition counts
/// are untracked by `krishiv-memory`, but bounded by the partition count
/// (bytes), like the other per-partition state.
fn monotonically_increasing_id() -> Binding {
    let counts = Mutex::new(Vec::<u64>::new());
    Binding::partitioned(DataType::Int64, move |x: &Inputs<'_>, partition| {
        let mut counts = counts
            .lock()
            .map_err(|_| internal("a row counter is poisoned".to_owned()))?;
        let index = partition as usize;
        if counts.len() <= index {
            counts.resize(index + 1, 0);
        }
        let first = counts[index];
        // Java's `long` arithmetic wraps; partitions past `2^31` wrap the
        // same way instead of panicking.
        let base = (u64::from(partition) << 33).wrapping_add(first);
        let rows = x.rows() as u64;
        counts[index] = first.wrapping_add(rows);
        Ok(Arc::new(Int64Array::from(
            (0..rows)
                .map(|i| base.wrapping_add(i) as i64)
                .collect::<Vec<_>>(),
        )) as ArrayRef)
    })
}

/// Binds the misc function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    if !args.is_empty() {
        return Some((listed, Err(bad_types(listed, args))));
    }
    // `typeof` never reaches execution: the analyzer rewrites it to a
    // string literal, so a call that arrives here is an analyzer bug.
    if listed == "typeof" || listed == "current_path" {
        return Some((
            listed,
            Err(unsupported(
                "typeof without the analyzer's literal rewrite".to_owned(),
            )),
        ));
    }
    let binding = match listed {
        "version" => version(opts),
        "current_database" | "current_schema" | "current_catalog" => catalog(listed),
        "input_file_name" => Ok(input_file_name()),
        "input_file_block_start" | "input_file_block_length" => Ok(input_file_block()),
        "spark_partition_id" => Ok(spark_partition_id()),
        "monotonically_increasing_id" => Ok(monotonically_increasing_id()),
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, binding))
}

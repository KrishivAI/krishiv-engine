// SPDX-License-Identifier: Apache-2.0
//! The embedded Python relation builders on a frame (`map_in_python`, `apply_in_python`,
//! `cogroup_apply_in_python`): the plan is built for a UDF of the right eval type and struct return type,
//! and refused otherwise. Running the UDF itself needs the Python runtime and is covered by the wheel's
//! `test_python_relation_udfs.py`.
#![forbid(unsafe_code)]

use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::PythonEvalType;
use krishiv_session::{DataFrame, ExplainMode, PythonUdfDef, Session};
use krishiv_types::{DataType, StructField, StructType};

fn is_class<T>(r: Result<T, KrishivError>, class: ErrorClass) -> bool {
    r.is_err_and(|e| e.error_class() == class)
}

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn udf(eval_type: PythonEvalType, return_type: DataType) -> PythonUdfDef {
    PythonUdfDef {
        name: "F".to_owned(),
        eval_type,
        return_type,
        deterministic: true,
        thread_safe: true,
    }
}

/// The declared output of a relation frame: `structure()`'s columns and types, nothing of its input.
fn assert_declared_output(df: &DataFrame) -> R {
    let schema = block_on(df.schema())?;
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| (f.name().as_str(), f.data_type().clone()))
        .collect();
    assert_eq!(
        fields,
        [
            ("id", arrow_schema::DataType::Int64),
            ("v", arrow_schema::DataType::Utf8View)
        ]
    );
    Ok(())
}

/// The explained plan names the relation node it was built as.
fn assert_plan_has(df: &DataFrame, node: &str) -> R {
    let text = block_on(df.explain(ExplainMode::Extended))?;
    assert!(text.contains(node), "{node} not in:\n{text}");
    Ok(())
}

fn structure() -> DataType {
    DataType::Struct(StructType::new([
        StructField::new("id", DataType::Long, true),
        StructField::new("v", DataType::String, true),
    ]))
}

#[test]
fn map_in_python_takes_a_map_kind_with_a_struct_result() -> R {
    let df = Session::new()?.range(0, 3, 1, None);
    let pandas = df.map_in_python(&udf(PythonEvalType::MapPandasIter, structure()), 1, false)?;
    assert_declared_output(&pandas)?;
    assert_plan_has(&pandas, "MapPartitions")?;
    let arrow = df.map_in_python(&udf(PythonEvalType::MapArrowIter, structure()), 1, false)?;
    assert_declared_output(&arrow)?;
    assert_plan_has(&arrow, "MapPartitions")?;
    // Barrier mode builds, and analysis refuses it by name.
    let barrier = df.map_in_python(&udf(PythonEvalType::MapArrowIter, structure()), 1, true)?;
    assert!(is_class(
        block_on(barrier.explain(ExplainMode::Extended)),
        ErrorClass::KrishivUnsupportedFeature
    ));
    Ok(())
}

#[test]
fn a_relation_refuses_the_wrong_eval_type_or_a_non_struct_result() -> R {
    let df = Session::new()?.range(0, 3, 1, None);
    let wrong_kind = df.map_in_python(
        &udf(PythonEvalType::GroupedMapPandas, structure()),
        1,
        false,
    );
    assert!(is_class(
        wrong_kind,
        ErrorClass::KrishivInvalidPythonRelation
    ));
    let not_struct = df.map_in_python(
        &udf(PythonEvalType::MapPandasIter, DataType::Long),
        1,
        false,
    );
    assert!(is_class(
        not_struct,
        ErrorClass::KrishivInvalidPythonRelation
    ));
    Ok(())
}

#[test]
fn apply_in_python_takes_grouped_map_kinds_but_not_rollup_or_cube() -> R {
    let df = Session::new()?.range(0, 3, 1, None);
    let key = df.col("id");
    for kind in [
        PythonEvalType::GroupedMapPandas,
        PythonEvalType::GroupedMapArrow,
        PythonEvalType::GroupedMapPandasIter,
        PythonEvalType::GroupedMapArrowIter,
    ] {
        let out = df
            .group_by([key.clone()])
            .apply_in_python(&udf(kind, structure()), 1)?;
        assert_declared_output(&out)?;
        assert_plan_has(&out, "GroupMap")?;
    }
    let wrong = df
        .group_by([key.clone()])
        .apply_in_python(&udf(PythonEvalType::MapPandasIter, structure()), 1);
    assert!(is_class(wrong, ErrorClass::KrishivInvalidPythonRelation));
    let def = udf(PythonEvalType::GroupedMapPandas, structure());
    assert!(is_class(
        df.rollup([key.clone()]).apply_in_python(&def, 1),
        ErrorClass::KrishivUnsupportedFeature
    ));
    assert!(is_class(
        df.cube([key]).apply_in_python(&def, 1),
        ErrorClass::KrishivUnsupportedFeature
    ));
    Ok(())
}

#[test]
fn cogroup_needs_matching_key_counts_and_a_cogrouped_kind() -> R {
    let s = Session::new()?;
    let (left, right) = (s.range(0, 3, 1, None), s.range(1, 4, 1, None));
    let def = udf(PythonEvalType::CogroupedMapPandas, structure());
    let pandas =
        left.cogroup_apply_in_python(&right, [left.col("id")], [right.col("id")], &def, 1)?;
    assert_declared_output(&pandas)?;
    assert_plan_has(&pandas, "CoGroupMap")?;
    let arrow = left.cogroup_apply_in_python(
        &right,
        [left.col("id")],
        [right.col("id")],
        &udf(PythonEvalType::CogroupedMapArrow, structure()),
        1,
    )?;
    assert_declared_output(&arrow)?;
    assert_plan_has(&arrow, "CoGroupMap")?;
    let uneven = left.cogroup_apply_in_python(
        &right,
        [left.col("id")],
        Vec::<krishiv_session::Column>::new(),
        &def,
        1,
    );
    assert!(is_class(uneven, ErrorClass::KrishivInvalidPythonRelation));
    let wrong = left.cogroup_apply_in_python(
        &right,
        [left.col("id")],
        [right.col("id")],
        &udf(PythonEvalType::MapPandasIter, structure()),
        1,
    );
    assert!(is_class(wrong, ErrorClass::KrishivInvalidPythonRelation));
    Ok(())
}

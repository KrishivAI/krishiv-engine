// SPDX-License-Identifier: Apache-2.0
//! The embedded Python relation builders on a frame (`map_in_python`, `apply_in_python`,
//! `cogroup_apply_in_python`): the plan is built for a UDF of the right eval type and struct return type,
//! and refused otherwise. Running the UDF itself needs the Python runtime and is covered by the wheel's
//! `test_python_relation_udfs.py`.
#![forbid(unsafe_code)]

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::PythonEvalType;
use krishiv_session::{PythonUdfDef, Session};
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

fn structure() -> DataType {
    DataType::Struct(StructType::new([
        StructField::new("id", DataType::Long, true),
        StructField::new("v", DataType::String, true),
    ]))
}

#[test]
fn map_in_python_takes_a_map_kind_with_a_struct_result() -> R {
    let df = Session::new()?.range(0, 3, 1, None);
    df.map_in_python(&udf(PythonEvalType::MapPandasIter, structure()), 1, false)?;
    df.map_in_python(&udf(PythonEvalType::MapArrowIter, structure()), 1, true)?;
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
    assert!(is_class(wrong_kind, ErrorClass::InternalError));
    let not_struct = df.map_in_python(
        &udf(PythonEvalType::MapPandasIter, DataType::Long),
        1,
        false,
    );
    assert!(is_class(not_struct, ErrorClass::InternalError));
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
        df.group_by([key.clone()])
            .apply_in_python(&udf(kind, structure()), 1)?;
    }
    let wrong = df
        .group_by([key.clone()])
        .apply_in_python(&udf(PythonEvalType::MapPandasIter, structure()), 1);
    assert!(is_class(wrong, ErrorClass::InternalError));
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
    left.cogroup_apply_in_python(&right, [left.col("id")], [right.col("id")], &def, 1)?;
    left.cogroup_apply_in_python(
        &right,
        [left.col("id")],
        [right.col("id")],
        &udf(PythonEvalType::CogroupedMapArrow, structure()),
        1,
    )?;
    let uneven = left.cogroup_apply_in_python(
        &right,
        [left.col("id")],
        Vec::<krishiv_session::Column>::new(),
        &def,
        1,
    );
    assert!(is_class(uneven, ErrorClass::InternalError));
    let wrong = left.cogroup_apply_in_python(
        &right,
        [left.col("id")],
        [right.col("id")],
        &udf(PythonEvalType::MapPandasIter, structure()),
        1,
    );
    assert!(is_class(wrong, ErrorClass::InternalError));
    Ok(())
}

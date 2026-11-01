// SPDX-License-Identifier: Apache-2.0
//! The array set and order signatures: `array_contains`, `array_distinct`,
//! `array_union`, `array_except`, `array_intersect`, `arrays_overlap`,
//! `sort_array` and `arrays_zip` (Spark 4.2's `ArrayContains`,
//! `ArrayDistinct`, `ArrayUnion`, `ArrayExcept`, `ArrayIntersect`,
//! `ArraysOverlap`, `SortArray` and `ArraysZip`). Checked against the
//! `array.sql` goldens and a Spark 4.2.0 run.
//!
//! The element type of every function but `arrays_zip` must order
//! (`DATATYPE_MISMATCH.INVALID_ORDERING_TYPE`: a map does not). An untyped
//! `NULL` is `NULL_TYPE` for `array_contains` and
//! `BINARY_ARRAY_DIFF_TYPES` for the two-array functions; `array_distinct`,
//! `sort_array` and `arrays_zip` cast it to an array of `NULL`s.
//!
//! `sort_array`'s order flag must fold (`NON_FOLDABLE_INPUT`); the analyzer
//! fills its default `true`, so the rule always sees two arguments.
//! `arrays_zip` names its fields `0`, `1`, ... here; the analyzer renames
//! the ones that are columns or fields and passes the names on as
//! trailing literals.

use krishiv_common::error::{
    DatatypeMismatchArrayFunctionDiffTypes, DatatypeMismatchBinaryArrayDiffTypes,
    DatatypeMismatchInvalidOrderingType, DatatypeMismatchNonFoldableInput,
    DatatypeMismatchNullType, DatatypeMismatchUnexpectedInputType, KrishivError,
};
use krishiv_types::{ArrayType, DataType, StructField, StructType};

use super::defs::{Spec, scalar, spark_orderable};
use super::hof::{call_sql, ordinal, quoted};
use crate::signature::{ArgType, CallTypes, Constant, InputBuilder, InputType, Inputs};

/// The `UNEXPECTED_INPUT_TYPE` error of argument `index`.
fn unexpected(name: &str, c: &CallTypes<'_>, index: usize) -> KrishivError {
    DatatypeMismatchUnexpectedInputType {
        sql_expr: call_sql(name, c),
        param_index: ordinal(index).to_owned(),
        required_type: "\"ARRAY\"".to_owned(),
        input_sql: format!("\"{}\"", c.sql(index).unwrap_or_default()),
        input_type: quoted(c.data_type(index)),
    }
    .into()
}

/// `INVALID_ORDERING_TYPE` for `data_type`.
fn not_orderable(name: &str, c: &CallTypes<'_>, data_type: &DataType) -> KrishivError {
    DatatypeMismatchInvalidOrderingType {
        sql_expr: call_sql(name, c),
        function_name: format!("`{name}`"),
        data_type: quoted(Some(data_type)),
    }
    .into()
}

/// The array argument `index`: its element type and `containsNull`, or the
/// `UNEXPECTED_INPUT_TYPE` error.
fn array_at<'a>(
    name: &str,
    c: &CallTypes<'a>,
    index: usize,
) -> Result<&'a ArrayType, KrishivError> {
    match c.data_type(index) {
        Some(DataType::Array(a)) => Ok(a),
        _ => Err(unexpected(name, c, index)),
    }
}

/// `array_contains(array, value)`: BOOLEAN, NULL for a NULL array or value,
/// and NULL for a miss when the array holds a NULL.
fn contains(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let name = "array_contains";
    if c.data_type(0).is_none() || c.data_type(1).is_none() {
        return Err(DatatypeMismatchNullType {
            sql_expr: call_sql(name, c),
            function_name: format!("`{name}`"),
        }
        .into());
    }
    let array = array_at(name, c, 0)?;
    let same = c
        .data_type(1)
        .is_some_and(|v| array.element_type().equals_ignore_nullability(v));
    if !same {
        return Err(DatatypeMismatchArrayFunctionDiffTypes {
            sql_expr: call_sql(name, c),
            function_name: format!("`{name}`"),
            data_type: "\"ARRAY\"".to_owned(),
            left_type: quoted(c.data_type(0)),
            right_type: quoted(c.data_type(1)),
        }
        .into());
    }
    if !spark_orderable(array.element_type()) {
        return Err(not_orderable(name, c, array.element_type()));
    }
    Ok(ArgType::new(
        DataType::Boolean,
        c.any_nullable() || array.contains_null(),
    ))
}

/// `array_distinct(array)`: the array without repeats, of the same type.
fn distinct(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let array = array_at("array_distinct", c, 0)?;
    if !spark_orderable(array.element_type()) {
        return Err(not_orderable("array_distinct", c, array.element_type()));
    }
    Ok(ArgType::new(
        DataType::Array(array.clone().into()),
        c.args()[0].nullable,
    ))
}

/// The result of a two-array function of this family: an array over the
/// shared element type (a BOOLEAN when `boolean`), where `holds_null` says
/// from the arguments' `containsNull` whether NULL can appear (for a
/// BOOLEAN: whether the result can be NULL).
fn pair(
    name: &str,
    c: &CallTypes<'_>,
    holds_null: fn(bool, bool) -> bool,
    boolean: bool,
) -> Result<ArgType, KrishivError> {
    let (Some(DataType::Array(left)), Some(DataType::Array(right))) =
        (c.data_type(0), c.data_type(1))
    else {
        return Err(different(name, c));
    };
    if !left
        .element_type()
        .equals_ignore_nullability(right.element_type())
    {
        return Err(different(name, c));
    }
    if !spark_orderable(left.element_type()) {
        return Err(not_orderable(name, c, left.element_type()));
    }
    let holds = holds_null(left.contains_null(), right.contains_null());
    if boolean {
        return Ok(ArgType::new(DataType::Boolean, c.any_nullable() || holds));
    }
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(left.element_type().clone(), holds).into()),
        c.any_nullable(),
    ))
}

/// `array_union(a, b)`: the distinct elements of both, as an array of the
/// shared element type that holds NULL when either did.
fn union(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pair("array_union", c, |l, r| l || r, false)
}

/// `array_except(a, b)`: the distinct elements of `a` not in `b`, typed as
/// `a`.
fn except(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pair("array_except", c, |l, _| l, false)
}

/// `array_intersect(a, b)`: the distinct elements of `a` that are in `b`,
/// holding NULL only when both could.
fn intersect(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pair("array_intersect", c, |l, r| l && r, false)
}

/// `arrays_overlap(a, b)`: whether they share an element; NULL when they do
/// not but a NULL element could have matched.
fn overlap(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    pair("arrays_overlap", c, |l, r| l || r, true)
}

/// `BINARY_ARRAY_DIFF_TYPES`: the arguments are not two arrays of one
/// element type.
fn different(name: &str, c: &CallTypes<'_>) -> KrishivError {
    DatatypeMismatchBinaryArrayDiffTypes {
        sql_expr: call_sql(name, c),
        function_name: format!("`{name}`"),
        array_type: "\"ARRAY\"".to_owned(),
        left_type: quoted(c.data_type(0)),
        right_type: quoted(c.data_type(1)),
    }
    .into()
}

/// `sort_array(array, ascendingOrder)`: the array sorted, NULL elements
/// first (last when descending).
fn sort(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let name = "sort_array";
    let array = array_at(name, c, 0)?;
    if !matches!(c.constant(1), Some(Constant::Boolean(_) | Constant::Null)) {
        return Err(DatatypeMismatchNonFoldableInput {
            sql_expr: call_sql(name, c),
            input_name: "`ascendingOrder`".to_owned(),
            input_type: "\"BOOLEAN\"".to_owned(),
            input_expr: format!("\"{}\"", c.sql(1).unwrap_or_default()),
        }
        .into());
    }
    if !spark_orderable(array.element_type()) {
        return Err(not_orderable(
            name,
            c,
            c.data_type(0).unwrap_or(&DataType::Null),
        ));
    }
    Ok(ArgType::new(
        DataType::Array(array.clone().into()),
        c.any_nullable(),
    ))
}

/// `arrays_zip(a, b, ...)`: an array of structs holding the `i`th element of
/// each array (NULL where an array is shorter), as long as the longest.
fn zip(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let mut fields = Vec::with_capacity(c.args().len());
    for i in 0..c.args().len() {
        let array = array_at("arrays_zip", c, i)?;
        fields.push(StructField::new(
            i.to_string(),
            array.element_type().clone(),
            true,
        ));
    }
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(DataType::Struct(StructType::new(fields)), false).into()),
        c.any_nullable(),
    ))
}

/// Every argument of `arrays_zip` is an array.
fn zip_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![InputType::Array; c.args().len()]
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(&["array_contains"], &Inputs::ArrayContains, contains);
    out.extend(scalar(
        &["array_distinct"],
        &Inputs::Expects(vec![InputType::Array]),
        distinct,
    ));
    out.extend(scalar(&["array_union"], &Inputs::ArrayPair, union));
    out.extend(scalar(&["array_except"], &Inputs::ArrayPair, except));
    out.extend(scalar(&["array_intersect"], &Inputs::ArrayPair, intersect));
    out.extend(scalar(&["arrays_overlap"], &Inputs::ArrayPair, overlap));
    out.extend(scalar(
        &["sort_array"],
        &Inputs::Expects(vec![InputType::Array, InputType::Strict(DataType::Boolean)]),
        sort,
    ));
    out.extend(scalar(
        &["arrays_zip"],
        &Inputs::Built(InputBuilder(zip_inputs)),
        zip,
    ));
    out
}

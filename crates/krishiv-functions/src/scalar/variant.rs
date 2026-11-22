// SPDX-License-Identifier: Apache-2.0
//! The variant functions: `parse_json`, `try_parse_json`,
//! `is_variant_null`, `variant_get` and `try_variant_get` (Spark 4.2's
//! `ParseJson`, `IsVariantNull`, `VariantGet`).
//!
//! `parse_json` builds Spark's exact bytes ([`VariantBuilder::from_json`]);
//! `try_parse_json` is NULL where it raises. `variant_get` reads the path
//! with Spark's `VariantPathParser` (a literal path once per bind, where a
//! bad one is `INVALID_VARIANT_GET_PATH`; a column per row, raising on the
//! first non-NULL row with a bad path), walks the encoded value by offsets, and casts what it finds with
//! the kernels' variant cast (`INVALID_VARIANT_CAST` for `variant_get`,
//! NULL for `try_variant_get`). Without a type, the result is a `VARIANT`
//! re-encoded with only the keys it uses, as Spark's `VariantBuilder` does.
//! `schema_of_variant` prints the kernels' [`Schema`]; `to_variant_object`
//! is the kernels' `castToVariant` with structs and maps allowed.

use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, StringViewBuilder};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::variant::{
    Edit, PathSegment, Schema, VariantColumn, VariantColumnBuilder, apply, cast_values, edit_path,
    extract, named_in, parse_path_or_raise, strip_nulls,
};
use krishiv_types::DataType as SparkType;
use krishiv_types::variant::{VariantBuilder, VariantRef};

use super::args::{Inputs, is_string, to_utf8};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "is_valid_variant",
    "is_variant_null",
    "parse_json",
    "schema_of_variant",
    "to_variant_object",
    "try_parse_json",
    "try_variant_array_append",
    "try_variant_get",
    "try_variant_insert",
    "try_variant_set",
    "variant_array_append",
    "variant_delete",
    "variant_get",
    "variant_insert",
    "variant_set",
    "variant_strip_nulls",
];

fn is_variant(t: &DataType) -> bool {
    krishiv_types::arrow::is_variant_struct(t)
}

fn variant_or_null(t: &DataType) -> bool {
    is_variant(t) || *t == DataType::Null
}

/// The string literal at argument `i`, `None` for a column or NULL.
fn string_literal(args: &[BindArg], i: usize) -> Option<Option<String>> {
    match args.get(i)? {
        BindArg::Literal(a) => {
            let a = to_utf8(a.as_ref()).ok()?;
            Some(a.is_valid(0).then(|| a.value(0).to_owned()))
        }
        BindArg::Column { .. } => None,
    }
}

/// `parse_json` / `try_parse_json` of every row of argument 0.
fn parse_json(x: &Inputs<'_>, fail: bool) -> Result<ArrayRef, KrishivError> {
    let text = x.strings(0)?;
    let mut out = VariantColumnBuilder::with_capacity(text.len());
    for t in &text {
        match t.map(VariantBuilder::from_json) {
            Some(Ok((m, v))) => out.append(&m, &v),
            Some(Err(e)) if fail => return Err(e),
            Some(Err(_)) | None => out.append_null(),
        }
    }
    out.finish()
}

fn schema_of_variant(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let column = x.column(0)?;
    let col = VariantColumn::new(column.as_ref())?;
    let mut out = StringViewBuilder::with_capacity(col.len());
    for i in 0..col.len() {
        match col.get(i)? {
            Some(v) => out.append_value(Schema::of(&v)?.sql()),
            None => out.append_null(),
        }
    }
    Ok(Arc::new(out.finish()))
}

/// The length of the value that starts `value`, from its header byte alone,
/// when that is fixed or declared up front (primitives and strings; `None`
/// for objects, arrays and anything unknown).
fn leading_value_len(value: &[u8]) -> Option<usize> {
    let header = *value.first()?;
    let declared = |at: usize| -> Option<usize> {
        let n = u32::from_le_bytes(value.get(at..at.checked_add(4)?)?.try_into().ok()?);
        usize::try_from(n).ok()?.checked_add(at)?.checked_add(4)
    };
    match header & 3 {
        0 => match header >> 2 {
            0..=2 => Some(1),
            3 => Some(2),
            4 => Some(3),
            5 | 11 | 14 => Some(5),
            6 | 7 | 12 | 13 | 17..=19 => Some(9),
            8 => Some(6),
            9 => Some(10),
            10 => Some(18),
            15 | 16 => declared(1),
            20 => Some(17),
            _ => None,
        },
        1 => Some(1 + usize::from(header >> 2)),
        _ => None,
    }
}

/// `is_valid_variant`: whether the bytes are a well-formed variant. Bytes
/// after a complete primitive value do not make it invalid (Spark checks
/// the value that starts the buffer); malformed bytes are `false`, never an
/// error, and SQL NULL is NULL.
fn is_valid_variant(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let column = x.column(0)?;
    let col = VariantColumn::new(column.as_ref())?;
    let mut out = BooleanBuilder::with_capacity(col.len());
    for i in 0..col.len() {
        match col.raw(i) {
            None => out.append_null(),
            Some((metadata, value)) => {
                let whole = krishiv_types::variant::VariantRef::new(metadata, value).is_ok();
                let prefix = || {
                    leading_value_len(value)
                        .filter(|n| *n < value.len())
                        .and_then(|n| value.get(..n))
                        .is_some_and(|v| {
                            krishiv_types::variant::VariantRef::new(metadata, v).is_ok()
                        })
                };
                out.append_value(whole || prefix());
            }
        }
    }
    Ok(Arc::new(out.finish()))
}

fn is_variant_null(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let column = x.column(0)?;
    let col = VariantColumn::new(column.as_ref())?;
    let mut out = BooleanBuilder::with_capacity(col.len());
    for i in 0..col.len() {
        // SQL NULL is not a variant null; the first value byte decides.
        out.append_value(match col.get(i)? {
            Some(v) => v.value_bytes().first() == Some(&0),
            None => false,
        });
    }
    Ok(Arc::new(out.finish()))
}

/// The path of a `variant_get` call: parsed once for a literal.
enum Path {
    /// A literal path (`None`: NULL, so every row is NULL).
    Literal(Option<Vec<PathSegment>>),
    /// A path per row.
    Column,
}

fn variant_get(
    x: &Inputs<'_>,
    path: &Path,
    to: &SparkType,
    fail: bool,
    name: &'static str,
    opts: &BindOptions,
) -> Result<ArrayRef, KrishivError> {
    let column = x.column(0)?;
    let col = VariantColumn::new(column.as_ref())?;
    let per_row = match path {
        Path::Column => Some(x.strings(1)?),
        Path::Literal(_) => None,
    };
    let mut found = Vec::with_capacity(col.len());
    for i in 0..col.len() {
        let Some(v) = col.get(i)? else {
            found.push(None);
            continue;
        };
        let owned;
        let steps: &[PathSegment] = match (path, &per_row) {
            (Path::Literal(None), _) => {
                found.push(None);
                continue;
            }
            (Path::Literal(Some(p)), _) => p,
            (Path::Column, Some(paths)) => {
                if paths.is_null(i) {
                    found.push(None);
                    continue;
                }
                owned = parse_path_or_raise(paths.value(i), name)?;
                &owned
            }
            (Path::Column, None) => return Err(super::args::internal("no path column".to_owned())),
        };
        found.push(extract(v, steps)?);
    }
    let zone = krishiv_kernels::datetime::Clocked {
        inner: opts.time_zone.as_ref(),
        query_start: opts.query_start_micros,
    };
    cast_values(&found, to, fail, &zone)
}

/// A `Decimal64` column as the same decimals on `Decimal128`; other columns as they are.
fn widen_decimal64(a: ArrayRef) -> Result<ArrayRef, KrishivError> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Decimal64Type;
    use arrow_array::{Array, Decimal128Array};
    let DataType::Decimal64(p, s) = a.data_type().clone() else {
        return Ok(a);
    };
    let wide: Decimal128Array = a
        .as_primitive::<Decimal64Type>()
        .iter()
        .map(|v| v.map(i128::from))
        .collect::<Decimal128Array>()
        .with_precision_and_scale(p, s)
        .map_err(|e| super::args::internal(format!("a widened decimal: {e}")))?;
    Ok(Arc::new(wide))
}

/// How an edit function writes its value argument.
#[derive(Clone, Copy)]
enum EditKind {
    Insert,
    Set(bool),
    Append,
}

/// The variant values of argument `i` of an edit call: a variant column as it is, any other type cast to `VARIANT`
/// (the analyzer checked it can be), an untyped NULL as all NULL.
fn variant_arg(x: &Inputs<'_>, i: usize) -> Result<Option<ArrayRef>, KrishivError> {
    let column = x.column(i)?;
    if *column.data_type() == DataType::Null {
        return Ok(None);
    }
    if is_variant(column.data_type()) {
        return Ok(Some(column));
    }
    let from = SparkType::from_arrow(column.data_type())?;
    krishiv_kernels::variant::cast_to_variant(column.as_ref(), &from).map(Some)
}

/// `variant_insert`, `variant_set`, `variant_array_append` and their `try_` forms over every row.
fn edit_rows(
    x: &Inputs<'_>,
    name: &'static str,
    kind: EditKind,
    try_: bool,
) -> Result<ArrayRef, KrishivError> {
    let column = x.column(0)?;
    if *column.data_type() == DataType::Null {
        return null_variants(x);
    }
    let col = VariantColumn::new(column.as_ref())?;
    let paths = if *x.column(1)?.data_type() == DataType::Null {
        None
    } else {
        Some(x.strings(1)?)
    };
    let value = variant_arg(x, 2)?;
    let values = value
        .as_ref()
        .map(|v| VariantColumn::new(v.as_ref()))
        .transpose()?;
    let mut out = VariantColumnBuilder::with_capacity(col.len());
    for i in 0..col.len() {
        let (Some(root), Some(paths), Some(values)) = (col.get(i)?, &paths, &values) else {
            out.append_null();
            continue;
        };
        let (Some(val), false) = (values.get(i)?, paths.is_null(i)) else {
            out.append_null();
            continue;
        };
        let text = paths.value(i);
        let steps = edit_path(text, name, matches!(kind, EditKind::Append))?;
        let edit = match kind {
            EditKind::Insert => Edit::Insert(val),
            EditKind::Set(create) => Edit::Set(val, create),
            EditKind::Append => Edit::Append(val),
        };
        match apply(root, &steps, edit).map_err(|e| named_in(e, name))? {
            Ok(Some((m, v))) => out.append(&m, &v),
            Ok(None) => match col.raw(i) {
                Some((m, v)) => out.append(m, v),
                None => out.append_null(),
            },
            Err(_) if try_ => out.append_null(),
            Err(failure) => return Err(failure.into_error(name, text)),
        }
    }
    out.finish()
}

/// `variant_delete(v, path1, ...)`: the paths apply left to right, a NULL path is skipped.
fn delete_rows(x: &Inputs<'_>, count: usize) -> Result<ArrayRef, KrishivError> {
    let column = x.column(0)?;
    if *column.data_type() == DataType::Null {
        return null_variants(x);
    }
    let col = VariantColumn::new(column.as_ref())?;
    let mut paths = Vec::with_capacity(count);
    for j in 1..count {
        paths.push(if *x.column(j)?.data_type() == DataType::Null {
            None
        } else {
            Some(x.strings(j)?)
        });
    }
    let mut out = VariantColumnBuilder::with_capacity(col.len());
    for i in 0..col.len() {
        let Some(root) = col.get(i)? else {
            out.append_null();
            continue;
        };
        let mut current: Option<(Vec<u8>, Vec<u8>)> = None;
        for path in paths.iter().flatten().filter(|p| !p.is_null(i)) {
            let steps = edit_path(path.value(i), "variant_delete", false)?;
            let at = match &current {
                Some((m, v)) => VariantRef::new(m, v)?,
                None => root,
            };
            if let Ok(Some(next)) =
                apply(at, &steps, Edit::Delete).map_err(|e| named_in(e, "variant_delete"))?
            {
                current = Some(next);
            }
        }
        match (current, col.raw(i)) {
            (Some((m, v)), _) => out.append(&m, &v),
            (None, Some((m, v))) => out.append(m, v),
            (None, None) => out.append_null(),
        }
    }
    out.finish()
}

/// `variant_strip_nulls(v, include_arrays)` over every row.
fn strip_rows(x: &Inputs<'_>, include_arrays: bool) -> Result<ArrayRef, KrishivError> {
    let column = x.column(0)?;
    if *column.data_type() == DataType::Null {
        return null_variants(x);
    }
    let col = VariantColumn::new(column.as_ref())?;
    let mut out = VariantColumnBuilder::with_capacity(col.len());
    for i in 0..col.len() {
        match col.get(i)? {
            Some(v) => {
                let (m, bytes) = strip_nulls(v, include_arrays)?;
                out.append(&m, &bytes);
            }
            None => out.append_null(),
        }
    }
    out.finish()
}

/// The boolean literal at argument `i`: `Some(None)` for a NULL one, `None` for a column.
fn boolean_literal(args: &[BindArg], i: usize) -> Option<Option<bool>> {
    use arrow_array::cast::AsArray;
    match args.get(i)? {
        BindArg::Literal(a) if *a.data_type() == DataType::Boolean => {
            Some(a.is_valid(0).then(|| a.as_boolean().value(0)))
        }
        BindArg::Literal(a) if a.is_null(0) => Some(None),
        _ => None,
    }
}

/// An all-NULL variant column of `rows` rows.
fn null_variants(x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let mut out = VariantColumnBuilder::with_capacity(x.rows());
    for _ in 0..x.rows() {
        out.append_null();
    }
    out.finish()
}

/// Binds the variant function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let variant = SparkType::Variant.to_arrow();
    let bound = match (listed, types.as_slice()) {
        ("parse_json" | "try_parse_json", [t]) if is_string(t) || *t == DataType::Null => {
            let fail = listed == "parse_json";
            Some(Binding::new(variant, move |x: &Inputs<'_>| {
                parse_json(x, fail)
            }))
        }
        ("schema_of_variant", [t]) if variant_or_null(t) => {
            Some(Binding::new(DataType::Utf8View, schema_of_variant))
        }
        ("to_variant_object", [t]) => {
            // The analyzer checked the type; its Spark form names the fields.
            let from = SparkType::from_arrow(t).ok();
            from.map(|from| {
                Binding::new(variant.clone(), move |x: &Inputs<'_>| {
                    krishiv_kernels::variant::cast_to_variant(x.column(0)?.as_ref(), &from)
                })
            })
        }
        ("is_valid_variant", [t]) if variant_or_null(t) => {
            Some(Binding::new(DataType::Boolean, is_valid_variant))
        }
        ("is_variant_null", [t]) if variant_or_null(t) => {
            Some(Binding::new(DataType::Boolean, is_variant_null))
        }
        ("variant_get" | "try_variant_get", [v, p, rest @ ..])
            if variant_or_null(v) && (is_string(p) || *p == DataType::Null) && rest.len() <= 1 =>
        {
            let fail = listed == "variant_get";
            let name: &'static str = if fail {
                "variant_get"
            } else {
                "try_variant_get"
            };
            // The analyzer checked the type literal.
            let to = match string_literal(args, 2) {
                Some(Some(t)) => krishiv_types::parse_data_type(&t)
                    .ok()
                    .map(|t| t.as_nullable()),
                None if rest.is_empty() => Some(SparkType::Variant),
                _ => None,
            };
            // A literal path that does not parse fails at bind, as Spark's
            // code generation evaluates it before any row.
            let path = match string_literal(args, 1) {
                Some(Some(p)) => match parse_path_or_raise(&p, name) {
                    Ok(p) => Path::Literal(Some(p)),
                    Err(e) => return Some((listed, Err(e))),
                },
                Some(None) => Path::Literal(None),
                None => Path::Column,
            };
            let opts = opts.clone();
            to.map(|to| {
                // Function results use the `Decimal128` carrier for every decimal (the analyzer's type), though the kernels
                // build small decimals on `Decimal64`.
                let declared = match to.to_arrow() {
                    DataType::Decimal64(p, s) => DataType::Decimal128(p, s),
                    other => other,
                };
                Binding::new(declared, move |x: &Inputs<'_>| {
                    let out = variant_get(x, &path, &to, fail, name, &opts)?;
                    widen_decimal64(out)
                })
            })
        }
        (
            "variant_insert"
            | "try_variant_insert"
            | "variant_set"
            | "try_variant_set"
            | "variant_array_append"
            | "try_variant_array_append",
            [v, p, _, rest @ ..],
        ) if variant_or_null(v)
            && (is_string(p) || *p == DataType::Null)
            && rest.len() <= usize::from(listed.ends_with("variant_set")) =>
        {
            let try_ = listed.starts_with("try_");
            let name: &'static str = NAMES
                .iter()
                .copied()
                .find(|n| *n == listed)
                .unwrap_or("variant_insert");
            // `create_if_missing` is a foldable boolean (the analyzer checked it); NULL makes every row NULL.
            let create = match (rest.len(), boolean_literal(args, 3)) {
                (0, _) => Some(Some(true)),
                (_, Some(c)) => Some(c),
                (_, None) => None,
            };
            let kind = match listed {
                "variant_insert" | "try_variant_insert" => Some(EditKind::Insert),
                "variant_array_append" | "try_variant_array_append" => Some(EditKind::Append),
                _ => create.flatten().map(EditKind::Set),
            };
            create.map(|_| {
                Binding::new(variant.clone(), move |x: &Inputs<'_>| match kind {
                    Some(kind) => edit_rows(x, name, kind, try_),
                    None => null_variants(x),
                })
            })
        }
        ("variant_delete", [v, paths @ ..])
            if variant_or_null(v)
                && !paths.is_empty()
                && paths.iter().all(|p| is_string(p) || *p == DataType::Null) =>
        {
            let count = args.len();
            Some(Binding::new(variant, move |x: &Inputs<'_>| {
                delete_rows(x, count)
            }))
        }
        ("variant_strip_nulls", [v, rest @ ..]) if variant_or_null(v) && rest.len() <= 1 => {
            let include = match (rest.len(), boolean_literal(args, 1)) {
                (0, _) => Some(Some(true)),
                (_, flag) => flag,
            };
            include.map(|include| {
                Binding::new(variant.clone(), move |x: &Inputs<'_>| match include {
                    Some(include) => strip_rows(x, include),
                    None => null_variants(x),
                })
            })
        }
        _ => None,
    };
    // Spark parses a foldable path once per expression, before any row: a bad literal path raises at bind, even
    // over NULL variants or no rows. A column path is parsed per row, after the NULL checks.
    let allow_root = listed.ends_with("variant_array_append");
    let literal_paths = match listed {
        "variant_delete" => 1..args.len(),
        "variant_insert"
        | "try_variant_insert"
        | "variant_set"
        | "try_variant_set"
        | "variant_array_append"
        | "try_variant_array_append" => 1..2,
        _ => 0..0,
    };
    // `variant_insert`, `variant_set` and `variant_array_append` are null-intolerant in Spark: a NULL literal
    // argument folds the call to NULL before the path is parsed. `variant_delete` raises regardless.
    let folds_to_null = listed != "variant_delete"
        && args.iter().any(|a| match a {
            BindArg::Literal(a) => *a.data_type() == DataType::Null || a.is_null(0),
            BindArg::Column { .. } => false,
        });
    let checked = bound.ok_or_else(|| bad_types(listed, args)).and_then(|b| {
        if !folds_to_null {
            for i in literal_paths {
                if let Some(Some(path)) = string_literal(args, i) {
                    edit_path(&path, listed, allow_root)?;
                }
            }
        }
        Ok(b)
    });
    Some((listed, checked))
}

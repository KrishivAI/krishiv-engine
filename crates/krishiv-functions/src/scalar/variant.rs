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

use std::sync::Arc;

use arrow_array::builder::BooleanBuilder;
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::variant::{
    PathSegment, VariantColumn, VariantColumnBuilder, cast_values, extract, parse_path_or_raise,
};
use krishiv_types::DataType as SparkType;
use krishiv_types::variant::VariantBuilder;

use super::args::{Inputs, is_string, to_utf8};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "is_variant_null",
    "parse_json",
    "try_parse_json",
    "try_variant_get",
    "variant_get",
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
    cast_values(&found, to, fail, opts.time_zone.as_ref())
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
                Some(Some(t)) => krishiv_types::parse_data_type(&t).ok(),
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
                Binding::new(to.to_arrow(), move |x: &Inputs<'_>| {
                    variant_get(x, &path, &to, fail, name, &opts)
                })
            })
        }
        _ => None,
    };
    Some((listed, bound.ok_or_else(|| bad_types(listed, args))))
}

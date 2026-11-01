// SPDX-License-Identifier: Apache-2.0
//! The spatial functions: `st_geomfromwkb`, `st_geogfromwkb`, `st_asbinary`,
//! `st_srid` and `st_setsrid` (Spark 4.2's `ST*` expressions).
//!
//! A geometry or geography value is the struct `{srid: int, wkb: binary}`
//! (see `krishiv_types::arrow`). The `from` functions validate the WKB with
//! [`krishiv_types::geo::rewrite`] (`WKB_PARSE_ERROR`, a geography's
//! coordinates in range) and store it little-endian; `st_asbinary` writes
//! `NDR` (the default) or `XDR`; any other endianness is
//! `ST_INVALID_ENDIANNESS_VALUE`. A NULL argument makes the row NULL. The
//! SRID of `st_geomfromwkb` and `st_setsrid` is a literal, already checked
//! by the signature (`ST_INVALID_SRID_VALUE`).

use std::sync::Arc;

use arrow_array::builder::{BinaryViewBuilder, Int32Builder};
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, BinaryViewArray, Int32Array, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use krishiv_common::error::{KrishivError, StInvalidEndiannessValue};
use krishiv_types::Srid;
use krishiv_types::geo::{ByteOrder, SpatialKind, rewrite};

use super::args::{Inputs, internal};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "st_asbinary",
    "st_geogfromwkb",
    "st_geomfromwkb",
    "st_setsrid",
    "st_srid",
];

/// The Arrow type of every spatial value.
fn spatial_type() -> DataType {
    krishiv_types::DataType::Geometry { srid: Srid::Any }.to_arrow()
}

/// Whether `t` is the struct of a spatial value.
fn is_spatial(t: &DataType) -> bool {
    *t == spatial_type()
}

/// Builds a column of spatial values; `None` is a NULL value.
fn spatial_column(rows: &[Option<(i32, Vec<u8>)>]) -> Result<ArrayRef, KrishivError> {
    let DataType::Struct(fields) = spatial_type() else {
        return Err(internal("the spatial type is not a struct".to_owned()));
    };
    let mut srids = Int32Builder::with_capacity(rows.len());
    let mut wkbs = BinaryViewBuilder::with_capacity(rows.len());
    let mut valid = Vec::with_capacity(rows.len());
    for row in rows {
        // The children are never null: a NULL value holds an empty one.
        srids.append_value(row.as_ref().map_or(0, |(s, _)| *s));
        wkbs.append_value(row.as_ref().map_or(&[][..], |(_, w)| w.as_slice()));
        valid.push(row.is_some());
    }
    let children: Vec<ArrayRef> = vec![Arc::new(srids.finish()), Arc::new(wkbs.finish())];
    StructArray::try_new(fields, children, Some(NullBuffer::from(valid)))
        .map(|a| Arc::new(a) as ArrayRef)
        .map_err(|e| internal(e.to_string()))
}

/// The SRID and WKB columns of a column of spatial values.
fn parts(column: &ArrayRef) -> Result<(Int32Array, BinaryViewArray), KrishivError> {
    let s = column
        .as_struct_opt()
        .ok_or_else(|| internal(format!("a spatial argument of type {}", column.data_type())))?;
    let srid = s.column(0).as_primitive_opt::<Int32Type>().cloned();
    let wkb = s
        .column(1)
        .as_binary_view_opt()
        .cloned()
        .ok_or_else(|| internal("a spatial value without a binary view WKB".to_owned()))?;
    srid.map(|s| (s, wkb))
        .ok_or_else(|| internal("a spatial value without an int SRID".to_owned()))
}

/// The `Int32` literal at argument `i`.
fn srid_literal(args: &[BindArg], i: usize) -> Option<i32> {
    match args.get(i)? {
        BindArg::Literal(a) => {
            let a = a.as_primitive_opt::<Int32Type>()?;
            a.is_valid(0).then(|| a.value(0))
        }
        BindArg::Column { .. } => None,
    }
}

/// The string literal at argument `i`.
fn string_literal(args: &[BindArg], i: usize) -> Option<String> {
    match args.get(i)? {
        BindArg::Literal(a) => {
            let a = super::args::to_utf8(a.as_ref()).ok()?;
            a.is_valid(0).then(|| a.value(0).to_owned())
        }
        BindArg::Column { .. } => None,
    }
}

/// Binds the spatial function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let is_binary = |t: &DataType| {
        matches!(
            t,
            DataType::Binary | DataType::BinaryView | DataType::LargeBinary
        )
    };
    // The first argument of the accessors: a spatial value or an untyped NULL.
    let spatial_or_null = |t: &DataType| is_spatial(t) || *t == DataType::Null;
    let bound = match (listed, types.as_slice()) {
        ("st_geomfromwkb" | "st_geogfromwkb", [wkb, ..])
            if is_binary(wkb)
                && (listed == "st_geogfromwkb" && types.len() == 1
                    || listed == "st_geomfromwkb" && matches!(types.len(), 1 | 2)) =>
        {
            let geography = listed == "st_geogfromwkb";
            let srid = if geography {
                Some(4326)
            } else if types.len() == 2 {
                srid_literal(args, 1)
            } else {
                Some(0)
            };
            srid.map(|srid| {
                let kind = if geography {
                    SpatialKind::Geography
                } else {
                    SpatialKind::Geometry
                };
                Binding::new(spatial_type(), move |x: &Inputs<'_>| {
                    let wkbs = x.binaries(0)?;
                    let rows = wkbs
                        .iter()
                        .map(|w| {
                            w.map(|w| rewrite(w, kind, ByteOrder::Ndr).map(|b| (srid, b)))
                                .transpose()
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    spatial_column(&rows)
                })
            })
        }
        ("st_asbinary", [g] | [g, DataType::Utf8 | DataType::Utf8View]) if spatial_or_null(g) => {
            let order = match string_literal(args, 1) {
                None if types.len() == 1 => Some(ByteOrder::Ndr),
                Some(e) if e.eq_ignore_ascii_case("NDR") => Some(ByteOrder::Ndr),
                Some(e) if e.eq_ignore_ascii_case("XDR") => Some(ByteOrder::Xdr),
                Some(e) => {
                    return Some((
                        listed,
                        Err(StInvalidEndiannessValue { endianness: e }.into()),
                    ));
                }
                None => None,
            };
            order.map(|order| {
                Binding::new(DataType::BinaryView, move |x: &Inputs<'_>| {
                    let column = x.column(0)?;
                    let mut out = BinaryViewBuilder::with_capacity(column.len());
                    if column.data_type() == &DataType::Null {
                        (0..column.len()).for_each(|_| out.append_null());
                    } else {
                        let (_, wkb) = parts(&column)?;
                        for row in 0..column.len() {
                            if column.is_null(row) {
                                out.append_null();
                            } else if order == ByteOrder::Ndr {
                                out.append_value(wkb.value(row));
                            } else {
                                // The stored bytes were validated on the way in.
                                out.append_value(rewrite(
                                    wkb.value(row),
                                    SpatialKind::Geometry,
                                    order,
                                )?);
                            }
                        }
                    }
                    Ok(Arc::new(out.finish()) as ArrayRef)
                })
            })
        }
        ("st_srid", [g]) if spatial_or_null(g) => {
            Some(Binding::new(DataType::Int32, move |x: &Inputs<'_>| {
                let column = x.column(0)?;
                let mut out = Int32Builder::with_capacity(column.len());
                if column.data_type() == &DataType::Null {
                    (0..column.len()).for_each(|_| out.append_null());
                } else {
                    let (srid, _) = parts(&column)?;
                    for row in 0..column.len() {
                        out.append_option((!column.is_null(row)).then(|| srid.value(row)));
                    }
                }
                Ok(Arc::new(out.finish()) as ArrayRef)
            }))
        }
        ("st_setsrid", [g, _]) if spatial_or_null(g) => srid_literal(args, 1).map(|new| {
            Binding::new(spatial_type(), move |x: &Inputs<'_>| {
                let column = x.column(0)?;
                if column.data_type() == &DataType::Null {
                    return spatial_column(&vec![None; column.len()]);
                }
                let (_, wkb) = parts(&column)?;
                let rows = (0..column.len())
                    .map(|row| (!column.is_null(row)).then(|| (new, wkb.value(row).to_vec())))
                    .collect::<Vec<_>>();
                spatial_column(&rows)
            })
        }),
        _ => None,
    };
    Some((listed, bound.ok_or_else(|| bad_types(listed, args))))
}

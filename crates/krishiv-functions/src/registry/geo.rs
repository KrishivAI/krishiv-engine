// SPDX-License-Identifier: Apache-2.0
//! The spatial function signatures: `st_geomfromwkb`, `st_geogfromwkb`,
//! `st_asbinary`, `st_srid` and `st_setsrid` (Spark 4.2's `ST*` expressions,
//! checked against the `st-functions.sql` golden).
//!
//! The SRID of `st_geomfromwkb` and `st_setsrid` must be foldable: it is the
//! result type (`geometry(srid)`, or `geography(4326)` for `st_setsrid` of a
//! geography), and an SRID Spark does not support is
//! `ST_INVALID_SRID_VALUE` at analysis.

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType, KrishivError,
};
use krishiv_types::{DataType, Srid};

use super::defs::{Spec, scalar};
use super::hof::{call_sql, quoted};
use crate::signature::{ArgType, CallTypes, Constant, InputType, Inputs};

/// The SRID argument `i`, a foldable integer.
fn srid_argument(name: &str, c: &CallTypes<'_>, i: usize) -> Result<i32, KrishivError> {
    match c.constant(i) {
        Some(Constant::Integral(n)) => {
            // An out-of-range value is no SRID either; the message prints it.
            Ok(i32::try_from(*n).unwrap_or(i32::MIN))
        }
        Some(Constant::Error(e)) => Err(e.clone()),
        _ => Err(DatatypeMismatchNonFoldableInput {
            sql_expr: call_sql(name, c),
            input_name: "`srid`".to_owned(),
            input_type: "\"INT\"".to_owned(),
            input_expr: format!("\"{}\"", c.sql(i).unwrap_or_default()),
        }
        .into()),
    }
}

/// `geometry(srid)`, `ST_INVALID_SRID_VALUE` for an unsupported one.
fn geometry(srid: i32) -> Result<DataType, KrishivError> {
    DataType::geometry(Srid::Fixed(srid))
}

fn nullable(c: &CallTypes<'_>) -> bool {
    c.args().first().is_none_or(|a| a.nullable)
}

/// `st_geomfromwkb(wkb[, srid])`: `geometry(srid)`, `geometry(0)` by default.
fn geom_from_wkb(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let srid = if c.args().len() > 1 {
        srid_argument("st_geomfromwkb", c, 1)?
    } else {
        0
    };
    Ok(ArgType::new(geometry(srid)?, nullable(c)))
}

/// `st_geogfromwkb(wkb)`: `geography(4326)`.
fn geog_from_wkb(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        DataType::geography(Srid::Fixed(4326))?,
        nullable(c),
    ))
}

/// The argument must be a geometry, a geography or an untyped `NULL`.
fn spatial_argument<'a>(
    name: &str,
    c: &CallTypes<'a>,
) -> Result<Option<&'a DataType>, KrishivError> {
    match c.data_type(0) {
        t @ (None | Some(DataType::Geometry { .. } | DataType::Geography { .. })) => Ok(t),
        t => Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql(name, c),
            param_index: "first".to_owned(),
            required_type: "(\"GEOGRAPHY\" or \"GEOMETRY\")".to_owned(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: quoted(t),
        }
        .into()),
    }
}

/// `st_asbinary(g[, endianness])`: a binary.
fn as_binary(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    spatial_argument("st_asbinary", c)?;
    Ok(ArgType::new(DataType::Binary, nullable(c)))
}

/// `st_srid(g)`: an int.
fn srid(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    spatial_argument("st_srid", c)?;
    Ok(ArgType::new(DataType::Integer, nullable(c)))
}

/// `st_setsrid(g, srid)`: the same kind with the new SRID.
fn set_srid(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let kind = spatial_argument("st_setsrid", c)?;
    let n = srid_argument("st_setsrid", c, 1)?;
    let t = match kind {
        Some(DataType::Geography { .. }) => DataType::geography(Srid::Fixed(n))?,
        _ => geometry(n)?,
    };
    Ok(ArgType::new(t, nullable(c)))
}

pub(super) fn specs() -> Vec<Spec> {
    let wkb = |with_srid: bool| {
        let mut v = vec![InputType::Strict(DataType::Binary)];
        if with_srid {
            v.push(InputType::Exact(DataType::Integer));
        }
        Inputs::Expects(v)
    };
    let mut out = scalar(&["st_geomfromwkb"], &wkb(true), geom_from_wkb);
    out.extend(scalar(&["st_geogfromwkb"], &wkb(false), geog_from_wkb));
    out.extend(scalar(
        &["st_asbinary"],
        &Inputs::Expects(vec![InputType::Any, InputType::Strict(DataType::String)]),
        as_binary,
    ));
    out.extend(scalar(&["st_srid"], &Inputs::Any, srid));
    out.extend(scalar(
        &["st_setsrid"],
        &Inputs::Expects(vec![InputType::Any, InputType::Exact(DataType::Integer)]),
        set_srid,
    ));
    out
}

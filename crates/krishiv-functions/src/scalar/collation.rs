// SPDX-License-Identifier: Apache-2.0
//! `__collation_key(value, type)`: the bytes operators compare in place of a collated value
//! (ADR 0043, design `P2-collations.md` section 3.3). Only the analyzer and the optimizer build
//! it: `type` is the Spark type of `value` as a literal, which names the collation of every
//! string leaf (the Arrow carrier is a plain string).

use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use std::collections::HashSet;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, ListArray};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::string::{
    collation_key, collation_key_type, keyed_map, keyed_map_type, push_collation_key,
};
use krishiv_types::Collation;

use super::args::{Inputs, internal, to_utf8};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "__collation_key",
    "__collation_map_keys",
    "__collation_greatest",
    "__collation_least",
    "__collation_array_set",
];

/// The first literal argument as a Spark type.
fn spark_type(t: &BindArg) -> Result<krishiv_types::DataType, KrishivError> {
    let BindArg::Literal(t) = t else {
        return Err(internal("a collation key's type is no literal".to_owned()));
    };
    let text = to_utf8(t.as_ref())?;
    if text.len() != 1 || text.is_null(0) {
        return Err(internal("a collation key's type is no string".to_owned()));
    }
    krishiv_types::parse_data_type(text.value(0))
}

/// The row-wise pick of `greatest` / `least` under a collation: NULLs are skipped and the first
/// of equal keys stays (Spark's `Greatest` / `Least` compare with `>` / `<` only).
fn pick(x: &Inputs<'_>, collation: Collation, greatest: bool) -> Result<ArrayRef, KrishivError> {
    let (a, b) = (x.strings(1)?, x.strings(2)?);
    let mut out = StringBuilder::new();
    let (mut ka, mut kb) = (Vec::new(), Vec::new());
    for row in 0..x.rows() {
        let chosen = match (a.is_null(row), b.is_null(row)) {
            (true, true) => None,
            (true, false) => Some(b.value(row)),
            (false, true) => Some(a.value(row)),
            (false, false) => {
                ka.clear();
                kb.clear();
                push_collation_key(collation, a.value(row), &mut ka)
                    .map_err(|e| e.error().clone())?;
                push_collation_key(collation, b.value(row), &mut kb)
                    .map_err(|e| e.error().clone())?;
                let takes_b = if greatest { kb > ka } else { kb < ka };
                Some(if takes_b { b.value(row) } else { a.value(row) })
            }
        };
        out.append_option(chosen);
    }
    Ok(Arc::new(out.finish()))
}

/// The element range of row `row` of `l`.
fn list_range(l: &ListArray, row: usize) -> Option<(usize, usize)> {
    let o = l.value_offsets();
    Some((
        usize::try_from(o[row]).ok()?,
        usize::try_from(o[row + 1]).ok()?,
    ))
}

/// The key at `i`, `None` for a NULL element.
fn key_at(v: &arrow_array::BinaryViewArray, i: usize) -> Option<&[u8]> {
    (!v.is_null(i)).then(|| v.value(i))
}

/// `__collation_array_set(op, a, keys(a), b, keys(b))`: `array_distinct`, `array_union`,
/// `array_intersect` or `array_except` where elements are equal when their keys are. The result
/// keeps the original elements of the first array that has them, in Spark's order.
fn array_set(op: &str, x: &Inputs<'_>) -> Result<ArrayRef, KrishivError> {
    let bad = || internal("a collation array set over a non-list or non-string key".to_owned());
    let (a, ka, b, kb) = (x.column(1)?, x.column(2)?, x.column(3)?, x.column(4)?);
    let (la, lka, lb, lkb) = (
        a.as_list_opt::<i32>().ok_or_else(bad)?,
        ka.as_list_opt::<i32>().ok_or_else(bad)?,
        b.as_list_opt::<i32>().ok_or_else(bad)?,
        kb.as_list_opt::<i32>().ok_or_else(bad)?,
    );
    let (va, vb) = (
        lka.values().as_binary_view_opt().ok_or_else(bad)?,
        lkb.values().as_binary_view_opt().ok_or_else(bad)?,
    );
    let two = op != "distinct";
    let mut picks: Vec<(usize, usize)> = Vec::new();
    let mut offsets = vec![0i32];
    let mut valid = Vec::with_capacity(x.rows());
    for row in 0..x.rows() {
        if la.is_null(row) || (two && lb.is_null(row)) {
            valid.push(false);
            offsets.push(i32::try_from(picks.len()).map_err(|_| bad())?);
            continue;
        }
        valid.push(true);
        let (sa, ea) = list_range(lka, row).ok_or_else(bad)?;
        let (sb, eb) = if two {
            list_range(lkb, row).ok_or_else(bad)?
        } else {
            (0, 0)
        };
        let in_b: HashSet<Option<&[u8]>> = (sb..eb).map(|i| key_at(vb, i)).collect();
        let mut seen: HashSet<Option<&[u8]>> = HashSet::new();
        for i in sa..ea {
            let k = key_at(va, i);
            let keep = match op {
                "intersect" => in_b.contains(&k),
                "except" => !in_b.contains(&k),
                _ => true,
            };
            if keep && seen.insert(k) {
                picks.push((0, i));
            }
        }
        if op == "union" {
            for i in sb..eb {
                if seen.insert(key_at(vb, i)) {
                    picks.push((1, i));
                }
            }
        }
        offsets.push(i32::try_from(picks.len()).map_err(|_| bad())?);
    }
    let values = if picks.is_empty() {
        arrow_array::new_empty_array(la.values().data_type())
    } else {
        arrow_select::interleave::interleave(&[la.values().as_ref(), lb.values().as_ref()], &picks)
            .map_err(|e| internal(e.to_string()))?
    };
    let DataType::List(field) = a.data_type() else {
        return Err(bad());
    };
    let nulls = valid.iter().any(|v| !v).then(|| NullBuffer::from(valid));
    Ok(Arc::new(ListArray::new(
        Arc::clone(field),
        OffsetBuffer::new(offsets.into()),
        values,
        nulls,
    )))
}

pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let bound = (|| match (listed, args) {
        ("__collation_key", [value, t]) => {
            let spark = spark_type(t)?;
            let ty: DataType = collation_key_type(&value.data_type());
            Ok(Binding::new(ty, move |x: &Inputs<'_>| {
                collation_key(&spark, &x.column(0)?).map_err(|e| e.error().clone())
            }))
        }
        ("__collation_map_keys", [value, t]) => {
            let spark = spark_type(t)?;
            let ty = keyed_map_type(&value.data_type());
            Ok(Binding::new(ty, move |x: &Inputs<'_>| {
                keyed_map(&spark, &x.column(0)?).map_err(|e| e.error().clone())
            }))
        }
        ("__collation_greatest" | "__collation_least", [t, _, _]) => {
            // A plain `STRING` is the binary collation an explicit `UTF8_BINARY` operand chose.
            let c = match spark_type(t)? {
                krishiv_types::DataType::CollatedString(c) => c,
                krishiv_types::DataType::String => Collation::UTF8_BINARY,
                _ => return Err(bad_types(listed, args)),
            };
            let greatest = listed == "__collation_greatest";
            Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                pick(x, c, greatest)
            }))
        }
        ("__collation_array_set", [op, a, _, _, _]) => {
            let BindArg::Literal(op) = op else {
                return Err(bad_types(listed, args));
            };
            let op = to_utf8(op.as_ref())?;
            if op.len() != 1 || op.is_null(0) {
                return Err(bad_types(listed, args));
            }
            let op = op.value(0).to_owned();
            Ok(Binding::new(a.data_type(), move |x: &Inputs<'_>| {
                array_set(&op, x)
            }))
        }
        _ => Err(bad_types(listed, args)),
    })();
    Some((listed, bound))
}

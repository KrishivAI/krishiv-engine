// SPDX-License-Identifier: Apache-2.0
//! Type checks and output conformance of the evaluator (port of the kernel's
//! `engine/ensure_data_types.rs` and `engine/arrow_utils/apply_schema.rs`, `delta_kernel`
//! 0.30.1, Apache-2.0).
//!
//! - [`check`] is the kernel's `ensure_data_types` in its `TypesAndNames` mode: a primitive
//!   matches its own Arrow type or one the reader may widen from (an integer to a wider one or
//!   to a double, a float to a double, an upcastable decimal, `Date32` to `timestamp_ntz`,
//!   `Int32` to `date`, `Int64` to `timestamp`, any timestamp unit or zone label); a string or
//!   binary matches its large and view forms; a struct matches by field name (extra Arrow
//!   fields are ignored).
//! - [`conform`] gives an array exactly the Arrow type of a kernel type: names, nullability,
//!   field ids and zone labels from the kernel side, children rebuilt without copies. Where the
//!   kernel's `apply_schema` leaves a widenable leaf as it is, this casts it, so the output
//!   always has the declared type (the kernel's `GetData` checks types).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, LargeListArray, ListArray, MapArray, StructArray};
use arrow_cast::{CastOptions, cast_with_options};
use arrow_schema::{DataType as ArrowType, Fields, TimeUnit};
use delta_kernel::schema::DataType;
use delta_kernel::{KernelError, KernelResult};

use super::arrow_err;
use crate::engine::relabel::{ARROW_UTC, relabel, with_zone};
use crate::engine::schema::to_arrow_type;

fn incorrect(expected: &dyn std::fmt::Display, got: &ArrowType) -> KernelError {
    KernelError::generic(format!(
        "Incorrect datatype. Expected {expected}, got {got}"
    ))
}

/// Whether a reader may widen `from` to `to` (both primitive Arrow types).
fn widens(from: &ArrowType, to: &ArrowType) -> bool {
    use ArrowType::{Date32, Decimal128, Float32, Float64, Int8, Int16, Int32, Int64, Timestamp};
    match (from, to) {
        (a, b) if a == b => true,
        (Timestamp(..), Timestamp(..))
        | (Int8, Int16 | Int32 | Int64 | Float64)
        | (Int16, Int32 | Int64 | Float64)
        | (Int32, Int64 | Float64 | Date32)
        | (Float32, Float64)
        | (Date32, Timestamp(_, None))
        | (Int64, Timestamp(TimeUnit::Microsecond, _)) => true,
        (_, Decimal128(p, s)) => {
            let (sp, ss) = match from {
                Decimal128(p, s) => (*p, *s),
                Int8 => (3, 0),
                Int16 => (5, 0),
                Int32 => (10, 0),
                Int64 => (20, 0),
                _ => return false,
            };
            *p >= sp && *s >= ss && i16::from(*p) - i16::from(sp) >= i16::from(*s) - i16::from(ss)
        }
        _ => false,
    }
}

/// Fails unless the Arrow type `got` can carry the kernel type `expected` (see the module
/// docs).
///
/// # Errors
/// The kernel's "Incorrect datatype" message.
pub(crate) fn check(expected: &DataType, got: &ArrowType) -> KernelResult<()> {
    match (expected, got) {
        (DataType::Primitive(_), _) if is_primitive(got) => {
            let want = to_arrow_type(expected)?;
            if widens(got, &want)
                || matches!(
                    (&want, got),
                    (ArrowType::Utf8, ArrowType::LargeUtf8 | ArrowType::Utf8View)
                        | (
                            ArrowType::Binary,
                            ArrowType::LargeBinary | ArrowType::BinaryView
                        )
                )
            {
                Ok(())
            } else {
                Err(incorrect(expected, got))
            }
        }
        (DataType::Array(a), ArrowType::List(f) | ArrowType::LargeList(f)) => {
            check(a.element_type(), f.data_type())
        }
        (DataType::Map(m), ArrowType::Map(entries, _)) => {
            let ArrowType::Struct(kv) = entries.data_type() else {
                return Err(incorrect(expected, got));
            };
            let [k, v] = kv.iter().collect::<Vec<_>>()[..] else {
                return Err(incorrect(expected, got));
            };
            check(m.key_type(), k.data_type())?;
            check(m.value_type(), v.data_type())
        }
        (DataType::Struct(s) | DataType::Variant(s), ArrowType::Struct(fields)) => {
            let mut found = 0;
            for f in fields {
                if let Some(k) = s.field(f.name()) {
                    check(k.data_type(), f.data_type())?;
                    found += 1;
                }
            }
            if found != s.num_fields() {
                let missing: Vec<&str> = s
                    .fields()
                    .map(|k| k.name().as_str())
                    .filter(|n| fields.iter().all(|f| f.name() != n))
                    .take(5)
                    .collect();
                return Err(KernelError::generic(format!(
                    "Missing Struct fields {} (Up to five missing fields shown)",
                    missing.join(", ")
                )));
            }
            Ok(())
        }
        _ => Err(incorrect(expected, got)),
    }
}

fn is_primitive(t: &ArrowType) -> bool {
    !matches!(
        t,
        ArrowType::Struct(_)
            | ArrowType::List(_)
            | ArrowType::LargeList(_)
            | ArrowType::ListView(_)
            | ArrowType::LargeListView(_)
            | ArrowType::FixedSizeList(..)
            | ArrowType::Map(..)
            | ArrowType::Union(..)
            | ArrowType::Dictionary(..)
            | ArrowType::RunEndEncoded(..)
    )
}

/// [`check`] when `expected` is given, then `array` unchanged.
pub(crate) fn validate(array: ArrayRef, expected: Option<&DataType>) -> KernelResult<ArrayRef> {
    if let Some(expected) = expected {
        check(expected, array.data_type())?;
    }
    Ok(array)
}

/// `array`, already [`check`]ed against `expected`, with every struct's children picked by
/// name in `expected`'s field order (through lists and maps too), so that the positional
/// [`conform`] cannot pair a child with another field's name. Arrow children that `expected`
/// does not name are dropped. An array whose struct fields are already in order is returned
/// as it is.
///
/// The kernel's `apply_schema` pairs children by position after its name-based check, so a
/// struct column stored as `{b, a}` and read as `{a, b}` swaps its values there; Krishiv's
/// readers reconcile files by name, so this only matters for batches built another way.
///
/// # Errors
/// An Arrow error rebuilding a struct, list or map.
pub(crate) fn select_by_name(array: ArrayRef, expected: &DataType) -> KernelResult<ArrayRef> {
    match (expected, array.data_type()) {
        (DataType::Struct(s) | DataType::Variant(s), ArrowType::Struct(fields)) => {
            let in_order = fields.len() == s.num_fields()
                && fields
                    .iter()
                    .zip(s.fields())
                    .all(|(f, k)| f.name() == k.name());
            let st = array.as_struct();
            let mut out_fields = Vec::with_capacity(s.num_fields());
            let mut columns = Vec::with_capacity(s.num_fields());
            let mut changed = !in_order;
            for k in s.fields() {
                let Some((i, f)) = fields.find(k.name()) else {
                    return Err(KernelError::generic(format!(
                        "Missing Struct fields {} (Up to five missing fields shown)",
                        k.name()
                    )));
                };
                let child = Arc::clone(st.column(i));
                let picked = select_by_name(Arc::clone(&child), k.data_type())?;
                changed |= !Arc::ptr_eq(&picked, &child);
                out_fields.push(Arc::new(
                    f.as_ref()
                        .clone()
                        .with_data_type(picked.data_type().clone()),
                ));
                columns.push(picked);
            }
            if !changed {
                return Ok(array);
            }
            // Never empty: a struct of no fields is always in order and unchanged.
            Ok(Arc::new(
                StructArray::try_new(out_fields.into(), columns, st.nulls().cloned())
                    .map_err(arrow_err)?,
            ))
        }
        (DataType::Array(a), ArrowType::List(f)) => {
            let l = array.as_list::<i32>();
            let values = select_by_name(Arc::clone(l.values()), a.element_type())?;
            if Arc::ptr_eq(&values, l.values()) {
                return Ok(array);
            }
            let field = Arc::new(
                f.as_ref()
                    .clone()
                    .with_data_type(values.data_type().clone()),
            );
            Ok(Arc::new(
                ListArray::try_new(field, l.offsets().clone(), values, l.nulls().cloned())
                    .map_err(arrow_err)?,
            ))
        }
        (DataType::Array(a), ArrowType::LargeList(f)) => {
            let l = array.as_list::<i64>();
            let values = select_by_name(Arc::clone(l.values()), a.element_type())?;
            if Arc::ptr_eq(&values, l.values()) {
                return Ok(array);
            }
            let field = Arc::new(
                f.as_ref()
                    .clone()
                    .with_data_type(values.data_type().clone()),
            );
            Ok(Arc::new(
                LargeListArray::try_new(field, l.offsets().clone(), values, l.nulls().cloned())
                    .map_err(arrow_err)?,
            ))
        }
        (DataType::Map(m), ArrowType::Map(entries_field, sorted)) => {
            let map = array.as_map();
            let keys = select_by_name(Arc::clone(map.keys()), m.key_type())?;
            let values = select_by_name(Arc::clone(map.values()), m.value_type())?;
            if Arc::ptr_eq(&keys, map.keys()) && Arc::ptr_eq(&values, map.values()) {
                return Ok(array);
            }
            let ArrowType::Struct(kv) = entries_field.data_type() else {
                return Err(KernelError::generic("Arrow map type wasn't a struct."));
            };
            let kv: Fields = kv
                .iter()
                .zip([keys.data_type(), values.data_type()])
                .map(|(f, t)| Arc::new(f.as_ref().clone().with_data_type(t.clone())))
                .collect();
            let entries = StructArray::try_new(
                kv.clone(),
                vec![keys, values],
                map.entries().nulls().cloned(),
            )
            .map_err(arrow_err)?;
            let entries_field = Arc::new(
                entries_field
                    .as_ref()
                    .clone()
                    .with_data_type(ArrowType::Struct(kv)),
            );
            Ok(Arc::new(
                MapArray::try_new(
                    entries_field,
                    map.offsets().clone(),
                    entries,
                    map.nulls().cloned(),
                    *sorted,
                )
                .map_err(arrow_err)?,
            ))
        }
        _ => Ok(array),
    }
}

/// Casts `array` to `target`, crossing zone labels Arrow cannot parse without `chrono-tz` as
/// `+00:00`. With `safe`, a value that does not convert becomes NULL.
///
/// # Errors
/// A cast Arrow does not support, or (not `safe`) a value that does not convert.
pub(crate) fn cast_to(array: &ArrayRef, target: &ArrowType, safe: bool) -> KernelResult<ArrayRef> {
    if array.data_type() == target {
        return Ok(Arc::clone(array));
    }
    let source = relabel(array, &with_zone(array.data_type(), ARROW_UTC)).map_err(arrow_err)?;
    let options = CastOptions {
        safe,
        ..CastOptions::default()
    };
    let cast =
        cast_with_options(&source, &with_zone(target, ARROW_UTC), &options).map_err(arrow_err)?;
    relabel(&cast, target).map_err(arrow_err)
}

/// `array` with exactly the Arrow type `target`, the converted form of a kernel type (see the
/// module docs). Struct children are matched by position.
///
/// # Errors
/// A shape or type mismatch, a struct field count mismatch, or nulls in a non-nullable child.
pub(crate) fn conform(array: &ArrayRef, target: &ArrowType) -> KernelResult<ArrayRef> {
    if array.data_type() == target {
        return Ok(Arc::clone(array));
    }
    match (array.data_type(), target) {
        (ArrowType::Struct(from), ArrowType::Struct(to)) => {
            if from.len() != to.len() {
                return Err(KernelError::generic(format!(
                    "Passed struct had {} columns, but the target type has {}",
                    from.len(),
                    to.len()
                )));
            }
            let s = array.as_struct();
            let columns = s
                .columns()
                .iter()
                .zip(to.iter())
                .map(|(c, f)| conform(c, f.data_type()))
                .collect::<KernelResult<Vec<_>>>()?;
            // Never empty: two structs of no fields have equal types and returned above.
            Ok(Arc::new(
                StructArray::try_new(to.clone(), columns, s.nulls().cloned()).map_err(arrow_err)?,
            ))
        }
        (ArrowType::List(_), ArrowType::List(to)) => {
            let l = array.as_list::<i32>();
            let values = conform(l.values(), to.data_type())?;
            Ok(Arc::new(
                ListArray::try_new(
                    Arc::clone(to),
                    l.offsets().clone(),
                    values,
                    l.nulls().cloned(),
                )
                .map_err(arrow_err)?,
            ))
        }
        (ArrowType::Map(..), ArrowType::Map(to, sorted)) => {
            let m = array.as_map();
            let entries: ArrayRef = Arc::new(m.entries().clone());
            let entries = conform(&entries, to.data_type())?;
            Ok(Arc::new(
                MapArray::try_new(
                    Arc::clone(to),
                    m.offsets().clone(),
                    entries.as_struct().clone(),
                    m.nulls().cloned(),
                    *sorted,
                )
                .map_err(arrow_err)?,
            ))
        }
        (
            ArrowType::LargeList(_) | ArrowType::ListView(_) | ArrowType::LargeListView(_),
            ArrowType::List(to),
        ) => {
            let as_list = ArrowType::List(Arc::new(
                to.as_ref()
                    .clone()
                    .with_data_type(list_element(array.data_type())?),
            ));
            conform(&cast_to(array, &as_list, false)?, target)
        }
        (from, to) if is_primitive(from) && is_primitive(to) => {
            let loose = matches!(
                (from, to),
                (ArrowType::LargeUtf8 | ArrowType::Utf8View, ArrowType::Utf8)
                    | (
                        ArrowType::LargeBinary | ArrowType::BinaryView,
                        ArrowType::Binary
                    )
            );
            if widens(from, to) || loose {
                cast_to(array, to, false)
            } else {
                Err(incorrect(to, from))
            }
        }
        (from, to) => Err(incorrect(to, from)),
    }
}

fn list_element(t: &ArrowType) -> KernelResult<ArrowType> {
    match t {
        ArrowType::List(f)
        | ArrowType::LargeList(f)
        | ArrowType::ListView(f)
        | ArrowType::LargeListView(f) => Ok(f.data_type().clone()),
        other => Err(KernelError::generic(format!("not a list: {other}"))),
    }
}

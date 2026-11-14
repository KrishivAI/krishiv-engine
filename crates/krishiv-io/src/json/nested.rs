// SPDX-License-Identifier: Apache-2.0
//! Values of JSON records as typed columns of any shape: structs from
//! objects, arrays from arrays and maps from objects, with Spark's partial
//! results (`spark.sql.json.enablePartialResults`, on by default since 3.4).
//!
//! Spark's `JacksonParser` rules, read at tag `v4.2.0` and checked against
//! `tests/fixtures/spark-4.2.0/json-nested.json`:
//! - a struct (`convertObject`) takes the members it names (exact names),
//!   ignores the others; a member that does not convert is null, and the
//!   struct is kept as a partial result;
//! - an array (`convertArray`) keeps a partial element (a struct or array
//!   with a bad part), but any element that fails outright fails the whole
//!   array;
//! - a map (`convertMap`) keeps partial values; a value that fails outright
//!   fails the whole map (Spark's key and value lists then differ in length);
//! - a value of the wrong JSON kind for its type fails outright; `null` is
//!   null at any level.
//!
//! A partial result anywhere makes the record malformed (its raw text goes
//! to the corrupt column), but its values are kept.

use std::sync::Arc;

use arrow_array::{ArrayRef, ListArray, MapArray, StructArray, new_null_array};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Fields};
use krishiv_common::error::KrishivError;

use super::convert::{JsonConverter, kind_of};
use super::decode::JVal;
use crate::csv::convert::{Column, Conv, Kind, Val};
use crate::error::{internal, unsupported};
use crate::text::spark_type_name;

/// What a column (or a part of one) converts to.
#[derive(Debug)]
pub(crate) enum Shape {
    /// A scalar: its kind and its Spark type name (for errors).
    Leaf(Kind, String),
    /// A struct: its fields by name.
    Struct(Vec<(String, Shape)>),
    /// An array of the element shape.
    List(Box<Shape>),
    /// A map with string keys, of the value shape.
    Map(Box<Shape>),
}

impl Shape {
    /// The shape of a column of `f`'s type.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a type the JSON reader cannot hold
    /// (a map with non-string keys, a variant, a calendar interval, a
    /// spatial type).
    pub(crate) fn of(f: &Field) -> Result<Self, KrishivError> {
        use krishiv_types::DataType as T;
        let spark = T::from_arrow_field(f).ok();
        if matches!(
            spark,
            Some(T::Variant | T::CalendarInterval | T::Geometry { .. } | T::Geography { .. })
        ) {
            return Err(refused(f));
        }
        Ok(match f.data_type() {
            DataType::Struct(fields) => Self::Struct(
                fields
                    .iter()
                    .map(|c| Ok((c.name().clone(), Self::of(c)?)))
                    .collect::<Result<_, KrishivError>>()?,
            ),
            DataType::List(e) | DataType::LargeList(e) => Self::List(Box::new(Self::of(e)?)),
            DataType::Map(entries, _) => {
                let DataType::Struct(kv) = entries.data_type() else {
                    return Err(refused(f));
                };
                match (kv.first(), kv.get(1)) {
                    (Some(k), Some(v))
                        if matches!(
                            k.data_type(),
                            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
                        ) =>
                    {
                        Self::Map(Box::new(Self::of(v)?))
                    }
                    _ => return Err(refused(f)),
                }
            }
            _ => Self::Leaf(kind_of(f)?, spark_type_name(f.data_type())),
        })
    }

    /// Whether the shape is a scalar (built by the flat column builders).
    pub(crate) fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(..))
    }
}

fn refused(f: &Field) -> KrishivError {
    unsupported(format!(
        "the JSON data source with a column of type {}",
        spark_type_name(f.data_type())
    ))
}

/// One converted value of any shape.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NVal {
    Null,
    Leaf(Val),
    Struct(Vec<NVal>),
    List(Vec<NVal>),
    Map(Vec<(String, NVal)>),
}

/// Why a value did not convert.
#[derive(Debug)]
pub(crate) enum Fail {
    /// Kept with its bad parts null; the record is malformed (the error is
    /// the first cause, for FAILFAST).
    Partial(NVal, Option<KrishivError>),
    /// Not kept: the value is null and the record malformed.
    Whole(Option<KrishivError>),
    /// The scan fails (an unsupported conversion, a region time zone).
    Fatal(KrishivError),
}

/// The first cause seen in a value.
fn first(bad: &mut Option<Option<KrishivError>>, cause: Option<KrishivError>) {
    if bad.is_none() {
        *bad = Some(cause);
    }
}

impl JsonConverter<'_> {
    /// `v` (absent when the member is missing) as a value of `shape`, for
    /// the member `name` of a record whose text is `record`.
    ///
    /// # Errors
    /// [`Fail`], as the module docs describe.
    pub(crate) fn value(
        &self,
        shape: &Shape,
        name: &str,
        v: Option<&JVal>,
        record: &str,
    ) -> Result<NVal, Fail> {
        let Some(v) = v else {
            return Ok(NVal::Null);
        };
        if matches!(v, JVal::Null) {
            return Ok(NVal::Null);
        }
        match (shape, v) {
            (Shape::Leaf(kind, target), _) => {
                match self.convert(kind, name, target, Some(v), record) {
                    Ok(Val::Null) => Ok(NVal::Null),
                    Ok(val) => Ok(NVal::Leaf(val)),
                    Err(Conv::Malformed) => Err(Fail::Whole(None)),
                    Err(Conv::Fatal(e)) => {
                        if matches!(
                            e.error_class().name(),
                            "KRISHIV_UNSUPPORTED_FEATURE" | "INVALID_TIMEZONE"
                        ) {
                            Err(Fail::Fatal(e))
                        } else {
                            Err(Fail::Whole(Some(e)))
                        }
                    }
                }
            }
            (Shape::Struct(fields), JVal::Object(members, _)) => {
                self.object(fields, members, record)
            }
            (Shape::List(element), JVal::Array(items, _)) => {
                let mut out = Vec::with_capacity(items.len());
                let mut bad = None;
                for item in items {
                    match self.value(element, name, Some(item), record) {
                        Ok(x) => out.push(x),
                        Err(Fail::Partial(x, cause)) => {
                            out.push(x);
                            first(&mut bad, cause);
                        }
                        Err(e) => return Err(e),
                    }
                }
                match bad {
                    Some(cause) => Err(Fail::Partial(NVal::List(out), cause)),
                    None => Ok(NVal::List(out)),
                }
            }
            (Shape::Map(value), JVal::Object(members, _)) => {
                let mut out = Vec::with_capacity(members.len());
                let mut bad = None;
                for (k, item) in members {
                    match self.value(value, k, Some(item), record) {
                        Ok(x) => out.push((k.clone(), x)),
                        Err(Fail::Partial(x, cause)) => {
                            out.push((k.clone(), x));
                            first(&mut bad, cause);
                        }
                        Err(e) => return Err(e),
                    }
                }
                match bad {
                    Some(cause) => Err(Fail::Partial(NVal::Map(out), cause)),
                    None => Ok(NVal::Map(out)),
                }
            }
            // The wrong JSON kind for a struct, array or map.
            _ => Err(Fail::Whole(None)),
        }
    }

    /// An object's members as a struct of `fields`: a member that fails
    /// outright is null, and the struct is then partial.
    pub(crate) fn object(
        &self,
        fields: &[(String, Shape)],
        members: &[(String, JVal)],
        record: &str,
    ) -> Result<NVal, Fail> {
        let mut out = vec![NVal::Null; fields.len()];
        let mut bad = None;
        for (k, v) in members {
            let Some(i) = fields.iter().position(|(n, _)| n == k) else {
                continue;
            };
            match self.value(&fields[i].1, k, Some(v), record) {
                Ok(x) => out[i] = x,
                Err(Fail::Partial(x, cause)) => {
                    out[i] = x;
                    first(&mut bad, cause);
                }
                Err(Fail::Whole(cause)) => {
                    out[i] = NVal::Null;
                    first(&mut bad, cause);
                }
                Err(e @ Fail::Fatal(_)) => return Err(e),
            }
        }
        match bad {
            Some(cause) => Err(Fail::Partial(NVal::Struct(out), cause)),
            None => Ok(NVal::Struct(out)),
        }
    }
}

/// An approximate size of a value, for the decoder's memory account.
pub(crate) fn size_of(v: &NVal) -> usize {
    match v {
        NVal::Null => 1,
        NVal::Leaf(Val::Text(s)) => s.len() + 16,
        NVal::Leaf(Val::Bytes(b)) => b.len() + 16,
        NVal::Leaf(_) => 16,
        NVal::Struct(vs) | NVal::List(vs) => 24 + vs.iter().map(size_of).sum::<usize>(),
        NVal::Map(kv) => {
            24 + kv
                .iter()
                .map(|(k, v)| k.len() + 16 + size_of(v))
                .sum::<usize>()
        }
    }
}

/// The Arrow array of `dt` holding `vals` (one per row).
///
/// # Errors
/// An internal error when Arrow refuses the arrays (a schema the shape
/// check let through).
pub(crate) fn build(dt: &DataType, vals: Vec<NVal>) -> Result<ArrayRef, KrishivError> {
    let rows = vals.len();
    if rows == 0 {
        return Ok(new_null_array(dt, 0));
    }
    let arrow = |e: arrow_schema::ArrowError| internal(format!("a JSON {dt} column: {e}"));
    Ok(match dt {
        DataType::Struct(fields) => {
            let valid: Vec<bool> = vals.iter().map(|v| matches!(v, NVal::Struct(_))).collect();
            let mut children: Vec<Vec<NVal>> = (0..fields.len())
                .map(|_| Vec::with_capacity(rows))
                .collect();
            for v in vals {
                match v {
                    NVal::Struct(parts) => {
                        for (child, part) in children.iter_mut().zip(parts) {
                            child.push(part);
                        }
                    }
                    _ => children.iter_mut().for_each(|c| c.push(NVal::Null)),
                }
            }
            let arrays = fields
                .iter()
                .zip(children)
                .map(|(f, c)| build(f.data_type(), c))
                .collect::<Result<Vec<_>, _>>()?;
            Arc::new(
                StructArray::try_new_with_length(fields.clone(), arrays, nulls(&valid), rows)
                    .map_err(arrow)?,
            )
        }
        DataType::List(element) => {
            let (offsets, valid, items) = flatten(vals, |v| match v {
                NVal::List(items) => Some(items),
                _ => None,
            })?;
            let child = build(element.data_type(), items)?;
            Arc::new(
                ListArray::try_new(Arc::clone(element), offsets, child, nulls(&valid))
                    .map_err(arrow)?,
            )
        }
        DataType::Map(entries, sorted) => {
            let DataType::Struct(kv) = entries.data_type() else {
                return Err(internal("a map without key and value fields"));
            };
            let (Some(kf), Some(vf)) = (kv.first(), kv.get(1)) else {
                return Err(internal("a map without key and value fields"));
            };
            let (offsets, valid, pairs) = flatten(vals, |v| match v {
                NVal::Map(pairs) => Some(pairs),
                _ => None,
            })?;
            let (keys, values): (Vec<NVal>, Vec<NVal>) = pairs
                .into_iter()
                .map(|(k, v)| (NVal::Leaf(Val::Text(k)), v))
                .unzip();
            let n = keys.len();
            let entries_array = StructArray::try_new_with_length(
                Fields::from(vec![Arc::clone(kf), Arc::clone(vf)]),
                vec![build(kf.data_type(), keys)?, build(vf.data_type(), values)?],
                None,
                n,
            )
            .map_err(arrow)?;
            Arc::new(
                MapArray::try_new(
                    Arc::clone(entries),
                    offsets,
                    entries_array,
                    nulls(&valid),
                    *sorted,
                )
                .map_err(arrow)?,
            )
        }
        _ => {
            let mut c = Column::new(dt)?;
            for v in &vals {
                match v {
                    NVal::Leaf(x) => c.append(x),
                    _ => c.append_null(),
                }
            }
            c.finish()
        }
    })
}

/// List-like values flattened: offsets, validity per value, the items.
type Flat<T> = (OffsetBuffer<i32>, Vec<bool>, Vec<T>);

/// The offsets, validity and flattened items of list-like values.
fn flatten<T>(
    vals: Vec<NVal>,
    items_of: impl Fn(NVal) -> Option<Vec<T>>,
) -> Result<Flat<T>, KrishivError> {
    let mut lengths = Vec::with_capacity(vals.len());
    let mut valid = Vec::with_capacity(vals.len());
    let mut out = Vec::new();
    for v in vals {
        match items_of(v) {
            Some(items) => {
                lengths.push(items.len());
                valid.push(true);
                out.extend(items);
            }
            None => {
                lengths.push(0);
                valid.push(false);
            }
        }
    }
    if i32::try_from(out.len()).is_err() {
        return Err(internal(
            "a JSON array column over 2^31 elements in one batch",
        ));
    }
    Ok((OffsetBuffer::from_lengths(lengths), valid, out))
}

/// The null buffer of `valid`, or none when every row is valid.
fn nulls(valid: &[bool]) -> Option<NullBuffer> {
    (!valid.iter().all(|v| *v)).then(|| NullBuffer::from(valid.to_vec()))
}

// SPDX-License-Identifier: Apache-2.0
//! Casts with array, map and struct types, as Spark's `Cast` runs them:
//! element by element (arrays), key and value by key and value (maps) and
//! field by field, by position (structs), each through the ordinary cast of
//! its type, and to `STRING` as `[1, null]`, `{k -> v}` and `{1, null}`.
//!
//! A cast between types that differ only in field names or nullability
//! relabels the data. Values no row reads (under a NULL struct, or in the
//! range of a NULL array) are never cast, so they raise nothing.

use std::fmt::Write as _;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, GenericListArray, ListArray, MapArray, OffsetSizeTrait, RecordBatch,
    StringViewArray, StructArray, UInt64Array,
};
use arrow_buffer::{ArrowNativeType, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, Field, FieldRef, Fields};
use krishiv_types::{DataType, DayTimeField, MapType, StructField, YearMonthField};

use super::cast::{CastParts, Precision, apply};
use super::{PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

/// Whether `t` is the carrier of an array, map or struct.
pub(super) fn is_nested(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::List(_) | ArrowType::LargeList(_) | ArrowType::Map(..) | ArrowType::Struct(_)
    )
}

/// `a` through `p`, where the source or the target is nested.
pub(super) fn cast(p: &CastParts, a: &ArrayRef) -> Result<ArrayRef> {
    if a.data_type() == &p.to && p.precision.is_empty() {
        return Ok(Arc::clone(a));
    }
    let refused = || unsupported(&format!("CAST from {} to {}", a.data_type(), p.to));
    match (a.data_type(), &p.to) {
        (ArrowType::List(_), ArrowType::List(to)) => list::<i32>(p, a, to),
        (ArrowType::LargeList(_), ArrowType::List(to)) => list::<i64>(p, a, to),
        (ArrowType::Map(..), ArrowType::Map(to, sorted)) => map(p, a, to, *sorted),
        (ArrowType::Struct(_), ArrowType::Struct(to)) => {
            let s = a.as_struct();
            if s.num_columns() != to.len() {
                return Err(refused());
            }
            let children = s
                .columns()
                .iter()
                .zip(to.iter())
                .enumerate()
                .map(|(i, (c, f))| {
                    apply(
                        &p.child(f.data_type(), p.precision.field(i)),
                        &masked(c, s.nulls())?,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let out = StructArray::try_new(to.clone(), children, s.nulls().cloned())
                .map_err(|e| internal(&e.to_string()))?;
            Ok(Arc::new(out))
        }
        (from, ArrowType::Utf8View | ArrowType::Utf8) if is_nested(from) => {
            let texts = texts(p, a)?;
            let out: ArrayRef = Arc::new(StringViewArray::from_iter(texts));
            if p.to == ArrowType::Utf8 {
                super::strings::from_view(&out)
            } else {
                Ok(out)
            }
        }
        _ => Err(refused()),
    }
}

/// `e`, of the nested type `from`, with every interval inside replaced by the
/// string `CAST(.. AS STRING)` writes for it. The Arrow carriers have no
/// fields, so this has to read them from the plan's type.
pub(super) fn interval_leaves(e: Arc<dyn PhysExpr>, from: DataType) -> Arc<dyn PhysExpr> {
    let to = leaves_type(&from, e.data_type());
    Arc::new(IntervalLeaves { e, from, to })
}

#[derive(Debug)]
struct IntervalLeaves {
    e: Arc<dyn PhysExpr>,
    from: DataType,
    to: ArrowType,
}

impl PhysExpr for IntervalLeaves {
    fn data_type(&self) -> &ArrowType {
        &self.to
    }

    fn is_constant(&self) -> bool {
        self.e.is_constant()
    }

    fn nullable(&self) -> bool {
        self.e.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.e.eval(batch, partition)?;
        let out = leaves(&self.from, v.array())?;
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}

/// `e`, already cast to `to`, with each interval cut to its end field. Both
/// families keep one carrier whatever their fields, so the cast itself leaves
/// the value as it was.
pub(super) fn narrowed(e: Arc<dyn PhysExpr>, to: &DataType) -> Arc<dyn PhysExpr> {
    if has_coarse_interval(to) {
        Arc::new(Narrowed { e, to: to.clone() })
    } else {
        e
    }
}

/// Whether `t` holds an interval that ends above its family's finest field.
fn has_coarse_interval(t: &DataType) -> bool {
    match t {
        DataType::YearMonthInterval { end, .. } => *end != YearMonthField::Month,
        DataType::DayTimeInterval { end, .. } => *end != DayTimeField::Second,
        DataType::Array(a) => has_coarse_interval(a.element_type()),
        DataType::Map(m) => {
            has_coarse_interval(m.key_type()) || has_coarse_interval(m.value_type())
        }
        DataType::Struct(s) => s
            .fields()
            .iter()
            .any(|f| has_coarse_interval(f.data_type())),
        _ => false,
    }
}

#[derive(Debug)]
struct Narrowed {
    e: Arc<dyn PhysExpr>,
    to: DataType,
}

impl PhysExpr for Narrowed {
    fn data_type(&self) -> &ArrowType {
        self.e.data_type()
    }

    fn is_constant(&self) -> bool {
        self.e.is_constant()
    }

    fn nullable(&self) -> bool {
        self.e.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.e.eval(batch, partition)?;
        let out = map_leaves(&self.to, v.array(), &super::cast::truncate_interval)?;
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}

/// `field`, carrying `t`.
fn retyped(field: &Field, t: ArrowType) -> Field {
    field.clone().with_data_type(t)
}

/// The fields of a map's entries as a key and a value type.
fn entry_types(from: &MapType) -> [&DataType; 2] {
    [from.key_type(), from.value_type()]
}

/// The carrier of `t`, of the Spark type `from`, once [`leaves`] has run.
fn leaves_type(from: &DataType, t: &ArrowType) -> ArrowType {
    let fields = |types: Vec<&DataType>, fields: &Fields| -> Fields {
        types
            .into_iter()
            .zip(fields.iter())
            .map(|(x, f)| Arc::new(retyped(f, leaves_type(x, f.data_type()))))
            .collect()
    };
    match (from, t) {
        (
            DataType::YearMonthInterval { .. }
            | DataType::DayTimeInterval { .. }
            | DataType::CalendarInterval,
            _,
        ) => ArrowType::Utf8View,
        (DataType::Array(x), ArrowType::List(f)) => ArrowType::List(Arc::new(retyped(
            f,
            leaves_type(x.element_type(), f.data_type()),
        ))),
        (DataType::Array(x), ArrowType::LargeList(f)) => ArrowType::LargeList(Arc::new(retyped(
            f,
            leaves_type(x.element_type(), f.data_type()),
        ))),
        (DataType::Map(m), ArrowType::Map(f, sorted)) => match f.data_type() {
            ArrowType::Struct(kv) => ArrowType::Map(
                Arc::new(retyped(
                    f,
                    ArrowType::Struct(fields(entry_types(m).to_vec(), kv)),
                )),
                *sorted,
            ),
            _ => t.clone(),
        },
        (DataType::Struct(s), ArrowType::Struct(fs)) => {
            let types = s.fields().iter().map(StructField::data_type).collect();
            ArrowType::Struct(fields(types, fs))
        }
        _ => t.clone(),
    }
}

/// What [`map_leaves`] does to an interval column of a Spark type.
type Leaf<'a> = dyn Fn(&DataType, &dyn Array) -> Result<ArrayRef> + 'a;

/// `a`, of the Spark type `from`, with its intervals as strings.
fn leaves(from: &DataType, a: &ArrayRef) -> Result<ArrayRef> {
    map_leaves(from, a, &super::cast::render_interval)
}

/// `a`, of the Spark type `from`, with `leaf` applied to its intervals.
fn map_leaves(from: &DataType, a: &ArrayRef, leaf: &Leaf<'_>) -> Result<ArrayRef> {
    let bad = |e: arrow_schema::ArrowError| internal(&e.to_string());
    fn structure(types: &[&DataType], s: &StructArray, leaf: &Leaf<'_>) -> Result<StructArray> {
        let columns = types
            .iter()
            .zip(s.columns())
            .map(|(t, c)| map_leaves(t, c, leaf))
            .collect::<Result<Vec<_>>>()?;
        let fields: Fields = s
            .fields()
            .iter()
            .zip(&columns)
            .map(|(f, c)| Arc::new(retyped(f, c.data_type().clone())))
            .collect();
        StructArray::try_new(fields, columns, s.nulls().cloned())
            .map_err(|e| internal(&e.to_string()))
    }
    fn list<O: OffsetSizeTrait>(x: &DataType, a: &ArrayRef, leaf: &Leaf<'_>) -> Result<ArrayRef> {
        let l = a.as_list::<O>();
        let values = map_leaves(x, l.values(), leaf)?;
        let (ArrowType::List(f) | ArrowType::LargeList(f)) = a.data_type() else {
            return Err(internal("a list without its field"));
        };
        let field = Arc::new(retyped(f, values.data_type().clone()));
        let out =
            GenericListArray::<O>::try_new(field, l.offsets().clone(), values, l.nulls().cloned())
                .map_err(|e| internal(&e.to_string()))?;
        Ok(Arc::new(out))
    }
    Ok(match (from, a.data_type()) {
        (
            DataType::YearMonthInterval { .. }
            | DataType::DayTimeInterval { .. }
            | DataType::CalendarInterval,
            _,
        ) => leaf(from, a.as_ref())?,
        (DataType::Array(x), ArrowType::List(_)) => list::<i32>(x.element_type(), a, leaf)?,
        (DataType::Array(x), ArrowType::LargeList(_)) => list::<i64>(x.element_type(), a, leaf)?,
        (DataType::Map(m), ArrowType::Map(f, sorted)) => {
            let map = a.as_map();
            let entries = structure(&entry_types(m), map.entries(), leaf)?;
            let field = Arc::new(retyped(f, entries.data_type().clone()));
            Arc::new(
                MapArray::try_new(
                    field,
                    map.offsets().clone(),
                    entries,
                    map.nulls().cloned(),
                    *sorted,
                )
                .map_err(bad)?,
            )
        }
        (DataType::Struct(s), ArrowType::Struct(_)) => {
            let types: Vec<&DataType> = s.fields().iter().map(StructField::data_type).collect();
            Arc::new(structure(&types, a.as_struct(), leaf)?)
        }
        _ => Arc::clone(a),
    })
}

/// `a` with its rows also NULL where `parent` is: the values a NULL parent
/// row holds are never read, so they must not be cast.
fn masked(a: &ArrayRef, parent: Option<&NullBuffer>) -> Result<ArrayRef> {
    let Some(parent) = parent.filter(|n| n.null_count() > 0) else {
        return Ok(Arc::clone(a));
    };
    let nulls = NullBuffer::union(a.nulls(), Some(parent));
    let data = a
        .to_data()
        .into_builder()
        .nulls(nulls)
        .build()
        .map_err(|e| internal(&e.to_string()))?;
    Ok(arrow_array::make_array(data))
}

/// The 32-bit offsets and values of a list or map whose rows read only
/// their own values: a NULL row with a non-empty range is dropped from the
/// values (they are taken again without it).
fn compact<O: OffsetSizeTrait>(
    offsets: &[O],
    nulls: Option<&NullBuffer>,
    values: &ArrayRef,
) -> Result<(OffsetBuffer<i32>, ArrayRef)> {
    let at = |i: usize| offsets.get(i).map_or(0, |o| o.as_usize());
    let rows = offsets.len().saturating_sub(1);
    let dirty = nulls.is_some_and(|n| (0..rows).any(|i| n.is_null(i) && at(i + 1) > at(i)));
    let too_big = || unsupported("an array batch with more than 2^31 values");
    if !dirty {
        let first = at(0);
        let narrow = offsets
            .iter()
            .map(|o| i32::try_from(o.as_usize() - first).map_err(|_| too_big()))
            .collect::<Result<Vec<_>>>()?;
        let len = at(rows) - first;
        return Ok((
            OffsetBuffer::new(ScalarBuffer::from(narrow)),
            values.slice(first, len),
        ));
    }
    let mut indices = Vec::new();
    let mut narrow = Vec::with_capacity(offsets.len());
    narrow.push(0_i32);
    for i in 0..rows {
        if nulls.is_none_or(|n| n.is_valid(i)) {
            indices.extend((at(i)..at(i + 1)).map(|v| v as u64));
        }
        narrow.push(i32::try_from(indices.len()).map_err(|_| too_big())?);
    }
    let taken = arrow_select::take::take(values.as_ref(), &UInt64Array::from(indices), None)
        .map_err(|e| internal(&e.to_string()))?;
    Ok((OffsetBuffer::new(ScalarBuffer::from(narrow)), taken))
}

fn list<O: OffsetSizeTrait>(p: &CastParts, a: &ArrayRef, to: &FieldRef) -> Result<ArrayRef> {
    let l = a.as_list::<O>();
    let (offsets, values) = compact(l.value_offsets(), l.nulls(), l.values())?;
    let values = apply(&p.child(to.data_type(), p.precision.element()), &values)?;
    let out = ListArray::try_new(Arc::clone(to), offsets, values, l.nulls().cloned())
        .map_err(|e| internal(&e.to_string()))?;
    Ok(Arc::new(out))
}

fn map(p: &CastParts, a: &ArrayRef, to: &FieldRef, sorted: bool) -> Result<ArrayRef> {
    let m = a.as_map();
    let entries: ArrayRef = Arc::new(m.entries().clone());
    let (offsets, entries) = compact(m.value_offsets(), m.nulls(), &entries)?;
    let entries = apply(&p.child(to.data_type(), p.precision.entries()), &entries)?;
    let out = MapArray::try_new(
        Arc::clone(to),
        offsets,
        entries.as_struct().clone(),
        m.nulls().cloned(),
        sorted,
    )
    .map_err(|e| internal(&e.to_string()))?;
    Ok(Arc::new(out))
}

/// Each row of `a` as Spark's `CAST(.. AS STRING)` writes it; `None` for a
/// NULL row.
fn texts(p: &CastParts, a: &ArrayRef) -> Result<Vec<Option<String>>> {
    let rows = |offsets: Vec<(usize, usize)>, items: Vec<Option<String>>, open, close| {
        offsets
            .into_iter()
            .enumerate()
            .map(|(i, (from, to))| {
                a.is_valid(i).then(|| {
                    let mut s = String::from(open);
                    for (k, item) in items.get(from..to).unwrap_or_default().iter().enumerate() {
                        if k > 0 {
                            s.push_str(", ");
                        }
                        s.push_str(item.as_deref().unwrap_or("null"));
                    }
                    s.push_str(close);
                    s
                })
            })
            .collect()
    };
    let ranges = |offsets: Vec<usize>| -> Vec<(usize, usize)> {
        offsets.windows(2).map(|w| (w[0], w[1])).collect()
    };
    Ok(match a.data_type() {
        ArrowType::List(_) => {
            let l = a.as_list::<i32>();
            let items = texts(p, l.values())?;
            let offsets = l.value_offsets().iter().map(|o| o.as_usize()).collect();
            rows(ranges(offsets), items, "[", "]")
        }
        ArrowType::LargeList(_) => {
            let l = a.as_list::<i64>();
            let items = texts(p, l.values())?;
            let offsets = l.value_offsets().iter().map(|o| o.as_usize()).collect();
            rows(ranges(offsets), items, "[", "]")
        }
        ArrowType::Map(..) => {
            let m = a.as_map();
            let keys = texts(p, m.keys())?;
            let values = texts(p, m.values())?;
            let pairs = keys
                .into_iter()
                .zip(values)
                .map(|(k, v)| {
                    let mut s = k.unwrap_or_else(|| "null".to_owned());
                    let _ = write!(s, " -> {}", v.as_deref().unwrap_or("null"));
                    Some(s)
                })
                .collect();
            let offsets = m.value_offsets().iter().map(|o| o.as_usize()).collect();
            rows(ranges(offsets), pairs, "{", "}")
        }
        ArrowType::Struct(_) => {
            let s = a.as_struct();
            let columns = s
                .columns()
                .iter()
                .map(|c| texts(p, &masked(c, s.nulls())?))
                .collect::<Result<Vec<_>>>()?;
            let width = columns.len();
            let items = (0..s.len())
                .flat_map(|row| columns.iter().map(move |c| c.get(row).cloned().flatten()))
                .collect();
            let offsets = (0..=s.len()).map(|row| row * width).collect();
            rows(ranges(offsets), items, "{", "}")
        }
        _ => {
            let out = apply(&p.child(&ArrowType::Utf8View, Precision::None), a)?;
            out.as_string_view()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect()
        }
    })
}

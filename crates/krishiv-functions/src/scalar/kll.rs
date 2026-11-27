// SPDX-License-Identifier: Apache-2.0
//! The KLL scalars (`kll_sketch_to_string_*`, `get_n_*`, `merge_*`, `get_quantile_*`, `get_rank_*`), after Spark's
//! `kllExpressions.scala` (v4.3.0); see `registry/kll.rs` for the signatures. A buffer that is no sketch of the
//! function's type is `KLL_INVALID_INPUT_SKETCH_BUFFER`; a rank outside `[0, 1]` is
//! `KLL_SKETCH_INVALID_QUANTILE_RANGE`.

use std::sync::Arc;

use arrow_array::builder::{Float32Builder, Float64Builder, Int64Builder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, BinaryArray, Int64Array};
use arrow_schema::{DataType, Field};
use krishiv_common::error::KrishivError;

use super::args::{Bin, Inputs};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};
use crate::sketch::kll::{Kll, KllType, Quantile};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "kll_sketch_to_string_bigint",
    "kll_sketch_to_string_float",
    "kll_sketch_to_string_double",
    "kll_sketch_get_n_bigint",
    "kll_sketch_get_n_float",
    "kll_sketch_get_n_double",
    "kll_sketch_merge_bigint",
    "kll_sketch_merge_float",
    "kll_sketch_merge_double",
    "kll_sketch_get_quantile_bigint",
    "kll_sketch_get_quantile_float",
    "kll_sketch_get_quantile_double",
    "kll_sketch_get_rank_bigint",
    "kll_sketch_get_rank_float",
    "kll_sketch_get_rank_double",
];

fn kind(name: &str) -> KllType {
    if name.ends_with("_bigint") {
        KllType::Bigint
    } else if name.ends_with("_float") {
        KllType::Float
    } else {
        KllType::Double
    }
}

fn elem_type(t: KllType) -> DataType {
    match t {
        KllType::Bigint => DataType::Int64,
        KllType::Float => DataType::Float32,
        KllType::Double => DataType::Float64,
    }
}

fn item(t: KllType) -> Arc<Field> {
    Arc::new(Field::new("element", elem_type(t), false))
}

/// A quantile's value as the element type; an empty sketch gives NaN (floats) and is invalid for `BIGINT`.
fn push_quantile(
    b: &mut dyn QuantileSink,
    q: Option<Quantile>,
    name: &str,
) -> Result<(), KrishivError> {
    b.push(q, name)
}

trait QuantileSink {
    fn push(&mut self, q: Option<Quantile>, name: &str) -> Result<(), KrishivError>;
}

struct LongSink<'a>(&'a mut Int64Builder);
struct FloatSink<'a>(&'a mut Float32Builder);
struct DoubleSink<'a>(&'a mut Float64Builder);

fn empty_sketch(name: &str) -> KrishivError {
    krishiv_common::error::KllInvalidInputSketchBuffer {
        function: format!("`{name}`"),
    }
    .into()
}

impl QuantileSink for LongSink<'_> {
    fn push(&mut self, q: Option<Quantile>, name: &str) -> Result<(), KrishivError> {
        match q {
            Some(Quantile::Long(v)) => self.0.append_value(v),
            _ => return Err(empty_sketch(name)),
        }
        Ok(())
    }
}
impl QuantileSink for FloatSink<'_> {
    fn push(&mut self, q: Option<Quantile>, _: &str) -> Result<(), KrishivError> {
        self.0.append_value(match q {
            Some(Quantile::Float(v)) => v,
            _ => f32::NAN,
        });
        Ok(())
    }
}
impl QuantileSink for DoubleSink<'_> {
    fn push(&mut self, q: Option<Quantile>, _: &str) -> Result<(), KrishivError> {
        self.0.append_value(match q {
            Some(Quantile::Double(v)) => v,
            _ => f64::NAN,
        });
        Ok(())
    }
}

/// Row `i`'s sketch bytes; a literal is held once and read at every row (no per-row copy).
fn bytes(b: &Bin, i: usize) -> Option<&[u8]> {
    match b {
        Bin::Array(a) => a.is_valid(i).then(|| a.value(i)),
        Bin::Scalar(s) => s.as_deref(),
    }
}

fn decode(t: KllType, a: &Bin, i: usize, name: &str) -> Result<Option<Kll>, KrishivError> {
    bytes(a, i).map(|b| Kll::from_bytes(t, b, name)).transpose()
}

/// `true` when the row-`i` second argument (a rank or a value) is a plain `Float64`/the element type, or a list of it.
fn second_ok(t: KllType, quantile: bool, d: &DataType) -> bool {
    let want = if quantile {
        DataType::Float64
    } else {
        elem_type(t)
    };
    match d {
        DataType::List(f) => f.data_type() == &want,
        d => d == &want,
    }
}

pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let t = kind(listed);
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        (n, [b]) if n.starts_with("kll_sketch_to_string_") && super::args::is_binary(b) => {
            Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
                let a = x.bin(0)?;
                let mut out = StringBuilder::new();
                for i in 0..x.rows() {
                    match decode(t, &a, i, n)? {
                        Some(s) => out.append_value(s.summary()),
                        None => out.append_null(),
                    }
                }
                Ok(Arc::new(out.finish()) as ArrayRef)
            }))
        }
        (n, [b]) if n.starts_with("kll_sketch_get_n_") && super::args::is_binary(b) => {
            Ok(Binding::new(DataType::Int64, move |x: &Inputs<'_>| {
                let a = x.bin(0)?;
                let mut out = Vec::with_capacity(x.rows());
                for i in 0..x.rows() {
                    out.push(
                        decode(t, &a, i, n)?.map(|s| i64::try_from(s.n()).unwrap_or(i64::MAX)),
                    );
                }
                Ok(Arc::new(Int64Array::from(out)) as ArrayRef)
            }))
        }
        (n, [a, b])
            if n.starts_with("kll_sketch_merge_")
                && super::args::is_binary(a)
                && super::args::is_binary(b) =>
        {
            Ok(Binding::new(DataType::Binary, move |x: &Inputs<'_>| {
                let (l, r) = (x.bin(0)?, x.bin(1)?);
                let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(x.rows());
                for i in 0..x.rows() {
                    if bytes(&l, i).is_none() || bytes(&r, i).is_none() {
                        out.push(None);
                        continue;
                    }
                    let (Some(mut left), Some(right)) =
                        (decode(t, &l, i, n)?, decode(t, &r, i, n)?)
                    else {
                        out.push(None);
                        continue;
                    };
                    left.merge(&right, n)?;
                    out.push(Some(left.to_bytes()));
                }
                Ok(Arc::new(BinaryArray::from_iter(out)) as ArrayRef)
            }))
        }
        (n, [b, r])
            if n.starts_with("kll_sketch_get_quantile_")
                && super::args::is_binary(b)
                && second_ok(t, true, r) =>
        {
            let is_list = matches!(r, DataType::List(_));
            let out_ty = if is_list {
                DataType::List(item(t))
            } else {
                elem_type(t)
            };
            Ok(Binding::new(out_ty, move |x: &Inputs<'_>| {
                let sketches = x.bin(0)?;
                let ranks = x.column(1)?;
                quantiles(t, n, x.rows(), &sketches, &ranks, is_list)
            }))
        }
        (n, [b, r])
            if n.starts_with("kll_sketch_get_rank_")
                && super::args::is_binary(b)
                && second_ok(t, false, r) =>
        {
            let is_list = matches!(r, DataType::List(_));
            let out_ty = if is_list {
                DataType::List(Arc::new(Field::new("element", DataType::Float64, false)))
            } else {
                DataType::Float64
            };
            Ok(Binding::new(out_ty, move |x: &Inputs<'_>| {
                let sketches = x.bin(0)?;
                let values = x.column(1)?;
                ranks(t, n, x.rows(), &sketches, &values, is_list)
            }))
        }
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}

fn quantiles(
    t: KllType,
    name: &str,
    rows: usize,
    sketches: &Bin,
    ranks: &ArrayRef,
    is_list: bool,
) -> Result<ArrayRef, KrishivError> {
    let mut lb = Int64Builder::new();
    let mut fb = Float32Builder::new();
    let mut db = Float64Builder::new();
    let mut lists: Vec<Option<usize>> = Vec::new(); // element count per row, None = NULL
    for i in 0..rows {
        let sketch = decode(t, sketches, i, name)?;
        let row_ranks: Option<Vec<f64>> = if is_list {
            let l = ranks.as_list::<i32>();
            // A NULL element has no defined rank (its slot holds an arbitrary value), so the row is NULL.
            l.is_valid(i)
                .then(|| l.value(i))
                .filter(|v| v.null_count() == 0)
                .map(|v| v.as_primitive::<Float64Type>().values().to_vec())
        } else {
            let p = ranks.as_primitive::<Float64Type>();
            p.is_valid(i).then(|| vec![p.value(i)])
        };
        let (Some(sketch), Some(row_ranks)) = (sketch, row_ranks) else {
            lists.push(None);
            if !is_list {
                match t {
                    KllType::Bigint => lb.append_null(),
                    KllType::Float => fb.append_null(),
                    KllType::Double => db.append_null(),
                }
            }
            continue;
        };
        for r in &row_ranks {
            let q = sketch.quantile(*r, name)?;
            match t {
                KllType::Bigint => push_quantile(&mut LongSink(&mut lb), q, name)?,
                KllType::Float => push_quantile(&mut FloatSink(&mut fb), q, name)?,
                KllType::Double => push_quantile(&mut DoubleSink(&mut db), q, name)?,
            }
        }
        lists.push(Some(row_ranks.len()));
    }
    let values: ArrayRef = match t {
        KllType::Bigint => Arc::new(lb.finish()),
        KllType::Float => Arc::new(fb.finish()),
        KllType::Double => Arc::new(db.finish()),
    };
    if !is_list {
        return Ok(values);
    }
    list_of(values, &lists, item(t))
}

fn list_of(
    values: ArrayRef,
    lists: &[Option<usize>],
    field: Arc<Field>,
) -> Result<ArrayRef, KrishivError> {
    let mut offsets = vec![0_i32];
    let mut validity = Vec::with_capacity(lists.len());
    let mut at = 0_i32;
    for n in lists {
        if let Some(n) = n {
            at += i32::try_from(*n).unwrap_or(i32::MAX);
        }
        offsets.push(at);
        validity.push(n.is_some());
    }
    let list = arrow_array::ListArray::try_new(
        field,
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        values,
        Some(validity.into()),
    )
    .map_err(|e| super::args::internal(e.to_string()))?;
    Ok(Arc::new(list))
}

fn ranks(
    t: KllType,
    name: &str,
    rows: usize,
    sketches: &Bin,
    values: &ArrayRef,
    is_list: bool,
) -> Result<ArrayRef, KrishivError> {
    let rank_of = |s: &Kll, v: &ArrayRef, j: usize| -> f64 {
        let r = match t {
            KllType::Bigint => s.rank(0.0, v.as_primitive::<Int64Type>().value(j)),
            KllType::Float => s.rank(f64::from(v.as_primitive::<Float32Type>().value(j)), 0),
            KllType::Double => s.rank(v.as_primitive::<Float64Type>().value(j), 0),
        };
        r.unwrap_or(f64::NAN)
    };
    let mut out = Float64Builder::new();
    let mut lists: Vec<Option<usize>> = Vec::new();
    for i in 0..rows {
        let Some(sketch) = decode(t, sketches, i, name)? else {
            lists.push(None);
            if !is_list {
                out.append_null();
            }
            continue;
        };
        if is_list {
            let l = values.as_list::<i32>();
            if !l.is_valid(i) {
                lists.push(None);
                continue;
            }
            let v = l.value(i);
            if v.null_count() > 0 {
                lists.push(None);
                continue;
            }
            for j in 0..v.len() {
                out.append_value(rank_of(&sketch, &v, j));
            }
            lists.push(Some(v.len()));
        } else if values.is_valid(i) {
            out.append_value(rank_of(&sketch, values, i));
        } else {
            out.append_null();
        }
    }
    let flat: ArrayRef = Arc::new(out.finish());
    if !is_list {
        return Ok(flat);
    }
    list_of(
        flat,
        &lists,
        Arc::new(Field::new("element", DataType::Float64, false)),
    )
}

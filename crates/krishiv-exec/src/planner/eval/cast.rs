// SPDX-License-Identifier: Apache-2.0
//! `CAST` / `TRY_CAST` through the kernels' cast.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::DataType as ArrowType;
use krishiv_kernels::cast::{CastMode as KernelMode, CastOptions, FixedOffset, cast};
use krishiv_plan::CastMode;
use krishiv_types::DataType;

use super::nested_cast;
use super::strings::{from_view, to_view};
use super::{EvalEnv, Input, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

/// The canonical Arrow carrier of `t` when several Arrow types store one
/// Spark `STRING` or `BINARY` (`Utf8`/`LargeUtf8`/`Utf8View`, or
/// `Binary`/`LargeBinary`/`BinaryView`): the type
/// [`DataType::to_arrow`](krishiv_types::DataType::to_arrow) maps the Spark
/// type to. `None` for every other type.
///
/// This is the one place that decides the physical type of an expression
/// whose operands are the same Spark type in different Arrow encodings:
/// comparisons, `IN`/`BETWEEN`, `CASE` branches and join keys all coerce to
/// this carrier, so the kernel always sees one encoding. Where one side
/// already is the carrier it is kept (no copy for the view side); the other
/// side goes through a cast.
pub(crate) fn canonical_carrier(t: &ArrowType) -> Option<ArrowType> {
    match t {
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => Some(ArrowType::Utf8View),
        ArrowType::Binary | ArrowType::LargeBinary | ArrowType::BinaryView => {
            Some(ArrowType::BinaryView)
        }
        _ => None,
    }
}

/// `e` with its values in `to`, for an encoding-only change of carrier
/// (both the same Spark `STRING` or `BINARY`): `e` itself when it already
/// is `to`, else a cast. The cast keeps every value (an encoding change
/// cannot fail on the values; a batch over 2 GiB in a 32-bit carrier is
/// `UNSUPPORTED`, as for the string kernels).
///
/// # Errors
/// `INTERNAL_ERROR` when `to` is not the canonical carrier of `e`'s type:
/// this wraps encoding changes only, never Spark-visible casts.
pub(crate) fn coerce_to(
    e: Arc<dyn PhysExpr>,
    to: &ArrowType,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    if e.data_type() == to {
        return Ok(e);
    }
    if canonical_carrier(e.data_type()) != Some(to.clone()) {
        return Err(internal(&format!(
            "a cast of {} to {to} is not an encoding change",
            e.data_type()
        )));
    }
    let mode = if env.ansi {
        CastMode::Ansi
    } else {
        CastMode::Legacy
    };
    let parts = parts(e.data_type(), to.clone(), mode, env)?;
    let nullable = e.nullable() || parts.mode != KernelMode::Ansi;
    Ok(Arc::new(Cast { e, parts, nullable }))
}

/// `(l, r)` with operands of one Spark `STRING` or `BINARY` in different
/// Arrow encodings both in the canonical carrier: sides already there are
/// kept, the others go through [`coerce_to`]. Operands of different Spark
/// types (or one untyped `NULL`) are returned unchanged, for the caller to
/// accept or reject as before.
pub(crate) fn unify(
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    env: &EvalEnv,
) -> Result<(Arc<dyn PhysExpr>, Arc<dyn PhysExpr>)> {
    let (lt, rt) = (l.data_type().clone(), r.data_type().clone());
    if lt == rt {
        return Ok((l, r));
    }
    match (canonical_carrier(&lt), canonical_carrier(&rt)) {
        (Some(a), Some(b)) if a == b => Ok((coerce_to(l, &a, env)?, coerce_to(r, &a, env)?)),
        _ => Ok((l, r)),
    }
}

#[derive(Debug)]
struct Cast {
    e: Arc<dyn PhysExpr>,
    parts: CastParts,
    nullable: bool,
}

/// Whether a cast between `from` and `to` reads the session time zone.
fn needs_zone(from: &ArrowType, to: &ArrowType) -> bool {
    let zoned = |t: &ArrowType| matches!(t, ArrowType::Timestamp(_, Some(_)));
    (zoned(from) && !matches!(to, ArrowType::Int64 | ArrowType::Timestamp(..)))
        || (zoned(to) && !matches!(from, ArrowType::Int64 | ArrowType::Timestamp(..)))
}

/// A cast's plan data, without its child: what [`NullIf`](super::cond::NullIf)
/// needs to cast an already-evaluated array.
#[derive(Debug, Clone)]
pub(super) struct CastParts {
    /// The result type.
    pub(super) to: ArrowType,
    /// The kernels' target: `Utf8` where the result is a `Utf8View` string.
    pub(super) kernel_to: ArrowType,
    pub(super) mode: KernelMode,
    pub(super) zone: FixedOffset,
}

/// The cast of `from` to `to` in `mode`, with the session zone of `env`.
pub(super) fn parts(
    from: &ArrowType,
    to: ArrowType,
    mode: CastMode,
    env: &super::EvalEnv,
) -> Result<CastParts> {
    let mode = match mode {
        CastMode::Ansi => KernelMode::Ansi,
        CastMode::Legacy => KernelMode::Legacy,
        CastMode::Try => KernelMode::Try,
        _ => return Err(internal("a cast mode the analyzer did not pin")),
    };
    let zone = match (needs_zone(from, &to), env.zone) {
        (_, Some(offset)) => FixedOffset(offset),
        (false, None) => FixedOffset(0),
        (true, None) => {
            return Err(unsupported(
                "casts that read a session time zone other than a fixed offset",
            ));
        }
    };
    let kernel_to = match &to {
        ArrowType::Utf8View => ArrowType::Utf8,
        ArrowType::BinaryView => ArrowType::Binary,
        other => other.clone(),
    };
    Ok(CastParts {
        to,
        kernel_to,
        mode,
        zone,
    })
}

impl CastParts {
    /// The cast of an element, value or field to `to`, in the same mode
    /// and zone.
    pub(super) fn child(&self, to: &ArrowType) -> Self {
        let kernel_to = match to {
            ArrowType::Utf8View => ArrowType::Utf8,
            ArrowType::BinaryView => ArrowType::Binary,
            other => other.clone(),
        };
        Self {
            to: to.clone(),
            kernel_to,
            mode: self.mode,
            zone: self.zone,
        }
    }
}

/// `a` through `p`.
pub(super) fn apply(p: &CastParts, a: &ArrayRef) -> Result<ArrayRef> {
    // Every value of a `void` array is NULL, of any type.
    if a.data_type() == &ArrowType::Null {
        return Ok(arrow_array::new_null_array(&p.to, a.len()));
    }
    if nested_cast::is_nested(a.data_type()) || nested_cast::is_nested(&p.to) {
        return nested_cast::cast(p, a);
    }
    let input = from_view(a)?;
    let opts = CastOptions {
        mode: p.mode,
        time_zone: &p.zone,
    };
    let out = cast(input.as_ref(), &p.kernel_to, &opts)?;
    let out: ArrayRef = match (out.data_type(), &p.to) {
        (a, b) if a == b => out,
        (ArrowType::Timestamp(..), ArrowType::Timestamp(..)) => {
            let data = out.to_data().into_builder().data_type(p.to.clone());
            arrow_array::make_array(data.build().map_err(|e| internal(&e.to_string()))?)
        }
        _ => to_view(&out)?,
    };
    if out.data_type() != &p.to {
        return Err(internal(&format!(
            "a cast to {} produced {}",
            p.to,
            out.data_type()
        )));
    }
    Ok(out)
}

pub(super) fn compile(
    e: Arc<dyn PhysExpr>,
    to: &DataType,
    mode: CastMode,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    let parts = parts(e.data_type(), to.to_arrow(), mode, input.env)?;
    // Every cast to a narrower or textual type can produce NULL in legacy
    // and `TRY_CAST` modes; ANSI keeps the input's nullability.
    let nullable = e.nullable() || parts.mode != KernelMode::Ansi;
    Ok(Arc::new(Cast { e, parts, nullable }))
}

impl PhysExpr for Cast {
    fn data_type(&self) -> &ArrowType {
        &self.parts.to
    }

    fn is_constant(&self) -> bool {
        self.e.is_constant()
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.e.eval(batch, partition)?;
        let out = apply(&self.parts, v.array())?;
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{
        ArrayRef, BinaryArray, Int32Array, LargeStringArray, RecordBatch, StringArray,
        StringViewArray,
    };
    use arrow_schema::{DataType as ArrowType, Field, Schema};
    use krishiv_common::config::Config;

    use super::super::{EvalEnv, PhysExpr, Value};
    use super::{canonical_carrier, coerce_to, unify};
    use crate::error::Result;

    /// An input column.
    #[derive(Debug)]
    struct Col(usize, ArrowType);

    impl PhysExpr for Col {
        fn data_type(&self) -> &ArrowType {
            &self.1
        }

        fn nullable(&self) -> bool {
            true
        }

        fn eval(&self, batch: &RecordBatch, _partition: u32) -> Result<Value> {
            Ok(Value::Array(Arc::clone(batch.column(self.0))))
        }
    }

    fn env() -> EvalEnv {
        EvalEnv::new(&Config::new(), Arc::from("test"))
    }

    /// A batch with one column per string and binary carrier.
    fn batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("u", ArrowType::Utf8, true),
            Field::new("l", ArrowType::LargeUtf8, true),
            Field::new("v", ArrowType::Utf8View, true),
            Field::new("b", ArrowType::Binary, true),
            Field::new("bv", ArrowType::BinaryView, true),
            Field::new("i", ArrowType::Int32, true),
        ]));
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![Some("a"), None])) as ArrayRef,
                Arc::new(LargeStringArray::from(vec![Some("b"), None])),
                Arc::new(StringViewArray::from(vec![Some("c"), None])),
                Arc::new(BinaryArray::from(vec![Some(&b"d"[..]), None])),
                Arc::new(arrow_array::BinaryViewArray::from(vec![
                    Some(&b"e"[..]),
                    None,
                ])),
                Arc::new(Int32Array::from(vec![Some(1), None])),
            ],
        )
        .unwrap()
    }

    fn col(index: usize, data_type: ArrowType) -> Arc<dyn PhysExpr> {
        Arc::new(Col(index, data_type))
    }

    /// Every string carrier maps to `Utf8View`, every binary one to
    /// `BinaryView`, and nothing else maps anywhere.
    #[test]
    fn canonical_carriers() {
        use ArrowType as T;
        for (t, want) in [
            (T::Utf8, Some(T::Utf8View)),
            (T::LargeUtf8, Some(T::Utf8View)),
            (T::Utf8View, Some(T::Utf8View)),
            (T::Binary, Some(T::BinaryView)),
            (T::LargeBinary, Some(T::BinaryView)),
            (T::BinaryView, Some(T::BinaryView)),
            (T::Int32, None),
            (T::Boolean, None),
            (T::Null, None),
            // A dictionary is not an encoding of the value type here.
            (T::Dictionary(Box::new(T::Int32), Box::new(T::Utf8)), None),
        ] {
            assert_eq!(canonical_carrier(&t), want, "{t}");
        }
    }

    /// `coerce_to` keeps the carrier (no copy) and converts the other
    /// carriers with their values kept; anything else is an engine bug.
    #[test]
    fn coerce_to_converts_encodings_only() {
        let env = env();
        let batch = batch();
        let view = col(2, ArrowType::Utf8View);
        assert!(Arc::ptr_eq(
            &coerce_to(Arc::clone(&view), &ArrowType::Utf8View, &env).unwrap(),
            &view
        ));
        for (index, from, values) in [
            (0, ArrowType::Utf8, vec![Some("a"), None]),
            (1, ArrowType::LargeUtf8, vec![Some("b"), None]),
        ] {
            let e = coerce_to(col(index, from), &ArrowType::Utf8View, &env).unwrap();
            assert_eq!(e.data_type(), &ArrowType::Utf8View);
            let got = e.eval(&batch, 0).unwrap().into_array(2).unwrap();
            assert_eq!(got.as_string_view().iter().collect::<Vec<_>>(), values);
        }
        let e = coerce_to(col(3, ArrowType::Binary), &ArrowType::BinaryView, &env).unwrap();
        let got = e.eval(&batch, 0).unwrap().into_array(2).unwrap();
        assert_eq!(
            got.as_binary_view().iter().collect::<Vec<_>>(),
            vec![Some(&b"d"[..]), None]
        );
        // A different Spark type is not an encoding change.
        assert!(coerce_to(col(5, ArrowType::Int32), &ArrowType::Utf8View, &env).is_err());
        assert!(coerce_to(col(0, ArrowType::Utf8), &ArrowType::BinaryView, &env).is_err());
    }

    /// `coerce_to` without ANSI keeps the values too: the encoding change
    /// itself never depends on the session mode.
    #[test]
    fn coerce_to_in_legacy_mode_keeps_values() {
        let mut config = Config::new();
        config.set("spark.sql.ansi.enabled", "false").unwrap();
        let env = EvalEnv::new(&config, Arc::from("test"));
        let batch = batch();
        let e = coerce_to(col(0, ArrowType::Utf8), &ArrowType::Utf8View, &env).unwrap();
        assert_eq!(e.data_type(), &ArrowType::Utf8View);
        let got = e.eval(&batch, 0).unwrap().into_array(2).unwrap();
        assert_eq!(
            got.as_string_view().iter().collect::<Vec<_>>(),
            vec![Some("a"), None]
        );
    }

    /// `unify` puts one Spark type's two encodings in the canonical
    /// carrier and leaves everything else for the caller to reject.
    #[test]
    fn unify_meets_mixed_encodings_in_the_view() {
        let env = env();
        let batch = batch();
        let (l, r) = unify(col(0, ArrowType::Utf8), col(2, ArrowType::Utf8View), &env).unwrap();
        assert_eq!(l.data_type(), &ArrowType::Utf8View);
        assert_eq!(r.data_type(), &ArrowType::Utf8View);
        for (e, want) in [(&l, [Some("a"), None]), (&r, [Some("c"), None])] {
            let got = e.eval(&batch, 0).unwrap().into_array(2).unwrap();
            assert_eq!(got.as_string_view().iter().collect::<Vec<_>>(), want);
        }
        // Equal types pass through untouched.
        let (l, r) = unify(col(0, ArrowType::Utf8), col(0, ArrowType::Utf8), &env).unwrap();
        assert_eq!(
            (l.data_type(), r.data_type()),
            (&ArrowType::Utf8, &ArrowType::Utf8)
        );
        // Different Spark types are not unified: the caller's check fails.
        let (l, r) = unify(col(5, ArrowType::Int32), col(0, ArrowType::Utf8), &env).unwrap();
        assert_eq!(
            (l.data_type(), r.data_type()),
            (&ArrowType::Int32, &ArrowType::Utf8)
        );
        // Neither side is a string or binary carrier.
        let (l, r) = unify(col(5, ArrowType::Int32), col(5, ArrowType::Int32), &env).unwrap();
        assert_eq!(
            (l.data_type(), r.data_type()),
            (&ArrowType::Int32, &ArrowType::Int32)
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
//! `substring(c, 1, n) IN (constants)` over a string view column, answered from the views of `c`
//! (a split of `cond.rs`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType as ArrowType;

use super::cond::view_head;
use super::logic::{long_bytes, view_parts};
use super::{Input, PhysExpr, Value};
use crate::error::Result;

/// `substring(c, 1, n) IN (constants)` over a string view column, answered from the views of `c`
/// without building the substrings (study 12 §3: q22's `substring(c_phone, 1, 2) IN (...)` spent
/// 0.15 s in the substring's array builder). The first `n` characters equal a literal of `n`
/// characters exactly when the value starts with its bytes, and equal a shorter literal exactly
/// when the value is that literal; a longer literal never matches.
#[derive(Debug)]
struct InPrefixViews {
    c: Arc<dyn PhysExpr>,
    /// Literals of `n` characters: the value starts with them.
    prefixes: Vec<Vec<u8>>,
    /// Literals of fewer than `n` characters: the value is one of them.
    exact: Vec<Vec<u8>>,
    /// The list holds a NULL.
    has_null: bool,
}

/// The column of `substring(c, 1, n)` with literal `1` and `n >= 1`, and `n`.
fn leading_substring(e: &krishiv_plan::Expr) -> Option<(&krishiv_plan::Expr, usize)> {
    use krishiv_plan::{ExprNode, FunctionId, Literal, ResolvedExpr};
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
        return None;
    };
    let FunctionId::Registry { name, .. } = &f.id else {
        return None;
    };
    if !(name.eq_ignore_ascii_case("substring") || name.eq_ignore_ascii_case("substr")) {
        return None;
    }
    // A collated string compares by its collation, never by bytes (ADR 0043).
    if !matches!(f.data_type, krishiv_types::DataType::String) {
        return None;
    }
    let [c, start, len] = f.args.as_slice() else {
        return None;
    };
    let int = |x: &krishiv_plan::Expr| match &x.node {
        ExprNode::Literal(Literal::Integer(v)) => Some(i64::from(*v)),
        _ => None,
    };
    (int(start)? == 1)
        .then(|| usize::try_from(int(len)?).ok().filter(|n| *n >= 1))
        .flatten()
        .map(|n| (c.as_ref(), n))
}

impl InPrefixViews {
    /// `None` unless `c` is a string view column and every element evaluates to one string.
    pub(super) fn try_new(
        c: Arc<dyn PhysExpr>,
        n: usize,
        items: &[Arc<dyn PhysExpr>],
    ) -> Option<Self> {
        if *c.data_type() != ArrowType::Utf8View {
            return None;
        }
        let empty = RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty()));
        let mut out = Self {
            c,
            prefixes: Vec::new(),
            exact: Vec::new(),
            has_null: false,
        };
        for item in items {
            let a = Arc::clone(item.eval(&empty, 0).ok()?.array());
            if a.len() != 1 {
                return None;
            }
            if a.is_null(0) || *a.data_type() == ArrowType::Null {
                out.has_null = true;
                continue;
            }
            if *a.data_type() != ArrowType::Utf8View {
                return None;
            }
            let text = a.as_string_view().value(0);
            match text.chars().count().cmp(&n) {
                std::cmp::Ordering::Equal => out.prefixes.push(text.as_bytes().to_vec()),
                std::cmp::Ordering::Less => out.exact.push(text.as_bytes().to_vec()),
                std::cmp::Ordering::Greater => {}
            }
        }
        Some(out)
    }
}

impl PhysExpr for InPrefixViews {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.c.nullable() || self.has_null
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.c.eval(batch, partition)?;
        let a = v.array();
        let (views, buffers) = view_parts(a.as_ref());
        let values = BooleanBuffer::collect_bool(views.len(), |i| {
            let Some(&view) = views.get(i) else {
                return false;
            };
            let inline = view.to_le_bytes();
            let (_, len) = view_head(view);
            let bytes = if len <= 12 {
                inline.get(4..).and_then(|b| b.get(..len as usize))
            } else {
                long_bytes(view, buffers)
            };
            bytes.is_some_and(|b| {
                self.prefixes.iter().any(|p| b.starts_with(p)) || self.exact.iter().any(|x| b == x)
            })
        });
        let nulls = if self.has_null {
            NullBuffer::union(
                a.logical_nulls().as_ref(),
                Some(&NullBuffer::new(values.clone())),
            )
        } else {
            a.logical_nulls()
        };
        let out: ArrayRef = Arc::new(BooleanArray::new(values, nulls));
        Ok(if v.is_scalar() {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}

/// [`InPrefixViews`] for `substring(c, 1, n) IN (constants)`, when the call has that shape and
/// `c` and the elements are string views.
pub(super) fn in_prefix(
    f: &krishiv_plan::ResolvedFunction,
    args: &[Arc<dyn PhysExpr>],
    input: Input<'_>,
) -> Result<Option<Arc<dyn PhysExpr>>> {
    let (Some(first), Some(items)) = (f.args.first(), args.get(1..)) else {
        return Ok(None);
    };
    let Some((column, n)) = leading_substring(first) else {
        return Ok(None);
    };
    if !items.iter().all(|e| e.is_constant()) {
        return Ok(None);
    }
    let c = super::compile(column, input)?;
    Ok(InPrefixViews::try_new(c, n, items).map(|e| Arc::new(e) as Arc<dyn PhysExpr>))
}

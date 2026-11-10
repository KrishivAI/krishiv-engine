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
    // A long list keeps the hash-set `IN` (one lookup per row); this scans the list per row.
    if items.len() > super::cond::IN_SET_THRESHOLD || !items.iter().all(|e| e.is_constant()) {
        return Ok(None);
    }
    let c = super::compile(column, input)?;
    Ok(InPrefixViews::try_new(c, n, items).map(|e| Arc::new(e) as Arc<dyn PhysExpr>))
}

#[cfg(test)]
mod tests {
    use arrow_array::StringViewArray;
    use arrow_schema::{Field, Schema};
    use krishiv_plan::Literal;

    use super::*;
    use crate::planner::eval::Column;

    fn text(t: &str) -> Arc<dyn PhysExpr> {
        Arc::new(crate::planner::eval::Lit::new(&Literal::String(t.to_owned())).unwrap())
    }

    fn column(data_type: ArrowType) -> Arc<dyn PhysExpr> {
        Arc::new(Column {
            index: 0,
            data_type,
            nullable: true,
        })
    }

    fn batch(values: Vec<Option<&str>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c",
            ArrowType::Utf8View,
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(StringViewArray::from(values))]).unwrap()
    }

    fn run(n: usize, items: &[&str], values: Vec<Option<&str>>) -> Vec<Option<bool>> {
        let items: Vec<_> = items.iter().map(|t| text(t)).collect();
        let e = InPrefixViews::try_new(column(ArrowType::Utf8View), n, &items).unwrap();
        let out = e.eval(&batch(values), 0).unwrap();
        let a = out.array().as_boolean();
        (0..a.len())
            .map(|i| a.is_valid(i).then(|| a.value(i)))
            .collect()
    }

    /// An n-character literal is a prefix test, a shorter one an equality, a longer one never
    /// matches; multi-byte characters count once and a value longer than 12 bytes reads its buffer.
    #[test]
    fn prefix_equality_and_never() {
        let long = "13 and then some longer value than twelve bytes";
        let got = run(
            2,
            &["13", "\u{e9}1", "1", "zzz"],
            vec![
                Some("13-x"),
                Some("1"),
                Some("\u{e9}1x"),
                Some("\u{e9}\u{e9}"),
                Some(""),
                Some(long),
                Some("zzzz"),
                None,
            ],
        );
        let t = Some(true);
        let f = Some(false);
        assert_eq!(got, [t, t, t, f, f, t, f, None]);
    }

    /// A NULL element turns every miss into NULL and leaves a hit true.
    #[test]
    fn a_null_element_makes_misses_null() {
        let items: Vec<Arc<dyn PhysExpr>> = vec![
            text("ab"),
            Arc::new(crate::planner::eval::Lit::new(&Literal::Null(None)).unwrap()),
        ];
        let e = InPrefixViews::try_new(column(ArrowType::Utf8View), 2, &items).unwrap();
        let out = e
            .eval(&batch(vec![Some("abc"), Some("xy"), None]), 0)
            .unwrap();
        let a = out.array().as_boolean();
        assert_eq!((a.value(0), a.is_valid(0)), (true, true));
        assert!(a.is_null(1) && a.is_null(2));
        assert!(e.nullable());
    }

    /// Anything but string views, or a non-string element, is not rewritten.
    #[test]
    fn other_carriers_fall_through() {
        let items = vec![text("ab")];
        assert!(InPrefixViews::try_new(column(ArrowType::Utf8), 2, &items).is_none());
        assert!(InPrefixViews::try_new(column(ArrowType::Int32), 2, &items).is_none());
        let int_item: Vec<Arc<dyn PhysExpr>> = vec![Arc::new(
            crate::planner::eval::Lit::new(&Literal::Integer(1)).unwrap(),
        )];
        assert!(InPrefixViews::try_new(column(ArrowType::Utf8View), 2, &int_item).is_none());
    }
}

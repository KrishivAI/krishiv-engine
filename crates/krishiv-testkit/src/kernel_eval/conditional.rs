// SPDX-License-Identifier: Apache-2.0
//! The conditional functions of the kernel evaluator (P2-T022): `CASE`,
//! `if`, `coalesce`, `ifnull`/`nvl`, `nvl2`, `nullif`, `nullifzero`,
//! `zeroifnull`, `nanvl` and the short-circuiting `AND`/`OR`, evaluated
//! lazily with `krishiv_kernels::conditional`: a branch runs only on the rows
//! that reach it, so an ANSI error in an untaken branch never surfaces.
//!
//! The evaluator evaluates a node "on a selection" by gathering the rows of
//! its leaves ([`Evaluator::eval_on`]); nested conditionals take the current
//! selection as their own root.

use std::cell::RefCell;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, new_null_array};
use arrow_schema::DataType;
use krishiv_kernels::KernelError;
use krishiv_kernels::compare::CompareKind;
use krishiv_kernels::conditional::{self, Lazy, Selection, When};
use krishiv_sql::ast;

use super::math::common_type;
use super::{
    Binder, EvalError, Evaluator, Node, PADDED_ROWS, Shape, Skip, is_integral, is_numeric,
    simple_string,
};

/// A conditional construct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CondFn {
    /// `CASE`: `whens` pairs of condition and value, then an optional `ELSE`.
    Case {
        whens: usize,
    },
    Coalesce,
    And,
    Or,
    IsNotNull,
    IsNan,
}

fn is_atomic(t: &DataType) -> bool {
    is_numeric(t)
        || matches!(
            t,
            DataType::Boolean | DataType::Binary | DataType::Date32 | DataType::Timestamp(..)
        )
}

/// The type the branches of a conditional take: the tightest common type;
/// in legacy mode a string and another atomic type meet as string.
fn branch_type(nodes: &[Node], ansi: bool) -> Option<DataType> {
    common_type(nodes, ansi).or_else(|| {
        let types: Vec<DataType> = nodes
            .iter()
            .map(Node::ty)
            .filter(|t| *t != DataType::Null)
            .collect();
        let promote = !ansi
            && types.contains(&DataType::Utf8)
            && types.iter().all(|t| *t == DataType::Utf8 || is_atomic(t));
        promote.then_some(DataType::Utf8)
    })
}

impl Binder<'_> {
    /// A `boolean` operand; `loose` also accepts a string (implicitly cast),
    /// which `AND`/`OR` do and `IF`/`CASE` do not.
    fn condition(self, n: Node, loose: bool) -> Result<Node, Skip> {
        match n.ty() {
            DataType::Boolean => Ok(n),
            DataType::Null => Ok(self.implicit(n, &DataType::Boolean)),
            DataType::Utf8 if loose => Ok(self.implicit(n, &DataType::Boolean)),
            t => Err(format!("analysis: {} as a condition", simple_string(&t))),
        }
    }

    fn cond(f: CondFn, args: Vec<Node>, ty: DataType) -> Node {
        Node::Cond(f, args, ty)
    }

    /// Types `values` to their common branch type.
    fn branches(self, values: Vec<Node>) -> Result<(Vec<Node>, DataType), Skip> {
        let ty = branch_type(&values, self.ansi)
            .ok_or_else(|| "analysis: branches of different types".to_owned())?;
        Ok((
            values.into_iter().map(|n| self.implicit(n, &ty)).collect(),
            ty,
        ))
    }

    /// `CASE [operand] WHEN ... THEN ... [ELSE ...] END`.
    pub(super) fn case_expr(
        self,
        operand: Option<&ast::Expr>,
        whens: &[ast::WhenClause],
        otherwise: Option<&ast::Expr>,
    ) -> Result<Node, Skip> {
        let operand = operand.map(|o| self.bind(o)).transpose()?;
        let mut conds = Vec::new();
        let mut values = Vec::new();
        for w in whens {
            let c = self.bind(&w.condition)?;
            conds.push(match &operand {
                Some(o) => self.compare(Some(CompareKind::Equal), o.clone(), c)?,
                None => self.condition(c, false)?,
            });
            values.push(self.bind(&w.result)?);
        }
        let has_else = otherwise.is_some();
        if let Some(e) = otherwise {
            values.push(self.bind(e)?);
        }
        self.case_node(conds, values, has_else)
    }

    /// A `CASE` from its conditions and values (the `ELSE` last, if any).
    fn case_node(self, conds: Vec<Node>, values: Vec<Node>, has_else: bool) -> Result<Node, Skip> {
        let n = conds.len();
        let (values, ty) = self.branches(values)?;
        let mut args = Vec::new();
        let mut values = values.into_iter();
        for c in conds {
            args.push(c);
            args.extend(values.next());
        }
        if has_else {
            args.extend(values.next());
        }
        Ok(Self::cond(CondFn::Case { whens: n }, args, ty))
    }

    /// `left AND right` / `left OR right`.
    pub(super) fn logical(self, and: bool, l: Node, r: Node) -> Result<Node, Skip> {
        let (l, r) = (self.condition(l, true)?, self.condition(r, true)?);
        let f = if and { CondFn::And } else { CondFn::Or };
        Ok(Self::cond(f, vec![l, r], DataType::Boolean))
    }

    /// The conditional function `name` applied to `args`, or `None`.
    pub(super) fn conditional_function(
        self,
        name: &str,
        args: &[&ast::Expr],
    ) -> Option<Result<Node, Skip>> {
        let known = matches!(
            name,
            "if" | "coalesce"
                | "ifnull"
                | "nvl"
                | "nvl2"
                | "nullif"
                | "nullifzero"
                | "zeroifnull"
                | "nanvl"
        );
        known.then(|| self.conditional_call(name, args))
    }

    fn conditional_call(self, name: &str, args: &[&ast::Expr]) -> Result<Node, Skip> {
        let arity = |ok: bool| -> Result<(), Skip> {
            ok.then_some(())
                .ok_or_else(|| format!("analysis: {name} with {} arguments", args.len()))
        };
        let bound = args
            .iter()
            .map(|e| self.bind(e))
            .collect::<Result<Vec<_>, _>>()?;
        let zero = || Node::Lit(Arc::new(arrow_array::Int32Array::from(vec![0])));
        match name {
            "if" => {
                arity(bound.len() == 3)?;
                let mut it = bound.into_iter();
                let (c, a, b) = (it.next(), it.next(), it.next());
                let (Some(c), Some(a), Some(b)) = (c, a, b) else {
                    return Err("if".to_owned());
                };
                self.case_node(vec![self.condition(c, false)?], vec![a, b], true)
            }
            "coalesce" | "ifnull" | "nvl" => {
                arity(if name == "coalesce" {
                    !bound.is_empty()
                } else {
                    bound.len() == 2
                })?;
                let (args, ty) = self.branches(bound)?;
                Ok(Self::cond(CondFn::Coalesce, args, ty))
            }
            "nvl2" => {
                arity(bound.len() == 3)?;
                let mut it = bound.into_iter();
                let (Some(a), Some(b), Some(c)) = (it.next(), it.next(), it.next()) else {
                    return Err("nvl2".to_owned());
                };
                let test = Self::cond(CondFn::IsNotNull, vec![a], DataType::Boolean);
                self.case_node(vec![test], vec![b, c], true)
            }
            "nullif" | "nullifzero" => {
                let (a, b) = if name == "nullif" {
                    arity(bound.len() == 2)?;
                    let mut it = bound.into_iter();
                    match (it.next(), it.next()) {
                        (Some(a), Some(b)) => (a, b),
                        _ => return Err("nullif".to_owned()),
                    }
                } else {
                    arity(bound.len() == 1)?;
                    let a = bound.into_iter().next().ok_or("nullifzero")?;
                    (a, zero())
                };
                let ty = a.ty();
                if ty == DataType::Null {
                    return Err("nullif of NULL".to_owned());
                }
                let eq = self.compare(Some(CompareKind::Equal), a.clone(), b)?;
                let null = Node::Lit(new_null_array(&ty, 1));
                self.case_node(vec![eq], vec![null, a], true)
            }
            "zeroifnull" => {
                arity(bound.len() == 1)?;
                let a = bound.into_iter().next().ok_or("zeroifnull")?;
                if !is_numeric(&a.ty()) {
                    return Err("zeroifnull of a non-numeric argument".to_owned());
                }
                let (args, ty) = self.branches(vec![a, zero()])?;
                Ok(Self::cond(CondFn::Coalesce, args, ty))
            }
            _ => {
                // nanvl(a, b): b where a is NaN.
                arity(bound.len() == 2)?;
                let mut it = bound.into_iter();
                let (Some(a), Some(b)) = (it.next(), it.next()) else {
                    return Err("nanvl".to_owned());
                };
                let ty = if a.ty() == DataType::Float32 && b.ty() == DataType::Float32 {
                    DataType::Float32
                } else if matches!(a.ty(), DataType::Float32 | DataType::Float64)
                    || is_integral(&a.ty())
                {
                    DataType::Float64
                } else {
                    return Err(format!("analysis: nanvl of {}", simple_string(&a.ty())));
                };
                let (a, b) = (self.implicit(a, &ty), self.implicit(b, &ty));
                let test = Self::cond(CondFn::IsNan, vec![a.clone()], DataType::Boolean);
                self.case_node(vec![test], vec![b, a], true)
            }
        }
    }
}

impl Evaluator<'_> {
    /// The rows a conditional starts from: the current selection, or every
    /// row of the leaves.
    fn root_selection(&self) -> Result<Selection, EvalError> {
        if let Some(sel) = self.frame.borrow().as_ref() {
            return Ok(sel.clone());
        }
        let rows = if self.shape == Shape::Padded {
            PADDED_ROWS
        } else {
            1
        };
        Ok(Selection::all(rows)?)
    }

    /// `node` evaluated on the rows of `sel` only.
    pub(super) fn eval_on(&self, node: &Node, sel: &Selection) -> Result<ArrayRef, EvalError> {
        let previous = self.frame.replace(Some(sel.clone()));
        let result = self.eval(node);
        self.frame.replace(previous);
        result
    }

    /// A conditional operand: `node` on whatever rows it is asked for. An
    /// evaluator limitation is kept aside in `deferred`, because the kernels
    /// carry only kernel errors.
    fn lazy_operand<'a>(
        &'a self,
        node: &'a Node,
        deferred: &'a RefCell<Option<EvalError>>,
    ) -> impl FnMut(&Selection) -> Result<ArrayRef, KernelError> + 'a {
        move |sel| match self.eval_on(node, sel) {
            Ok(a) => Ok(a),
            Err(EvalError::Kernel(k)) => Err(k),
            Err(other) => {
                *deferred.borrow_mut() = Some(other);
                Err(KernelError::internal(
                    "a conditional operand is not evaluable",
                ))
            }
        }
    }

    /// Evaluates conditional `f` over `args`.
    pub(super) fn cond_fn(
        &self,
        f: CondFn,
        args: &[Node],
        ty: &DataType,
    ) -> Result<ArrayRef, EvalError> {
        match f {
            CondFn::IsNotNull | CondFn::IsNan => {
                let v = self.eval(&args[0])?;
                // A NullArray has no validity buffer, but every row is NULL.
                let all_null = v.data_type() == &DataType::Null;
                let flags: BooleanArray = (0..v.len())
                    .map(|i| {
                        Some(match f {
                            CondFn::IsNotNull => !all_null && v.is_valid(i),
                            _ => !all_null && v.is_valid(i) && is_nan(v.as_ref(), i),
                        })
                    })
                    .collect();
                return Ok(Arc::new(flags));
            }
            _ => {}
        }
        let deferred: RefCell<Option<EvalError>> = RefCell::new(None);
        let root = self.root_selection()?;
        let result = self.run_conditional(f, args, ty, &root, &deferred);
        match result {
            Ok(a) => Ok(a),
            Err(k) => Err(deferred.take().unwrap_or(EvalError::Kernel(k))),
        }
    }

    #[expect(
        clippy::let_and_return,
        reason = "binding the result ends the operand borrows before the temporaries drop"
    )]
    fn run_conditional(
        &self,
        f: CondFn,
        args: &[Node],
        ty: &DataType,
        root: &Selection,
        deferred: &RefCell<Option<EvalError>>,
    ) -> Result<ArrayRef, KernelError> {
        let mut ops: Vec<Box<dyn Lazy + '_>> = args
            .iter()
            .map(|n| Box::new(self.lazy_operand(n, deferred)) as Box<dyn Lazy + '_>)
            .collect();
        let out = match f {
            CondFn::Case { whens } => {
                let otherwise = (args.len() > whens.saturating_mul(2))
                    .then(|| ops.pop())
                    .flatten();
                let mut pairs: Vec<When<'_>> = Vec::new();
                for chunk in ops.chunks_mut(2) {
                    if let [c, v] = chunk {
                        pairs.push(When {
                            condition: c.as_mut(),
                            value: v.as_mut(),
                        });
                    }
                }
                let mut otherwise = otherwise;
                conditional::case_when(
                    root,
                    &mut pairs,
                    otherwise.as_mut().map(|o| o.as_mut() as &mut dyn Lazy),
                    ty,
                )
            }
            CondFn::Coalesce => {
                let mut refs: Vec<&mut dyn Lazy> = Vec::new();
                for o in &mut ops {
                    refs.push(o.as_mut());
                }
                let out = conditional::coalesce(root, &mut refs, ty);
                out
            }
            CondFn::And | CondFn::Or => {
                let mut it = ops.iter_mut();
                let (Some(l), Some(r)) = (it.next(), it.next()) else {
                    return Err(KernelError::internal("and/or arity"));
                };
                let out = if f == CondFn::And {
                    conditional::and(root, l.as_mut(), r.as_mut())
                } else {
                    conditional::or(root, l.as_mut(), r.as_mut())
                };
                out.map(|b| Arc::new(b) as ArrayRef)
            }
            _ => Err(KernelError::internal("not a conditional")),
        };
        out
    }
}

fn is_nan(a: &dyn Array, i: usize) -> bool {
    match a.data_type() {
        DataType::Float64 => a.as_primitive::<Float64Type>().value(i).is_nan(),
        DataType::Float32 => a.as_primitive::<Float32Type>().value(i).is_nan(),
        _ => false,
    }
}

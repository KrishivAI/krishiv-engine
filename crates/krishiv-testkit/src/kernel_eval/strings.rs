// SPDX-License-Identifier: Apache-2.0
//! The string functions of the kernel evaluator (P2-T023a): `upper`, `lower`,
//! `length`, the `trim` family, `startswith`/`endswith`/`contains`,
//! `replace`, `lpad`/`rpad`, `substring`, `concat`/`||`, `like` and a
//! literal-delimiter `split`, typed with Spark 4.2's implicit casts
//! (`StringTypeWithCollation` arguments take any atomic type as its string
//! rendering; binary stays binary only where Spark has a binary overload).

use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, BooleanArray, cast::AsArray};
use arrow_schema::{DataType, Field};
use krishiv_kernels::Operand;
use krishiv_kernels::string::{
    self, BinOperand, MatchKind, PadSide, StrOperand, TrimSide, literal_delimiter,
};
use krishiv_sql::ast;

use super::{Binder, EvalError, Evaluator, Node, Shape, Skip, is_numeric, scalar, simple_string};

/// What `string_call` reports for a name that is not a string function.
const NOT_A_STRING_FUNCTION: &str = "\0";

/// A string function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum StrFn {
    Upper,
    Lower,
    CharLength,
    ByteLength,
    Trim(TrimSide),
    Match(MatchKind),
    MatchBinary(MatchKind),
    Replace,
    Pad(PadSide),
    PadBinary(PadSide),
    Substring,
    SubstringBinary,
    Concat,
    ConcatBinary,
    Like(char),
    /// `Some(delimiter)`, or `None` for a NULL delimiter (an all-NULL result).
    Split(Option<String>),
    /// `NOT`.
    Not,
}

/// `array<string>`, the type of `split`.
pub(super) fn string_list() -> DataType {
    DataType::List(std::sync::Arc::new(Field::new(
        "item",
        DataType::Utf8,
        false,
    )))
}

impl Binder {
    /// A type Spark converts to string implicitly.
    fn to_string_node(self, n: Node) -> Result<Node, Skip> {
        match n.ty() {
            DataType::Utf8 => Ok(n),
            DataType::Null => Ok(self.implicit(n, &DataType::Utf8)),
            t if is_numeric(&t)
                || matches!(
                    t,
                    DataType::Boolean
                        | DataType::Binary
                        | DataType::Date32
                        | DataType::Timestamp(..)
                ) =>
            {
                Ok(self.implicit(n, &DataType::Utf8))
            }
            t => Err(format!("{} as a string argument", simple_string(&t))),
        }
    }

    fn str_arg(self, e: &ast::Expr) -> Result<Node, Skip> {
        self.to_string_node(self.bind(e)?)
    }

    fn int_arg(self, e: &ast::Expr) -> Result<Node, Skip> {
        let n = self.bind(e)?;
        match n.ty() {
            DataType::Int32 => Ok(n),
            DataType::Null | DataType::Utf8 => Ok(self.implicit(n, &DataType::Int32)),
            t if is_numeric(&t) => Ok(self.implicit(n, &DataType::Int32)),
            t => Err(format!("{} as an int argument", simple_string(&t))),
        }
    }

    fn call(f: StrFn, args: Vec<Node>, ty: DataType) -> Node {
        Node::Str(f, args, ty)
    }

    /// The string function `name` applied to `args`, or `None` when `name`
    /// is not one of them.
    pub(super) fn string_function(
        self,
        name: &str,
        args: &[&ast::Expr],
    ) -> Option<Result<Node, Skip>> {
        match self.string_call(name, args) {
            Err(why) if why == NOT_A_STRING_FUNCTION => None,
            result => Some(result),
        }
    }

    fn string_call(self, name: &str, args: &[&ast::Expr]) -> Result<Node, Skip> {
        let arity = |ok: bool| -> Result<(), Skip> {
            ok.then_some(())
                .ok_or_else(|| format!("analysis: {name} with {} arguments", args.len()))
        };
        let text = DataType::Utf8;
        let is_binary = |n: &Node| n.ty() == DataType::Binary;
        match name {
            "upper" | "ucase" | "lower" | "lcase" => {
                arity(args.len() == 1)?;
                let f = if name.starts_with('u') {
                    StrFn::Upper
                } else {
                    StrFn::Lower
                };
                Ok(Self::call(f, vec![self.str_arg(args[0])?], text))
            }
            "length" | "char_length" | "character_length" => {
                arity(args.len() == 1)?;
                let n = self.bind(args[0])?;
                if is_binary(&n) {
                    return Ok(Self::call(StrFn::ByteLength, vec![n], DataType::Int32));
                }
                let n = self.to_string_node(n)?;
                Ok(Self::call(StrFn::CharLength, vec![n], DataType::Int32))
            }
            "trim" | "ltrim" | "rtrim" | "btrim" => {
                arity((1..=2).contains(&args.len()))?;
                let side = match name {
                    "ltrim" => TrimSide::Leading,
                    "rtrim" => TrimSide::Trailing,
                    _ => TrimSide::Both,
                };
                let mut a = args
                    .iter()
                    .map(|e| self.str_arg(e))
                    .collect::<Result<Vec<_>, _>>()?;
                // `trim(chars, str)`, `ltrim`, `rtrim` take the trim string
                // first (also `TRIM(c FROM s)`); `btrim(str, chars)` last.
                if a.len() == 2 && name != "btrim" {
                    a.swap(0, 1);
                }
                Ok(Self::call(StrFn::Trim(side), a, text))
            }
            "startswith" | "endswith" | "contains" => {
                arity(args.len() == 2)?;
                let kind = match name {
                    "startswith" => MatchKind::StartsWith,
                    "endswith" => MatchKind::EndsWith,
                    _ => MatchKind::Contains,
                };
                let (l, r) = (self.bind(args[0])?, self.bind(args[1])?);
                if is_binary(&l) && is_binary(&r) {
                    return Ok(Self::call(
                        StrFn::MatchBinary(kind),
                        vec![l, r],
                        DataType::Boolean,
                    ));
                }
                let (l, r) = (self.to_string_node(l)?, self.to_string_node(r)?);
                Ok(Self::call(
                    StrFn::Match(kind),
                    vec![l, r],
                    DataType::Boolean,
                ))
            }
            "replace" => {
                arity((2..=3).contains(&args.len()))?;
                let mut a = args
                    .iter()
                    .map(|e| self.str_arg(e))
                    .collect::<Result<Vec<_>, _>>()?;
                if a.len() == 2 {
                    a.push(Node::Lit(std::sync::Arc::new(
                        arrow_array::StringArray::from(vec![""]),
                    )));
                }
                Ok(Self::call(StrFn::Replace, a, text))
            }
            "lpad" | "rpad" => {
                arity((2..=3).contains(&args.len()))?;
                let side = if name == "lpad" {
                    PadSide::Left
                } else {
                    PadSide::Right
                };
                let (s, len) = (self.bind(args[0])?, self.int_arg(args[1])?);
                let fill = args.get(2).map(|e| self.bind(e)).transpose()?;
                if is_binary(&s) && fill.as_ref().is_none_or(is_binary) {
                    let mut a = vec![s, len];
                    a.extend(fill);
                    return Ok(Self::call(StrFn::PadBinary(side), a, DataType::Binary));
                }
                let mut a = vec![self.to_string_node(s)?, len];
                if let Some(f) = fill {
                    a.push(self.to_string_node(f)?);
                }
                Ok(Self::call(StrFn::Pad(side), a, text))
            }
            "substring" | "substr" => {
                arity((2..=3).contains(&args.len()))?;
                let s = self.bind(args[0])?;
                let ints = args[1..]
                    .iter()
                    .map(|e| self.int_arg(e))
                    .collect::<Result<Vec<_>, _>>()?;
                let binary = is_binary(&s);
                let mut a = vec![if binary { s } else { self.to_string_node(s)? }];
                a.extend(ints);
                Ok(if binary {
                    Self::call(StrFn::SubstringBinary, a, DataType::Binary)
                } else {
                    Self::call(StrFn::Substring, a, text)
                })
            }
            "concat" | "||" => {
                if args.is_empty() {
                    // `concat()` is the empty string.
                    return Ok(Node::Lit(std::sync::Arc::new(
                        arrow_array::StringArray::from(vec![""]),
                    )));
                }
                let nodes = args
                    .iter()
                    .map(|e| self.bind(e))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(self.concat(nodes)?)
            }
            "like" => {
                arity((2..=3).contains(&args.len()))?;
                let escape = match args.get(2) {
                    None => "\\".to_owned(),
                    Some(e) => match self.fold_string(e)? {
                        Some(s) => s,
                        None => return Err("NULL escape".to_owned()),
                    },
                };
                self.like(args[0], args[1], &escape)
            }
            "split" => self.split(args),
            _ => Err(NOT_A_STRING_FUNCTION.to_owned()),
        }
    }

    /// `concat(args)` / `l || r`: binary when every argument is binary.
    pub(super) fn concat(self, nodes: Vec<Node>) -> Result<Node, Skip> {
        if nodes.iter().all(|n| n.ty() == DataType::Binary) {
            return Ok(Self::call(StrFn::ConcatBinary, nodes, DataType::Binary));
        }
        let nodes = nodes
            .into_iter()
            .map(|n| self.to_string_node(n))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::call(StrFn::Concat, nodes, DataType::Utf8))
    }

    /// `expr LIKE pattern ESCAPE escape`.
    pub(super) fn like(
        self,
        expr: &ast::Expr,
        pattern: &ast::Expr,
        escape: &str,
    ) -> Result<Node, Skip> {
        let mut chars = escape.chars();
        let (Some(escape), None) = (chars.next(), chars.next()) else {
            return Err("analysis: escape is not one character".to_owned());
        };
        let (s, p) = (self.str_arg(expr)?, self.str_arg(pattern)?);
        Ok(Self::call(
            StrFn::Like(escape),
            vec![s, p],
            DataType::Boolean,
        ))
    }

    /// `NOT node`.
    pub(super) fn negate(node: Node) -> Node {
        Self::call(StrFn::Not, vec![node], DataType::Boolean)
    }

    /// `split(str, delimiter[, limit])` with a constant literal delimiter.
    fn split(self, args: &[&ast::Expr]) -> Result<Node, Skip> {
        if !(2..=3).contains(&args.len()) {
            return Err(format!("analysis: split with {} arguments", args.len()));
        }
        let s = self.str_arg(args[0])?;
        let delimiter = match self.fold_string(args[1])? {
            None => None,
            Some(regex) => Some(
                literal_delimiter(&regex)
                    .ok_or_else(|| format!("split by the regular expression {regex:?}"))?,
            ),
        };
        let limit = match args.get(2) {
            Some(e) => self.int_arg(e)?,
            None => Node::Lit(std::sync::Arc::new(arrow_array::Int32Array::from(vec![-1]))),
        };
        Ok(Self::call(
            StrFn::Split(delimiter),
            vec![s, limit],
            string_list(),
        ))
    }

    /// The constant string `e` evaluates to (`None` for NULL).
    fn fold_string(self, e: &ast::Expr) -> Result<Option<String>, Skip> {
        let node = self.str_arg(e)?;
        let ev = Evaluator::folder(self);
        let v = ev.eval(&node).map_err(|_| {
            if ev.invalid_utf8.get() {
                "documented divergence (ADR 0023): STRING holds valid UTF-8 only".to_owned()
            } else {
                "a constant argument that fails to evaluate".to_owned()
            }
        })?;
        Ok(v.is_valid(0)
            .then(|| v.as_string::<i32>().value(0).to_owned()))
    }

    /// The typed `[NOT] LIKE` / `ILIKE` expression of a parsed
    /// `ExprKind::Like`.
    pub(super) fn like_expr(
        self,
        expr: &ast::Expr,
        pattern: &ast::Expr,
        escape: Option<&str>,
        negated: bool,
        case_insensitive: bool,
    ) -> Result<Node, Skip> {
        if case_insensitive {
            return Err("ILIKE".to_owned());
        }
        let node = self.like(expr, pattern, escape.unwrap_or("\\"))?;
        Ok(if negated { Self::negate(node) } else { node })
    }
}

impl Evaluator {
    /// An evaluator for constant folding at bind time.
    pub(super) fn folder(b: Binder) -> Self {
        Self {
            ctx: krishiv_kernels::KernelCtx::new(),
            ansi: b.ansi,
            shape: Shape::Arrays,
            invalid_utf8: std::cell::Cell::new(false),
        }
    }

    /// Whether argument `i` of `n` is presented to the kernel as a scalar in
    /// this evaluator's shape (`LeftScalar`: the first; `RightScalar`: all
    /// but the first).
    fn scalar_arg(&self, i: usize, n: usize) -> bool {
        n > 1
            && match self.shape {
                Shape::LeftScalar => i == 0,
                Shape::RightScalar => i > 0,
                Shape::Arrays | Shape::Padded => false,
            }
    }

    fn str_operand<'a>(&self, a: &'a ArrayRef, i: usize, n: usize) -> StrOperand<'a, i32> {
        let s = a.as_string::<i32>();
        if self.scalar_arg(i, n) {
            StrOperand::Scalar(s.is_valid(0).then(|| s.value(0)))
        } else {
            StrOperand::Array(s)
        }
    }

    fn bin_operand<'a>(&self, a: &'a ArrayRef, i: usize, n: usize) -> BinOperand<'a, i32> {
        let b = a.as_binary::<i32>();
        if self.scalar_arg(i, n) {
            BinOperand::Scalar(b.is_valid(0).then(|| b.value(0)))
        } else {
            BinOperand::Array(b)
        }
    }

    fn int_operand<'a>(&self, a: &'a ArrayRef, i: usize, n: usize) -> Operand<'a, Int32Type> {
        if self.scalar_arg(i, n) {
            Operand::Scalar(scalar::<Int32Type>(a.as_ref()))
        } else {
            Operand::Array(a.as_primitive::<Int32Type>())
        }
    }

    /// Evaluates string function `f` over `args`.
    pub(super) fn string_fn(&self, f: &StrFn, args: &[Node]) -> Result<ArrayRef, EvalError> {
        let v = args
            .iter()
            .map(|a| self.eval(a))
            .collect::<Result<Vec<_>, _>>()?;
        let n = v.len();
        let (s, b, int) = (
            |i: usize| self.str_operand(&v[i], i, n),
            |i: usize| self.bin_operand(&v[i], i, n),
            |i: usize| self.int_operand(&v[i], i, n),
        );
        let out: ArrayRef = match f {
            StrFn::Upper => std::sync::Arc::new(string::upper(v[0].as_string::<i32>(), &self.ctx)?),
            StrFn::Lower => std::sync::Arc::new(string::lower(v[0].as_string::<i32>(), &self.ctx)?),
            StrFn::CharLength => std::sync::Arc::new(string::char_length(v[0].as_string::<i32>())),
            StrFn::ByteLength => std::sync::Arc::new(string::byte_length(v[0].as_binary::<i32>())),
            StrFn::Trim(side) => {
                std::sync::Arc::new(string::trim(*side, s(0), (n > 1).then(|| s(1)))?)
            }
            StrFn::Match(kind) => std::sync::Arc::new(string::match_str(*kind, s(0), s(1))?),
            StrFn::MatchBinary(kind) => {
                std::sync::Arc::new(string::match_binary(*kind, b(0), b(1))?)
            }
            StrFn::Replace => std::sync::Arc::new(string::replace(s(0), s(1), s(2))?),
            StrFn::Pad(side) => {
                std::sync::Arc::new(string::pad(*side, s(0), int(1), (n > 2).then(|| s(2)))?)
            }
            StrFn::PadBinary(side) => std::sync::Arc::new(string::pad_binary(
                *side,
                b(0),
                int(1),
                (n > 2).then(|| b(2)),
            )?),
            StrFn::Substring => {
                std::sync::Arc::new(string::substring(s(0), int(1), (n > 2).then(|| int(2)))?)
            }
            StrFn::SubstringBinary => std::sync::Arc::new(string::substring_binary(
                b(0),
                int(1),
                (n > 2).then(|| int(2)),
            )?),
            StrFn::Concat => {
                let parts: Vec<_> = (0..n).map(s).collect();
                std::sync::Arc::new(string::concat(&parts)?)
            }
            StrFn::ConcatBinary => {
                let parts: Vec<_> = (0..n).map(b).collect();
                std::sync::Arc::new(string::concat_binary(&parts)?)
            }
            StrFn::Like(escape) => std::sync::Arc::new(string::like(s(0), s(1), *escape)?),
            StrFn::Split(Some(d)) => std::sync::Arc::new(string::split_literal(s(0), d, int(1))?),
            StrFn::Split(None) => arrow_array::new_null_array(&string_list(), v[0].len()),
            StrFn::Not => {
                let x = v[0].as_boolean();
                std::sync::Arc::new(BooleanArray::new(!x.values(), x.nulls().cloned()))
            }
        };
        Ok(out)
    }
}

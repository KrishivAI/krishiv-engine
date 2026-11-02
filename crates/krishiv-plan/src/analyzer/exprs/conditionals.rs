// SPDX-License-Identifier: Apache-2.0
//! Conditional functions (`if`, `nvl`, `nvl2`, `nullif`, `coalesce`, ...) and their Spark result types.

use super::*;

/// A conditional function rewritten during analysis (Spark's
/// `RuntimeReplaceable`).
#[derive(Clone, Copy)]
pub(super) enum Conditional {
    /// `if(c, t, e)`: `CASE WHEN c THEN t ELSE e END`.
    If,
    /// `nvl(a, b)` / `ifnull(a, b)`: `coalesce(a, b)`.
    Coalesce,
    /// `nvl2(x, t, e)`: `CASE WHEN x IS NOT NULL THEN t ELSE e END`.
    Nvl2,
    /// `nullif(a, b)`: `a` at its own type with the `=`-cast `b` to compare
    /// against.
    NullIf,
    /// `nullifzero(x)`: `nullif(x, 0)`.
    NullIfZero,
    /// `zeroifnull(x)`: `coalesce(x, 0)` at the met type.
    ZeroIfNull,
    /// `nanvl(a, b)`: both sides are `float`/`double` already.
    Nanvl,
    /// `decode(expr, search, result, ...)`: a `CASE` over null-safe
    /// equality (the two-argument form is the charset decoder, a function).
    Decode,
}

impl Conditional {
    /// Which conditional `name` is, ASCII case-insensitively (function
    /// names are); `None` for any other function, `coalesce` included.
    pub(super) fn of(name: &str, args: usize) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "decode" if args >= 3 => Self::Decode,
            "if" => Self::If,
            "nvl" | "ifnull" => Self::Coalesce,
            "nvl2" => Self::Nvl2,
            "nullif" => Self::NullIf,
            "nullifzero" => Self::NullIfZero,
            "zeroifnull" => Self::ZeroIfNull,
            "nanvl" => Self::Nanvl,
            _ => return None,
        })
    }
}

/// The target type of a cast-named function call (`float(x)`,
/// `int(x)`, ...), ASCII case-insensitively; `None` for any other
/// function. Spark analyzes these names as `CAST(x AS target)` (its
/// `RuntimeReplaceable`): `decimal` is `decimal(10,0)`.
pub(super) fn cast_target(name: &str) -> Option<DataType> {
    Some(match name.to_ascii_lowercase().as_str() {
        "boolean" => DataType::Boolean,
        "tinyint" => DataType::Byte,
        "smallint" => DataType::Short,
        "int" => DataType::Integer,
        "bigint" => DataType::Long,
        "float" => DataType::Float,
        "double" => DataType::Double,
        "decimal" => DataType::Decimal(DecimalType::USER_DEFAULT),
        "string" => DataType::String,
        "binary" => DataType::Binary,
        "date" => DataType::Date,
        "timestamp" => DataType::Timestamp,
        "time" => DataType::Time { precision: 6 },
        _ => return None,
    })
}

/// Whether `t` is a numeric type.
pub(super) fn is_conditional_numeric(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal(_)
    )
}

/// The cast `b` needs for a legacy `nullif(a, b)` comparison that `=`
/// refuses (`None` when no legacy shape applies): the string side meets
/// the other side's type, a number meets a boolean at `BOOLEAN`. `None` as
/// the cast (rather than a type) leaves `b` as it is: `a` is the side the
/// evaluator casts in place.
pub(super) fn legacy_nullif_cast(a: &Bound, b: &Bound) -> Option<Option<DataType>> {
    let (x, y) = (a.ty.data_type.as_ref(), b.ty.data_type.as_ref());
    match (x, y) {
        (Some(DataType::String), _) => Some(None),
        (_, Some(DataType::String)) => Some(x.cloned()),
        (Some(DataType::Boolean), Some(n)) if is_conditional_numeric(n) => {
            Some(Some(DataType::Boolean))
        }
        (Some(n), Some(DataType::Boolean)) if is_conditional_numeric(n) => Some(None),
        _ => None,
    }
}

/// The literal zero of `t`: what `zeroifnull` meets its argument with (`'0'`
/// for a legacy `STRING`). `None` for any other type, which a met argument
/// never has (it always meets the hidden `INT`, so never stays a narrower
/// integral type).
pub(super) fn zero_literal(t: &DataType, origin: Origin) -> Option<Arc<Expr>> {
    let literal = match t {
        DataType::Integer => Literal::Integer(0),
        DataType::Long => Literal::Long(0),
        DataType::Float => Literal::Float(0.0),
        DataType::Double => Literal::Double(0.0),
        DataType::Decimal(d) => Literal::Decimal {
            unscaled: 0,
            data_type: *d,
        },
        DataType::String => Literal::String("0".to_owned()),
        _ => return None,
    };
    Some(Arc::new(
        Expr::new(ExprNode::Literal(literal)).with_origin(origin),
    ))
}

impl Resolver<'_> {
    /// A conditional function rewritten during analysis (Spark's
    /// `RuntimeReplaceable`): `if` and `nvl2` become `CASE WHEN`, `nvl` and
    /// `ifnull` become `coalesce`, `zeroifnull` becomes `coalesce` with a
    /// zero, and `nullif` / `nullifzero` become a same-typed `nullif` the
    /// evaluator runs once. `coalesce` keeps its name (the evaluator lowers
    /// it lazily), as does `nanvl` (both sides are `float`/`double`).
    pub(super) fn conditional_call(
        &mut self,
        which: Conditional,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
    ) -> Result<Bound, AnalysisError> {
        let n = f.args.len();
        if matches!(which, Conditional::Decode) {
            // A scalar function has no `DISTINCT` or `FILTER` form.
            for (present, syntax) in [
                (f.distinct, "DISTINCT"),
                (f.filter.is_some(), "FILTER CLAUSE"),
            ] {
                if present {
                    return Err(AnalysisError::new(
                        InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                            pretty_name: "`decode`".to_owned(),
                            syntax: syntax.to_owned(),
                        },
                        e.origin,
                    ));
                }
            }
            return self.decode_case(e, &kids[..n]);
        }
        // The arity and the input checks quote the call as written.
        let (mut call_kids, ty) = self.typed_call(entry, e, &kids[..n])?;
        call_kids.extend(kids.get(n).cloned());
        let origin = e.origin;
        let node = |node: ExprNode| Arc::new(Expr::new(node).with_origin(origin));
        // The call as written (before casts), naming the output and quoting
        // errors: never the rewritten form.
        let shown = node(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: entry.id.clone(),
                kind: entry.kind,
                args: kids[..n].iter().map(Bound::shown).collect(),
                distinct: f.distinct,
                filter: kids.get(n).map(Bound::shown),
                within_group: Vec::new(),
                data_type: ty.data_type.clone().unwrap_or(DataType::Null),
                nullable: ty.nullable,
                deterministic: entry.deterministic,
            },
        )));
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        let resolved =
            |id: FunctionId, args: Vec<Arc<Expr>>, distinct: bool, filter: Option<Arc<Expr>>| {
                node(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id,
                        kind: FunctionKind::Scalar,
                        args,
                        distinct,
                        filter,
                        within_group: Vec::new(),
                        data_type: data_type.clone(),
                        nullable: ty.nullable,
                        deterministic: entry.deterministic,
                    },
                )))
            };
        let expr = match which {
            // `if(c, t, e)`: the branches' common type, as `when` types it.
            Conditional::If => {
                let args = call_kids[..n].iter().map(|k| Arc::clone(&k.expr)).collect();
                resolved(
                    FunctionId::Builtin(Builtin::When),
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
            // `nvl(a, b)` / `ifnull(a, b)`: `coalesce`, cast likewise.
            Conditional::Coalesce => {
                let args = call_kids[..n].iter().map(|k| Arc::clone(&k.expr)).collect();
                self.coalesce_node(
                    origin,
                    e,
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                    &ty,
                )?
            }
            // `nvl2(x, t, e)`: `t` where `x` is not null, else `e`.
            Conditional::Nvl2 => {
                let cond = self.operator_call(Builtin::IsNotNull, origin, &call_kids[..1])?;
                let mut args = vec![cond.expr];
                args.extend(call_kids[1..n].iter().map(|k| Arc::clone(&k.expr)));
                resolved(
                    FunctionId::Builtin(Builtin::When),
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
            // `nullif(a, b)`: `NULL` where `a = b`, else `a` (evaluated
            // once by the evaluator); `b` carries the analyzer's own `=`
            // cast, `a` keeps its type.
            Conditional::NullIf => {
                let other =
                    self.nullif_other(&call_kids[0], &call_kids[1], &shown, kids, f, origin)?;
                resolved(
                    entry.id.clone(),
                    vec![Arc::clone(&call_kids[0].expr), other],
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
            // `nullifzero(x)`: `nullif(x, 0)`, named as written.
            Conditional::NullIfZero => {
                let zero = Bound {
                    expr: node(ExprNode::Literal(Literal::Integer(0))),
                    ty: arg_type(DataType::Integer, false),
                    agg: None,
                    win: None,
                    outer: None,
                    lateral: Vec::new(),
                    shown: None,
                };
                let other = self.nullif_other(&call_kids[0], &zero, &shown, kids, f, origin)?;
                resolved(
                    entry.id.clone(),
                    vec![Arc::clone(&call_kids[0].expr), other],
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
            // `zeroifnull(x)`: `coalesce(x, 0)` at the met type.
            Conditional::ZeroIfNull => {
                let Some(to) = ty.data_type.as_ref() else {
                    return Err(unsupported("zeroifnull of an untyped NULL", origin));
                };
                let Some(zero) = zero_literal(to, origin) else {
                    return Err(unsupported(format!("zeroifnull of {}", to.sql()), origin));
                };
                let args = vec![Arc::clone(&call_kids[0].expr), zero];
                self.coalesce_node(
                    origin,
                    e,
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                    &ty,
                )?
            }
            // `nanvl(a, b)`: `b` where `a` is NaN, else `a`; both sides are
            // already `float`/`double`.
            Conditional::Nanvl => {
                let args = call_kids[..n].iter().map(|k| Arc::clone(&k.expr)).collect();
                resolved(
                    entry.id.clone(),
                    args,
                    f.distinct,
                    call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                )
            }
            Conditional::Decode => return self.decode_case(e, &kids[..n]),
        };
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        Ok(Bound {
            expr,
            ty: ArgType {
                data_type: ty.data_type.clone(),
                nullable: ty.nullable,
            },
            agg: kids.iter().find_map(|k| k.agg),
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
            shown: Some(shown),
        })
    }

    /// A cast-named function call (`float(x)`, `int(x)`, ...), rewritten
    /// during analysis to `CAST(x AS target)` in the session's default
    /// mode (Spark's `RuntimeReplaceable`): the plan only ever holds the
    /// `Cast` node, typed by the cast path. The output column is named by
    /// the argument, and cast errors quote the argument, as Spark does;
    /// the arity error quotes the call. A pair the cast path does not
    /// support fails exactly as `CAST(x AS target)` fails, apart from the
    /// quoted expression.
    pub(super) fn cast_named_call(
        &mut self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
        to: DataType,
    ) -> Result<Bound, AnalysisError> {
        let n = f.args.len();
        // The arity check quotes the call as written, as `typed_call` does.
        if !entry.arity.accepts(n) {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: format!("`{}`", entry.id.name()),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: n.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            ));
        }
        let arg = &kids[0];
        let mode = if self.ctx.conf.ansi {
            CastMode::Ansi
        } else {
            CastMode::Legacy
        };
        if let Err(bad) = coerce::can_cast(arg.ty.data_type.as_ref(), &to, mode) {
            let error = cast_mismatch(
                bad,
                format!("\"{}\"", pretty(&arg.shown(), self.ctx.conf.session_zone)),
                quoted_type(arg.ty.data_type.as_ref()),
                quoted_type(Some(&to)),
            );
            return Err(AnalysisError::new(error, e.origin));
        }
        let nullable = arg.ty.nullable
            || arg
                .ty
                .data_type
                .as_ref()
                .is_some_and(|from| coerce::force_nullable(from, &to));
        let expr = Arc::new(
            Expr::new(ExprNode::Cast {
                expr: Arc::clone(&arg.expr),
                to: to.clone(),
                mode,
            })
            .with_origin(e.origin),
        );
        Ok(Bound {
            expr,
            ty: arg_type(to, nullable),
            agg: arg.agg,
            win: arg.win,
            outer: arg.outer,
            lateral: arg.lateral.clone(),
            shown: Some(arg.shown()),
        })
    }

    /// A resolved `coalesce` call: `nvl` / `ifnull` / `zeroifnull` keep the
    /// result type their own casts met at.
    pub(super) fn coalesce_node(
        &mut self,
        origin: Origin,
        e: &Arc<Expr>,
        args: Vec<Arc<Expr>>,
        distinct: bool,
        filter: Option<Arc<Expr>>,
        ty: &ArgType,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let id = match self
            .ctx
            .functions
            .lookup(&["coalesce".to_owned()], self.ctx.profile)
        {
            FunctionLookup::Found(entry) => entry.id,
            _ => return Err(unsupported("function coalesce", e.origin)),
        };
        Ok(Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id,
                    kind: FunctionKind::Scalar,
                    args,
                    distinct,
                    filter,
                    within_group: Vec::new(),
                    data_type: ty.data_type.clone().unwrap_or(DataType::Null),
                    nullable: ty.nullable,
                    deterministic: true,
                },
            )))
            .with_origin(origin),
        ))
    }

    /// The second argument of `nullif(a, b)`: `b` cast to the type `=`
    /// compares at, through the analyzer's own comparison coercion. In
    /// legacy mode `=` leaves two shapes to P2-T071d (a string against
    /// another type, a boolean against a number); there Spark casts the
    /// string to the other side's type and the number to `BOOLEAN`, which
    /// this mirrors so `nullif` keeps Spark's rows.
    pub(super) fn nullif_other(
        &mut self,
        a: &Bound,
        b: &Bound,
        shown: &Arc<Expr>,
        kids: &[Bound],
        f: &FunctionCall,
        origin: Origin,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let args = [arg_of(a), arg_of(b)];
        let ansi = self.ctx.conf.ansi;
        let rules = coerce::Rules {
            ansi,
            case_sensitive: self.ctx.conf.case_sensitive,
        };
        match coerce::operator(
            Builtin::Eq,
            &args,
            rules,
            self.ctx.conf.decimal_precision_loss,
        ) {
            Ok(typed) => Ok(self.implicit_cast(b, typed.casts[1].clone()).expr),
            Err(m) => {
                if !ansi && let Some(to) = legacy_nullif_cast(a, b) {
                    return Ok(self.implicit_cast(b, to).expr);
                }
                Err(self.mismatch(m, shown, kids, &f.name, origin))
            }
        }
    }
}

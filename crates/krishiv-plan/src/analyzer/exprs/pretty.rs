// SPDX-License-Identifier: Apache-2.0
//! Spark's rendering of resolved expressions (`pretty`), as `toPrettySQL` names them.

use super::*;

/// Spark's generated name for an unaliased expression (`toPrettySQL`):
/// `(a + 1)`, `count(1)`, `CASE WHEN ... END`. Built bottom-up with an
/// explicit stack, once per distinct node. Timestamp literals render in
/// `zone`, the session's resolved zone, through the same cast-to-string
/// kernels `CAST(ts AS STRING)` uses.
pub(in crate::analyzer) fn pretty(e: &Expr, zone: Option<SessionZone>) -> String {
    let mut done: HashMap<usize, String> = HashMap::new();
    let mut stack = vec![(e, false)];
    while let Some((x, expanded)) = stack.pop() {
        if done.contains_key(&addr(x)) {
            continue;
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| kids.push(&**c));
        if !expanded {
            stack.push((x, true));
            stack.extend(kids.into_iter().rev().map(|c| (c, false)));
            continue;
        }
        let k: Vec<&str> = kids
            .iter()
            .map(|c| done.get(&addr(*c)).map_or("", String::as_str))
            .collect();
        let mut s = pretty_node(x, &k, zone);
        if s.len() > MAX_NAME {
            // A shared subexpression repeated in a name doubles it per level:
            // names stay bounded (Spark's would not).
            let mut end = MAX_NAME;
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            s.truncate(end);
            s.push_str("...");
        }
        done.insert(addr(x), s);
    }
    done.remove(&addr(e)).unwrap_or_default()
}

/// The names of the outer references of a subquery plan, in order, each
/// once.
pub(super) fn outer_names(plan: &crate::LogicalPlan) -> String {
    use crate::walk::{Node, visit_distinct};
    fn collect(node: Node<'_>) -> Vec<(AttrId, String)> {
        let mut names: Vec<(AttrId, String)> = Vec::new();
        let _ = visit_distinct(node, |n| {
            if let Node::Expr(e) = n
                && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &e.node
                && !names.iter().any(|(id, _)| *id == c.id)
            {
                names.push((c.id, c.name.clone()));
            }
            std::ops::ControlFlow::Continue(())
        });
        names
    }
    let mut names = collect(Node::Plan(plan));
    // Spark lists each reference of a set operation's branches, not once per
    // column. `INTERSECT` and `EXCEPT` are a semi or an anti join by now.
    let mut union = None;
    let _ = visit_distinct(Node::Plan(plan), |n| {
        let Node::Plan(p) = n else {
            return std::ops::ControlFlow::Continue(());
        };
        match &p.node {
            crate::PlanNode::Union { inputs, .. } => union = Some(inputs.clone()),
            crate::PlanNode::Join {
                left,
                right,
                kind: crate::JoinKind::LeftSemi | crate::JoinKind::LeftAnti,
                ..
            } => union = Some(vec![Arc::clone(left), Arc::clone(right)]),
            _ => return std::ops::ControlFlow::Continue(()),
        }
        std::ops::ControlFlow::Break(())
    });
    // Only when a branch reads the outer query: over an uncorrelated set
    // operation, the references above it are listed as anywhere else.
    if union.is_some_and(|inputs: Vec<_>| {
        inputs
            .iter()
            .any(|input| !collect(Node::Plan(input)).is_empty())
    }) {
        names = Vec::new();
        let _ = visit_distinct(Node::Plan(plan), |n| {
            if let Node::Expr(e) = n
                && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &e.node
            {
                names.push((c.id, c.name.clone()));
            }
            std::ops::ControlFlow::Continue(())
        });
    }
    names
        .into_iter()
        .map(|(_, n)| n)
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn pretty_node(x: &Expr, k: &[&str], zone: Option<SessionZone>) -> String {
    match &x.node {
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
            c.name.clone()
        }
        ExprNode::Column(c) => c.name.last().cloned().unwrap_or_default(),
        ExprNode::Literal(v) => pretty_literal(v, zone),
        ExprNode::TypedLiteral { kind, value } => {
            let keyword = match kind {
                TypedLiteralKind::Date => "DATE",
                TypedLiteralKind::Timestamp => "TIMESTAMP",
                TypedLiteralKind::TimestampNtz => "TIMESTAMP_NTZ",
                TypedLiteralKind::TimestampLtz => "TIMESTAMP_LTZ",
                TypedLiteralKind::Time => "TIME",
            };
            // Spark names a time by its value, not by the text written.
            if *kind == TypedLiteralKind::Time
                && let Some(n) = krishiv_kernels::cast::parse_time(value)
            {
                return format!("{keyword} '{}'", krishiv_kernels::cast::format_time(n));
            }
            format!("{keyword} '{value}'")
        }
        // Spark's pretty SQL shows a cast to an array, map or struct type as
        // its operand (`CAST(array() AS ARRAY<FLOAT>)` is `array()`).
        ExprNode::Cast { to, mode, .. }
            if *mode != CastMode::Try
                && matches!(
                    to,
                    DataType::Array(_) | DataType::Map(_) | DataType::Struct(_)
                ) =>
        {
            k[0].to_owned()
        }
        ExprNode::Cast { to, mode, .. } => {
            let cast = if *mode == CastMode::Try {
                "TRY_CAST"
            } else {
                "CAST"
            };
            format!("{cast}({} AS {})", k[0], to.sql())
        }
        // An unresolved alias keeps `AS name` (a `struct` call rewritten
        // to `named_struct` shows its children as written); a resolved
        // one is its expression (its output names the attribute).
        ExprNode::Alias { name, .. } => format!("{} AS {name}", k[0]),
        ExprNode::Resolved(ResolvedExpr::Alias { .. }) => k[0].to_owned(),
        ExprNode::Resolved(ResolvedExpr::StateInput { .. }) => k.join(", "),
        ExprNode::Field { name, .. } | ExprNode::Resolved(ResolvedExpr::Field { name, .. }) => {
            format!("{}.{}", k[0], name)
        }
        // As written (`update_fields(base, ...)`): the `shown` form of an
        // update, before the analyzer rewrites it.
        ExprNode::UpdateFields { ops, .. } => {
            let mut values = k[1..].iter();
            let rendered: Vec<String> = ops
                .iter()
                .map(|op| match op {
                    FieldOp::With { name, .. } => format!(
                        "WithField({name}, {})",
                        values.next().copied().unwrap_or("?")
                    ),
                    FieldOp::Drop { name } => format!("DropField({name})"),
                })
                .collect();
            format!("update_fields({}, {})", k[0], rendered.join(", "))
        }
        ExprNode::Subquery { kind, plan } => {
            // Spark lists the outer references.
            let outer = outer_names(plan);
            match kind {
                SubqueryKind::Scalar => format!("scalarsubquery({outer})"),
                SubqueryKind::Exists => format!("exists({outer})"),
                SubqueryKind::In(_) if k.len() == 1 => {
                    format!("({} IN (listquery({outer})))", k[0])
                }
                SubqueryKind::In(_) => {
                    let fields: Vec<String> = k.iter().map(|v| format!("'{v}', {v}")).collect();
                    format!(
                        "(named_struct({}) IN (listquery({outer})))",
                        fields.join(", ")
                    )
                }
            }
        }
        ExprNode::Star(_) => "*".to_owned(),
        // Spark's `lambdafunction(body, arg1, ...)`: bound variables render
        // as `namedlambdavariable()` (Spark's `NamedLambdaVariable.sql`
        // ignores the written name); an unbound lambda keeps its written
        // names, as error quotes of not-yet-bound calls show them.
        ExprNode::Lambda { params, .. } => {
            let mut args = vec![k[0].to_owned()];
            let bound = params.iter().any(|p| p.id != AttrId(u64::MAX));
            args.extend(params.iter().map(|p| {
                if bound {
                    "namedlambdavariable()".to_owned()
                } else {
                    p.name.clone()
                }
            }));
            format!("lambdafunction({})", args.join(", "))
        }
        ExprNode::Resolved(ResolvedExpr::LambdaVar(_)) => "namedlambdavariable()".to_owned(),
        ExprNode::Window(w) => pretty_window(w, k, zone),
        ExprNode::Function(FunctionCall {
            name, qualifier, ..
        }) if name == "__subscript" && qualifier.is_empty() && k.len() == 2 => {
            format!("{}[{}]", k[0], k[1])
        }
        ExprNode::Function(FunctionCall {
            name,
            qualifier,
            arg_names,
            distinct,
            filter,
            within_group,
            ..
        }) => {
            let qualified = if qualifier.is_empty() {
                name.clone()
            } else {
                format!("{}.{name}", qualifier.join("."))
            };
            // Operators are never qualified, so only a bare name matches.
            let op = if qualifier.is_empty() {
                Builtin::from_name(name)
            } else {
                None
            };
            // The argument children come first; the filter child follows
            // them when the call has one, and any `WITHIN GROUP`
            // ordering keys follow it (a named-argument rearrangement
            // fills defaults, so the written count undercounts `mask`).
            let n = k
                .len()
                .saturating_sub(usize::from(filter.is_some()) + within_group.len());
            let text = call(
                &qualified,
                op,
                *distinct,
                &k[..n.min(k.len())],
                Some(arg_names),
                None,
            );
            // The filter child follows the arguments, when the call has
            // one; any `WITHIN GROUP` ordering keys follow it.
            let mut rest = k.get(n.min(k.len())..).unwrap_or_default().iter();
            let filter = filter.as_ref().and_then(|_| rest.next());
            let mut text = text;
            if !within_group.is_empty() {
                let orders: Vec<String> = within_group
                    .iter()
                    .zip(rest)
                    .map(|(o, key)| within_key(name, o, key))
                    .collect();
                text = format!("{text} WITHIN GROUP (ORDER BY {})", orders.join(", "));
            }
            // Spark appends the filter after the whole function text.
            match filter {
                Some(filter) => format!("{text} FILTER (WHERE {filter})"),
                None => text,
            }
        }
        // Spark's `lag`/`lead` show `input, offset, default`, and
        // `nth_value` `input, offset` then ` ignore nulls` when set.
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.kind == FunctionKind::Window
                && matches!(f.id.name(), "lag" | "lead" | "nth_value")
                && f.filter.is_none() =>
        {
            let shown = if f.id.name() == "nth_value" { 2 } else { 3 };
            let n = shown.min(k.len());
            let text = call(
                f.id.name(),
                None,
                f.distinct,
                &k[..n],
                None,
                Some(&f.data_type),
            );
            let ignore = matches!(
                f.args.get(shown).map(|a| &a.node),
                Some(ExprNode::Literal(Literal::Boolean(true)))
            );
            if ignore && f.id.name() == "nth_value" {
                format!("{text} ignore nulls")
            } else {
                text
            }
        }
        // Spark's `ExtractValue` shows `child[ordinal]`.
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.id.name() == "__subscript" && k.len() == 2 && f.filter.is_none() =>
        {
            format!("{}[{}]", k[0], k[1])
        }
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let op = match &f.id {
                FunctionId::Builtin(b) => Some(*b),
                FunctionId::Registry { name, .. } => builtin_op(name, f.args.len()),
                FunctionId::Udf { .. } => None,
            };
            // As above: the argument children come first, then the
            // filter child, then any ordering keys.
            let n = k
                .len()
                .saturating_sub(usize::from(f.filter.is_some()) + f.within_group.len());
            // `aggregate`/`reduce` of three arguments show the identity
            // `finish` Spark's constructor fills in, once the merge is bound
            // (see `hof`): the call then reads as the four-argument form.
            let identity = matches!(f.id.name(), "aggregate" | "reduce")
                && f.args.len() == 3
                && k.len() >= 3
                && matches!(&f.args[2].node, ExprNode::Lambda { params, .. } if params.iter().any(|p| p.id != AttrId(u64::MAX)));
            let owned;
            let kids: &[&str] = if identity {
                owned = vec![
                    k[0],
                    k[1],
                    k[2],
                    "lambdafunction(namedlambdavariable(), namedlambdavariable())",
                ];
                &owned
            } else {
                &k[..n.min(k.len())]
            };
            let text = call(f.id.name(), op, f.distinct, kids, None, Some(&f.data_type));
            // The filter child follows the arguments, when the call has
            // one; any `WITHIN GROUP` ordering keys follow it.
            let mut rest = k.get(n.min(k.len())..).unwrap_or_default().iter();
            let filter = f.filter.as_ref().and_then(|_| rest.next());
            let mut text = text;
            if !f.within_group.is_empty() {
                // `mode` keeps Spark's `Mode.sql` form: the ordering is
                // the tie-break and takes no arguments.
                if f.id.name() == "mode" {
                    if let (Some(name), Some(o)) = (rest.next(), f.within_group.first()) {
                        let desc = if o.direction == SortDirection::Descending {
                            " DESC"
                        } else {
                            ""
                        };
                        text = format!("mode() WITHIN GROUP (ORDER BY {name}{desc})");
                    }
                } else {
                    let orders: Vec<String> = f
                        .within_group
                        .iter()
                        .zip(rest)
                        .map(|(o, key)| within_key(f.id.name(), o, key))
                        .collect();
                    text = format!("{text} WITHIN GROUP (ORDER BY {})", orders.join(", "));
                }
            }
            // Spark appends the filter after the whole function text.
            match filter {
                Some(filter) => format!("{text} FILTER (WHERE {filter})"),
                None => text,
            }
        }
    }
}

/// `f OVER (...)` as Spark names it: the function (`RANK()`,
/// `DENSE_RANK()` and `PERCENT_RANK()` upper case, as Spark's `RankLike`
/// writes them), then the window as `WindowSpecDefinition.sql` writes it.
/// `k` holds the children's names in [`Expr::for_each_child`] order.
pub(super) fn pretty_window(
    w: &crate::WindowExpr,
    k: &[&str],
    zone: Option<SessionZone>,
) -> String {
    use crate::{FrameBound, FrameUnits, WindowSpecRef};
    let function = match &w.function.node {
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.kind == FunctionKind::Window
                && ["rank", "dense_rank", "percent_rank"].contains(&f.id.name()) =>
        {
            format!("{}()", f.id.name().to_uppercase())
        }
        _ => k[0].to_owned(),
    };
    let spec = match &w.window {
        WindowSpecRef::Named(name) => return format!("{function} OVER {name}"),
        WindowSpecRef::Spec(spec) => spec,
    };
    let mut parts = Vec::new();
    let mut rest = k.get(1..).unwrap_or_default().iter();
    if !spec.partition_by.is_empty() {
        let names: Vec<&str> = rest
            .by_ref()
            .take(spec.partition_by.len())
            .copied()
            .collect();
        parts.push(format!("PARTITION BY {}", names.join(", ")));
    }
    if !spec.order_by.is_empty() {
        let names: Vec<String> = spec
            .order_by
            .iter()
            .zip(rest.by_ref())
            .map(|(o, n)| format!("{n} {}", order_sql(o)))
            .collect();
        parts.push(format!("ORDER BY {}", names.join(", ")));
    }
    let Some(f) = &spec.frame else {
        // An unresolved window, as Spark's errors show one.
        parts.push("unspecifiedframe$()".to_owned());
        return format!("{function} OVER ({})", parts.join(" "));
    };
    {
        let bound = |b: &FrameBound, rest: &mut std::slice::Iter<'_, &str>| match b {
            FrameBound::UnboundedPreceding => "UNBOUNDED PRECEDING".to_owned(),
            FrameBound::CurrentRow => "CURRENT ROW".to_owned(),
            FrameBound::UnboundedFollowing => "UNBOUNDED FOLLOWING".to_owned(),
            FrameBound::Offset(e) => {
                let own = rest.next().copied().unwrap_or_default();
                match crate::expr::negated(e) {
                    Some(n) => format!("{} PRECEDING", pretty(n, zone)),
                    None => format!("{own} FOLLOWING"),
                }
            }
        };
        let units = match f.units {
            FrameUnits::Rows => "ROWS",
            FrameUnits::Range => "RANGE",
        };
        let lower = bound(&f.lower, &mut rest);
        let upper = bound(&f.upper, &mut rest);
        parts.push(format!("{units} BETWEEN {lower} AND {upper}"));
    }
    format!("{function} OVER ({})", parts.join(" "))
}

/// One `WITHIN GROUP` ordering key as Spark names it. `percentile_cont` and
/// `percentile_disc` print the key and ` DESC` for a descending order (their
/// own `sql`); the other ordered-set aggregates print the full sort order.
pub(super) fn within_key(function: &str, o: &crate::SortOrder, key: &str) -> String {
    if function.eq_ignore_ascii_case("percentile_cont")
        || function.eq_ignore_ascii_case("percentile_disc")
    {
        let desc = if o.direction == SortDirection::Descending {
            " DESC"
        } else {
            ""
        };
        format!("{key}{desc}")
    } else {
        format!("{key} {}", order_sql(o))
    }
}

/// `ASC NULLS FIRST` and the like (Spark's `SortOrder.sql` after the key).
pub(in crate::analyzer) const fn order_sql(o: &crate::SortOrder) -> &'static str {
    use crate::{NullOrdering, SortDirection};
    match (o.direction, o.nulls) {
        (SortDirection::Ascending, NullOrdering::NullsFirst) => "ASC NULLS FIRST",
        (SortDirection::Ascending, NullOrdering::NullsLast) => "ASC NULLS LAST",
        (SortDirection::Descending, NullOrdering::NullsFirst) => "DESC NULLS FIRST",
        (SortDirection::Descending, NullOrdering::NullsLast) => "DESC NULLS LAST",
    }
}

pub(super) fn call(
    name: &str,
    op: Option<Builtin>,
    distinct: bool,
    k: &[&str],
    arg_names: Option<&[Option<String>]>,
    data_type: Option<&DataType>,
) -> String {
    let infix = |sym: &str| format!("({} {sym} {})", k[0], k[1]);
    // `[NOT] LIKE|ILIKE ANY|SOME|ALL` with literal patterns: Spark's
    // `LikeAny`-family name, `likeany(child)` (the child is `lower(child)`
    // for ILIKE; the patterns are not shown), in both its name and SQL form.
    let quantified = |q: QuantifiedLike| {
        let not = if q.negated { "not" } else { "" };
        let any = if q.all { "all" } else { "any" };
        format!("{not}like{any}({})", k[0])
    };
    match (op, k.len()) {
        (Some(op), n) if n >= 1 && op.quantified_like().is_some() => {
            quantified(op.quantified_like().unwrap_or_default())
        }
        (Some(Builtin::Eq), 2) => infix("="),
        (Some(Builtin::EqNullSafe), 2) => infix("<=>"),
        (Some(Builtin::Lt), 2) => infix("<"),
        (Some(Builtin::LtEq), 2) => infix("<="),
        (Some(Builtin::Gt), 2) => infix(">"),
        (Some(Builtin::GtEq), 2) => infix(">="),
        (Some(Builtin::And), 2) => infix("AND"),
        (Some(Builtin::Or), 2) => infix("OR"),
        (Some(Builtin::Add), 2) => infix("+"),
        // Spark's interval rewrites print without parentheses.
        (
            Some(
                Builtin::DateAddYmInterval
                | Builtin::TimestampAddYmInterval
                | Builtin::TimestampAddInterval,
            ),
            2,
        ) => format!("{} + {}", k[0], k[1]),
        (Some(Builtin::DatetimeSub), 2) => format!("{} - {}", k[0], k[1]),
        (Some(Builtin::Subtract), 2) => infix("-"),
        (Some(Builtin::Multiply), 2) => infix("*"),
        (Some(Builtin::Divide), 2) => infix("/"),
        (Some(Builtin::Remainder), 2) => infix("%"),
        (Some(Builtin::IntegralDivide), 2) => infix("div"),
        (Some(Builtin::Like), 2) => format!("{} LIKE {}", k[0], k[1]),
        (Some(Builtin::Not), 1) => format!("(NOT {})", k[0]),
        (Some(Builtin::Negative), 1) => format!("(- {})", k[0]),
        (Some(Builtin::IsNull), 1) => format!("({} IS NULL)", k[0]),
        (Some(Builtin::IsNotNull), 1) => format!("({} IS NOT NULL)", k[0]),
        (Some(Builtin::In), n) if n > 0 => format!("({} IN ({}))", k[0], k[1..].join(", ")),
        (Some(Builtin::When), _) => {
            let mut s = "CASE".to_owned();
            for pair in k.chunks(2) {
                match pair {
                    [c, v] => s.push_str(&format!(" WHEN {c} THEN {v}")),
                    [e] => s.push_str(&format!(" ELSE {e}")),
                    _ => {}
                }
            }
            s + " END"
        }
        // Spark's own spellings of `if` and `positive`.
        _ if name.eq_ignore_ascii_case("if") && k.len() == 3 => {
            format!("(IF({}, {}, {}))", k[0], k[1], k[2])
        }
        _ if name.eq_ignore_ascii_case("positive") && k.len() == 1 => format!("(+ {})", k[0]),
        // A third argument is the precision the analyzer appends for a
        // `time(p)`: not part of the call as written.
        _ if name.eq_ignore_ascii_case("extract") && (2..=3).contains(&k.len()) => {
            format!("extract({} FROM {})", k[0], k[1])
        }
        // `trim` family: one argument is `trim(src)` / `ltrim(src)` /
        // `rtrim(src)` / `btrim(...)`; two are Spark's `TRIM(...)` form.
        // The analyzer keeps `trim(trimStr, src)` (characters first, as
        // Spark's two-parameter call is) and `btrim(src, chars)` last.
        _ if name.eq_ignore_ascii_case("trim") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("trim({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("trim") && k.len() == 2 => {
            format!("TRIM(BOTH {} FROM {})", k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("ltrim") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("ltrim({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("ltrim") && k.len() == 2 => {
            format!("TRIM(LEADING {} FROM {})", k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("rtrim") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("rtrim({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("rtrim") && k.len() == 2 => {
            format!("TRIM(TRAILING {} FROM {})", k[0], k[1])
        }
        // Spark fills in the missing arguments: `substr(s, p)` is
        // `substr(s, p, 2147483647)`, `lpad(s, n)` is `lpad(s, n, ' ')`
        // (`X'00'` for binary), `replace(s, search)` is
        // `replace(s, search, '')`, `round(x)` is `round(x, 0)` and `log(x)`
        // is `LOG(E(), x)`.
        _ if name.eq_ignore_ascii_case("substr") && k.len() == 2 => {
            format!("substr({}, {}, 2147483647)", k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("substring") && k.len() == 2 => {
            format!("substring({}, {}, 2147483647)", k[0], k[1])
        }
        _ if (name.eq_ignore_ascii_case("lpad") || name.eq_ignore_ascii_case("rpad"))
            && k.len() == 2 =>
        {
            let pad = if data_type == Some(&DataType::Binary) {
                "X'00'"
            } else {
                " "
            };
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{}, {}, {pad})", name.to_lowercase(), k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("replace") && k.len() == 2 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("replace({d}{}, {}, )", k[0], k[1])
        }
        _ if (name.eq_ignore_ascii_case("round") || name.eq_ignore_ascii_case("bround"))
            && k.len() == 1 =>
        {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{}, 0)", name.to_lowercase(), k[0])
        }
        _ if name.eq_ignore_ascii_case("ceil") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("CEIL({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("floor") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("FLOOR({d}{})", k[0])
        }
        _ if name.eq_ignore_ascii_case("log") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("LOG({d}E(), {})", k[0])
        }
        _ if name.eq_ignore_ascii_case("log") && k.len() == 2 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("LOG({d}{}, {})", k[0], k[1])
        }
        _ if name.eq_ignore_ascii_case("ln") && k.len() == 1 => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("ln({d}{})", k[0])
        }
        // Spark's name of `first(x, ignoreNulls)` leaves the flag out.
        _ if k.len() == 2
            && matches!(k[1], "true" | "false")
            && ["first", "last", "first_value", "last_value", "any_value"]
                .iter()
                .any(|n| name.eq_ignore_ascii_case(n)) =>
        {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_lowercase(), k[0])
        }
        // `JsonToStructs` has the JSON text as its only child: the schema and
        // the options are not shown.
        // A map schema is the exception: Spark 4.2 names the column `entries`.
        _ if name.eq_ignore_ascii_case("from_json")
            && k.get(1).is_some_and(|schema| {
                let text = schema.trim_matches('\'').trim_start();
                text.get(..3)
                    .is_some_and(|head| head.eq_ignore_ascii_case("map"))
                    && text[3..].trim_start().starts_with('<')
            }) =>
        {
            "entries".to_owned()
        }
        _ if name.eq_ignore_ascii_case("from_json") && !k.is_empty() => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("from_json({d}{})", k[0])
        }
        // `StructsToJson` and `SchemaOfJson` likewise show only their first
        // child, not the options.
        _ if (name.eq_ignore_ascii_case("to_json")
            || name.eq_ignore_ascii_case("schema_of_json"))
            && !k.is_empty() =>
        {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_lowercase(), k[0])
        }
        // The bitwise operators show infix (`(20181117 >> 2)`), `~`
        // prefix; `to_varchar` shows as `to_char`.
        _ if k.len() == 2 && matches!(name, "&" | "|" | "^" | "<<" | ">>" | ">>>") => infix(name),
        _ if k.len() == 1 && name == "~" => format!("~{}", k[0]),
        _ if name.eq_ignore_ascii_case("to_varchar") => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("to_char({d}{})", k.join(", "))
        }
        // `current_database` and `current_schema` are one expression
        // (Spark's `CurrentDatabase`, shown as `current_schema`);
        // `current_catalog` shows without parentheses.
        _ if name.eq_ignore_ascii_case("current_database") => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("current_schema({d}{})", k.join(", "))
        }
        _ if name.eq_ignore_ascii_case("current_catalog") && k.is_empty() && !distinct => {
            "current_catalog".to_owned()
        }
        // `RLike` prints its function name upper case, whichever spelling
        // (`rlike`, `regexp`, `regexp_like`) the query used.
        _ if matches!(
            name.to_ascii_lowercase().as_str(),
            "rlike" | "regexp" | "regexp_like"
        ) =>
        {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_uppercase(), named(k, arg_names))
        }
        // Single-argument math (`sec`, `csc`, `cot`, `exp`, ...) and
        // two-argument math (`power`, `atan2`, `hypot`) print upper case,
        // as Spark's `UnaryMathExpression` / `BinaryMathExpression` do;
        // aliases (`pow`, `sign`) and the rest print lower case.
        _ if is_upper_math(name) => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_uppercase(), named(k, arg_names))
        }
        _ => {
            let d = if distinct { "DISTINCT " } else { "" };
            format!("{}({d}{})", name.to_lowercase(), named(k, arg_names))
        }
    }
}

/// The arguments `k` with their `name => ` prefixes, when `arg_names[i]`
/// names one.
pub(super) fn named(k: &[&str], arg_names: Option<&[Option<String>]>) -> String {
    k.iter()
        .enumerate()
        .map(
            |(i, v)| match arg_names.and_then(|n| n.get(i)).and_then(Option::as_ref) {
                Some(n) => format!("{n} => {v}"),
                None => (*v).to_owned(),
            },
        )
        .collect::<Vec<_>>()
        .join(", ")
}

/// The default of the optional trailing integer argument of a regular-expression
/// function called with `n` arguments: the group index of `regexp_extract` /
/// `regexp_extract_all` (1) and `regexp_instr` (0), the start position of
/// `regexp_replace` (1).
pub(super) fn regexp_default_index(name: &str, n: usize) -> Option<i32> {
    match (name, n) {
        ("regexp_extract" | "regexp_extract_all", 2) | ("regexp_replace", 3) => Some(1),
        ("regexp_instr", 2) => Some(0),
        _ => None,
    }
}

/// Whether Spark prints `name` upper case: the `UnaryMathExpression` /
/// `BinaryMathExpression` functions without an alias (`sec` is `SEC`,
/// `power` is `POWER`), while aliases (`pow`, `sign`) and `ln` stay lower.
pub(super) fn is_upper_math(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "sqrt"
            | "cbrt"
            | "exp"
            | "expm1"
            | "log10"
            | "log2"
            | "log1p"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
            | "cot"
            | "csc"
            | "sec"
            | "degrees"
            | "radians"
            | "rint"
            | "signum"
            | "power"
            | "atan2"
            | "hypot"
            | "log"
            | "pi"
            | "e"
    )
}

pub(super) fn pretty_literal(v: &Literal, zone: Option<SessionZone>) -> String {
    match v {
        // A URL's credentials never reach a name (ADR 0038).
        Literal::String(s) => krishiv_common::secret::SecretUrl::parse(s)
            .map_or_else(|| s.clone(), |url| url.to_string()),
        Literal::Null(_) => "NULL".to_owned(),
        Literal::Boolean(v) => v.to_string(),
        Literal::Byte(v) => v.to_string(),
        Literal::Short(v) => v.to_string(),
        Literal::Integer(v) => v.to_string(),
        Literal::Long(v) => v.to_string(),
        Literal::Float(v) if v.is_finite() => crate::spark_fmt::float(*v),
        Literal::Float(v) => format!("CAST('{}' AS FLOAT)", special_float(f64::from(*v))),
        Literal::Double(v) if v.is_finite() => crate::spark_fmt::double(*v),
        Literal::Double(v) => format!("CAST('{}' AS DOUBLE)", special_float(*v)),
        Literal::Decimal {
            unscaled,
            data_type,
        } => crate::spark_fmt::decimal(*unscaled, u32::from(data_type.scale())),
        Literal::Binary(b) => {
            use std::fmt::Write as _;
            let mut out = String::from("X'");
            for byte in b {
                let _ = write!(out, "{byte:02X}");
            }
            out.push('\'');
            out
        }
        Literal::Date(days) => crate::spark_fmt::date(*days),
        // An instant renders in the session zone (a fixed offset; UTC when
        // unset), as Spark's `toPrettySQL` of a timestamp literal does.
        Literal::Timestamp(us) => {
            crate::spark_fmt::timestamp(SessionZone::local_micros(zone, *us), false)
        }
        Literal::TimestampNtz(us) => crate::spark_fmt::timestamp(*us, true),
        _ => v.to_string(),
    }
}

/// `NaN`, `Infinity` or `-Infinity`, as Spark spells them in a name.
pub(super) fn special_float(v: f64) -> &'static str {
    if v.is_nan() {
        "NaN"
    } else if v > 0.0 {
        "Infinity"
    } else {
        "-Infinity"
    }
}

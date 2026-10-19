// SPDX-License-Identifier: Apache-2.0
//! The pretty-printer: a tree in the style of Spark's `treeString`.
//!
//! Origins are not printed. Subquery plans are printed under the node that
//! uses them, behind a `:` rail, before the node's children. Read options
//! and file paths print as their typed values (ADR 0038): a secret as `*********(redacted)`, a reference
//! as `secret('scope', 'key')`, a URL with `*********(redacted)` for its credentials; no
//! pattern is applied to options. SQL text, which cannot be typed, goes
//! through [`krishiv_common::redact`]. A plan or expression deeper than
//! [`crate::MAX_DEPTH`] prints as a one-line marker instead of recursing.
//!
//! Plans can share subtrees (`df.join(df)`), so printing them as trees
//! could take exponential space. A relation reached from several places is
//! printed once, marked `(shared #n)`, and referred to as `Shared #n,
//! printed above` elsewhere; one expression's text is cut at 64 KiB.

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter, Write};

use krishiv_common::secret::SecretUrl;

use crate::walk::{self, Node};
use crate::{
    Builtin, CastMode, DayTimeField, Expr, ExprNode, FunctionCall, FunctionId, Grouping,
    JoinCondition, Literal, LogicalPlan, NullOrdering, Plan, PlanNode, ReadSource, ResolvedExpr,
    ResolvedFunction, SortDirection, SortOrder, SqlArgs, SubqueryKind, YearMonthField,
};

/// Printed instead of a node deeper than [`crate::MAX_DEPTH`].
const TOO_DEEP: &str = "<nested deeper than the maximum depth>";

impl Display for Plan {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Printer::new(self.root()).tree(f, self.root(), "", "")
    }
}

impl Display for LogicalPlan {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if walk::too_deep(Node::Plan(self)) {
            return writeln!(f, "{TOO_DEEP}");
        }
        Printer::new(self).tree(f, self, "", "")
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if walk::too_deep(Node::Expr(self)) {
            return f.write_str(TOO_DEEP);
        }
        Text(self).fmt(f)
    }
}

fn key(plan: &LogicalPlan) -> *const LogicalPlan {
    std::ptr::from_ref(plan)
}

/// Prints a plan tree. A relation shared through `Arc` by several parents
/// (a DAG, e.g. `df.join(df)`) is printed in full once, marked
/// `(shared #n)`, and as `Shared #n, printed above` at its other places, so
/// the output is linear in the distinct relations rather than exponential.
struct Printer {
    shared: HashSet<*const LogicalPlan>,
    labels: HashMap<*const LogicalPlan, usize>,
}

impl Printer {
    /// Finds the relations reached from more than one place.
    fn new(root: &LogicalPlan) -> Self {
        let mut reached: HashMap<*const LogicalPlan, usize> = HashMap::new();
        let mut stack = vec![root];
        while let Some(plan) = stack.pop() {
            let mut next = subqueries(plan);
            plan.for_each_child(&mut |c| next.push(c));
            for child in next {
                let count = reached.entry(key(child)).or_insert(0);
                *count += 1;
                if *count == 1 {
                    stack.push(child);
                }
            }
        }
        Self {
            shared: reached
                .into_iter()
                .filter(|&(_, n)| n > 1)
                .map(|(k, _)| k)
                .collect(),
            labels: HashMap::new(),
        }
    }

    /// Writes `plan` on a line starting with `first`, and its subtrees with
    /// `rest` as the rail prefix.
    fn tree(
        &mut self,
        f: &mut Formatter<'_>,
        plan: &LogicalPlan,
        first: &str,
        rest: &str,
    ) -> fmt::Result {
        if let Some(n) = self.labels.get(&key(plan)) {
            return writeln!(f, "{first}Shared #{n}, printed above");
        }
        write!(f, "{first}{}", NodeLine(plan))?;
        if self.shared.contains(&key(plan)) {
            let n = self.labels.len() + 1;
            self.labels.insert(key(plan), n);
            write!(f, " (shared #{n})")?;
        }
        writeln!(f)?;
        let subqueries = subqueries(plan);
        let mut children = Vec::new();
        plan.for_each_child(&mut |c| children.push(c));
        let inner = format!("{rest}:  ");
        for (i, sub) in subqueries.iter().enumerate() {
            self.branch(f, sub, &inner, i + 1 == subqueries.len())?;
        }
        for (i, child) in children.iter().enumerate() {
            self.branch(f, child, rest, i + 1 == children.len())?;
        }
        Ok(())
    }

    fn branch(
        &mut self,
        f: &mut Formatter<'_>,
        plan: &LogicalPlan,
        rest: &str,
        last: bool,
    ) -> fmt::Result {
        let (head, tail) = if last { ("+- ", "   ") } else { (":- ", ":  ") };
        self.tree(f, plan, &format!("{rest}{head}"), &format!("{rest}{tail}"))
    }
}

/// The distinct subquery plans in `plan`'s own expressions, in order,
/// without recursing and visiting a shared expression once.
fn subqueries(plan: &LogicalPlan) -> Vec<&LogicalPlan> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = Vec::new();
    plan.for_each_expr(&mut |e| stack.push(&**e));
    stack.reverse();
    let mut next = Vec::new();
    while let Some(e) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(e)) {
            continue;
        }
        if let ExprNode::Subquery { plan, .. } = &e.node {
            out.push(&**plan);
        }
        e.for_each_child(&mut |c| next.push(&**c));
        stack.extend(next.drain(..).rev());
    }
    out
}

/// One node's line, without its children.
struct NodeLine<'a>(&'a LogicalPlan);

impl Display for NodeLine<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.0.node {
            PlanNode::Read(source) => write!(f, "Read {}", SourceText(source)),
            PlanNode::OneRowRelation => f.write_str("OneRowRelation"),
            PlanNode::Sql { query, args } => {
                // Query text reaches sinks only as `RedactedSql`, which the
                // SQL front end builds from tokens; the printer shows none.
                write!(f, "Sql [{} bytes]", query.len())?;
                match args {
                    SqlArgs::None => Ok(()),
                    SqlArgs::Positional(v) => write!(f, ", [{}]", List(v)),
                    SqlArgs::Named(m) => {
                        f.write_str(", {")?;
                        for (i, (k, v)) in m.iter().enumerate() {
                            write!(f, "{}{k} = {}", sep(i), Text(v))?;
                        }
                        f.write_str("}")
                    }
                }
            }
            PlanNode::Scan(scan) => {
                write!(f, "Scan {} [", SourceText(&scan.source))?;
                for (i, c) in scan.output.iter().enumerate() {
                    write!(f, "{}{}#{}", sep(i), c.name, c.id.0)?;
                }
                f.write_str("]")?;
                scan.version
                    .map_or(Ok(()), |v| write!(f, " @ version {}", v.0))
            }
            PlanNode::Project { exprs, .. } => write!(f, "Project [{}]", List(exprs)),
            PlanNode::WithColumns { columns, .. } => {
                f.write_str("WithColumns [")?;
                for (i, (name, e)) in columns.iter().enumerate() {
                    write!(f, "{}{name} = {}", sep(i), Text(e))?;
                }
                f.write_str("]")
            }
            PlanNode::Filter { predicate, .. } => write!(f, "Filter {}", Text(predicate)),
            PlanNode::Aggregate {
                grouping,
                aggregates,
                ..
            } => {
                f.write_str("Aggregate ")?;
                match grouping {
                    Grouping::GroupBy(v) => write!(f, "[{}]", List(v))?,
                    Grouping::Rollup(v) => write!(f, "ROLLUP [{}]", List(v))?,
                    Grouping::Cube(v) => write!(f, "CUBE [{}]", List(v))?,
                    Grouping::GroupingSets { exprs, sets } => {
                        write!(f, "[{}] GROUPING SETS (", List(exprs))?;
                        for (i, s) in sets.iter().enumerate() {
                            write!(f, "{}({})", sep(i), List(s))?;
                        }
                        f.write_str(")")?;
                    }
                }
                write!(f, ", [{}]", List(aggregates))
            }
            PlanNode::Join {
                kind, condition, ..
            } => {
                write!(f, "Join {kind:?}")?;
                match condition {
                    JoinCondition::None => Ok(()),
                    JoinCondition::On(e) => write!(f, ", {}", Text(e)),
                    JoinCondition::Using(cols) => write!(f, ", USING ({})", cols.join(", ")),
                }
            }
            PlanNode::Sort { order, global, .. } => {
                f.write_str("Sort [")?;
                for (i, o) in order.iter().enumerate() {
                    write!(f, "{}{o}", sep(i))?;
                }
                write!(f, "], {global}")
            }
            PlanNode::Limit { limit, .. } => write!(f, "Limit {}", Text(limit)),
            PlanNode::SubqueryAlias { alias, columns, .. } => {
                write!(f, "SubqueryAlias {}", Name(std::slice::from_ref(alias)))?;
                if columns.is_empty() {
                    return Ok(());
                }
                write!(f, " ({})", columns.join(", "))
            }
            PlanNode::CteRef(r) => {
                write!(f, "CteRef {} [", Name(std::slice::from_ref(&r.name)))?;
                for (i, c) in r.output.iter().enumerate() {
                    write!(f, "{}{}#{}", sep(i), c.name, c.id.0)?;
                }
                f.write_str("]")
            }
            PlanNode::Shift { offset, .. } => write!(f, "Shift +{offset}"),
            PlanNode::Hint { name, params, .. } => {
                write!(f, "Hint {}", Name(std::slice::from_ref(name)))?;
                if params.is_empty() {
                    return Ok(());
                }
                write!(f, " ({})", List(params))
            }
            PlanNode::JoinHint { strategy, .. } => {
                write!(f, "JoinHint (strategy={})", strategy.name())
            }
            PlanNode::Union {
                by_name,
                allow_missing_columns,
                ..
            } => {
                f.write_str("Union")?;
                if *by_name {
                    f.write_str(" byName")?;
                }
                if *allow_missing_columns {
                    f.write_str(" allowMissingColumns")?;
                }
                Ok(())
            }
            PlanNode::Distinct { .. } => f.write_str("Distinct"),
            PlanNode::WithCte { ctes, .. } => {
                f.write_str("WithCte [")?;
                for (i, c) in ctes.iter().enumerate() {
                    write!(f, "{}{}", sep(i), c.name)?;
                }
                f.write_str("]")
            }
        }
    }
}

struct SourceText<'a>(&'a ReadSource);

impl Display for SourceText<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let options = match self.0 {
            ReadSource::Table { name, options } => {
                write!(f, "{}", Name(name))?;
                options
            }
            ReadSource::Files {
                format,
                paths,
                options,
            } => {
                write!(f, "{format} [")?;
                for (i, p) in paths.iter().enumerate() {
                    write!(f, "{}{p}", sep(i))?;
                }
                f.write_str("]")?;
                options
            }
        };
        if !options.is_empty() {
            f.write_str(" {")?;
            for (i, (k, v)) in options.iter().enumerate() {
                write!(f, "{}{k}={v}", sep(i))?;
            }
            f.write_str("}")?;
        }
        Ok(())
    }
}

/// `, ` before every item but the first.
const fn sep(i: usize) -> &'static str {
    if i == 0 { "" } else { ", " }
}

/// Comma-separated expressions.
struct List<'a>(&'a [std::sync::Arc<Expr>]);

impl Display for List<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, e) in self.0.iter().enumerate() {
            write!(f, "{}{}", sep(i), Text(e))?;
        }
        Ok(())
    }
}

/// A multi-part name, back-quoting parts that are not plain identifiers.
struct Name<'a>(&'a [String]);

impl Display for Name<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, part) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_char('.')?;
            }
            let plain = !part.is_empty()
                && part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !part.starts_with(|c: char| c.is_ascii_digit());
            if plain {
                f.write_str(part)?;
            } else {
                write!(f, "`{}`", part.replace('`', "``"))?;
            }
        }
        Ok(())
    }
}

/// A string with `\`, `'` and control characters escaped, on one line.
struct Escaped<'a>(&'a str);

impl Display for Escaped<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for c in self.0.chars() {
            match c {
                '\\' => f.write_str("\\\\")?,
                '\'' => f.write_str("\\'")?,
                '\n' => f.write_str("\\n")?,
                '\r' => f.write_str("\\r")?,
                '\t' => f.write_str("\\t")?,
                c if c.is_control() => write!(f, "\\u{:04x}", u32::from(c))?,
                c => f.write_char(c)?,
            }
        }
        Ok(())
    }
}

/// An expression's text, without the depth check of `Expr`'s `Display`.
/// It is written from an explicit stack of pieces, so printing never
/// recurses on expression depth.
struct Text<'a>(&'a Expr);

/// A piece of an expression's text.
enum Piece<'a> {
    Expr(&'a Expr),
    Str(&'static str),
    Owned(String),
}

/// The most text one expression prints. An expression whose subtrees are
/// shared (a DAG) can expand to exponentially long text; it is cut here.
const MAX_EXPR_TEXT: usize = 64 * 1024;

impl Display for Text<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut stack = vec![Piece::Expr(self.0)];
        let mut pieces = Vec::new();
        let mut written = 0;
        while let Some(piece) = stack.pop() {
            let text: &str = match &piece {
                Piece::Str(s) => s,
                Piece::Owned(s) => s,
                Piece::Expr(_) => "",
            };
            if written + text.len() > MAX_EXPR_TEXT {
                // Write only what fits, then the marker.
                let mut end = MAX_EXPR_TEXT - written;
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                f.write_str(&text[..end])?;
                return f.write_str("...(truncated)");
            }
            written += text.len();
            match piece {
                Piece::Str(s) => f.write_str(s)?,
                Piece::Owned(s) => f.write_str(&s)?,
                Piece::Expr(e) => {
                    expr_pieces(e, &mut pieces);
                    stack.extend(pieces.drain(..).rev());
                }
            }
        }
        Ok(())
    }
}

/// A literal's text, never formatting more of a long string or binary
/// value than an expression can print: the clipped value already overflows
/// [`MAX_EXPR_TEXT`], so the writer cuts it and adds the marker.
fn literal_text(v: &Literal) -> String {
    match v {
        Literal::String(s) if s.len() > MAX_EXPR_TEXT => {
            let mut end = MAX_EXPR_TEXT + 1;
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            Literal::String(s[..end].to_owned()).to_string()
        }
        Literal::Binary(b) if b.len() > MAX_EXPR_TEXT => {
            Literal::Binary(b[..=MAX_EXPR_TEXT].to_vec()).to_string()
        }
        v => v.to_string(),
    }
}

/// Appends `args` separated by `, `.
fn list<'a>(args: &'a [std::sync::Arc<Expr>], out: &mut Vec<Piece<'a>>) {
    for (i, e) in args.iter().enumerate() {
        if i > 0 {
            out.push(Piece::Str(", "));
        }
        out.push(Piece::Expr(e));
    }
}

/// Appends the pieces of one expression node, its children as
/// `Piece::Expr`.
fn expr_pieces<'a>(e: &'a Expr, out: &mut Vec<Piece<'a>>) {
    match &e.node {
        ExprNode::Column(c) => {
            let id = c.plan_id.map(|id| format!("@plan{id}")).unwrap_or_default();
            out.push(Piece::Owned(format!("{}{id}", Name(&c.name))));
        }
        ExprNode::Literal(v) => out.push(Piece::Owned(literal_text(v))),
        ExprNode::Star(s) => {
            let q = if s.qualifier.is_empty() {
                String::new()
            } else {
                format!("{}.", Name(&s.qualifier))
            };
            let id = s.plan_id.map(|id| format!("@plan{id}")).unwrap_or_default();
            out.push(Piece::Owned(format!("{q}*{id}")));
        }
        ExprNode::Function(FunctionCall {
            name,
            args,
            distinct,
            filter,
        }) => call(
            name,
            Builtin::from_name(name),
            args,
            *distinct,
            filter.as_deref(),
            out,
        ),
        ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
            id,
            args,
            distinct,
            filter,
            ..
        })) => {
            let op = match id {
                FunctionId::Builtin(b) => Some(*b),
                FunctionId::Registry { .. } => None,
            };
            call(id.name(), op, args, *distinct, filter.as_deref(), out);
        }
        ExprNode::Cast { expr, to, mode } => {
            out.push(Piece::Str(match mode {
                CastMode::Try => "TRY_CAST(",
                CastMode::Default | CastMode::Legacy | CastMode::Ansi => "CAST(",
            }));
            out.push(Piece::Expr(expr));
            out.push(Piece::Owned(format!(" AS {})", to.sql())));
        }
        ExprNode::Alias { expr, name } => {
            out.push(Piece::Expr(expr));
            out.push(Piece::Owned(format!(
                " AS {}",
                Name(std::slice::from_ref(name))
            )));
        }
        ExprNode::Subquery { kind, .. } => match kind {
            SubqueryKind::Scalar => out.push(Piece::Str("scalar-subquery")),
            SubqueryKind::Exists => out.push(Piece::Str("exists")),
            SubqueryKind::In(values) => {
                out.push(Piece::Str("(("));
                list(values, out);
                out.push(Piece::Str(") IN (list-query))"));
            }
        },
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            out.push(Piece::Owned(format!("{}#{}", c.name, c.id.0)));
        }
        ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => {
            out.push(Piece::Owned(format!("outer({}#{})", c.name, c.id.0)));
        }
        ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
            out.push(Piece::Expr(expr));
            out.push(Piece::Owned(format!(" AS {}#{}", output.name, output.id.0)));
        }
    }
}

/// Appends a call: operators infix or in their SQL form, others as
/// `name(args)`.
fn call<'a>(
    name: &str,
    op: Option<Builtin>,
    args: &'a [std::sync::Arc<Expr>],
    distinct: bool,
    filter: Option<&'a Expr>,
    out: &mut Vec<Piece<'a>>,
) {
    let arg = |i: usize| Piece::Expr(&args[i]);
    match (op, args.len()) {
        (Some(op), 2) if is_infix(op) && !distinct => {
            out.extend([Piece::Str("("), arg(0), Piece::Str(" ")]);
            out.extend([
                Piece::Str(op.name()),
                Piece::Str(" "),
                arg(1),
                Piece::Str(")"),
            ]);
        }
        (Some(Builtin::Not), 1) => out.extend([Piece::Str("(NOT "), arg(0), Piece::Str(")")]),
        (Some(Builtin::Negative), 1) => {
            out.extend([Piece::Str("(- "), arg(0), Piece::Str(")")]);
        }
        (Some(Builtin::In), n) if n >= 1 => {
            out.extend([Piece::Str("("), arg(0), Piece::Str(" IN (")]);
            list(&args[1..], out);
            out.push(Piece::Str("))"));
        }
        (Some(Builtin::Between), 3) => out.extend([
            Piece::Str("("),
            arg(0),
            Piece::Str(" BETWEEN "),
            arg(1),
            Piece::Str(" AND "),
            arg(2),
            Piece::Str(")"),
        ]),
        (Some(Builtin::When), n) if n >= 2 => {
            out.push(Piece::Str("CASE"));
            let (pairs, rest) = args.as_chunks::<2>();
            for [cond, value] in pairs {
                out.extend([Piece::Str(" WHEN "), Piece::Expr(cond)]);
                out.extend([Piece::Str(" THEN "), Piece::Expr(value)]);
            }
            if let [e] = rest {
                out.extend([Piece::Str(" ELSE "), Piece::Expr(e)]);
            }
            out.push(Piece::Str(" END"));
        }
        _ => {
            out.push(Piece::Owned(format!("{name}(")));
            if distinct {
                out.push(Piece::Str("DISTINCT "));
            }
            list(args, out);
            out.push(Piece::Str(")"));
        }
    }
    if let Some(e) = filter {
        out.extend([
            Piece::Str(" FILTER (WHERE "),
            Piece::Expr(e),
            Piece::Str(")"),
        ]);
    }
}

impl Display for SortOrder {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let dir = match self.direction {
            SortDirection::Ascending => "ASC",
            SortDirection::Descending => "DESC",
        };
        let nulls = match self.nulls {
            NullOrdering::NullsFirst => "NULLS FIRST",
            NullOrdering::NullsLast => "NULLS LAST",
        };
        write!(f, "{} {dir} {nulls}", Text(&self.expr))
    }
}

/// Whether the pretty-printer writes `op` between its two arguments.
const fn is_infix(op: Builtin) -> bool {
    matches!(
        op,
        Builtin::Eq
            | Builtin::EqNullSafe
            | Builtin::Lt
            | Builtin::LtEq
            | Builtin::Gt
            | Builtin::GtEq
            | Builtin::And
            | Builtin::Or
            | Builtin::Add
            | Builtin::Subtract
            | Builtin::Multiply
            | Builtin::Divide
            | Builtin::Remainder
            | Builtin::IntegralDivide
            | Builtin::Like
            | Builtin::ILike
            | Builtin::RLike
    )
}

/// Microseconds per second.
const MICROS_PER_SECOND: i64 = 1_000_000;

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null(None) => f.write_str("NULL"),
            Self::Null(Some(t)) => write!(f, "CAST(NULL AS {})", t.sql()),
            Self::Boolean(v) => write!(f, "{v}"),
            Self::Byte(v) => write!(f, "{v}Y"),
            Self::Short(v) => write!(f, "{v}S"),
            Self::Integer(v) => write!(f, "{v}"),
            Self::Long(v) => write!(f, "{v}L"),
            Self::Float(v) if v.is_finite() => write!(f, "{v:?}F"),
            Self::Float(v) => write!(f, "CAST('{}' AS FLOAT)", special(f64::from(*v))),
            Self::Double(v) if v.is_finite() => write!(f, "{v:?}D"),
            Self::Double(v) => write!(f, "CAST('{}' AS DOUBLE)", special(*v)),
            Self::Decimal {
                unscaled,
                data_type,
            } => write_decimal(f, *unscaled, data_type.scale()),
            // A URL's credentials never print, even in a literal.
            Self::String(s) => match SecretUrl::parse(s) {
                Some(url) => write!(f, "'{}'", Escaped(&url.to_string())),
                None => write!(f, "'{}'", Escaped(s)),
            },
            Self::Binary(b) => {
                f.write_str("X'")?;
                b.iter().try_for_each(|byte| write!(f, "{byte:02X}"))?;
                f.write_char('\'')
            }
            Self::Date(days) => {
                let (y, m, d) = civil_from_days(i64::from(*days));
                let sign = if y < 0 { "-" } else { "" };
                write!(f, "DATE '{sign}{:04}-{m:02}-{d:02}'", y.unsigned_abs())
            }
            Self::Timestamp(us) => write!(f, "TIMESTAMP_MICROS({us})"),
            Self::TimestampNtz(us) => write!(f, "TIMESTAMP_NTZ_MICROS({us})"),
            Self::YearMonthInterval { months, start, end } => {
                let sign = if *months < 0 { "-" } else { "" };
                let m = months.unsigned_abs();
                let value = match (start, end) {
                    (YearMonthField::Year, YearMonthField::Month) => {
                        format!("{}-{}", m / 12, m % 12)
                    }
                    (YearMonthField::Year, YearMonthField::Year) => (m / 12).to_string(),
                    (YearMonthField::Month, _) => m.to_string(),
                };
                write!(
                    f,
                    "INTERVAL '{sign}{value}' {}",
                    fields(ym_name(*start), ym_name(*end))
                )
            }
            Self::DayTimeInterval { micros, start, end } => {
                let sign = if *micros < 0 { "-" } else { "" };
                write!(
                    f,
                    "INTERVAL '{sign}{}' {}",
                    day_time_value(micros.unsigned_abs(), *start, *end),
                    fields(dt_name(*start), dt_name(*end))
                )
            }
        }
    }
}

/// `NaN`, `Infinity` or `-Infinity`, as Spark spells them.
fn special(v: f64) -> &'static str {
    if v.is_nan() {
        "NaN"
    } else if v > 0.0 {
        "Infinity"
    } else {
        "-Infinity"
    }
}

const fn ym_name(f: YearMonthField) -> &'static str {
    match f {
        YearMonthField::Year => "YEAR",
        YearMonthField::Month => "MONTH",
    }
}

const fn dt_name(f: DayTimeField) -> &'static str {
    match f {
        DayTimeField::Day => "DAY",
        DayTimeField::Hour => "HOUR",
        DayTimeField::Minute => "MINUTE",
        DayTimeField::Second => "SECOND",
    }
}

/// `START` or `START TO END`.
fn fields(start: &str, end: &str) -> String {
    if start == end {
        start.to_owned()
    } else {
        format!("{start} TO {end}")
    }
}

/// The ANSI interval string of `micros` for fields `start..=end`: the
/// leading field unbounded, the others in their range, e.g. `1 02:03:04.5`.
fn day_time_value(micros: u64, start: DayTimeField, end: DayTimeField) -> String {
    let units = [86_400, 3_600, 60, 1].map(|s: u64| s * MICROS_PER_SECOND.unsigned_abs());
    let (first, last) = (start as usize, end as usize);
    let mut rest = micros;
    let mut out = String::new();
    for (i, unit) in units.iter().enumerate().take(last + 1).skip(first) {
        let whole = rest / unit;
        rest %= unit;
        match i {
            _ if i == first => write!(out, "{whole}"),
            1 => write!(out, " {whole:02}"),
            _ => write!(out, ":{whole:02}"),
        }
        .ok();
    }
    let fraction = if end == DayTimeField::Second { rest } else { 0 };
    if fraction > 0 {
        let digits = format!("{fraction:06}");
        out.push('.');
        out.push_str(digits.trim_end_matches('0'));
    }
    out
}

/// Writes `unscaled * 10^-scale` in plain notation, e.g. `0.06`.
fn write_decimal(f: &mut Formatter<'_>, unscaled: i128, scale: u8) -> fmt::Result {
    let digits = unscaled.unsigned_abs().to_string();
    let scale = usize::from(scale);
    let sign = if unscaled < 0 { "-" } else { "" };
    if scale == 0 {
        return write!(f, "{sign}{digits}");
    }
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = padded.split_at(padded.len() - scale);
    write!(f, "{sign}{int}.{frac}")
}

/// Proleptic Gregorian `(year, month, day)` of a day count since 1970-01-01
/// (H. Hinnant's `civil_from_days`).
const fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    (y, m, d)
}

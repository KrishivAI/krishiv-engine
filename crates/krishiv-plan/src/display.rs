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
    Builtin, CastMode, DayTimeField, Expr, ExprNode, FrameBound, FrameUnits, FunctionCall,
    FunctionId, Grouping, JoinCondition, JoinKind, Literal, LogicalPlan, NullOrdering, Plan,
    PlanNode, ReadSource, ResolvedExpr, ResolvedFunction, SortDirection, SortOrder, SqlArgs,
    SubqueryKind, WindowSpec, WindowSpecRef, YearMonthField,
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

/// Text to append to a relation's line, if any.
type Note<'n> = &'n dyn Fn(&LogicalPlan) -> Option<String>;

/// A plan printed with a note after each relation's line
/// ([`LogicalPlan::display_with`]).
pub struct Annotated<'a> {
    plan: &'a LogicalPlan,
    note: Note<'a>,
}

impl fmt::Debug for Annotated<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Annotated").finish_non_exhaustive()
    }
}

impl Display for Annotated<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if walk::too_deep(Node::Plan(self.plan)) {
            return writeln!(f, "{TOO_DEEP}");
        }
        let mut printer = Printer::new(self.plan);
        printer.note = Some(self.note);
        printer.tree(f, self.plan, "", "")
    }
}

impl LogicalPlan {
    /// The plan as `Display` prints it, each relation's line followed by
    /// `, ` and `note(relation)` when that is `Some` (`EXPLAIN COST` adds
    /// the estimated statistics this way).
    #[must_use]
    pub fn display_with<'a>(&'a self, note: Note<'a>) -> Annotated<'a> {
        Annotated { plan: self, note }
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
struct Printer<'n> {
    shared: HashSet<*const LogicalPlan>,
    labels: HashMap<*const LogicalPlan, usize>,
    /// Text to append to a relation's line.
    note: Option<Note<'n>>,
}

impl Printer<'_> {
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
            note: None,
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
        if let Some(text) = self.note.and_then(|note| note(plan)) {
            write!(f, ", {text}")?;
        }
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
            PlanNode::PartialAggregate {
                keys, aggregates, ..
            } => write!(
                f,
                "PartialAggregate [{}], [{}]",
                List(keys),
                List(aggregates)
            ),
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
                match kind {
                    JoinKind::Existence(id) => write!(f, "Join ExistenceJoin(exists#{})", id.0)?,
                    _ => write!(f, "Join {kind:?}")?,
                }
                match condition {
                    JoinCondition::None => Ok(()),
                    JoinCondition::On(e) => write!(f, ", {}", Text(e)),
                    JoinCondition::Using(cols) => write!(f, ", USING ({})", cols.join(", ")),
                    JoinCondition::Natural => write!(f, ", NATURAL"),
                }
            }
            PlanNode::Sort { order, global, .. } => {
                f.write_str("Sort [")?;
                for (i, o) in order.iter().enumerate() {
                    write!(f, "{}{o}", sep(i))?;
                }
                write!(f, "], {global}")
            }
            PlanNode::Window {
                exprs,
                partition_by,
                order_by,
                ..
            } => {
                write!(f, "Window [{}], [{}], [", List(exprs), List(partition_by))?;
                for (i, o) in order_by.iter().enumerate() {
                    write!(f, "{}{o}", sep(i))?;
                }
                f.write_str("]")
            }
            PlanNode::Limit { limit, .. } => write!(f, "Limit {}", Text(limit)),
            PlanNode::Offset { offset, .. } => write!(f, "Offset {}", Text(offset)),
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
            PlanNode::PipeOperator { .. } => f.write_str("PipeOperator"),
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
            // Spark's `Intersect false`, `Except All true`, ...
            PlanNode::SetOp { op, all, .. } => match (op, all) {
                (crate::SetOperation::Intersect, false) => f.write_str("Intersect false"),
                (crate::SetOperation::Intersect, true) => f.write_str("Intersect All true"),
                (crate::SetOperation::Except, false) => f.write_str("Except false"),
                (crate::SetOperation::Except, true) => f.write_str("Except All true"),
            },
            PlanNode::ReplicateRows { count, .. } => {
                write!(f, "Generate replicaterows({})", Text(count))
            }
            // Spark's `Generate explode(a), false, [col#3]`.
            PlanNode::Generate {
                kind,
                outer,
                arg,
                output,
                ..
            } => {
                write!(
                    f,
                    "Generate {}{}({}), {}, [",
                    kind.name(),
                    if *outer { "_outer" } else { "" },
                    Text(arg),
                    outer
                )?;
                for (i, c) in output.iter().enumerate() {
                    write!(f, "{}{}#{}", sep(i), c.name, c.id.0)?;
                }
                f.write_str("]")
            }
            // Spark's `Expand [[a, null, 1], [a, b, 0]], [a, b, spark_grouping_id]`.
            PlanNode::Expand {
                projections,
                output,
                ..
            } => {
                f.write_str("Expand [")?;
                for (i, p) in projections.iter().enumerate() {
                    write!(f, "{}[", sep(i))?;
                    for (j, e) in p.iter().enumerate() {
                        write!(f, "{}{}", sep(j), Text(e))?;
                    }
                    f.write_str("]")?;
                }
                f.write_str("], [")?;
                for (i, c) in output.iter().enumerate() {
                    write!(f, "{}{}#{}", sep(i), c.name, c.id.0)?;
                }
                f.write_str("]")
            }
            PlanNode::WithCte { ctes, .. } => {
                f.write_str("WithCte [")?;
                for (i, c) in ctes.iter().enumerate() {
                    write!(f, "{}{}", sep(i), c.name)?;
                }
                f.write_str("]")
            }
            PlanNode::Pivot {
                pivot_columns,
                values,
                aggregates,
                ..
            } => {
                f.write_str("Pivot [")?;
                for (i, e) in pivot_columns.iter().enumerate() {
                    write!(f, "{}{}", sep(i), Text(e))?;
                }
                f.write_str("], [")?;
                for (i, v) in values.iter().enumerate() {
                    write!(f, "{}[", sep(i))?;
                    for (j, e) in v.exprs.iter().enumerate() {
                        write!(f, "{}{}", sep(j), Text(e))?;
                    }
                    if let Some(alias) = &v.alias {
                        write!(f, " AS {alias}")?;
                    }
                    f.write_str("]")?;
                }
                f.write_str("], [")?;
                for (i, e) in aggregates.iter().enumerate() {
                    write!(f, "{}{}", sep(i), Text(e))?;
                }
                f.write_str("]")
            }
            PlanNode::Unpivot {
                value_columns,
                name_column,
                column_sets,
                include_nulls,
                ..
            } => {
                if *include_nulls {
                    f.write_str("UnpivotIncludeNulls ")?;
                } else {
                    f.write_str("Unpivot ")?;
                }
                write!(f, "[{} FOR {name_column} IN (", value_columns.join(", "))?;
                for (i, s) in column_sets.iter().enumerate() {
                    write!(f, "{}[", sep(i))?;
                    for (j, e) in s.columns.iter().enumerate() {
                        write!(f, "{}{}", sep(j), Text(e))?;
                    }
                    if let Some(alias) = &s.alias {
                        write!(f, " AS {alias}")?;
                    }
                    f.write_str("]")?;
                }
                f.write_str(")]")
            }
            // Spark's `LocalRelation [col1, col2]`, which an inline table
            // becomes; the row count tells single-row tables apart.
            PlanNode::InlineTable { rows, columns } => {
                write!(
                    f,
                    "InlineTable [{}] ({} rows)",
                    columns.join(", "),
                    rows.len()
                )
            }
            PlanNode::TableFunction { name, args } => {
                write!(f, "TableFunction {name} [{}]", List(args))
            }
            PlanNode::Range {
                start,
                end,
                step,
                num_slices,
                ..
            } => match num_slices {
                Some(n) => write!(f, "Range [{start}, {end}, step {step}, slices {n}]"),
                None => write!(f, "Range [{start}, {end}, step {step}]"),
            },
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

/// Appends `args` separated by `, `, each with its `name => ` prefix when
/// `arg_names[i]` is one.
fn named_list<'a>(
    args: &'a [std::sync::Arc<Expr>],
    arg_names: &'a [Option<String>],
    out: &mut Vec<Piece<'a>>,
) {
    for (i, e) in args.iter().enumerate() {
        if i > 0 {
            out.push(Piece::Str(", "));
        }
        if let Some(n) = arg_names.get(i).and_then(Option::as_ref) {
            out.push(Piece::Owned(format!("{n} => ")));
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
        ExprNode::TypedLiteral { kind, value } => {
            let keyword = match kind {
                crate::TypedLiteralKind::Date => "DATE",
                crate::TypedLiteralKind::Timestamp => "TIMESTAMP",
                crate::TypedLiteralKind::TimestampNtz => "TIMESTAMP_NTZ",
                crate::TypedLiteralKind::TimestampLtz => "TIMESTAMP_LTZ",
                crate::TypedLiteralKind::Time => "TIME",
            };
            out.push(Piece::Owned(format!("{keyword} '{value}'")));
        }
        ExprNode::Star(s) => {
            let q = if s.qualifier.is_empty() {
                String::new()
            } else {
                format!("{}.", Name(&s.qualifier))
            };
            let id = s.plan_id.map(|id| format!("@plan{id}")).unwrap_or_default();
            let mut text = format!("{q}*{id}");
            if !s.except.is_empty() {
                let cols: Vec<String> =
                    s.except.iter().map(|path| Name(path).to_string()).collect();
                text.push_str(&format!(" EXCEPT ({})", cols.join(", ")));
            }
            if let Some(n) = s.position {
                text.push_str(&format!(" @{n}"));
            }
            if !s.replace.is_empty() {
                let cols: Vec<&str> = s.replace.iter().map(|(n, _)| n.as_str()).collect();
                text.push_str(&format!(" REPLACE ({})", cols.join(", ")));
            }
            out.push(Piece::Owned(text));
        }
        ExprNode::Function(FunctionCall {
            name,
            qualifier,
            args,
            arg_names,
            distinct,
            filter,
            within_group,
        }) => {
            let qualified = if qualifier.is_empty() {
                name.clone()
            } else {
                format!("{}.{name}", Name(qualifier))
            };
            call(
                &qualified,
                Builtin::from_name(name),
                args,
                arg_names,
                *distinct,
                filter.as_deref(),
                within_group,
                out,
            );
        }
        ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
            id,
            args,
            distinct,
            filter,
            within_group,
            ..
        })) => {
            // `mode` keeps Spark's `Mode.sql` form: the ordering is the
            // tie-break (ascending written as descending) and takes no
            // arguments (`mode() WITHIN GROUP (ORDER BY col DESC)`).
            if id.name() == "mode" && !within_group.is_empty() {
                out.push(Piece::Str("mode() WITHIN GROUP (ORDER BY "));
                if let Some(o) = within_group.first() {
                    out.push(Piece::Expr(&o.expr));
                    if o.direction == SortDirection::Descending {
                        out.push(Piece::Str(" DESC"));
                    }
                }
                out.push(Piece::Str(")"));
                if let Some(e) = filter {
                    out.extend([
                        Piece::Str(" FILTER (WHERE "),
                        Piece::Expr(e),
                        Piece::Str(")"),
                    ]);
                }
                return;
            }
            let op = match id {
                FunctionId::Builtin(b) => Some(*b),
                FunctionId::Registry { .. } | FunctionId::Udf { .. } => None,
            };
            call(
                id.name(),
                op,
                args,
                &[],
                *distinct,
                filter.as_deref(),
                within_group,
                out,
            );
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
        ExprNode::Field { base, name } => {
            out.push(Piece::Expr(base));
            out.push(Piece::Owned(format!(
                ".{}",
                Name(std::slice::from_ref(name))
            )));
        }
        ExprNode::UpdateFields { base, ops } => {
            out.push(Piece::Str("update_fields("));
            out.push(Piece::Expr(base));
            for op in ops {
                out.push(Piece::Str(", "));
                match op {
                    crate::FieldOp::With { name, value } => {
                        out.push(Piece::Owned(format!(
                            "WithField({}, ",
                            Name(std::slice::from_ref(name))
                        )));
                        out.push(Piece::Expr(value));
                        out.push(Piece::Str(")"));
                    }
                    crate::FieldOp::Drop { name } => {
                        out.push(Piece::Owned(format!(
                            "DropField({})",
                            Name(std::slice::from_ref(name))
                        )));
                    }
                }
            }
            out.push(Piece::Str(")"));
        }
        ExprNode::Resolved(ResolvedExpr::StateInput { columns, .. }) => {
            out.push(Piece::Str("state("));
            list(columns, out);
            out.push(Piece::Str(")"));
        }
        ExprNode::Resolved(ResolvedExpr::Field {
            base,
            name,
            ordinal,
            ..
        }) => {
            out.push(Piece::Expr(base));
            out.push(Piece::Owned(format!(
                ".{}#{ordinal}",
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
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::LambdaVar(c)) => {
            out.push(Piece::Owned(format!("{}#{}", c.name, c.id.0)));
        }
        ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => {
            out.push(Piece::Owned(format!("outer({}#{})", c.name, c.id.0)));
        }
        ExprNode::Lambda { params, body } => {
            let names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
            out.push(Piece::Owned(format!("({}) -> ", names.join(", "))));
            out.push(Piece::Expr(body));
        }
        ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
            out.push(Piece::Expr(expr));
            out.push(Piece::Owned(format!(" AS {}#{}", output.name, output.id.0)));
        }
        ExprNode::Window(w) => {
            out.push(Piece::Expr(&w.function));
            match &w.window {
                WindowSpecRef::Named(name) => out.push(Piece::Owned(format!(
                    " OVER {}",
                    Name(std::slice::from_ref(name))
                ))),
                WindowSpecRef::Spec(spec) => window_pieces(spec, out),
            }
        }
    }
}

/// ` OVER (...)` as Spark's `WindowSpecDefinition.sql` writes it: the
/// partitioning, the ordering and the frame, separated by spaces (an
/// unspecified frame writes nothing).
fn window_pieces<'a>(spec: &'a WindowSpec, out: &mut Vec<Piece<'a>>) {
    out.push(Piece::Str(" OVER ("));
    let mut first = true;
    let mut gap = |out: &mut Vec<Piece<'a>>| {
        if !std::mem::take(&mut first) {
            out.push(Piece::Str(" "));
        }
    };
    if !spec.partition_by.is_empty() {
        gap(out);
        out.push(Piece::Str("PARTITION BY "));
        list(&spec.partition_by, out);
    }
    if !spec.order_by.is_empty() {
        gap(out);
        out.push(Piece::Str("ORDER BY "));
        for (i, o) in spec.order_by.iter().enumerate() {
            if i > 0 {
                out.push(Piece::Str(", "));
            }
            out.push(Piece::Expr(&o.expr));
            out.push(Piece::Str(order_suffix(o)));
        }
    }
    if let Some(frame) = &spec.frame {
        gap(out);
        out.push(Piece::Str(match frame.units {
            FrameUnits::Rows => "ROWS BETWEEN ",
            FrameUnits::Range => "RANGE BETWEEN ",
        }));
        bound_pieces(&frame.lower, out);
        out.push(Piece::Str(" AND "));
        bound_pieces(&frame.upper, out);
    }
    out.push(Piece::Str(")"));
}

/// A frame bound as Spark writes it: a negated value is `n PRECEDING`, any
/// other value `n FOLLOWING`.
fn bound_pieces<'a>(b: &'a FrameBound, out: &mut Vec<Piece<'a>>) {
    match b {
        FrameBound::UnboundedPreceding => out.push(Piece::Str("UNBOUNDED PRECEDING")),
        FrameBound::CurrentRow => out.push(Piece::Str("CURRENT ROW")),
        FrameBound::UnboundedFollowing => out.push(Piece::Str("UNBOUNDED FOLLOWING")),
        FrameBound::Offset(e) => match crate::expr::negated(e) {
            Some(n) => out.extend([Piece::Expr(n), Piece::Str(" PRECEDING")]),
            None => out.extend([Piece::Expr(e), Piece::Str(" FOLLOWING")]),
        },
    }
}

/// ` ASC NULLS FIRST` and the like.
pub(crate) const fn order_suffix(o: &SortOrder) -> &'static str {
    match (o.direction, o.nulls) {
        (SortDirection::Ascending, NullOrdering::NullsFirst) => " ASC NULLS FIRST",
        (SortDirection::Ascending, NullOrdering::NullsLast) => " ASC NULLS LAST",
        (SortDirection::Descending, NullOrdering::NullsFirst) => " DESC NULLS FIRST",
        (SortDirection::Descending, NullOrdering::NullsLast) => " DESC NULLS LAST",
    }
}

/// Appends a call: operators infix or in their SQL form, others as
/// `name(args)`. A named argument prints as `name => value`; a quantified
/// `LIKE`/`ILIKE` prints in its SQL form (`a LIKE ANY (p, ...)`).
#[expect(
    clippy::too_many_arguments,
    reason = "one call site per expression form, every field of the call"
)]
fn call<'a>(
    name: &str,
    op: Option<Builtin>,
    args: &'a [std::sync::Arc<Expr>],
    arg_names: &'a [Option<String>],
    distinct: bool,
    filter: Option<&'a Expr>,
    within_group: &'a [SortOrder],
    out: &mut Vec<Piece<'a>>,
) {
    let arg = |i: usize| Piece::Expr(&args[i]);
    // `to_varchar` is Spark's `to_char` under its alias (the goldens show
    // `to_char(78.12, $99.99)` for `to_varchar(78.12, '$99.99')`).
    let name = if name == "to_varchar" {
        "to_char"
    } else {
        name
    };
    // `a [NOT] LIKE|ILIKE ANY|SOME|ALL (p, ...)`, as Spark writes the form.
    match (op, args.len()) {
        // The key of `aes_encrypt`/`aes_decrypt` never reaches a plan, an
        // error message or a log (ADR 0042).
        _ if matches!(name, "aes_encrypt" | "aes_decrypt" | "try_aes_decrypt")
            && args.len() >= 2
            && !distinct =>
        {
            out.extend([
                Piece::Owned(format!("{name}(")),
                arg(0),
                Piece::Str(", ***"),
            ]);
            for a in &args[2..] {
                out.extend([Piece::Str(", "), Piece::Expr(a)]);
            }
            out.push(Piece::Str(")"));
        }
        (Some(op), n) if n >= 1 && op.quantified_like().is_some() => {
            let q = op.quantified_like().unwrap_or_default();
            out.extend([arg(0), Piece::Str(if q.negated { " NOT " } else { " " })]);
            out.extend([
                Piece::Str(if q.insensitive { "ILIKE " } else { "LIKE " }),
                Piece::Str(if q.all { "ALL (" } else { "ANY (" }),
            ]);
            list(&args[1..], out);
            out.push(Piece::Str(")"));
        }
        (Some(op), 2) if is_infix(op) && !distinct => {
            out.extend([Piece::Str("("), arg(0), Piece::Str(" ")]);
            out.extend([
                Piece::Str(op.name()),
                Piece::Str(" "),
                arg(1),
                Piece::Str(")"),
            ]);
        }
        // The bitwise operators are registry functions; Spark still shows
        // them infix (`(20181117 >> 2)`), `~` prefix.
        (None, 2) if !distinct => match registry_infix(name) {
            Some(symbol) => {
                out.extend([Piece::Str("("), arg(0), Piece::Str(" ")]);
                out.extend([Piece::Str(symbol), Piece::Str(" "), arg(1), Piece::Str(")")]);
            }
            None => {
                out.push(Piece::Owned(format!("{name}(")));
                list(args, out);
                out.push(Piece::Str(")"));
            }
        },
        (None, 1) if name == "~" && !distinct => {
            out.extend([Piece::Str("~"), arg(0)]);
        }
        (Some(Builtin::Not), 1) => out.extend([Piece::Str("(NOT "), arg(0), Piece::Str(")")]),
        (Some(Builtin::Negative), 1) => {
            out.extend([Piece::Str("(- "), arg(0), Piece::Str(")")]);
        }
        // Unary `-` lowers as `"-"` with one argument (see `builtin_op`).
        _ if name == "-" && args.len() == 1 => {
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
            named_list(args, arg_names, out);
            out.push(Piece::Str(")"));
        }
    }
    if !within_group.is_empty() {
        out.push(Piece::Str(" WITHIN GROUP (ORDER BY "));
        for (i, o) in within_group.iter().enumerate() {
            if i > 0 {
                out.push(Piece::Str(", "));
            }
            out.push(Piece::Expr(&o.expr));
            out.push(Piece::Str(order_suffix(o)));
        }
        out.push(Piece::Str(")"));
    }
    // Spark appends the filter after the whole function text.
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

/// The infix symbol of a registry bitwise operator (`&`, `|`, `^`,
/// `<<`, `>>`, `>>>`), if `name` is one.
fn registry_infix(name: &str) -> Option<&'static str> {
    match name {
        "&" => Some("&"),
        "|" => Some("|"),
        "^" => Some("^"),
        "<<" => Some("<<"),
        ">>" => Some(">>"),
        ">>>" => Some(">>>"),
        _ => None,
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
            Self::Float(v) if v.is_finite() => write!(f, "{}F", crate::spark_fmt::float(*v)),
            Self::Float(v) => write!(f, "CAST('{}' AS FLOAT)", special(f64::from(*v))),
            Self::Double(v) if v.is_finite() => write!(f, "{}D", crate::spark_fmt::double(*v)),
            Self::Double(v) => write!(f, "CAST('{}' AS DOUBLE)", special(*v)),
            Self::Decimal {
                unscaled,
                data_type,
            } => f.write_str(
                crate::spark_fmt::decimal(*unscaled, u32::from(data_type.scale())).as_str(),
            ),
            // A URL's credentials never print, even in a literal.
            Self::String(s) => match SecretUrl::parse(s) {
                Some(url) => write!(f, "'{}'", Escaped(&url.to_string())),
                None => write!(f, "'{}'", Escaped(s)),
            },
            Self::Map(pairs) => {
                f.write_str("MAP(")?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "'{}', '{}'", Escaped(k), Escaped(v))?;
                }
                f.write_str(")")
            }
            Self::Binary(b) => {
                f.write_str("X'")?;
                b.iter().try_for_each(|byte| write!(f, "{byte:02X}"))?;
                f.write_char('\'')
            }
            Self::Date(days) => f.write_str(crate::spark_fmt::date(*days).as_str()),
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

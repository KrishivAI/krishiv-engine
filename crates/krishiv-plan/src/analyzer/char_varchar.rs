// SPDX-License-Identifier: Apache-2.0
//! `CHAR(n)` / `VARCHAR(n)` columns (design P2-T001b). The column's type is `string`; the declared type travels
//! beside it, as Spark carries it in a field's metadata (`__CHAR_VARCHAR_TYPE_STRING`): [`Resolver::declared`]
//! maps the attributes that hold such a column to their catalog string (`char(5)`).
//!
//! Read-side padding (`spark.sql.readSideCharPadding`): a scan of a table with `char(n)` columns is read through
//! `rpad(c, greatest(char_length(c), n), ' ')`, so a `CHAR(5)` holding `'N'` reads as `'N    '` while a longer
//! value (a file written elsewhere, or under `charVarcharAsString`) is left as it is: Spark's
//! `readSidePadding` only pads. The padded column is a new attribute with the column's name that carries the
//! same declared type. The declared type follows an attribute through a bare-column alias
//! ([`Resolver::carry_declared`]) and a second instance of a relation (`Resolver::shifted`).

use std::sync::Arc;

use super::AnalysisError;
use super::catalog::FunctionLookup;
use super::exprs::Bound;
use super::resolve::{Attr, Rel, Resolver};
use crate::{AttrId, BoundColumn, Expr, ExprNode, Literal, Origin, ResolvedExpr, ResolvedFunction};
use krishiv_types::DataType;

/// The most spaces a comparison literal is padded with during analysis; a wider pad is an `rpad` call.
const MAX_LITERAL_PADDING: i32 = 1 << 16;

/// `n` of a declared `char(n)`; `None` for `varchar(n)`, nested types and other strings.
pub(super) fn char_length(declared: &str) -> Option<i32> {
    declared
        .strip_prefix("char(")?
        .strip_suffix(')')?
        .trim()
        .parse()
        .ok()
}

/// A column's declared type as far as `CHAR` / `VARCHAR` matter, read from its catalog string
/// (`array<struct<x:char(2)>>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclaredType {
    /// `char(n)`.
    Char(i32),
    /// `varchar(n)`.
    Varchar(i32),
    /// `struct<...>`: the fields in order.
    Struct(Vec<DeclaredType>),
    /// `array<...>`.
    Array(Box<DeclaredType>),
    /// `map<key, value>`.
    Map(Box<DeclaredType>, Box<DeclaredType>),
    /// Anything else: no field of it is limited.
    Other,
}

impl DeclaredType {
    /// Reads a catalog string; a part it cannot follow is [`DeclaredType::Other`].
    #[must_use]
    pub fn parse(declared: &str) -> Self {
        let d = declared.trim();
        let lower = d.to_ascii_lowercase();
        let number = |inner: &str| inner.trim().parse::<i32>().ok().filter(|n| *n >= 0);
        if let Some(n) = lower
            .strip_prefix("char(")
            .and_then(|r| r.strip_suffix(')'))
            .and_then(number)
        {
            return Self::Char(n);
        }
        if let Some(n) = lower
            .strip_prefix("varchar(")
            .and_then(|r| r.strip_suffix(')'))
            .and_then(number)
        {
            return Self::Varchar(n);
        }
        let inner = |prefix: &str| {
            lower
                .starts_with(prefix)
                .then(|| d.get(prefix.len()..d.len().checked_sub(1)?))
                .flatten()
                .filter(|_| d.ends_with('>'))
        };
        if let Some(body) = inner("array<") {
            return Self::Array(Box::new(Self::parse(body)));
        }
        if let Some(body) = inner("map<")
            && let [k, v] = split_top(body, ',').as_slice()
        {
            return Self::Map(Box::new(Self::parse(k)), Box::new(Self::parse(v)));
        }
        if let Some(body) = inner("struct<") {
            let fields = split_top(body, ',')
                .into_iter()
                .map(|f| {
                    // The type follows the field name's `:` (outside a backquoted name), up to the field's
                    // `NOT NULL` or `COMMENT`.
                    top_level(f)
                        .find(|&(_, c)| c == ':')
                        .map_or(Self::Other, |(i, _)| Self::parse(field_type(&f[i + 1..])))
                })
                .collect();
            return Self::Struct(fields);
        }
        Self::Other
    }

    /// Whether a `CHAR` or `VARCHAR` limit applies somewhere in the type.
    #[must_use]
    pub fn is_limited(&self) -> bool {
        match self {
            Self::Char(_) | Self::Varchar(_) => true,
            Self::Struct(f) => f.iter().any(Self::is_limited),
            Self::Array(e) => e.is_limited(),
            Self::Map(k, v) => k.is_limited() || v.is_limited(),
            Self::Other => false,
        }
    }

    /// Whether a `CHAR` applies somewhere in the type, the only limit a read pads for.
    fn has_char(&self) -> bool {
        match self {
            Self::Char(_) => true,
            Self::Struct(f) => f.iter().any(Self::has_char),
            Self::Array(e) => e.has_char(),
            Self::Map(k, v) => k.has_char() || v.has_char(),
            Self::Varchar(_) | Self::Other => false,
        }
    }
}

/// A struct field's type without the modifiers after it (`char(2) not null comment 'c'` is `char(2)`): the
/// text up to the first space outside `<...>`, `(...)` and quotes. Only types that are [`DeclaredType::Other`]
/// anyway (`interval day to second`) hold such a space.
fn field_type(s: &str) -> &str {
    let s = s.trim_start();
    top_level(s)
        .find(|&(_, c)| c.is_whitespace())
        .map_or(s, |(i, _)| &s[..i])
}

/// `s` split at each `sep` outside `<...>`, `(...)` and quotes.
fn split_top(s: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    for (i, c) in top_level(s) {
        if c == sep {
            parts.push(&s[start..i]);
            start = i + c.len_utf8();
        }
    }
    parts.push(&s[start..]);
    parts
}

/// The characters of `s` outside `<...>` and `(...)` and outside quoted text: a `'...'` or `"..."` string (a
/// backslash escapes the next character, as in a catalog string's `COMMENT`) and a `` `...` `` name (a doubled
/// backquote stays inside it). The brackets themselves are not yielded.
fn top_level(s: &str) -> impl Iterator<Item = (usize, char)> + '_ {
    let mut depth = 0_i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut chars = s.char_indices().peekable();
    std::iter::from_fn(move || {
        while let Some((i, c)) = chars.next() {
            if let Some(q) = quote {
                if escaped {
                    escaped = false;
                } else if c == '\\' && q != '`' {
                    escaped = true;
                } else if c == q {
                    if q == '`' && chars.peek().is_some_and(|&(_, n)| n == '`') {
                        chars.next();
                    } else {
                        quote = None;
                    }
                }
                continue;
            }
            match c {
                '\'' | '"' | '`' => quote = Some(c),
                '<' | '(' => depth += 1,
                '>' | ')' => depth -= 1,
                _ if depth == 0 => return Some((i, c)),
                _ => {}
            }
        }
        None
    })
}

impl Resolver<'_> {
    /// The field metadata the analyzer keeps per attribute (the declared `CHAR`/`VARCHAR` type, the column
    /// comment) passes from `from` to `to`, a new attribute standing for the same column.
    pub(super) fn carry_column_meta(&mut self, from: AttrId, to: AttrId) {
        if let Some(d) = self.declared.get(&from).cloned() {
            self.declared.insert(to, d);
        }
        if let Some(t) = self.comments.get(&from).cloned() {
            self.comments.insert(to, t);
        }
    }

    /// `rel`, a scan, read through `rpad(c, n, ' ')` for each of its `char(n)` columns; `rel` itself when it has
    /// none, when `spark.sql.legacy.charVarcharAsString` is on (the padding rule is off) or when the registry has
    /// no `rpad`.
    pub(super) fn pad_char_columns(
        &mut self,
        rel: Rel,
        origin: Origin,
    ) -> Result<Rel, AnalysisError> {
        if self.ctx.conf.char_varchar_as_string {
            return Ok(rel);
        }
        let padded: Vec<(usize, DeclaredType)> = rel
            .out
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                let d = DeclaredType::parse(self.declared.get(&a.col.id)?);
                d.has_char().then_some((i, d))
            })
            .collect();
        if padded.is_empty() {
            return Ok(rel);
        }
        if ["rpad", "greatest", "char_length"]
            .iter()
            .any(|f| self.function_entry(f).is_none())
        {
            return Ok(rel);
        }
        let mut exprs = Vec::with_capacity(rel.out.len());
        let mut out: Vec<Attr> = Vec::with_capacity(rel.out.len());
        for (i, a) in rel.out.iter().enumerate() {
            let Some((_, declared)) = padded.iter().find(|(p, _)| *p == i) else {
                exprs.push(super::ops::col_expr(&a.col, Origin::UNKNOWN));
                out.push(a.clone());
                continue;
            };
            let read = super::ops::col_expr(&a.col, Origin::UNKNOWN);
            let call = match self.read_pad(read, declared, &a.col.data_type, a.col.nullable) {
                Ok(Some(call)) => call,
                Ok(None) => return Ok(rel),
                Err(()) => {
                    return Err(super::bind::unsupported(
                        "reads of a CHAR column whose declared type does not match the schema",
                        origin,
                    ));
                }
            };
            let col = BoundColumn {
                id: self.fresh(),
                name: a.col.name.clone(),
                data_type: a.col.data_type.clone(),
                nullable: a.col.nullable,
            };
            self.carry_column_meta(a.col.id, col.id);
            exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: call,
                    output: col.clone(),
                },
            ))));
            out.push(Attr {
                col,
                qual: Arc::clone(&a.qual),
                source: a.source.clone(),
                duplicate: a.duplicate,
            });
        }
        let mut padded = self.projection(origin, &rel, exprs, out);
        // The hidden columns (a file table's `_metadata`) stay reachable through the padding projection.
        padded.hidden = rel.hidden;
        Ok(padded)
    }
}

/// The column `e` is, or, when `internal_cast`, casts to its own `string` type (the read projection of a
/// permanent view casts its columns to their stored types). A cast the query wrote drops the declared type, as
/// in Spark.
fn bare_column(e: &Expr, internal_cast: bool) -> Option<&BoundColumn> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c),
        ExprNode::Cast { expr, to, .. } if internal_cast && *to == DataType::String => {
            match &expr.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) if c.data_type == DataType::String => {
                    Some(c)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The column whose comment `e` keeps: a bare column, or, when `internal_cast`, the engine's cast of one (a
/// permanent view's read projection: Spark gives that alias the view's stored field metadata, which holds the
/// comment the column had when the view was stored). A cast the query wrote keeps none, as in Spark.
fn commented_column(e: &Expr, internal_cast: bool) -> Option<&BoundColumn> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c),
        ExprNode::Cast { expr, .. } if internal_cast => match &expr.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c),
            _ => None,
        },
        _ => None,
    }
}

/// The column `c` and length `n` of the read-side padding call `rpad(c, greatest(char_length(c), n), ' ')`,
/// matched whole: a user's `rpad` of another shape is a plain string.
fn read_padded_column(e: &Expr) -> Option<(&BoundColumn, i32)> {
    fn call<'e>(e: &'e Expr, name: &str, arity: usize) -> Option<&'e [Arc<Expr>]> {
        let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
            return None;
        };
        (matches!(f.id, crate::FunctionId::Registry { .. })
            && f.id.name() == name
            && f.args.len() == arity)
            .then_some(f.args.as_slice())
    }
    fn column(e: &Expr) -> Option<&BoundColumn> {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c),
            _ => None,
        }
    }
    let [arg, len, blank] = call(e, "rpad", 3)? else {
        return None;
    };
    let [chars, n] = call(len, "greatest", 2)? else {
        return None;
    };
    let [counted] = call(chars, "char_length", 1)? else {
        return None;
    };
    let (c, counted) = (column(arg)?, column(counted)?);
    match (&n.node, &blank.node) {
        (ExprNode::Literal(Literal::Integer(n)), ExprNode::Literal(Literal::String(b)))
            if c.id == counted.id && b == " " =>
        {
            Some((c, *n))
        }
        _ => None,
    }
}

/// `x AS name` where `x` is a cast the engine built in ANSI mode before analysis.
fn is_internal_cast_alias(e: &Expr) -> bool {
    matches!(&e.node, ExprNode::Alias { expr, .. }
        if matches!(&unstored(expr).node, ExprNode::Cast { mode: crate::CastMode::Ansi, .. }))
}

/// `e` without the [`ExprNode::StoredField`] of a view's read projection.
fn unstored(e: &Expr) -> &Expr {
    match &e.node {
        ExprNode::StoredField { expr, .. } => expr,
        _ => e,
    }
}

/// The comment a view stored for the column `written` aliases (`Some(None)`: it stored none), if `written` is an
/// item of a view's read projection.
fn stored_comment(written: Option<&Expr>) -> Option<Option<&str>> {
    match &written?.node {
        ExprNode::Alias { expr, .. } => match &expr.node {
            ExprNode::StoredField { comment, .. } => Some(comment.as_deref()),
            _ => None,
        },
        _ => None,
    }
}

/// The declared length `n` of a bare `char(n)` attribute operand; `None` for any other operand (an expression,
/// a nested field, a `varchar` column).
fn bare_char(
    declared: &std::collections::HashMap<crate::AttrId, Arc<str>>,
    b: &Bound,
) -> Option<i32> {
    if b.shown.is_some() {
        return None;
    }
    match &b.expr.node {
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
            char_length(declared.get(&c.id)?)
        }
        _ => None,
    }
}

/// A written string literal operand.
fn string_literal(b: &Bound) -> Option<&str> {
    match &b.expr.node {
        ExprNode::Literal(Literal::String(s)) if b.shown.is_none() => Some(s),
        _ => None,
    }
}

impl Resolver<'_> {
    /// The registry entry of the builtin `name`, `None` when there is none.
    fn function_entry(&self, name: &str) -> Option<crate::analyzer::catalog::FunctionEntry> {
        match self
            .ctx
            .functions
            .lookup(&[name.to_owned()], self.ctx.profile)
        {
            FunctionLookup::Found(entry) => Some(entry),
            _ => None,
        }
    }

    /// A call of the builtin `name`; `None` when the registry has none.
    fn named_call(
        &self,
        name: &str,
        args: Vec<Arc<Expr>>,
        data_type: DataType,
        nullable: bool,
    ) -> Option<Arc<Expr>> {
        let entry = self.function_entry(name)?;
        Some(Arc::new(Expr::new(ExprNode::Resolved(
            ResolvedExpr::Function(ResolvedFunction {
                id: entry.id.clone(),
                kind: entry.kind,
                args,
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type,
                nullable,
                deterministic: true,
            }),
        ))))
    }

    fn int_literal(n: i32) -> Arc<Expr> {
        Arc::new(Expr::literal(Literal::Integer(n)))
    }

    /// `rpad(arg, len, ' ')`.
    fn rpad_to(&self, arg: Arc<Expr>, len: Arc<Expr>, nullable: bool) -> Option<Arc<Expr>> {
        let blank = Arc::new(Expr::literal(Literal::String(" ".to_owned())));
        self.named_call("rpad", vec![arg, len, blank], DataType::String, nullable)
    }

    /// `rpad(arg, n, ' ')` (it cuts a longer value to `n`, as Spark's comparison padding does).
    fn rpad_call(&self, arg: Arc<Expr>, n: i32, nullable: bool) -> Option<Arc<Expr>> {
        self.rpad_to(arg, Self::int_literal(n), nullable)
    }

    /// The read-side form: `rpad(arg, greatest(char_length(arg), n), ' ')` pads and never cuts.
    fn read_pad_call(&self, arg: Arc<Expr>, n: i32, nullable: bool) -> Option<Arc<Expr>> {
        let chars = self.named_call(
            "char_length",
            vec![Arc::clone(&arg)],
            DataType::Integer,
            nullable,
        )?;
        let len = self.named_call(
            "greatest",
            vec![chars, Self::int_literal(n)],
            DataType::Integer,
            false,
        )?;
        self.rpad_to(arg, len, nullable)
    }

    /// `if(cond, a, b)`: the resolved form is the `When` operator, not a registry call.
    fn when_call(&self, args: Vec<Arc<Expr>>, data_type: DataType) -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: crate::FunctionId::Builtin(crate::Builtin::When),
                kind: crate::FunctionKind::Scalar,
                args,
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type,
                nullable: true,
                deterministic: true,
            },
        ))))
    }

    /// `arg` read as Spark's `processStringForCharVarchar` with only `readSidePadding` set: each `CHAR` in it pads
    /// and a `VARCHAR` does not. `Ok(None)` when the registry lacks a function, `Err` when `declared` does not
    /// follow `ty`.
    fn read_pad(
        &mut self,
        arg: Arc<Expr>,
        declared: &DeclaredType,
        ty: &DataType,
        nullable: bool,
    ) -> Result<Option<Arc<Expr>>, ()> {
        Ok(match (declared, ty) {
            (DeclaredType::Char(n), DataType::String) => self.read_pad_call(arg, *n, nullable),
            (DeclaredType::Struct(fields), DataType::Struct(st))
                if fields.len() == st.fields().len() =>
            {
                let mut args = Vec::with_capacity(fields.len() * 2);
                for (ordinal, (d, f)) in fields.iter().zip(st.fields().iter()).enumerate() {
                    let field_nullable = f.nullable() || nullable;
                    let read = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                        base: Arc::clone(&arg),
                        ordinal,
                        name: f.name().to_owned(),
                        data_type: f.data_type().clone(),
                        nullable: field_nullable,
                    })));
                    let value = if d.has_char() {
                        match self.read_pad(read, d, f.data_type(), field_nullable)? {
                            Some(v) => v,
                            None => return Ok(None),
                        }
                    } else {
                        read
                    };
                    args.push(Arc::new(Expr::literal(Literal::String(
                        f.name().to_owned(),
                    ))));
                    args.push(value);
                }
                let built = self.named_call("named_struct", args, ty.clone(), false);
                match built {
                    Some(s) if nullable => {
                        let test = self.named_call("isnull", vec![arg], DataType::Boolean, false);
                        let null = Arc::new(Expr::literal(Literal::Null(Some(ty.clone()))));
                        test.map(|t| self.when_call(vec![t, null, s], ty.clone()))
                    }
                    other => other,
                }
            }
            (DeclaredType::Array(e), DataType::Array(a)) => {
                self.read_pad_array(arg, e, a.element_type(), a.contains_null(), ty, nullable)?
            }
            (DeclaredType::Map(k, v), DataType::Map(m)) => {
                let (key_ty, value_ty) = (m.key_type().clone(), m.value_type().clone());
                let contains_null = m.value_contains_null();
                let array = |e: &DataType, n: bool| {
                    DataType::Array(krishiv_types::ArrayType::new(e.clone(), n).into())
                };
                let keys_ty = array(&key_ty, false);
                let values_ty = array(&value_ty, contains_null);
                let keys = self.named_call(
                    "map_keys",
                    vec![Arc::clone(&arg)],
                    keys_ty.clone(),
                    nullable,
                );
                let values = self.named_call("map_values", vec![arg], values_ty.clone(), nullable);
                let (Some(keys), Some(values)) = (keys, values) else {
                    return Ok(None);
                };
                let keys = if k.has_char() {
                    self.read_pad_array(keys, k, &key_ty, false, &keys_ty, nullable)?
                } else {
                    Some(keys)
                };
                let values = if v.has_char() {
                    self.read_pad_array(values, v, &value_ty, contains_null, &values_ty, nullable)?
                } else {
                    Some(values)
                };
                let (Some(keys), Some(values)) = (keys, values) else {
                    return Ok(None);
                };
                self.named_call("map_from_arrays", vec![keys, values], ty.clone(), nullable)
            }
            _ => return Err(()),
        })
    }

    /// `transform(arr, x -> <x read as declared>)`.
    fn read_pad_array(
        &mut self,
        arr: Arc<Expr>,
        declared: &DeclaredType,
        element: &DataType,
        contains_null: bool,
        ty: &DataType,
        nullable: bool,
    ) -> Result<Option<Arc<Expr>>, ()> {
        if !declared.has_char() {
            return Ok(Some(arr));
        }
        let var = BoundColumn {
            id: self.fresh(),
            name: "x".to_owned(),
            data_type: element.clone(),
            nullable: contains_null,
        };
        let read = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::LambdaVar(
            var.clone(),
        ))));
        let Some(body) = self.read_pad(read, declared, element, contains_null)? else {
            return Ok(None);
        };
        let lambda = Arc::new(Expr::new(ExprNode::Lambda {
            params: vec![var],
            body,
        }));
        Ok(self.named_call("transform", vec![arr, lambda], ty.clone(), nullable))
    }

    /// The declared type follows `c AS x` (Spark keeps the field metadata through an alias of a bare column).
    /// `written` are the select items before resolution: a cast passes the declared type on only if the engine
    /// built it in ANSI mode before analysis (a permanent view's read projection, `cast_ansi`); the casts a
    /// query writes reach the analyzer in the default or `TRY` mode.
    pub(super) fn carry_declared(&mut self, items: &[Bound], written: &[Arc<Expr>]) {
        let aligned = items.len() == written.len();
        for (i, b) in items.iter().enumerate() {
            self.carry_declared_item(b, aligned.then(|| &*written[i]));
        }
    }

    /// `b` without the engine's own cast of a `string` column to `string` (a permanent view's read projection,
    /// `written` an ANSI cast under an alias, [`is_internal_cast_alias`]): a no-op whose resolved form a stored
    /// plan (a DataFrame's temporary view over the view) could not tell from a user's cast, which drops the
    /// declared type. Without it the stored plan holds the bare column and keeps the declared `CHAR`/`VARCHAR`.
    pub(super) fn without_internal_identity_cast(b: Bound, written: &Expr) -> Bound {
        if !is_internal_cast_alias(written) {
            return b;
        }
        let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &b.expr.node else {
            return b;
        };
        let ExprNode::Cast {
            expr: inner, to, ..
        } = &expr.node
        else {
            return b;
        };
        let identity = *to == DataType::String
            && matches!(&inner.node, ExprNode::Resolved(ResolvedExpr::Column(c)) if c.data_type == DataType::String);
        if !identity {
            return b;
        }
        let alias = Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
            expr: Arc::clone(inner),
            output: output.clone(),
        }))
        .with_origin(b.expr.origin);
        Bound {
            expr: Arc::new(alias),
            ..b
        }
    }

    /// [`Self::carry_declared`] for one item, `written` its form before resolution when known. The select list
    /// calls it as each item is bound, so a later item that reads the alias (a lateral column alias) sees it.
    pub(super) fn carry_declared_item(&mut self, b: &Bound, written: Option<&Expr>) {
        let internal_cast = written.is_some_and(is_internal_cast_alias);
        // An item already resolved comes from a stored analyzed plan (a DataFrame's temporary view): its
        // read-side padding call stands for the padded column.
        let stored = written.is_some_and(|w| matches!(w.node, ExprNode::Resolved(_)));
        // A view's read projection gives each column the metadata the view stored (Spark's `explicitMetadata`):
        // its comment, or none, whatever the column under it has.
        if let (Some(stored), ExprNode::Resolved(ResolvedExpr::Alias { output, .. })) =
            (stored_comment(written), &b.expr.node)
        {
            match stored {
                Some(t) => {
                    self.comments.insert(output.id, Arc::from(t));
                }
                None => {
                    self.comments.remove(&output.id);
                }
            }
        // A column comment follows the same alias of a bare column (Spark's `Alias` keeps a named child's
        // metadata; any other expression starts without it), and a stored plan's padded `CHAR` column.
        } else if !self.comments.is_empty()
            && let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &b.expr.node
            && let Some(c) = commented_column(expr, internal_cast)
                .or_else(|| read_padded_column(expr).filter(|_| stored).map(|(c, _)| c))
            && let Some(t) = self.comments.get(&c.id).cloned()
        {
            self.comments.insert(output.id, t);
        }
        if self.declared.is_empty() {
            return;
        }
        if let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &b.expr.node
            && let Some(c) = bare_column(expr, internal_cast).or_else(|| {
                let (c, n) = read_padded_column(expr).filter(|_| stored)?;
                (self.declared.get(&c.id).and_then(|d| char_length(d)) == Some(n)).then_some(c)
            })
            && let Some(d) = self.declared.get(&c.id).cloned()
        {
            self.declared.insert(output.id, d);
        }
    }

    /// Spark's `ApplyCharTypePadding` for `c IN (...)`, `kids` being the value and then the list; `None` when
    /// nothing pads.
    ///
    /// Over a bare `char(n)` value and a list of constants, the value and the literals pad to the longest of `n`
    /// and the literals, a value already that long staying as it is. Over a list of bare attributes, all of them
    /// `char`, each pads to the longest declared length. Any other list is left alone. The operands as Spark shows
    /// them (`shown`) are kept, so names do not change.
    ///
    /// A `NULL` in the list turns all of that off, and so does a constant that is not a string (a real Spark 4.2.0
    /// run: `c IN ('a', NULL)` matches no padded `'a'`, and `c IN (1)` casts `c`).
    pub(super) fn pad_char_in(&self, kids: &[Bound]) -> Result<Option<Vec<Bound>>, AnalysisError> {
        if self.ctx.conf.char_varchar_as_string || self.declared.is_empty() {
            return Ok(None);
        }
        let Some((value, list)) = kids.split_first() else {
            return Ok(None);
        };
        if list.is_empty() {
            return Ok(None);
        }
        let Some(n) = bare_char(&self.declared, value) else {
            return Ok(None);
        };
        // A list of `char` attributes: each pads to the longest.
        let lengths: Option<Vec<i32>> = list.iter().map(|k| bare_char(&self.declared, k)).collect();
        if let Some(lengths) = lengths {
            let target = lengths.iter().copied().fold(n, i32::max);
            let mut out = Vec::with_capacity(kids.len());
            for (k, len) in kids.iter().zip(std::iter::once(n).chain(lengths)) {
                out.push(if len < target {
                    match self.pad_operand(k, target) {
                        Some(p) => p,
                        None => return Ok(None),
                    }
                } else {
                    k.clone()
                });
            }
            return Ok(Some(out));
        }
        // Spark's run (4.2.0): a `NULL` anywhere in the list leaves everything unpadded.
        let mut widths = Vec::with_capacity(list.len());
        for k in list {
            let Some(s) = self.constant_string(k)? else {
                return Ok(None);
            };
            widths.push(i32::try_from(s.chars().count()).ok());
        }
        let mut target = n;
        for w in &widths {
            let Some(l) = w else { return Ok(None) };
            target = target.max(*l);
        }
        let mut out = Vec::with_capacity(kids.len());
        out.push(if n < target {
            match self.pad_operand(value, target) {
                Some(p) => p,
                None => return Ok(None),
            }
        } else {
            value.clone()
        });
        for (k, w) in list.iter().zip(&widths) {
            out.push(match w {
                Some(l) if *l < target => match self.pad_string_literal(k, target) {
                    Some(p) => p,
                    None => return Ok(None),
                },
                _ => k.clone(),
            });
        }
        Ok(Some(out))
    }

    /// The text of the constant string `k` (a literal, or a call that folds to one); `None` for a `NULL`, another
    /// type or a constant the analyzer cannot fold.
    fn constant_string(&self, k: &Bound) -> Result<Option<String>, AnalysisError> {
        if let Some(s) = string_literal(k) {
            return Ok(Some(s.to_owned()));
        }
        if k.ty.data_type != Some(DataType::String) {
            return Ok(None);
        }
        Ok(match self.fold(&k.expr, self.ctx.conf.ansi) {
            Some(krishiv_functions::signature::Constant::String(s)) => Some(s),
            _ => None,
        })
    }

    /// `k` through `rpad(k, to, ' ')`, still shown as it was written.
    fn pad_operand(&self, k: &Bound, to: i32) -> Option<Bound> {
        let e = self.rpad_call(Arc::clone(&k.expr), to, k.ty.nullable)?;
        Some(Bound {
            expr: e,
            shown: Some(k.shown()),
            ..k.clone()
        })
    }

    /// The string literal `k` padded to `to`, still shown as it was written; a constant that is no literal goes
    /// through `rpad`. A wide declaration (`CHAR(2147483647)`) would build a huge literal while analyzing even an
    /// empty table: past `MAX_LITERAL_PADDING` the literal pads at run time instead (same value, `rpad`).
    fn pad_string_literal(&self, k: &Bound, to: i32) -> Option<Bound> {
        let Some(text) = string_literal(k) else {
            return self.pad_operand(k, to);
        };
        let width = i32::try_from(text.chars().count()).ok()?;
        if to.saturating_sub(width) > MAX_LITERAL_PADDING {
            return self.pad_operand(k, to);
        }
        let padded = format!("{text}{}", " ".repeat(usize::try_from(to - width).ok()?));
        Some(Bound {
            expr: Arc::new(Expr::literal(Literal::String(padded)).with_origin(k.expr.origin)),
            shown: Some(k.shown()),
            ..k.clone()
        })
    }

    /// `ApplyCharTypePadding` also pads a comparison of two nested columns whose `CHAR` lengths differ, field by
    /// field. That is not implemented, so such a comparison is refused rather than answered unpadded.
    pub(super) fn refuse_nested_char_comparison(
        &self,
        kids: &[Bound],
        origin: Origin,
    ) -> Result<(), AnalysisError> {
        if self.ctx.conf.char_varchar_as_string || self.declared.is_empty() {
            return Ok(());
        }
        // The lengths are compared, not the catalog strings: struct field names do not change the padding.
        let nested = |b: &Bound| -> Option<DeclaredType> {
            if b.shown.is_some() {
                return None;
            }
            match &b.expr.node {
                ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
                    let d = self.declared.get(&c.id)?;
                    let parsed = DeclaredType::parse(d);
                    (char_length(d).is_none() && parsed.has_char()).then_some(parsed)
                }
                _ => None,
            }
        };
        let types: Vec<DeclaredType> = kids.iter().filter_map(nested).collect();
        if types.len() == kids.len() && types.windows(2).any(|w| w[0] != w[1]) {
            return Err(super::bind::unsupported(
                "comparisons of nested CHAR columns of different lengths",
                origin,
            ));
        }
        Ok(())
    }

    /// Spark's `ApplyCharTypePadding` for the comparison of `kids` (`=`, `<=>`, `<`, `<=`, `>`, `>=`): only a bare
    /// `char(n)` attribute is padded. Two of them pad the shorter to the longer; against a string literal of
    /// length L the column pads up to L when n < L, and the literal pads up to n when n > L. The operands as
    /// Spark shows them (`shown`) are kept, so names do not change. `None` when nothing pads.
    pub(super) fn pad_char_comparison(&self, kids: &[Bound]) -> Option<Vec<Bound>> {
        if self.ctx.conf.char_varchar_as_string || self.declared.is_empty() {
            return None;
        }
        let [a, b] = kids else { return None };
        let pad = |k: &Bound, to: i32| self.pad_operand(k, to);
        let literal = |k: &Bound, to: i32| self.pad_string_literal(k, to);
        let length = |s: &str| i32::try_from(s.chars().count()).ok();
        match (bare_char(&self.declared, a), bare_char(&self.declared, b)) {
            (Some(la), Some(lb)) if la < lb => Some(vec![pad(a, lb)?, b.clone()]),
            (Some(la), Some(lb)) if lb < la => Some(vec![a.clone(), pad(b, la)?]),
            (Some(n), None) => {
                let l = length(string_literal(b)?)?;
                match n.cmp(&l) {
                    std::cmp::Ordering::Less => Some(vec![pad(a, l)?, b.clone()]),
                    std::cmp::Ordering::Greater => Some(vec![a.clone(), literal(b, n)?]),
                    std::cmp::Ordering::Equal => None,
                }
            }
            (None, Some(n)) => {
                let l = length(string_literal(a)?)?;
                match n.cmp(&l) {
                    std::cmp::Ordering::Less => Some(vec![a.clone(), pad(b, l)?]),
                    std::cmp::Ordering::Greater => Some(vec![literal(a, n)?, b.clone()]),
                    std::cmp::Ordering::Equal => None,
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod declared_tests {
    use super::DeclaredType;

    #[test]
    fn field_modifiers_quotes_and_names_keep_the_limits() {
        let t = DeclaredType::parse(
            "struct<x: char(2) not null comment 'a,b (c) <d>', `y:z`: varchar(3) comment 'it\\'s', w: int>",
        );
        let DeclaredType::Struct(f) = &t else {
            panic!("{t:?}")
        };
        assert_eq!(f.len(), 3, "{f:?}");
        assert!(matches!(f[0], DeclaredType::Char(2)), "{f:?}");
        assert!(matches!(f[1], DeclaredType::Varchar(3)), "{f:?}");
        assert!(matches!(f[2], DeclaredType::Other), "{f:?}");
        let n = DeclaredType::parse("array<struct<a: struct<`b``c`: char(1) comment ')'>>>");
        assert!(n.is_limited(), "{n:?}");
        assert!(matches!(
            DeclaredType::parse("struct<i: interval day to second>"),
            DeclaredType::Struct(ref f) if matches!(f.as_slice(), [DeclaredType::Other])
        ));
    }
}

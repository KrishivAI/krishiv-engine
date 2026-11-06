// SPDX-License-Identifier: Apache-2.0
//! The single collation of a call's string arguments (Spark's `CollationTypeCoercion`, ADR 0043,
//! design `P2-collations.md` section 3.2).
//!
//! Each string argument has a collation and a strength: *explicit* (`COLLATE`), *implicit* (a
//! column) or *default* (a literal). The strongest wins; two different explicit collations are
//! `COLLATION_MISMATCH.EXPLICIT`, two different implicit ones are indeterminate
//! (`INDETERMINATE_COLLATION_IN_EXPRESSION`). The call then runs on plain strings (the carrier is
//! the same) and a string result takes the collation.

use krishiv_common::error::{
    CollationMismatchExplicit, DatatypeMismatchUnexpectedInputType,
    IndeterminateCollationInExpression,
};
use krishiv_types::{Collation, CollationFamily};

use super::*;

/// How firmly an operand carries its collation, weakest first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Strength {
    Default,
    Implicit,
    Explicit,
}

/// Why the operands have no single collation.
#[derive(Debug, PartialEq)]
enum Conflict {
    /// Different explicit collations, in order of appearance.
    Explicit(Vec<Collation>),
    /// Different implicit collations.
    Indeterminate,
}

/// Calls that compare their string arguments, or key on them. They keep the collated types so
/// the execution guard sees them (units 3a and 3b lift it).
const COMPARING: &[&str] = &[
    "==",
    "<=>",
    "<",
    "<=",
    ">",
    ">=",
    "in",
    "between",
    "greatest",
    "least",
    "nullif",
    "equal_null",
    "min",
    "max",
    "min_by",
    "max_by",
    "mode",
    "collect_set",
    "listagg",
    "approx_count_distinct",
];

/// Comparisons that run on collation keys when their collation has a key built from existing
/// functions (`UTF8_LCASE`, `_RTRIM`). `greatest`, `least`, `nullif` and `equal_null` return or
/// reuse the operands and wait for unit 3b.
const KEYED: &[&str] = &["==", "<=>", "<", "<=", ">", ">=", "in", "between"];

/// Calls that fail on operands without one collation (Spark's `canContainIndeterminateCollation`);
/// any other call over such operands runs on the plain strings.
const FAILS_INDETERMINATE: &[&str] = &[
    "contains",
    "startswith",
    "endswith",
    "find_in_set",
    "instr",
    "locate",
    "position",
    "replace",
    "substring_index",
    "translate",
    "trim",
    "ltrim",
    "rtrim",
    "btrim",
    "upper",
    "ucase",
    "lower",
    "lcase",
    "initcap",
];

/// Calls whose result depends on the collation (Spark's `CollationSupport` and the pattern
/// matchers). Their collation-aware kernels belong to unit 5, so until then they refuse a
/// non-binary collation instead of matching bytes.
const DEPENDS: &[&str] = &[
    "contains",
    "startswith",
    "endswith",
    "instr",
    "locate",
    "position",
    "replace",
    "split_part",
    "find_in_set",
    "translate",
    "substring_index",
    "trim",
    "ltrim",
    "rtrim",
    "btrim",
    "like",
    "ilike",
    "rlike",
    "like_any",
    "like_all",
    "not_like_any",
    "not_like_all",
    "ilike_any",
    "ilike_all",
    "not_ilike_any",
    "not_ilike_all",
    "regexp",
    "regexp_like",
    "regexp_replace",
    "regexp_extract",
    "regexp_extract_all",
    "regexp_count",
    "regexp_instr",
    "regexp_substr",
    "str_to_map",
    "split",
];

/// Case mapping uses the collator's locale under ICU (unit 6).
const ICU_CASE: &[&str] = &["upper", "ucase", "lower", "lcase", "initcap"];

/// Calls that search a string and so refuse an accent-insensitive ICU collation.
const SEARCHES: &[&str] = &[
    "contains",
    "startswith",
    "endswith",
    "instr",
    "locate",
    "position",
    "replace",
    "split_part",
    "str_to_map",
    "substring_index",
    "translate",
];

/// Calls that may take a nested value holding a collated string: they only count or pick.
const NESTED_OK: &[&str] = &[
    "size",
    "cardinality",
    "array_size",
    "isnull",
    "isnotnull",
    "__subscript",
    "element_at",
    "try_element_at",
    "get",
];

/// String functions whose result is an ordinary binary string whatever the collation of their
/// arguments (their return type in Spark is the default string type; `soundex`, `format_number`,
/// `uuid` and the charset `decode` are covered by the `collations-*` goldens). Not `listagg`
/// (golden `listagg-collations`: it returns the collated type), and `decode` only with two
/// arguments, see [`binary_result`].
const BINARY_RESULT: &[&str] = &[
    "soundex",
    "format_number",
    "base64",
    "decode",
    "chr",
    "char",
    "hex",
    "md5",
    "sha",
    "sha1",
    "sha2",
    "uuid",
    "date_format",
    "to_json",
    "to_csv",
    "to_xml",
];

/// Whether a call of `name` with `args` arguments returns an ordinary binary string. `decode` is
/// the charset decoder only with two arguments; with three or more it is Oracle's `DECODE`, a
/// `CASE` whose result keeps the collation of its branches.
fn binary_result(name: &str, args: usize) -> bool {
    BINARY_RESULT.contains(&name) && !(name == "decode" && args != 2)
}

/// Calls that read the type of their argument: it stays collated.
const INSPECTS: &[&str] = &["collation", "typeof"];

/// The comparison key of `x` under `c`, as an expression over plain strings.
fn key_of(x: &Arc<Expr>, c: Collation, origin: crate::Origin) -> Expr {
    let lit =
        |t: &str| Expr::new(ExprNode::Literal(Literal::String(t.to_owned()))).with_origin(origin);
    let mut key = Expr::new(ExprNode::Cast {
        expr: Arc::clone(x),
        to: DataType::String,
        mode: CastMode::Ansi,
    })
    .with_origin(origin);
    if c.rtrim() {
        key = Expr::function("rtrim", vec![key]).with_origin(origin);
    }
    if matches!(c.family(), CollationFamily::Utf8Lcase) {
        key = Expr::function(
            "replace",
            vec![
                Expr::function("lower", vec![key]).with_origin(origin),
                lit("\u{3c2}"),
                lit("\u{3c3}"),
            ],
        )
        .with_origin(origin);
    }
    key
}

/// How many leading arguments of `name` only select (a null test) and so take no part in the
/// collation of the result: `nvl2(selector, if_not_null, if_null)`.
fn selector_args(name: &str) -> usize {
    usize::from(name.eq_ignore_ascii_case("nvl2"))
}

/// Whether a lookup in a map compares keys of a collated type (`m[k]`, `element_at`): the
/// key compare is bytes until unit 3b, so only array indexing and plain-key maps are exempt.
fn map_lookup_compares_collation(kids: &[Bound]) -> bool {
    kids.iter().any(
        |k| matches!(&k.ty.data_type, Some(DataType::Map(m)) if m.key_type().contains_collation()),
    ) || (kids
        .iter()
        .any(|k| matches!(k.ty.data_type, Some(DataType::Map(_))))
        && kids
            .iter()
            .any(|k| matches!(k.ty.data_type, Some(DataType::CollatedString(_)))))
}

fn is_string(t: Option<&DataType>) -> bool {
    matches!(t, Some(DataType::String | DataType::CollatedString(_)))
}

/// Strengths already computed, by expression address: plans are DAGs, and a shared child must
/// not be walked once per parent (`shared_expressions_stay_linear`).
type Memo = std::collections::HashMap<usize, Option<(Collation, Strength)>>;

/// The collation and strength `e` carries, when it is a string.
fn strength_of(e: &Expr, memo: &mut Memo) -> Option<(Collation, Strength)> {
    let key = std::ptr::from_ref(e) as usize;
    if let Some(done) = memo.get(&key) {
        return *done;
    }
    let out = strength_of_uncached(e, memo);
    memo.insert(key, out);
    out
}

fn strength_of_uncached(e: &Expr, memo: &mut Memo) -> Option<(Collation, Strength)> {
    match &e.node {
        ExprNode::Collate { collation, .. } => Some((*collation, Strength::Explicit)),
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
            match &c.data_type {
                DataType::String => Some((Collation::UTF8_BINARY, Strength::Implicit)),
                DataType::CollatedString(c) => Some((*c, Strength::Implicit)),
                _ => None,
            }
        }
        ExprNode::Literal(Literal::String(_)) => Some((Collation::UTF8_BINARY, Strength::Default)),
        ExprNode::Alias { expr, .. } => strength_of(expr, memo),
        // A written `CAST(x AS STRING COLLATE c)` (bound, so never in `Default` mode) is
        // explicit; the analyzer's own implicit cast, shown in `Default` mode, takes its
        // child's strength.
        ExprNode::Cast {
            to: DataType::CollatedString(c),
            mode,
            ..
        } if *mode != CastMode::Default => Some((*c, Strength::Explicit)),
        // A written cast to plain `STRING` ends the inherited collation: the result is a binary
        // string, implicit when the child is a string and default otherwise.
        ExprNode::Cast {
            expr,
            to: DataType::String,
            mode,
        } if *mode != CastMode::Default => Some((
            Collation::UTF8_BINARY,
            match strength_of(expr, memo) {
                Some(_) => Strength::Implicit,
                None => Strength::Default,
            },
        )),
        ExprNode::Cast { expr, to, .. } if is_string(Some(to)) => strength_of(expr, memo),
        ExprNode::Function(f)
            if {
                let name = f.name.to_ascii_lowercase();
                INSPECTS.contains(&name.as_str()) || binary_result(&name, f.args.len())
            } =>
        {
            // `collation(x)` and `typeof(x)` return an ordinary string, and so do the
            // binary-result functions (`soundex`, `md5`, ...): the collation of the arguments
            // stops here.
            Some((Collation::UTF8_BINARY, Strength::Default))
        }
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if INSPECTS.contains(&f.id.name()) || binary_result(f.id.name(), f.args.len()) =>
        {
            Some((Collation::UTF8_BINARY, Strength::Default))
        }
        // A call that does not return a string carries no collation, whatever its arguments are
        // (`length(c)`, a boolean test).
        ExprNode::Resolved(ResolvedExpr::Function(f)) if !is_string(Some(&f.data_type)) => None,
        ExprNode::Function(_) | ExprNode::Resolved(ResolvedExpr::Function(_)) => {
            let skip = match &e.node {
                ExprNode::Function(f) => selector_args(&f.name),
                ExprNode::Resolved(ResolvedExpr::Function(f)) => selector_args(f.id.name()),
                _ => 0,
            };
            let mut kids = Vec::new();
            let mut index = 0;
            e.for_each_child(&mut |k| {
                if index >= skip {
                    kids.extend(strength_of(k, memo));
                }
                index += 1;
            });
            merge(&kids).ok().flatten()
        }
        _ => None,
    }
}

/// The collation of operands with these strengths.
fn merge(items: &[(Collation, Strength)]) -> Result<Option<(Collation, Strength)>, Conflict> {
    let distinct = |s: Strength| {
        let mut out: Vec<Collation> = Vec::new();
        for (c, x) in items {
            if *x == s && !out.contains(c) {
                out.push(*c);
            }
        }
        out
    };
    let explicit = distinct(Strength::Explicit);
    if explicit.len() > 1 {
        return Err(Conflict::Explicit(explicit));
    }
    if let Some(c) = explicit.first() {
        return Ok(Some((*c, Strength::Explicit)));
    }
    let implicit = distinct(Strength::Implicit);
    if implicit.len() > 1 {
        return Err(Conflict::Indeterminate);
    }
    if let Some(c) = implicit.first() {
        return Ok(Some((*c, Strength::Implicit)));
    }
    Ok(distinct(Strength::Default)
        .first()
        .map(|c| (*c, Strength::Default)))
}

impl Resolver<'_> {
    /// `expr_node_inner`, with the collation rule for a function call over collated strings.
    pub(super) fn expr_node(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        in_agg: bool,
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let ExprNode::Function(f) = &e.node else {
            return self.expr_node_inner(e, kids, in_agg, window_fn, cx);
        };
        let n = f.args.len().min(kids.len());
        let name = f.name.to_ascii_lowercase();
        // Comparisons of `UTF8_LCASE` / `_RTRIM` strings compare their keys (design 3.3, unit 3a).
        if KEYED.contains(&name.as_str())
            && !f.distinct
            && kids[..n]
                .iter()
                .any(|k| matches!(k.ty.data_type, Some(DataType::CollatedString(_))))
            && let Some(bound) = self.keyed_comparison(e, kids, n, &name, in_agg, window_fn, cx)?
        {
            return Ok(bound);
        }
        if f.distinct
            || COMPARING.contains(&name.as_str())
            || INSPECTS.contains(&name.as_str())
            || !kids[..n].iter().any(|k| {
                k.ty.data_type
                    .as_ref()
                    .is_some_and(DataType::contains_collation)
            })
        {
            return self.expr_node_inner(e, kids, in_agg, window_fn, cx);
        }
        // A collated string inside an array, map or struct is read by field or element only:
        // the nested compare, key and array functions are not collation-aware yet.
        if (!NESTED_OK.contains(&name.as_str()) || map_lookup_compares_collation(&kids[..n]))
            && kids[..n].iter().any(|k| {
                k.ty.data_type.as_ref().is_some_and(|t| {
                    !matches!(t, DataType::CollatedString(_)) && t.contains_collation()
                })
            })
        {
            return Err(unsupported(
                "a collated string inside a nested value",
                e.origin,
            ));
        }
        // A collated string against a plain array or map (`array_contains(arr, c)`) is a
        // comparison the array functions would make as bytes.
        if (!NESTED_OK.contains(&name.as_str()) || map_lookup_compares_collation(&kids[..n]))
            && kids[..n]
                .iter()
                .any(|k| matches!(k.ty.data_type, Some(DataType::CollatedString(_))))
            && kids[..n].iter().any(|k| {
                k.ty.data_type.as_ref().is_some_and(|t| {
                    !matches!(t, DataType::String | DataType::CollatedString(_))
                        && t.any_type(&|x| matches!(x, DataType::String))
                })
            })
        {
            return Err(unsupported(
                "a collated string against a nested value",
                e.origin,
            ));
        }
        let mut memo = Memo::new();
        let (chosen, indeterminate) = self.choose_collation(e, kids, n, &name)?;
        if SEARCHES.contains(&name.as_str())
            && matches!(
                chosen.map(Collation::family),
                Some(CollationFamily::Icu { ai: true, .. })
            )
            && let Some(i) = kids[..n]
                .iter()
                .position(|k| matches!(k.ty.data_type, Some(DataType::CollatedString(_))))
        {
            let shown = with_children(e, kids[..n].iter().map(Bound::shown).collect());
            let zone = self.ctx.conf.session_zone;
            return Err(AnalysisError::new(
                DatatypeMismatchUnexpectedInputType {
                    sql_expr: format!("\"{}\"", pretty(&shown, zone)),
                    param_index: coerce::ordinal(i),
                    required_type: "\"STRING\"".to_owned(),
                    input_sql: format!("\"{}\"", pretty(&kids[i].shown(), zone)),
                    input_type: coerce::quoted_type(kids[i].ty.data_type.as_ref()),
                },
                e.origin,
            ));
        }
        if let Some(c) = chosen
            && !c.is_utf8_binary()
            && (DEPENDS.contains(&name.as_str())
                || (ICU_CASE.contains(&name.as_str())
                    && matches!(c.family(), CollationFamily::Icu { .. })))
        {
            return Err(unsupported(
                format!("{name} over a collated string"),
                e.origin,
            ));
        }
        // The call runs on plain strings: the carrier is one.
        let plain: Vec<Bound> = kids
            .iter()
            .map(|k| {
                let mut k = k.clone();
                let own = match &k.ty.data_type {
                    Some(DataType::CollatedString(c)) => Some(*c),
                    Some(DataType::String) => Some(Collation::UTF8_BINARY),
                    _ => None,
                };
                // A string of another collation is shown cast to the call's.
                // Only a literal is shown cast (a column's implicit cast is not), and the charset
                // of `encode`/`decode`/`to_binary` is no collated operand.
                let literal = matches!(
                    strength_of(&k.shown(), &mut memo),
                    Some((_, Strength::Default))
                );
                if let (Some(own), Some(want)) = (own, chosen)
                    && own != want
                    && literal
                    && !matches!(
                        name.as_str(),
                        "encode" | "decode" | "to_binary" | "try_to_binary"
                    )
                {
                    let shown = Expr::new(ExprNode::Cast {
                        expr: k.shown(),
                        to: DataType::collated_string(want),
                        mode: CastMode::Default,
                    })
                    .with_origin(e.origin);
                    k.shown = Some(Arc::new(shown));
                }
                if matches!(k.ty.data_type, Some(DataType::CollatedString(_))) {
                    k.ty.data_type = Some(DataType::String);
                }
                k
            })
            .collect();
        let mut bound = self.expr_node_inner(e, &plain, in_agg, window_fn, cx)?;
        if let Some(t) = bound.ty.data_type.as_ref() {
            // A nested result would drop the collation of its strings; an indeterminate string
            // result has no type here (Spark carries an indeterminate type).
            let nested = chosen.is_some_and(|c| !c.is_utf8_binary())
                && !matches!(t, DataType::String)
                && t.any_type(&|x| matches!(x, DataType::String));
            if nested || (indeterminate && *t == DataType::String) {
                return Err(unsupported(
                    "a collated or indeterminate string inside a result",
                    e.origin,
                ));
            }
        }
        if let Some(c) = chosen
            && !c.is_utf8_binary()
            && bound.ty.data_type == Some(DataType::String)
            && !binary_result(&name, f.args.len())
            && window_fn
        {
            // The re-tag cast would hide the window function from the window binder.
            return Err(unsupported(
                "a window function over a collated string",
                e.origin,
            ));
        }
        if let Some(c) = chosen
            && !c.is_utf8_binary()
            && bound.ty.data_type == Some(DataType::String)
            && !binary_result(&name, f.args.len())
        {
            let to = DataType::collated_string(c);
            let shown = bound.shown();
            bound.expr = Arc::new(
                Expr::new(ExprNode::Cast {
                    expr: Arc::clone(&bound.expr),
                    to: to.clone(),
                    mode: CastMode::Ansi,
                })
                .with_origin(e.origin),
            );
            bound.shown = Some(shown);
            bound.ty.data_type = Some(to);
        }
        Ok(bound)
    }
}

impl Resolver<'_> {
    /// A comparison of strings whose collation has a key: each string operand is replaced by its
    /// key and the ordinary comparison runs on the keys. `None` when the collation has no such
    /// key here (ICU), which leaves the operands collated for the execution guard to refuse.
    ///
    /// The key of `UTF8_LCASE` is Spark's `lowerCaseCodePoints`: the simple lower case of each
    /// code point, with the final sigma `ς` read as `σ` (and `İ` lowering to `i` + U+0307, as the
    /// function does); `_RTRIM` first drops the trailing spaces.
    #[expect(
        clippy::too_many_arguments,
        reason = "the arguments of `expr_node`, plus the call's"
    )]
    fn keyed_comparison(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        n: usize,
        name: &str,
        in_agg: bool,
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Option<Bound>, AnalysisError> {
        // The keys are for string-to-string comparisons. Against a timestamp, a number or a binary
        // the other operand is cast to the string's type, and a lowered or trimmed key would be
        // cast instead of the text (`ts = '2020-01-01T10:00:00' COLLATE UTF8_LCASE`): those stay
        // with the execution guard. An untyped NULL is no obstacle.
        if !kids[..n]
            .iter()
            .all(|k| k.ty.data_type.is_none() || is_string(k.ty.data_type.as_ref()))
        {
            return Ok(None);
        }
        // `x BETWEEN lo AND hi` is `x >= lo AND x <= hi` to Spark, each half coerced on its own: with
        // different collations on the two halves (`x BETWEEN 'A' COLLATE UTF8_BINARY AND 'c'`) one
        // collation for all three would compare a half in the wrong one. Those stay with the guard.
        if name == "between" && n >= 3 {
            let lower = [kids[0].clone(), kids[1].clone()];
            let upper = [kids[0].clone(), kids[2].clone()];
            if self.choose_collation(e, &lower, 2, name)?.0
                != self.choose_collation(e, &upper, 2, name)?.0
            {
                return Ok(None);
            }
        }
        let (chosen, _) = self.choose_collation(e, kids, n, name)?;
        let Some(c) = chosen else {
            return Ok(None);
        };
        if !matches!(
            c.family(),
            CollationFamily::Utf8Binary | CollationFamily::Utf8Lcase
        ) {
            return Ok(None);
        }
        let mut keyed = Vec::with_capacity(kids.len());
        for (i, k) in kids.iter().enumerate() {
            if i >= n || !is_string(k.ty.data_type.as_ref()) {
                keyed.push(k.clone());
                continue;
            }
            let mut b = self.expr(&Arc::new(key_of(&k.expr, c, e.origin)), cx)?;
            b.shown = Some(k.shown());
            b.carry_markers(std::slice::from_ref(k));
            keyed.push(b);
        }
        self.expr_node_inner(e, &keyed, in_agg, window_fn, cx)
            .map(Some)
    }

    /// The one collation of the string operands of the call `e` (its first `n` arguments),
    /// and whether they were indeterminate for a call that tolerates that.
    ///
    /// # Errors
    /// `COLLATION_MISMATCH.EXPLICIT`, or `INDETERMINATE_COLLATION_IN_EXPRESSION` for the calls
    /// Spark fails on.
    fn choose_collation(
        &self,
        e: &Arc<Expr>,
        kids: &[Bound],
        n: usize,
        name: &str,
    ) -> Result<(Option<Collation>, bool), AnalysisError> {
        let mut memo = Memo::new();
        let items: Vec<(Collation, Strength)> = kids[selector_args(name).min(n)..n]
            .iter()
            .filter(|k| is_string(k.ty.data_type.as_ref()))
            .filter_map(|k| {
                // A subquery, variable, field or element: not a shape the pass reads through,
                // but Spark counts it as implicit, so it is never skipped.
                strength_of(&k.shown(), &mut memo).or(match &k.ty.data_type {
                    Some(DataType::CollatedString(c)) => Some((*c, Strength::Implicit)),
                    _ => Some((Collation::UTF8_BINARY, Strength::Implicit)),
                })
            })
            .collect();
        let mut indeterminate = false;
        let chosen = match merge(&items) {
            Ok(c) => c.map(|(c, _)| c),
            Err(Conflict::Explicit(list)) => {
                let types: Vec<String> = list
                    .iter()
                    .map(|c| format!("\"STRING COLLATE {}\"", c.name()))
                    .collect();
                return Err(AnalysisError::new(
                    CollationMismatchExplicit {
                        explicit_types: types.join(", "),
                    },
                    e.origin,
                ));
            }
            Err(Conflict::Indeterminate) if !FAILS_INDETERMINATE.contains(&name) => {
                indeterminate = true;
                None
            }
            Err(Conflict::Indeterminate) => {
                let shown = with_children(e, kids[..n].iter().map(Bound::shown).collect());
                return Err(AnalysisError::new(
                    IndeterminateCollationInExpression {
                        expr: format!("\"{}\"", pretty(&shown, self.ctx.conf.session_zone)),
                    },
                    e.origin,
                ));
            }
        };
        Ok((chosen, indeterminate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AttrId;

    fn leaf(c: Collation) -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
            crate::BoundColumn {
                id: AttrId(1),
                name: "c".to_owned(),
                data_type: DataType::collated_string(c),
                nullable: true,
            },
        ))))
    }

    /// `e = concat(e, e)` sixty times shares one child per level: a walk that expands the
    /// sharing would visit 2^60 nodes.
    #[test]
    fn strength_of_a_shared_expression_dag_is_linear() {
        let lcase = Collation::parse("UTF8_LCASE").unwrap();
        let mut e = leaf(lcase);
        for _ in 0..60 {
            e = Arc::new(Expr::new(ExprNode::Function(crate::FunctionCall {
                name: "concat".to_owned(),
                qualifier: Vec::new(),
                args: vec![Arc::clone(&e), Arc::clone(&e)],
                arg_names: Vec::new(),
                distinct: false,
                filter: None,
                within_group: Vec::new(),
            })));
        }
        let got = strength_of(&e, &mut Memo::new());
        assert_eq!(got, Some((lcase, Strength::Implicit)));
    }

    #[test]
    fn collation_and_typeof_end_the_propagation() {
        let lcase = Collation::parse("UTF8_LCASE").unwrap();
        for name in ["collation", "typeof"] {
            let call = Expr::function(name, vec![Expr::clone(&leaf(lcase))]);
            let got = strength_of(&call, &mut Memo::new());
            assert_eq!(
                got,
                Some((Collation::UTF8_BINARY, Strength::Default)),
                "{name}"
            );
        }
    }
}

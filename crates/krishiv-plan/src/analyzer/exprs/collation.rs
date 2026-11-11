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

/// Searches that see a `UTF8_LCASE` operand through its lower-cased key: Spark's
/// `lowercaseContains` / `lowercaseStartsWith` / `lowercaseEndsWith` compare the lower-cased
/// code points (unit 5a). `_RTRIM` and the other searches stay with the guard.
const KEYED_SEARCH: &[&str] = &["contains", "startswith", "endswith"];

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
/// arguments (checked against the Spark 4.2 `collations-*` goldens).
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
    "listagg",
    "string_agg",
];

/// Calls that read the type of their argument: it stays collated.
const INSPECTS: &[&str] = &["collation", "typeof"];

/// Internal calls the collation code builds over collated operands: they read the types as they
/// are.
const INTERNAL: &[&str] = &[
    "__collation_key",
    "__collation_map_keys",
    "__collation_greatest",
    "__collation_least",
];

/// Calls that build a nested value of their operands.
const CONSTRUCTORS: &[&str] = &["array", "struct", "named_struct", "map"];

/// Whether the string operands of the constructor `name` take one collation each where they are
/// merged into one element type: all of `array`'s, each of `map`'s keys and values; a struct's
/// fields are independent (Spark coerces no `CreateNamedStruct`).
fn one_collation(kids: &[Bound], name: &str) -> bool {
    let same = |it: &mut dyn Iterator<Item = &Bound>| {
        let mut types = it.filter_map(|k| k.ty.data_type.as_ref());
        types.next().is_none_or(|first| types.all(|t| t == first))
    };
    match name {
        "array" => same(&mut kids.iter()),
        "map" => same(&mut kids.iter().step_by(2)) && same(&mut kids.iter().skip(1).step_by(2)),
        _ => true,
    }
}

/// A call of `name` over `args` and the literal Spark type `t`.
fn typed_call(name: &str, x: &Arc<Expr>, t: &DataType, origin: crate::Origin) -> Arc<Expr> {
    let lit = Expr::new(ExprNode::Literal(Literal::String(t.sql()))).with_origin(origin);
    Arc::new(Expr::function(name, vec![Expr::clone(x), lit]).with_origin(origin))
}

/// The comparison key of `x` read as `t` (design 3.3): the bytes of a binary string, else
/// `__collation_key`, which is `lowerCaseCodePoints` and the right trim.
fn key_of(x: &Arc<Expr>, t: &DataType, origin: crate::Origin) -> Arc<Expr> {
    if matches!(t, DataType::String) {
        return Arc::new(
            Expr::new(ExprNode::Cast {
                expr: Arc::clone(x),
                to: DataType::String,
                mode: CastMode::Ansi,
            })
            .with_origin(origin),
        );
    }
    typed_call("__collation_key", x, t, origin)
}

/// The comparison key of `x` under the collation `c` (`IN (subquery)` pairs).
pub(super) fn key_of_collation(x: &Arc<Expr>, c: Collation, origin: crate::Origin) -> Arc<Expr> {
    key_of(x, &string_of(c), origin)
}

/// `t` with a binary collation as `STRING`, else the collated string.
fn string_of(c: Collation) -> DataType {
    if c.is_utf8_binary() {
        DataType::String
    } else {
        DataType::collated_string(c)
    }
}

/// Whether every collated leaf of `t` has a key here (`UTF8_LCASE`, `_RTRIM`).
fn has_key(t: &DataType) -> bool {
    !t.any_type(&|x| {
        matches!(x, DataType::CollatedString(c) if !matches!(
            c.family(),
            CollationFamily::Utf8Binary | CollationFamily::Utf8Lcase
        ) || c.is_indeterminate())
    })
}

/// The collations `e` names explicitly (`COLLATE`, a written cast to a collated string), or
/// `None` when it carries no collation at all: no column (implicit) and nothing explicit.
fn carried_collations(e: &Expr) -> Option<Vec<Collation>> {
    let (mut named, mut column) = (Vec::new(), false);
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        if let crate::walk::Node::Expr(x) = n {
            match &x.node {
                // Spark gives a column and a scalar subquery implicit strength.
                ExprNode::Resolved(ResolvedExpr::Column(_) | ResolvedExpr::OuterColumn(_))
                | ExprNode::Subquery { .. } => {
                    column = true;
                }
                ExprNode::Collate { collation, .. } => named.push(*collation),
                ExprNode::Cast {
                    to: DataType::CollatedString(c),
                    mode,
                    ..
                } if *mode != CastMode::Default => named.push(*c),
                _ => {}
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    (column || !named.is_empty()).then_some(named)
}

/// The collation whose key an `IN (subquery)` pair compares through: the collated side of a
/// string pair when it is `UTF8_LCASE` or `_RTRIM` (Spark compares the pair in that collation).
pub(super) fn subquery_key_collation(
    left: Option<&DataType>,
    right: &DataType,
) -> Option<Collation> {
    if !is_string(left) || !is_string(Some(right)) {
        return None;
    }
    // Only a pair in one and the same collation is keyed. A pair of different collations is
    // Spark's COLLATION_MISMATCH / INDETERMINATE_COLLATION_IN_EXPRESSION (or an explicit
    // `COLLATE` winning), which keying one side would answer wrongly.
    match (left, right) {
        (Some(DataType::CollatedString(l)), DataType::CollatedString(r))
            if l == r && (l.rtrim() || matches!(l.family(), CollationFamily::Utf8Lcase)) =>
        {
            Some(*l)
        }
        _ => None,
    }
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
        && kids.iter().any(|k| is_collated(k.ty.data_type.as_ref())))
}

/// A collated string that is more than the display-only `UTF8_BINARY` marker.
fn is_collated(t: Option<&DataType>) -> bool {
    t.is_some_and(DataType::is_collated_string)
}

/// The display-only `UTF8_BINARY` marker (ADR 0043, update 2026-11-07).
fn is_marker(t: Option<&DataType>) -> bool {
    matches!(t, Some(DataType::CollatedString(c)) if c.is_utf8_binary())
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
            if INSPECTS.contains(&f.name.to_ascii_lowercase().as_str())
                || BINARY_RESULT.contains(&f.name.to_ascii_lowercase().as_str()) =>
        {
            // `collation(x)` and `typeof(x)` return an ordinary string, and so do the
            // binary-result functions (`soundex`, `md5`, ...): the collation of the arguments
            // stops here.
            Some((Collation::UTF8_BINARY, Strength::Default))
        }
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if INSPECTS.contains(&f.id.name()) || BINARY_RESULT.contains(&f.id.name()) =>
        {
            Some((Collation::UTF8_BINARY, Strength::Default))
        }
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

/// Whether `e` is a string literal (under casts and collate) that cannot begin with U+0307 or end
/// with `i`/`I`, so no lower-cased match of it can split the pair `İ` lower-cases to (a pattern
/// ending in `İ` is refused too: stricter than needed, since its lower-case ends on the dot).
fn safe_search_pattern(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Literal(Literal::String(s)) => {
            !s.starts_with('\u{307}') && !s.ends_with(['i', 'I', '\u{130}'])
        }
        ExprNode::Cast { expr, .. } | ExprNode::Collate { expr, .. } => safe_search_pattern(expr),
        _ => false,
    }
}

/// A copy of the `i`-th argument of the call `e` (a name for error messages).
fn arg_of(e: &Expr, i: usize) -> Expr {
    match &e.node {
        ExprNode::Function(f) => f.args.get(i).map_or_else(
            || Expr::new(ExprNode::Literal(Literal::Null(None))),
            |a| Expr::clone(a),
        ),
        _ => Expr::new(ExprNode::Literal(Literal::Null(None))),
    }
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
        if INTERNAL.contains(&name.as_str()) {
            return self.expr_node_inner(e, kids, in_agg, window_fn, cx);
        }
        if let Some(bound) = self.collated_call(e, kids, n, &name, in_agg, window_fn, cx)? {
            return Ok(bound);
        }
        // Comparisons of `UTF8_LCASE` / `_RTRIM` strings compare their keys (design 3.3, unit 3a).
        if (KEYED.contains(&name.as_str()) || KEYED_SEARCH.contains(&name.as_str()))
            && !f.distinct
            && kids[..n].iter().any(|k| {
                k.ty.data_type
                    .as_ref()
                    .is_some_and(DataType::contains_collation)
            })
            && let Some(bound) = self.keyed_comparison(e, kids, n, &name, in_agg, window_fn, cx)?
        {
            return Ok(bound);
        }
        // `instr` / `locate` / `position` over `UTF8_LCASE` strings answer positions in the
        // original text: the kernel `__locate_lcase` matches the lower-cased code points (unit 5b).
        if matches!(
            name.as_str(),
            "instr" | "locate" | "position" | "find_in_set"
        ) && !f.distinct
            && (2..=3).contains(&n)
            && (n == 2 || name != "find_in_set")
            && kids[..2].iter().all(|k| is_string(k.ty.data_type.as_ref()))
            && kids[..2]
                .iter()
                .any(|k| is_collated(k.ty.data_type.as_ref()))
            && let Some(bound) = self.locate_lcase(e, kids, n, &name, in_agg, window_fn, cx)?
        {
            return Ok(bound);
        }
        // Spark's `CreateArray` / `CreateNamedStruct` / `CreateMap`: a constructor over operands of
        // one collation keeps it in the result type, which comparisons and keys then read.
        // A `map` of two or more collated keys stays refused: its duplicate-key check still compares
        // bytes, where Spark's compares keys (`DUPLICATED_MAP_KEY` for 'a' and 'A' under UTF8_LCASE).
        let keyed_map_dedupe = name == "map"
            && n > 2
            && kids[..n].iter().step_by(2).any(|k| {
                k.ty.data_type
                    .as_ref()
                    .is_some_and(DataType::contains_collation)
            });
        if CONSTRUCTORS.contains(&name.as_str())
            && !keyed_map_dedupe
            && one_collation(&kids[..n], &name)
        {
            return self.expr_node_inner(e, kids, in_agg, window_fn, cx);
        }
        if f.distinct
            || COMPARING.contains(&name.as_str())
            || INSPECTS.contains(&name.as_str())
            || !kids[..n].iter().any(|k| {
                is_marker(k.ty.data_type.as_ref())
                    || k.ty
                        .data_type
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
                .any(|k| is_collated(k.ty.data_type.as_ref()))
            && kids[..n].iter().any(|k| {
                k.ty.data_type.as_ref().is_some_and(|t| {
                    !t.is_string_type() && t.any_type(&|x| matches!(x, DataType::String))
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
                .position(|k| is_collated(k.ty.data_type.as_ref()))
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
                        to: if want.is_utf8_binary() {
                            DataType::String
                        } else {
                            DataType::collated_string(want)
                        },
                        mode: CastMode::Default,
                    })
                    .with_origin(e.origin);
                    k.shown = Some(Arc::new(shown));
                }
                if is_marker(k.ty.data_type.as_ref()) || is_collated(k.ty.data_type.as_ref()) {
                    k.ty.data_type = Some(DataType::String);
                }
                k
            })
            .collect();
        // Spark casts an operand to the chosen collation only if it has another one; the result
        // type is the function's own, which for the string functions is the head string
        // operand's type after that cast. So the head carries the marker if it already had it or
        // was cast to `UTF8_BINARY` (the cast target is the written type): `elt(2, s, c)` with a
        // plain `s` is plain, `translate(lcase_col, c COLLATE UTF8_BINARY, ..)` is marked.
        let marked = {
            let skip = selector_args(&name).min(n);
            let own = |k: &Bound| match k.ty.data_type.as_ref() {
                Some(DataType::CollatedString(c)) => Some(*c),
                Some(DataType::String) => Some(Collation::UTF8_BINARY),
                _ => None,
            };
            kids[skip..n]
                .iter()
                .find(|k| own(k).is_some())
                .is_some_and(|head| is_marker(head.ty.data_type.as_ref()) || own(head) != chosen)
        };
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
            && !BINARY_RESULT.contains(&name.as_str())
            && window_fn
        {
            // The re-tag cast would hide the window function from the window binder.
            return Err(unsupported(
                "a window function over a collated string",
                e.origin,
            ));
        }
        // The result is re-tagged with the chosen collation. For `UTF8_BINARY` that is the
        // display-only marker (Spark prints `upper(c)` of a `STRING COLLATE UTF8_BINARY` column
        // that way), kept only when `marked` and outside windows and the inspecting functions.
        if let Some(c) = chosen
            && bound.ty.data_type == Some(DataType::String)
            && !BINARY_RESULT.contains(&name.as_str())
            && if c.is_utf8_binary() {
                marked && !window_fn && !INSPECTS.contains(&name.as_str())
            } else {
                true
            }
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
        // Nested operands (`array(c) = array('A')`): the key has the shape of the value. Only
        // equality: nested ordering stays with the guard.
        let nested = kids[..n].iter().find_map(|k| {
            k.ty.data_type
                .as_ref()
                .filter(|t| t.contains_collation() && !t.is_string_type())
        });
        if let Some(t) = nested {
            // Spark merges the collations leaf by leaf, by strength (`CollationTypeCoercion.
            // mergeStructurally`); that merge is not built here, so the operands are keyed with
            // `t` only where it is the one collation: an operand carrying none (no column, no
            // `COLLATE`) defers to it, any other must be of type `t` as written (not coerced to
            // it) and name only collations of `t`.
            let agrees = |k: &Bound| {
                let shown = k.shown();
                let carried = carried_collations(&shown);
                carried.is_none()
                    || (k.ty.data_type.as_ref() == Some(t)
                        && !matches!(
                            k.expr.node,
                            ExprNode::Cast {
                                mode: CastMode::Default,
                                ..
                            }
                        )
                        && carried.is_some_and(|named| {
                            named.iter().all(|c| {
                                t.any_type(&|x| {
                                    x == &DataType::collated_string(*c)
                                        || (c.is_utf8_binary() && x == &DataType::String)
                                })
                            })
                        }))
            };
            if !matches!(name, "==" | "<=>")
                || !has_key(t)
                || !kids[..n].iter().all(|k| {
                    k.ty.data_type
                        .as_ref()
                        .is_none_or(|u| u.collation_key_type() == t.collation_key_type())
                        && agrees(k)
                })
            {
                return Ok(None);
            }
            let t = t.clone();
            return self
                .compare_keys(e, kids, n, |_| t.clone(), in_agg, window_fn, cx)
                .map(Some);
        }
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
        // `x BETWEEN lo AND hi` is `x >= lo AND x <= hi` to Spark, each half coerced on its own.
        if name == "between" && n >= 3 {
            let (l, u) = (
                [kids[0].clone(), kids[1].clone()],
                [kids[0].clone(), kids[2].clone()],
            );
            if self.choose_collation(e, &l, 2, name)?.0 != self.choose_collation(e, &u, 2, name)?.0
            {
                let f = |op: &str, i: usize| {
                    Arc::new(
                        Expr::function(op, vec![arg_of(e, 0), arg_of(e, i)]).with_origin(e.origin),
                    )
                };
                let ge = self.expr_node(&f(">=", 1), &l, in_agg, window_fn, cx)?;
                let le = self.expr_node(&f("<=", 2), &u, in_agg, window_fn, cx)?;
                let mut both = self.operator_call(Builtin::And, e.origin, &[ge, le])?;
                both.shown = Some(with_children(
                    e,
                    kids[..n].iter().map(Bound::shown).collect(),
                ));
                return Ok(Some(both));
            }
        }
        let (chosen, _) = self.choose_collation(e, kids, n, name)?;
        let Some(c) = chosen else {
            return Ok(None);
        };
        if !has_key(&string_of(c))
            || (KEYED_SEARCH.contains(&name)
                && (c.rtrim() || !matches!(c.family(), CollationFamily::Utf8Lcase)))
        {
            return Ok(None);
        }
        // Lower-casing `İ` (U+0130) gives `i` + U+0307, and Spark refuses a match that starts or
        // ends inside that pair (`lowercaseMatchLengthFrom` / `lowercaseMatchUntil`). Only a
        // literal pattern that cannot start with U+0307 or end with `i` is safe on the key; any
        // other pattern stays with the execution guard.
        if KEYED_SEARCH.contains(&name)
            && !kids.get(1).is_some_and(|k| safe_search_pattern(&k.expr))
        {
            return self.search_lcase(e, kids, name, in_agg, window_fn, cx);
        }
        self.compare_keys(e, kids, n, |_| string_of(c), in_agg, window_fn, cx)
            .map(Some)
    }

    /// Replaces each typed operand of the call by its key read as `type_of(i)` and runs it.
    #[expect(
        clippy::too_many_arguments,
        reason = "the arguments of `expr_node`, plus the operand type"
    )]
    fn compare_keys(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        n: usize,
        type_of: impl Fn(usize) -> DataType,
        in_agg: bool,
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let search = matches!(
            &e.node,
            ExprNode::Function(f) if KEYED_SEARCH.contains(&f.name.to_ascii_lowercase().as_str())
        );
        let mut keyed = Vec::with_capacity(kids.len());
        for (i, k) in kids.iter().enumerate() {
            if i >= n || k.ty.data_type.is_none() {
                keyed.push(k.clone());
                continue;
            }
            let target = type_of(i);
            let mut b = self.expr(&key_of(&k.expr, &target, e.origin), cx)?;
            b.shown = Some(k.shown());
            // A literal of another collation is shown cast to the call's, as in the plain path.
            if search
                && matches!(
                    strength_of(&k.shown(), &mut Memo::new()),
                    Some((_, Strength::Default))
                )
                && k.ty.data_type.as_ref() != Some(&target)
            {
                b.shown = Some(Arc::new(
                    Expr::new(ExprNode::Cast {
                        expr: k.shown(),
                        to: target,
                        mode: CastMode::Default,
                    })
                    .with_origin(e.origin),
                ));
            }
            b.carry_markers(std::slice::from_ref(k));
            keyed.push(b);
        }
        self.expr_node_inner(e, &keyed, in_agg, window_fn, cx)
    }

    /// `greatest`/`least`/`nullif`/`equal_null` and map lookups over a collated value: `None`
    /// when `name` is another call or nothing here has a key.
    #[expect(
        clippy::too_many_arguments,
        reason = "the arguments of `expr_node`, plus the call's"
    )]
    fn collated_call(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        n: usize,
        name: &str,
        in_agg: bool,
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Option<Bound>, AnalysisError> {
        let strings = kids[..n]
            .iter()
            .all(|k| k.ty.data_type.is_none() || is_string(k.ty.data_type.as_ref()));
        let any_collated = kids[..n]
            .iter()
            .any(|k| is_collated(k.ty.data_type.as_ref()));
        if matches!(
            name,
            "element_at" | "try_element_at" | "__subscript" | "map_contains_key"
        ) && n == 2
            && let Some(DataType::Map(m)) = kids[0].ty.data_type.clone()
            && m.key_type().is_collated_string()
            && has_key(m.key_type())
            && kids[1].ty.data_type.as_ref() == Some(m.key_type())
        {
            let t = DataType::Map(Arc::clone(&m));
            let map = typed_call("__collation_map_keys", &kids[0].expr, &t, e.origin);
            let mut keyed = vec![
                self.expr(&map, cx)?,
                self.expr(&key_of(&kids[1].expr, m.key_type(), e.origin), cx)?,
            ];
            keyed[0].shown = Some(kids[0].shown());
            keyed[1].shown = Some(kids[1].shown());
            return self
                .expr_node_inner(e, &keyed, in_agg, window_fn, cx)
                .map(Some);
        }
        if !any_collated || !strings {
            return Ok(None);
        }
        match name {
            "equal_null" if n == 2 => {
                let f = Arc::new(
                    Expr::function("<=>", vec![arg_of(e, 0), arg_of(e, 1)]).with_origin(e.origin),
                );
                let mut b = self.expr_node(&f, kids, in_agg, window_fn, cx)?;
                b.shown = Some(with_children(
                    e,
                    kids[..n].iter().map(Bound::shown).collect(),
                ));
                Ok(Some(b))
            }
            // Spark's `NullIf` is `if(a = b, NULL, a)`.
            "nullif" if n == 2 => {
                let eq = Arc::new(
                    Expr::function("==", vec![arg_of(e, 0), arg_of(e, 1)]).with_origin(e.origin),
                );
                let Some(cond) =
                    self.keyed_comparison(&eq, kids, 2, "==", in_agg, window_fn, cx)?
                else {
                    return Ok(None);
                };
                let null = self.expr(
                    &Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Null(None))).with_origin(e.origin),
                    ),
                    cx,
                )?;
                let call = Arc::new(
                    Expr::function("if", vec![arg_of(e, 0), arg_of(e, 1), arg_of(e, 0)])
                        .with_origin(e.origin),
                );
                let value = Bound {
                    ty: kids[0].ty.clone(),
                    ..kids[0].clone()
                };
                let mut b =
                    self.expr_node_inner(&call, &[cond, null, value], in_agg, window_fn, cx)?;
                b.shown = Some(with_children(
                    e,
                    kids[..n].iter().map(Bound::shown).collect(),
                ));
                Ok(Some(b))
            }
            "greatest" | "least" if n >= 2 => {
                let (chosen, _) = self.choose_collation(e, kids, n, name)?;
                let Some(c) = chosen.filter(|c| has_key(&string_of(*c))) else {
                    return Ok(None);
                };
                let func = if name == "greatest" {
                    "__collation_greatest"
                } else {
                    "__collation_least"
                };
                let t = Expr::new(ExprNode::Literal(Literal::String(string_of(c).sql())))
                    .with_origin(e.origin);
                let plain = |k: &Bound| {
                    Expr::new(ExprNode::Cast {
                        expr: Arc::clone(&k.expr),
                        to: DataType::String,
                        mode: CastMode::Ansi,
                    })
                    .with_origin(e.origin)
                };
                let mut acc = plain(&kids[0]);
                for k in &kids[1..n] {
                    acc =
                        Expr::function(func, vec![t.clone(), acc, plain(k)]).with_origin(e.origin);
                }
                let mut b = self.expr(&Arc::new(acc), cx)?;
                let to = string_of(c);
                b.expr = Arc::new(
                    Expr::new(ExprNode::Cast {
                        expr: Arc::clone(&b.expr),
                        to: to.clone(),
                        mode: CastMode::Ansi,
                    })
                    .with_origin(e.origin),
                );
                b.ty.data_type = Some(to);
                b.shown = Some(with_children(
                    e,
                    kids[..n].iter().map(Bound::shown).collect(),
                ));
                Ok(Some(b))
            }
            _ => Ok(None),
        }
    }

    /// `contains` / `startswith` / `endswith` with a pattern the lower-cased key cannot answer
    /// (a column, or a literal that may split `İ`'s pair) as the `__*_lcase` kernel call.
    fn search_lcase(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        name: &str,
        in_agg: bool,
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Option<Bound>, AnalysisError> {
        if kids.len() != 2 {
            // The wrong argument count is the generic rules' error (`WRONG_NUM_ARGS`).
            return Ok(None);
        }
        let mut plain = Vec::with_capacity(2);
        for k in &kids[..2] {
            let cast = Expr::new(ExprNode::Cast {
                expr: Arc::clone(&k.expr),
                to: DataType::String,
                mode: CastMode::Ansi,
            })
            .with_origin(e.origin);
            let mut b = self.expr(&Arc::new(cast), cx)?;
            b.shown = Some(k.shown());
            plain.push(b);
        }
        let call = Expr::function(
            format!("__{name}_lcase"),
            plain.iter().map(|k| (*k.expr).clone()).collect(),
        )
        .with_origin(e.origin);
        let mut b = self.expr_node_inner(&Arc::new(call), &plain, in_agg, window_fn, cx)?;
        b.shown = Some(with_children(e, kids.iter().map(Bound::shown).collect()));
        Ok(Some(b))
    }

    /// `instr` / `locate` / `position` as `__locate_lcase(substr, str, start)` when the strings
    /// compare in `UTF8_LCASE` (not `_RTRIM`); `None` leaves the call to the other rules.
    #[expect(clippy::too_many_arguments, reason = "the arguments of `expr_node`")]
    fn locate_lcase(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        n: usize,
        name: &str,
        in_agg: bool,
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Option<Bound>, AnalysisError> {
        if kids.len() != n || n < 2 || (name == "instr" && n != 2) {
            return Ok(None);
        }
        let (chosen, _) = self.choose_collation(e, kids, 2, name)?;
        let Some(c) = chosen else {
            return Ok(None);
        };
        if c.rtrim() || !matches!(c.family(), CollationFamily::Utf8Lcase) {
            return Ok(None);
        }
        let (sub, text) = if name == "instr" { (1, 0) } else { (0, 1) };
        let find_in_set = name == "find_in_set";
        let plain = |this: &mut Self, k: &Bound| -> Result<Bound, AnalysisError> {
            let cast = Expr::new(ExprNode::Cast {
                expr: Arc::clone(&k.expr),
                to: DataType::String,
                mode: CastMode::Ansi,
            })
            .with_origin(e.origin);
            let mut b = this.expr(&Arc::new(cast), cx)?;
            b.shown = Some(k.shown());
            Ok(b)
        };
        let mut keyed = vec![plain(self, &kids[sub])?, plain(self, &kids[text])?];
        if !find_in_set {
            keyed.push(if n == 3 && name != "instr" {
                kids[2].clone()
            } else {
                let one = Expr::new(ExprNode::Literal(Literal::Integer(1))).with_origin(e.origin);
                self.expr(&Arc::new(one), cx)?
            });
        }
        let call = Expr::function(
            if find_in_set {
                "__find_in_set_lcase"
            } else {
                "__locate_lcase"
            },
            keyed.iter().map(|k| (*k.expr).clone()).collect(),
        )
        .with_origin(e.origin);
        let mut b = self.expr_node_inner(&Arc::new(call), &keyed, in_agg, window_fn, cx)?;
        b.shown = Some(with_children(e, kids.iter().map(Bound::shown).collect()));
        Ok(Some(b))
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

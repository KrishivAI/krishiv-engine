// SPDX-License-Identifier: Apache-2.0
//! The grouped pivot the DataFrame front ends build (PySpark's `groupBy(...).pivot(col, values).agg(...)`,
//! the native `GroupedData::pivot` and the Connect `Aggregate` with `GROUP_TYPE_PIVOT`): the same
//! [`PlanNode::Pivot`] the SQL `PIVOT` clause lowers to, so values, naming, casts and errors are the
//! analyzer's.
//!
//! The node groups by every input column neither the pivot column nor an aggregate reads, so the
//! input is first narrowed to the grouping keys, a copy of the pivot column and the columns the
//! aggregates read. The copy keeps an aggregate that reads the pivot column itself
//! (`count("course")`) from taking it out of the grouping; a grouping key an aggregate reads is
//! grouped by a hidden copy and renamed back.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::{InvalidUsageOfStarOrRegex, KrishivError, KrishivUnsupportedFeature};

use crate::{ColumnRef, Expr, ExprNode, Literal, LogicalPlan, Origin, PivotValue, PlanNode, Star};

fn refuse(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

/// The name the pivot column's copy has in the narrowed input (never an output column).
const PIVOT_COPY: &str = "__krishiv_pivot";

/// The unresolved column references under `e`, and whether a `*` is under it. Each distinct node is
/// visited once: expressions are DAGs, and a shared child would otherwise be walked once per path.
fn references(e: &Expr, out: &mut Vec<Expr>, star: &mut bool) {
    let mut seen: HashSet<*const Expr> = HashSet::new();
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(e) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(e)) {
            continue;
        }
        match &e.node {
            ExprNode::Column(_) => out.push(e.clone()),
            ExprNode::Star(_) => *star = true,
            _ => {}
        }
        e.for_each_child(&mut |child| stack.push(child));
    }
}

/// The last name part of a bare or aliased column, the name its output has.
fn output_name(e: &Expr) -> Option<String> {
    match &e.node {
        ExprNode::Column(c) => c.name.last().cloned(),
        ExprNode::Alias { name, .. } => Some(name.clone()),
        _ => None,
    }
}

/// The column named `name` (one name part, so dots in it are kept).
fn named(name: &str) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Column(ColumnRef::new(vec![
        name.to_owned(),
    ]))))
}

/// The pivot of `aggregates` over `input` grouped by `keys`, on `column` with `values`. `origin`
/// names the nodes this builds under the returned one (the returned node is the caller's to wrap).
///
/// # Errors
/// A `*` in an aggregate (`INVALID_USAGE_OF_STAR_OR_REGEX`), or a shape the narrowing cannot keep
/// exact (a nested field or qualified column an aggregate reads, a column named like the internal
/// copy, same-named columns of different DataFrames, an unnamed key beside a clash, a value named
/// like a key an aggregate reads): `KRISHIV_UNSUPPORTED_FEATURE`.
pub fn grouped_pivot(
    input: &Arc<LogicalPlan>,
    keys: &[Expr],
    column: &Expr,
    values: Vec<PivotValue>,
    aggregates: Vec<Arc<Expr>>,
    case_sensitive: bool,
    mut origin: impl FnMut() -> Origin,
) -> Result<PlanNode, KrishivError> {
    let mut read = Vec::new();
    let mut star = false;
    for a in &aggregates {
        references(a, &mut read, &mut star);
    }
    if star {
        // Spark's DataFrame pivot does not expand `count(*)` (SQL's PIVOT clause does).
        return Err(InvalidUsageOfStarOrRegex {
            elem: "'*'".to_owned(),
            pretty_name: "Pivot".to_owned(),
        }
        .into());
    }
    if read
        .iter()
        .any(|r| matches!(&r.node, ExprNode::Column(c) if c.name.len() > 1))
    {
        // Narrowing the input by a dotted name would flatten `s.x` and drop the struct `s`.
        return Err(refuse(
            "a pivot aggregate that reads a nested field or qualified column",
        ));
    }
    let fold = |n: String| if case_sensitive { n } else { n.to_lowercase() };
    let read_folded: Vec<Option<String>> = read.iter().map(|r| output_name(r).map(fold)).collect();
    let read_names: Vec<String> = read_folded.iter().flatten().cloned().collect();
    if read_names.contains(&fold(PIVOT_COPY.to_owned())) {
        return Err(refuse(
            "a pivot aggregate that reads a column named like the internal pivot copy",
        ));
    }
    // Two references of one name from different DataFrames (a cross join's `left["v"]`, `right["v"]`)
    // cannot both survive a narrowing that keeps one column of each name. Each name is folded once
    // and its first source remembered, so the check is linear in the references.
    let source = |r: &Expr| match &r.node {
        ExprNode::Column(c) => c.plan_id.map(|id| id.0),
        _ => None,
    };
    let mut first_source: std::collections::HashMap<&str, Option<i64>> =
        std::collections::HashMap::new();
    for (r, name) in read.iter().zip(&read_folded) {
        let Some(name) = name else { continue };
        if *first_source.entry(name).or_insert_with(|| source(r)) != source(r) {
            return Err(refuse(
                "a pivot aggregate that reads same-named columns of different DataFrames",
            ));
        }
    }
    let names: Vec<Option<String>> = keys.iter().map(output_name).collect();
    let clash: Vec<bool> = names
        .iter()
        .map(|n| {
            n.as_ref()
                .is_some_and(|n| read_names.contains(&fold(n.clone())))
        })
        .collect();
    let any_clash = clash.iter().any(|c| *c);
    if any_clash && names.iter().any(Option::is_none) {
        return Err(refuse(
            "a pivot aggregate that reads a grouping column beside an unnamed grouping expression",
        ));
    }
    // The internal copies must not share a name with a column the projection retains.
    let key_names: Vec<String> = names.iter().flatten().cloned().map(fold).collect();
    if key_names.contains(&fold(PIVOT_COPY.to_owned()))
        || (0..keys.len()).any(|i| {
            let c = fold(format!("__krishiv_key_{i}"));
            key_names.contains(&c) || read_names.contains(&c)
        })
    {
        return Err(refuse(
            "a grouping column named like an internal pivot copy (`__krishiv_pivot`, `__krishiv_key_N`)",
        ));
    }
    if any_clash {
        // The keys are restored by name above the pivot, so no pivot output may carry a key's name.
        // Spark names an output by the value's text, then `_` and the aggregate's name when there
        // are several aggregates (`PivotTransformer.outputName`).
        let agg_names: Vec<Option<String>> = aggregates.iter().map(|a| output_name(a)).collect();
        let collides = values.iter().any(|v| {
            let Some(ExprNode::Literal(Literal::String(t))) = v.exprs.first().map(|e| &e.node)
            else {
                return false;
            };
            if aggregates.len() == 1 {
                key_names.contains(&fold(t.clone()))
            } else {
                agg_names
                    .iter()
                    .flatten()
                    .any(|n| key_names.contains(&fold(format!("{t}_{n}"))))
            }
        });
        if collides {
            return Err(refuse(
                "a pivot output named like a grouping column that is restored by name",
            ));
        }
    }
    let copy = |i: usize| format!("__krishiv_key_{i}");
    let mut kept: Vec<Arc<Expr>> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        if clash[i] {
            let inner = match &key.node {
                ExprNode::Alias { expr, .. } => (**expr).clone(),
                _ => key.clone(),
            };
            kept.push(Arc::new(inner.alias(copy(i))));
        } else {
            kept.push(Arc::new(key.clone()));
        }
    }
    kept.push(Arc::new(column.clone().alias(PIVOT_COPY)));
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (r, name) in read.into_iter().zip(read_folded) {
        let Some(name) = name else { continue };
        if seen.insert(name) {
            kept.push(Arc::new(r));
        }
    }
    let wrap =
        |node: PlanNode, origin: Origin| Arc::new(LogicalPlan::new(node).with_origin(origin));
    let narrowed = wrap(
        PlanNode::Project {
            input: Arc::clone(input),
            exprs: kept,
        },
        origin(),
    );
    let pivot = PlanNode::Pivot {
        input: narrowed,
        pivot_columns: vec![named(PIVOT_COPY)],
        values,
        aggregates,
    };
    if !any_clash {
        return Ok(pivot);
    }
    let pivoted = wrap(pivot, origin());
    // The keys back under their names, then the pivoted columns.
    let mut out: Vec<Arc<Expr>> = Vec::new();
    let mut except: Vec<Vec<String>> = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let Some(name) = name else { continue };
        if clash[i] {
            out.push(Arc::new(Expr::clone(&named(&copy(i))).alias(name)));
            except.push(vec![copy(i)]);
        } else {
            out.push(named(name));
            except.push(vec![name.clone()]);
        }
    }
    out.push(Arc::new(Expr::new(ExprNode::Star(Star {
        qualifier: Vec::new(),
        except,
        replace: Vec::new(),
        position: None,
        regex: None,
        plan_id: None,
    }))));
    Ok(PlanNode::Project {
        input: pivoted,
        exprs: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlanId;

    fn col(name: &str) -> Expr {
        Expr::new(ExprNode::Column(ColumnRef::new(vec![name.to_owned()])))
    }

    fn col_of(plan: i64, name: &str) -> Expr {
        let mut c = ColumnRef::new(vec![name.to_owned()]);
        c.plan_id = Some(PlanId(plan));
        Expr::new(ExprNode::Column(c))
    }

    fn input() -> Arc<LogicalPlan> {
        Arc::new(LogicalPlan::new(PlanNode::OneRowRelation))
    }

    fn text(v: &str) -> PivotValue {
        PivotValue {
            exprs: vec![Arc::new(Expr::new(ExprNode::Literal(Literal::String(
                v.to_owned(),
            ))))],
            alias: None,
        }
    }

    fn build(
        keys: &[Expr],
        values: Vec<PivotValue>,
        aggs: Vec<Expr>,
    ) -> Result<PlanNode, KrishivError> {
        grouped_pivot(
            &input(),
            keys,
            &col("p"),
            values,
            aggs.into_iter().map(Arc::new).collect(),
            false,
            Origin::default,
        )
    }

    fn refused(r: Result<PlanNode, KrishivError>, needle: &str) {
        let msg = r.expect_err("refused").to_string();
        assert!(msg.contains(needle), "{msg}");
    }

    #[test]
    fn a_plain_pivot_narrows_the_input_and_is_the_pivot_node() {
        let node = build(&[col("k")], vec![text("a")], vec![col("v")]).expect("builds");
        let PlanNode::Pivot { input, .. } = node else {
            panic!("a Pivot node");
        };
        let PlanNode::Project { exprs, .. } = &input.node else {
            panic!("a narrowing Project");
        };
        assert_eq!(exprs.len(), 3, "key, pivot copy, read column");
    }

    #[test]
    fn a_star_aggregate_is_refused() {
        let star = Expr::new(ExprNode::Star(Star {
            qualifier: Vec::new(),
            except: Vec::new(),
            replace: Vec::new(),
            position: None,
            regex: None,
            plan_id: None,
        }));
        let r = build(&[col("k")], vec![text("a")], vec![star]);
        assert!(r.is_err());
    }

    #[test]
    fn a_qualified_column_read_is_refused() {
        let q = Expr::new(ExprNode::Column(ColumnRef::new(vec![
            "s".to_owned(),
            "x".to_owned(),
        ])));
        refused(
            build(&[col("k")], vec![text("a")], vec![q]),
            "nested field or qualified column",
        );
    }

    #[test]
    fn a_read_named_like_the_internal_copy_is_refused() {
        refused(
            build(&[col("k")], vec![text("a")], vec![col(PIVOT_COPY)]),
            "internal pivot copy",
        );
    }

    #[test]
    fn same_named_columns_of_different_dataframes_are_refused() {
        refused(
            build(
                &[col("k")],
                vec![text("a")],
                vec![col_of(1, "v"), col_of(2, "v")],
            ),
            "different DataFrames",
        );
    }

    #[test]
    fn an_unnamed_key_beside_a_clash_is_refused() {
        let unnamed = Expr::new(ExprNode::Literal(Literal::String("x".to_owned())));
        refused(
            build(&[col("k"), unnamed], vec![text("a")], vec![col("k")]),
            "unnamed grouping expression",
        );
    }

    #[test]
    fn a_value_named_like_a_clashing_key_is_refused() {
        refused(
            build(&[col("k")], vec![text("K")], vec![col("k")]),
            "named like a grouping column",
        );
    }

    #[test]
    fn a_key_an_aggregate_reads_is_grouped_by_a_copy_and_restored() {
        let node = build(&[col("k"), col("j")], vec![text("a")], vec![col("k")]).expect("builds");
        let PlanNode::Project { input, exprs } = node else {
            panic!("a restoring Project");
        };
        assert!(matches!(&input.node, PlanNode::Pivot { .. }));
        // The two keys then the star over the pivoted columns.
        assert_eq!(exprs.len(), 3);
        let PlanNode::Pivot {
            input: narrowed, ..
        } = &input.node
        else {
            panic!("pivot");
        };
        let PlanNode::Project { exprs: kept, .. } = &narrowed.node else {
            panic!("narrow");
        };
        assert!(
            kept.iter()
                .any(|e| output_name(e).as_deref() == Some("__krishiv_key_0"))
        );
    }

    #[test]
    fn an_aliased_key_an_aggregate_reads_is_copied_from_the_inner_expression() {
        let aliased = col("k").alias("kk");
        let node = build(&[aliased], vec![text("a")], vec![col("kk")]).expect("builds");
        assert!(matches!(node, PlanNode::Project { .. }));
    }

    #[test]
    fn a_value_named_like_a_restored_key_is_refused_for_one_aggregate() {
        // keys k, a; `sum(k)` reads key k, so keys are restored by name; value "a" is a key's name.
        refused(
            build(&[col("k"), col("a")], vec![text("a")], vec![col("k")]),
            "named like a grouping column",
        );
    }

    #[test]
    fn several_aggregates_name_outputs_by_value_and_aggregate() {
        let named_agg = |n: &str, c: &str| col(c).alias(n);
        // Spark names the outputs `k_total` and `k_n`: neither is the key `k`, so it builds.
        let r = build(
            &[col("k")],
            vec![text("k")],
            vec![named_agg("total", "k"), named_agg("n", "v")],
        );
        assert!(r.is_ok(), "{r:?}");
    }

    #[test]
    fn a_grouping_column_named_like_an_internal_copy_is_refused() {
        refused(
            build(&[col("__krishiv_pivot")], vec![text("a")], vec![col("v")]),
            "internal pivot copy",
        );
        refused(
            build(&[col("__krishiv_key_0")], vec![text("a")], vec![col("v")]),
            "internal pivot copy",
        );
    }
}

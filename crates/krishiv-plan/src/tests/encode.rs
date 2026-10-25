// SPDX-License-Identifier: Apache-2.0
//! The canonical encoding and the structural hash (owner decisions 3 and 4):
//! relative `plan_id` references, stability across processes, bottom-up
//! subtree keys.

use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_common::secret::OptionValue;
use krishiv_types::DataType;

use super::depth::self_joins;
use crate::{
    Builtin, ColumnRef, Expr, ExprNode, FunctionId, FunctionKind, JoinCondition, JoinKind, Literal,
    LogicalPlan, Origin, Plan, PlanId, PlanNode, ReadSource, ResolvedExpr, ResolvedFunction,
    StructuralHash, SubqueryKind, YearMonthField,
};

// ---- plan_id references ---------------------------------------------------

fn df(id: i64) -> LogicalPlan {
    LogicalPlan::table(["t"]).with_origin(Origin::plan(PlanId(id)))
}

fn df_col(id: i64) -> Expr {
    Expr::new(ExprNode::Column(ColumnRef {
        name: vec!["a".to_owned()],
        plan_id: Some(PlanId(id)),
    }))
}

fn key(p: LogicalPlan) -> (StructuralHash, Vec<u8>) {
    let p = Plan::try_new(p).unwrap();
    (p.structural_hash(), p.canonical_encoding())
}

/// `df1.join(df2, df1["a"] == df2["a"])` with the given plan ids and
/// reference targets.
fn self_join(ids: (i64, i64), refs: (i64, i64)) -> LogicalPlan {
    let cond = Expr::builtin(Builtin::Eq, vec![df_col(refs.0), df_col(refs.1)]);
    df(ids.0).join(
        df(ids.1),
        JoinKind::Inner,
        JoinCondition::On(Arc::new(cond)),
    )
}

#[test]
fn plan_id_references_encode_their_target_not_their_value() {
    let base = key(self_join((1, 2), (1, 2)));
    assert_eq!(key(self_join((7, 3), (7, 3))), base, "renumbered");
    assert_ne!(key(self_join((1, 2), (1, 1))), base, "retargeted");
    assert_ne!(key(self_join((1, 2), (2, 1))), base, "swapped");
    // References found nowhere are told apart by first appearance.
    let unbound = key(self_join((1, 2), (8, 9)));
    assert_eq!(key(self_join((1, 2), (5, 6))), unbound);
    assert_ne!(key(self_join((1, 2), (8, 8))), unbound);
}

/// A reference held by a join to a `plan_id` carried twice in its subtree
/// (`df.join(df)`) differs from one carried once: all carriers are keyed.
#[test]
fn every_carrier_of_a_reference_is_keyed() {
    let at_join = |right_id| {
        let left = df(1).filter(df_col(1));
        let cond = Expr::builtin(Builtin::Eq, vec![df_col(1), df_col(1)]);
        left.join(
            df(right_id),
            JoinKind::Inner,
            JoinCondition::On(Arc::new(cond)),
        )
    };
    assert_ne!(key(at_join(1)), key(at_join(2)));
    assert_eq!(key(at_join(2)), key(at_join(3)), "renumbered");
}

/// A deterministic random stream for [`resolution_follows_the_key`].
struct Stream(u64);

impl Stream {
    fn pick(&mut self, n: u64) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % n
    }
}

/// A plan whose shape comes from `shape` and whose `plan_id`s (carried by
/// relations from a small alphabet, referenced at every level) come from
/// `ids`, so two `ids` streams give two id assignments of one shape.
fn shaped(shape: &mut Stream, ids: &mut Stream, depth: u32) -> LogicalPlan {
    let reference = |ids: &mut Stream| df_col(i64::try_from(1 + ids.pick(3)).unwrap());
    let plan = match if depth == 0 { 0 } else { shape.pick(3) } {
        0 => LogicalPlan::table(["t"]),
        1 => shaped(shape, ids, depth - 1).filter(reference(ids)),
        _ => {
            let left = shaped(shape, ids, depth - 1);
            let right = shaped(shape, ids, depth - 1);
            let cond = Expr::builtin(Builtin::Eq, vec![reference(ids), reference(ids)]);
            left.join(right, JoinKind::Inner, JoinCondition::On(Arc::new(cond)))
        }
    };
    match ids.pick(4) {
        0 => plan,
        id => plan.with_origin(Origin::plan(PlanId(i64::try_from(id).unwrap()))),
    }
}

/// How one reference resolves, as an offset in its operator's subtree.
#[derive(Debug, PartialEq)]
enum Resolution {
    Unresolved,
    At(usize),
    Ambiguous,
}

/// Every reference of `plan`, in order, resolved by the nearest-carrier
/// rule and by the ambiguity rule (an error when several relations carry
/// the `plan_id`).
fn resolve(plan: &LogicalPlan) -> Vec<(Option<usize>, Resolution)> {
    fn subtree<'a>(p: &'a LogicalPlan, out: &mut Vec<&'a LogicalPlan>) {
        out.push(p);
        p.for_each_child(&mut |c| subtree(c, out));
    }
    let mut operators = Vec::new();
    subtree(plan, &mut operators);
    let mut out = Vec::new();
    for op in operators {
        let mut below = Vec::new();
        subtree(op, &mut below);
        let mut refs = Vec::new();
        op.for_each_expr(&mut |e| refs.push(Arc::clone(e)));
        while let Some(e) = refs.pop() {
            if let ExprNode::Column(ColumnRef {
                plan_id: Some(id), ..
            }) = &e.node
            {
                let at: Vec<usize> = below
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.origin.plan_id == Some(*id))
                    .map(|(i, _)| i)
                    .collect();
                let strict = match at.as_slice() {
                    [] => Resolution::Unresolved,
                    [one] => Resolution::At(*one),
                    _ => Resolution::Ambiguous,
                };
                out.push((at.first().copied(), strict));
            }
            e.for_each_child(&mut |c| refs.push(Arc::clone(c)));
        }
    }
    out
}

/// Property: two id assignments of one shape with equal keys resolve every
/// reference alike, under both rules. Checked over 5,000 random cases, of
/// which many collide in key, so the property is not vacuous.
#[test]
fn resolution_follows_the_key() {
    let mut equal_keys = 0;
    for seed in 1..=5_000_u64 {
        let build = |ids: u64| {
            let mut shape = Stream(seed);
            shaped(&mut shape, &mut Stream(ids), 3)
        };
        let (a, b) = (build(seed * 2 + 1), build(seed * 3 + 7));
        if key(a.clone()) == key(b.clone()) {
            equal_keys += 1;
            assert_eq!(resolve(&a), resolve(&b), "{a:?}\n{b:?}");
        }
    }
    assert!(equal_keys > 1_000, "{equal_keys} equal-key pairs");
}

/// The review's I1 case: a reference resolves in the subtree of the
/// operator that holds it, so reusing a `plan_id` elsewhere matters.
#[test]
fn duplicated_plan_ids_do_not_collide() {
    let filtered = |id| df(id).filter(df_col(1));
    let a = df(1).join(filtered(1), JoinKind::Inner, JoinCondition::None);
    let b = df(1).join(filtered(2), JoinKind::Inner, JoinCondition::None);
    assert_ne!(key(a), key(b));
}

#[test]
fn subquery_references_reach_enclosing_operators() {
    let correlated = |outer_id, ref_id| {
        let sub = df(5).filter(df_col(ref_id));
        df(outer_id).filter(Expr::subquery(SubqueryKind::Exists, sub))
    };
    assert_eq!(key(correlated(1, 1)), key(correlated(2, 2)), "renumbered");
    assert_ne!(
        key(correlated(1, 1)),
        key(correlated(1, 5)),
        "inner vs outer"
    );
    assert_ne!(
        key(correlated(1, 1)),
        key(correlated(1, 9)),
        "outer vs unbound"
    );
}

#[test]
fn subtree_keys_are_the_keys_of_the_subtrees() {
    let mut plan = df(0);
    for i in 1..400 {
        plan = plan
            .filter(df_col(i - 1))
            .with_origin(Origin::plan(PlanId(i)));
    }
    let plan = Plan::try_new(plan).unwrap();
    let subtrees = plan.subtree_hashes();
    assert_eq!(subtrees.len(), 400);
    for (node, hash) in subtrees.iter().step_by(37) {
        let alone = Plan::try_new(LogicalPlan::clone(node)).unwrap();
        assert_eq!(alone.structural_hash(), *hash);
    }
    assert_eq!(subtrees[0].1, plan.structural_hash());
}

/// A resolved `name(a)` of `kind`, with registry handle `id`.
fn resolved_call(name: &str, id: u32, kind: FunctionKind, arg: &str) -> LogicalPlan {
    let func = ResolvedFunction {
        id: FunctionId::Registry {
            name: name.to_owned(),
            id,
        },
        kind,
        args: vec![Arc::new(Expr::column([arg]))],
        distinct: false,
        filter: None,
        data_type: DataType::String,
        nullable: true,
        deterministic: true,
    };
    let call = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(func)));
    LogicalPlan::table(["t"]).project(vec![call])
}

/// A function's key is its name, kind and arguments, never its registry
/// handle, so a cache shared across builds cannot confuse two functions.
#[test]
fn resolved_function_keys_ignore_registry_ids() {
    use FunctionKind::{Aggregate, Scalar};
    let base = key(resolved_call("upper", 1, Scalar, "a"));
    assert_eq!(key(resolved_call("upper", 99, Scalar, "a")), base, "id");
    assert_ne!(key(resolved_call("lower", 1, Scalar, "a")), base, "name");
    assert_ne!(key(resolved_call("upper", 1, Aggregate, "a")), base, "kind");
    assert_ne!(key(resolved_call("upper", 1, Scalar, "b")), base, "args");
}

#[test]
fn literals_compare_by_bits() {
    let p = |v| key(LogicalPlan::table(["t"]).filter(Expr::from(Literal::Double(v))));
    assert_ne!(p(0.0), p(-0.0));
    assert_eq!(p(f64::NAN), p(f64::NAN));
}

#[test]
fn interval_qualifiers_are_part_of_the_key() {
    let ym = |start, end| {
        key(
            LogicalPlan::table(["t"]).filter(Expr::from(Literal::YearMonthInterval {
                months: 12,
                start,
                end,
            })),
        )
    };
    assert_ne!(
        ym(YearMonthField::Year, YearMonthField::Year),
        ym(YearMonthField::Month, YearMonthField::Month)
    );
}

/// The key is a pure function of the encoding layout: these values change
/// only with [`crate::ENCODING_VERSION`].
#[test]
fn keys_are_stable() {
    let plan = self_join((1, 2), (1, 2))
        .filter(Expr::builtin(
            Builtin::Gt,
            vec![Expr::column(["a"]), Literal::Integer(1).into()],
        ))
        .project(vec![Expr::column(["a"]).alias("b")]);
    let (hash, encoding) = key(plan);
    assert_eq!(encoding.len(), 461, "{encoding:?}");
    assert_eq!(hash.to_string(), "c8beb826b28467220b04a0c87ce3719b");
    // Options other than inline secrets, whose keys are per process.
    let options = [
        ("header", OptionValue::Plain("true".to_owned())),
        ("password", OptionValue::reference("prod", "pg")),
    ];
    let read = LogicalPlan::new(PlanNode::Read(ReadSource::Table {
        name: vec!["t".to_owned()],
        options: options.map(|(k, v)| (k.to_owned(), v)).into(),
    }));
    let (hash, encoding) = key(read);
    assert_eq!(encoding.len(), 124, "{encoding:?}");
    assert_eq!(hash.to_string(), "ba4505febd5716e05d133affe7ea86c9");
}

/// A plan whose subtrees are shared (`p = p.join(p)` 30 times) is keyed in
/// time linear in its distinct nodes, and keys like the same plan built as
/// a tree.
#[test]
fn shared_subtrees_are_encoded_once() {
    let start = Instant::now();
    let dag = Plan::try_new(self_joins(30)).unwrap();
    let again = Plan::try_new(self_joins(30)).unwrap();
    assert!(dag.structurally_eq(&again));
    assert_eq!(dag.structural_hash(), again.structural_hash());
    assert_eq!(dag.subtree_hashes().len(), 31);
    assert!(dag.canonical_encoding().len() < 4096);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );

    /// The same shape with no sharing.
    fn tree(n: usize) -> LogicalPlan {
        if n == 0 {
            return LogicalPlan::table(["t"]);
        }
        tree(n - 1).join(tree(n - 1), JoinKind::Inner, JoinCondition::None)
    }
    let (shared, unshared) = (
        Plan::try_new(self_joins(5)).unwrap(),
        Plan::try_new(tree(5)).unwrap(),
    );
    assert!(shared.structurally_eq(&unshared));
    assert_eq!(shared.structural_hash(), unshared.structural_hash());
    assert!(!shared.structurally_eq(&Plan::try_new(tree(4)).unwrap()));
}

/// A shared expression (`e = e and e` 30 times) is encoded once.
#[test]
fn shared_expressions_are_encoded_once() {
    let start = Instant::now();
    let e = (0..30).fold(Arc::new(Expr::column(["a"])), |e, _| {
        Arc::new(Expr::new(ExprNode::Function(crate::FunctionCall {
            name: Builtin::And.name().to_owned(),
            args: vec![Arc::clone(&e), e],
            distinct: false,
            filter: None,
        })))
    });
    let plan = Plan::try_new(LogicalPlan::table(["t"]).filter(e)).unwrap();
    let _ = plan.structural_hash();
    assert!(plan.canonical_encoding().len() < 4096);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
}

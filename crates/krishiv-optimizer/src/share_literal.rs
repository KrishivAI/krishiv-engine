// SPDX-License-Identifier: Apache-2.0
//! `ShareLiteralInstances`: instances of one CTE that differ only in the
//! literal a group key is compared with share one computation over all the
//! literals (perf-1 lever A, TPC-DS q31, q75).
//!
//! ```text
//! ... Shift(B[g = 1]) ... Shift(B[g = 2]) ... Shift(B[g = 3]) ...
//!   =>  D := B[g IN (1, 2, 3)], g kept in D's output
//!       ... Shift(Project(Filter g = 1 (D))) ... Shift(Project(Filter g = 2 (D))) ...
//! ```
//!
//! TPC-DS q31 reads the CTE `ss` three times, with `d_qoy = 1`, `2` and `3`
//! on its group key. Pushdown moves each literal below the aggregate into
//! the `date_dim` branch, so the three bodies differ deep inside and
//! `ShareCte` (equal bodies, or bodies that differ only above an equal core)
//! shares none of them: the facts are scanned and aggregated three times.
//! Here the conjunct `g' = k` on the group key's lineage becomes
//! `g' IN (k_1 .. k_n)` in one definition `D`, which keeps the key in its
//! output, and each instance reads `D` through its own `g = k_i` above it;
//! `ShareCte`, which runs right after, then shares `D` through its core path
//! (each instance keeps its `Shift` and the filter and projection over a
//! `CteRef` of `D`).
//!
//! # Why it is exact
//!
//! The lineage of `g` from the top of the body down to the conjunct passes
//! only nodes a filter on `g` commutes with: projections that pass `g`
//! through (or rename it), filters, a grouped aggregate with `g` a bare
//! group key, `DISTINCT`, positional `UNION`s (every branch carries the
//! conjunct), inner and cross joins, the preserved side of an outer join,
//! the left side of a semi, anti, existence or single join, and `Shift`s.
//! So `Filter(g = k_i)(D)` keeps exactly the rows instance `i` computes.
//!
//! No new error: below the core aggregate the instances are equal apart from
//! the literal, so every row `D`'s input evaluates an expression on is a row
//! some instance evaluated that expression on (the definition computes the
//! union of the instances' rows at each node), and `g = k_i` over a column
//! cannot raise. The aggregate list is the exception: `D` computes every
//! instance's items over every literal's groups, so an item only some
//! instances output must not raise on the others' groups. Such an item is a
//! column or a built-in `count`, `min` or `max` whose arguments run on any row
//! (`runs_on_other_groups`); a `sum`, `avg` or anything else that only some
//! instances read keeps the group unshared (an ANSI `sum` could overflow on
//! another literal's rows).
//!
//! # When it applies
//!
//! Two or more instances (`Shift`s, and with `direct` the places that read
//! a shifted body without one, as `ShareCte` counts them) whose bodies' cores
//! are an aggregate grouped by `g` (or a `DISTINCT` / positional `UNION`
//! with `g` in its output) and are equal once the conjunct is set aside
//! (an aggregate's output lists may differ by items that cannot raise: `D`
//! outputs their union), with
//! at least two distinct literals. Literals equal in two instances are one
//! `IN` element; instances that are equal outright are `ShareCte`'s.
//!
//! Applied only when `ShareCte` then shares every rewritten instance;
//! otherwise the plan is left as it was. The rule runs `ShareCte` itself and
//! names its definitions `__shared_lit_<n>`, which the physical planner
//! always reads from one shared plan: where a runtime filter reaches a
//! reference, a copy per reference would compute every literal's groups.
//! Runs once, right before `ShareCte`,
//! only with a session configuration, `krishiv.sql.optimizer.shareCteInstances`
//! and `krishiv.optimizer.shareLiteralInstances` on. The rewritten instances'
//! cores are `CteRef`s, so a second run finds nothing: the plan stays a
//! fixed point.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Grouping, JoinKind,
    Literal, LogicalPlan, Origin, PlanNode, ResolvedExpr, normalized_subtree_hash, runs_on_any_row,
};
use krishiv_types::DataType;

use crate::expr::{
    and_all, builtin, call, column, conjuncts, has_outer_reference, lit, output_of, recall,
    strip_alias,
};
use crate::share_cte::{self, Member, Sharer, core, shifts, worth_sharing};

/// `root` with its groups of instances that differ only in a group key's
/// literal rewritten to read one definition, then shared by `ShareCte`;
/// `root` itself when there are none, or when `ShareCte` would not share
/// every rewritten instance.
pub(crate) fn share(
    root: Arc<LogicalPlan>,
    direct: bool,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let members = shifts(&root, direct);
    if members.len() < 2 {
        return Ok(root);
    }
    let candidates: Vec<Candidate> = members
        .iter()
        .enumerate()
        .flat_map(|(i, m)| candidates(i, m))
        .collect();
    let groups = groups(&members, &candidates);
    if groups.is_empty() {
        return Ok(root);
    }
    let mut replace: crate::expr::AddrMap<LogicalPlan, Arc<LogicalPlan>> = Default::default();
    // Each definition's output ids, and the definition.
    let mut expected: Vec<(Vec<AttrId>, Arc<LogicalPlan>)> = Vec::new();
    for group in &groups {
        let Some((def, readers)) = rewrite_group(&members, &candidates, group) else {
            continue;
        };
        let Some(out) = def.resolved_output() else {
            continue;
        };
        expected.push((out.iter().map(|c| c.id).collect(), Arc::clone(&def)));
        for (ptr, plan) in readers {
            replace.insert(ptr, plan);
        }
    }
    if replace.is_empty() {
        return Ok(root);
    }
    let mut sharer = Sharer {
        replace: &replace,
        done: Default::default(),
    };
    let rewritten = sharer.go(Arc::clone(&root))?;
    // The rewrite pays only where `ShareCte` shares the definition with
    // every instance that now reads it: otherwise each would compute the
    // definition over all the literals. `ShareCte` replaces the definition
    // (one `Arc`) wherever an instance reads it, so it is shared when it is
    // referenced and no longer computed in the main query.
    let shared = share_cte::share(Arc::clone(&rewritten), direct)?;
    if !all_shared(&expected, &shared) {
        return Ok(root);
    }
    // Named apart, so the physical planner shares them even where a runtime
    // filter reaches a reference: a copy per reference would compute every
    // literal's groups.
    let names: HashMap<Vec<AttrId>, ()> = expected.into_iter().map(|(ids, _)| (ids, ())).collect();
    let mut renamer = Renamer {
        defs: &names,
        taken: cte_names(&shared),
        names: HashMap::new(),
        done: HashMap::new(),
    };
    renamer.go(shared)
}

/// Whether `ShareCte` shared every definition in `expected` (its output ids,
/// the definition) in `shared`: each is referenced and no longer computed in
/// the main query.
fn all_shared(expected: &[(Vec<AttrId>, Arc<LogicalPlan>)], shared: &Arc<LogicalPlan>) -> bool {
    let refs = cte_refs(shared);
    let main = match &shared.node {
        PlanNode::WithCte { input, .. } => input,
        _ => shared,
    };
    expected
        .iter()
        .all(|(ids, def)| refs.get(ids).copied().unwrap_or(0) >= 1 && !reaches(main, def))
}

/// The name prefix of the definitions this rule adds: the physical planner
/// always reads them from one shared plan.
pub(crate) const LITERAL_PREFIX: &str = "__shared_lit_";

/// Renames the definitions (by their source ids) and their references to
/// `__shared_lit_<n>`.
struct Renamer<'a> {
    defs: &'a HashMap<Vec<AttrId>, ()>,
    /// The names the plan's `WITH`s and references already use: a user CTE
    /// named `__shared_lit_0` keeps its name and the new ones skip it.
    taken: HashSet<String>,
    /// Old name to new name.
    names: HashMap<String, String>,
    done: HashMap<*const LogicalPlan, (Arc<LogicalPlan>, Arc<LogicalPlan>)>,
}

impl Renamer<'_> {
    fn name(&mut self, old: &str) -> String {
        if let Some(name) = self.names.get(old) {
            return name.clone();
        }
        let mut n = self.names.len();
        let name = loop {
            let name = format!("{LITERAL_PREFIX}{n}");
            if self.taken.insert(name.clone()) {
                break name;
            }
            n += 1;
        };
        self.names.insert(old.to_owned(), name.clone());
        name
    }

    fn go(&mut self, plan: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
        let key = Arc::as_ptr(&plan);
        if let Some((_, out)) = self.done.get(&key) {
            return Ok(Arc::clone(out));
        }
        let rebuilt = Arc::clone(&plan)
            .map_children(|c| self.go(c))?
            .map_expressions(|e| self.subqueries(e))?;
        let renamed = match &rebuilt.node {
            PlanNode::CteRef(r) if self.defs.contains_key(&r.source) => {
                let name = self.name(&r.name);
                Arc::new(
                    LogicalPlan::new(PlanNode::CteRef(krishiv_plan::CteRef { name, ..r.clone() }))
                        .with_origin(rebuilt.origin),
                )
            }
            PlanNode::WithCte { input, ctes } => {
                let mut changed = false;
                let ctes = ctes
                    .iter()
                    .map(|c| {
                        let ids: Option<Vec<AttrId>> = c
                            .plan
                            .resolved_output()
                            .map(|o| o.iter().map(|x| x.id).collect());
                        match ids {
                            Some(ids) if self.defs.contains_key(&ids) => {
                                changed = true;
                                krishiv_plan::Cte {
                                    name: self.name(&c.name),
                                    plan: Arc::clone(&c.plan),
                                }
                            }
                            _ => c.clone(),
                        }
                    })
                    .collect();
                if changed {
                    Arc::new(
                        LogicalPlan::new(PlanNode::WithCte {
                            input: Arc::clone(input),
                            ctes,
                        })
                        .with_origin(rebuilt.origin),
                    )
                } else {
                    rebuilt
                }
            }
            _ => rebuilt,
        };
        self.done.insert(key, (plan, Arc::clone(&renamed)));
        Ok(renamed)
    }

    fn subqueries(&mut self, e: Arc<Expr>) -> Result<Arc<Expr>, KrishivError> {
        if !crate::expr::has_subquery(&e) {
            return Ok(e);
        }
        e.transform_up(&mut |x| match &x.node {
            ExprNode::Subquery { kind, plan } => {
                let rewritten = self.go(Arc::clone(plan))?;
                if Arc::ptr_eq(&rewritten, plan) {
                    return Ok(x);
                }
                Ok(Arc::new(Expr {
                    node: ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: rewritten,
                    },
                    origin: x.origin,
                }))
            }
            _ => Ok(x),
        })
    }
}

/// One way an instance may join a group: the group key `g` (at the core's
/// output, or the aggregate's input), its literal, and the core with the
/// conjunct set aside (`key`).
struct Candidate {
    member: usize,
    g: BoundColumn,
    literal: Literal,
    /// The conjunct `g' = literal` as written (its call, for `g = k` above).
    conjunct: Arc<Expr>,
    key: Arc<LogicalPlan>,
}

/// The candidates of member `i`: one per group key whose lineage holds a
/// literal conjunct.
fn candidates(i: usize, m: &Member) -> Vec<Candidate> {
    let top = core(&m.body);
    let keys: Vec<BoundColumn> = match &top.node {
        PlanNode::Aggregate {
            grouping: Grouping::GroupBy(keys),
            ..
        } => keys
            .iter()
            .filter_map(|k| match &k.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.clone()),
                _ => None,
            })
            .collect(),
        PlanNode::Distinct { .. } | PlanNode::Union { by_name: false, .. } => {
            top.resolved_output().unwrap_or_default()
        }
        _ => return Vec::new(),
    };
    let mut out = Vec::new();
    for g in keys {
        if g.data_type.is_nested() {
            continue;
        }
        let mut lift = Lift::new(Mode::Mark);
        let Some(key) = lift.core(top, g.id) else {
            continue;
        };
        // Instances' conjuncts were pushed down from different places of the
        // query, so their origins differ: keys compare without them.
        let key = canonical_states(&Unorigin::default().plan(&key)).0;
        let Some((literal, conjunct)) = lift.single() else {
            continue;
        };
        out.push(Candidate {
            member: i,
            g,
            literal,
            conjunct,
            key,
        });
    }
    out
}

/// Groups of candidates (indices) with one key column and equal keys, each
/// member in one group at most, each with two or more distinct literals.
fn groups(members: &[Member], candidates: &[Candidate]) -> Vec<Vec<usize>> {
    let mut buckets: HashMap<(AttrId, u128), Vec<Vec<usize>>> = HashMap::new();
    for (i, c) in candidates.iter().enumerate() {
        let groups = buckets
            .entry((c.g.id, normalized_subtree_hash(&c.key)))
            .or_default();
        match groups
            .iter_mut()
            .find(|g| equal(&candidates[g[0]].key, &c.key))
        {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    let mut all: Vec<Vec<usize>> = buckets.into_values().flatten().collect();
    // A stable order: by first candidate.
    all.sort_by_key(|g| g[0]);
    let mut taken: HashSet<usize> = HashSet::new();
    let mut out = Vec::new();
    for g in all {
        let g: Vec<usize> = g
            .into_iter()
            .filter(|&c| !taken.contains(&candidates[c].member))
            .collect();
        let mut literals: Vec<&Literal> = Vec::new();
        for &c in &g {
            if !literals.contains(&&candidates[c].literal) {
                literals.push(&candidates[c].literal);
            }
        }
        let uses: u64 = g.iter().map(|&c| members[candidates[c].member].uses).sum();
        if literals.len() < 2 || uses < 2 {
            continue;
        }
        taken.extend(g.iter().map(|&c| candidates[c].member));
        out.push(g);
    }
    out
}

/// The definition of `group` and, per instance, the pointer to replace and
/// its rewritten instance; `None` when the definition is not worth sharing.
#[expect(
    clippy::type_complexity,
    reason = "a definition and its (pointer, replacement) pairs"
)]
fn rewrite_group(
    members: &[Member],
    candidates: &[Candidate],
    group: &[usize],
) -> Option<(
    Arc<LogicalPlan>,
    Vec<(*const LogicalPlan, Arc<LogicalPlan>)>,
)> {
    let first = &candidates[group[0]];
    let mut literals: Vec<Literal> = Vec::new();
    for &c in group {
        if !literals.contains(&candidates[c].literal) {
            literals.push(candidates[c].literal.clone());
        }
    }
    let top = core(&members[first.member].body);
    let mut lift = Lift::new(Mode::In(&literals));
    let lifted = lift.core(top, first.g.id)?;
    let def = match &lifted.node {
        PlanNode::Aggregate {
            input,
            grouping,
            aggregates,
        } => {
            // An item only some instances output runs on the other literals'
            // groups too, which those instances never computed: only an item
            // that cannot raise may (`runs_on_other_groups`); otherwise the
            // lists must be equal. Items match by what they compute
            // (`item_key`), not by output id: where a projection was
            // collapsed into one instance's aggregate, its outputs carry the
            // projection's ids (TPC-DS q04, q11, q74 once a dropped first
            // union input left an aliasing projection).
            // Each instance's state ids (eager aggregation's, fresh per
            // instance) under the canonical names its comparison key used.
            let states: HashMap<usize, HashMap<AttrId, AttrId>> = group
                .iter()
                .map(|&c| {
                    (
                        c,
                        canonical_states(core(&members[candidates[c].member].body)).1,
                    )
                })
                .collect();
            let none = HashMap::new();
            let states_of = |c: usize| states.get(&c).unwrap_or(&none);
            let lists: Vec<Vec<Arc<Expr>>> = group
                .iter()
                .filter_map(|&c| match &core(&members[candidates[c].member].body).node {
                    PlanNode::Aggregate { aggregates, .. } => Some(
                        aggregates
                            .iter()
                            .map(|e| item_key(e, states_of(c)))
                            .collect(),
                    ),
                    _ => None,
                })
                .collect();
            for &c in group {
                if let PlanNode::Aggregate { aggregates, .. } =
                    &core(&members[candidates[c].member].body).node
                {
                    for e in aggregates {
                        let k = item_key(e, states_of(c));
                        let everywhere = lists.iter().all(|l| l.contains(&k));
                        if !everywhere && !runs_on_other_groups(e) {
                            return None;
                        }
                    }
                }
            }
            // The union of the instances' outputs, then the key.
            let mut items: Vec<Arc<Expr>> = aggregates.clone();
            let first_states = states_of(group[0]);
            let mut keys: Vec<Arc<Expr>> =
                items.iter().map(|e| item_key(e, first_states)).collect();
            let mut ids: HashSet<AttrId> = items
                .iter()
                .filter_map(|e| output_of(e).map(|c| c.id))
                .collect();
            for &c in group {
                if let PlanNode::Aggregate { aggregates, .. } =
                    &core(&members[candidates[c].member].body).node
                {
                    for e in aggregates {
                        let k = item_key(e, states_of(c));
                        if keys.contains(&k) {
                            continue;
                        }
                        // An item of this instance alone that reads its own
                        // state columns: the definition does not have them.
                        if reads_any(e, states_of(c)) {
                            return None;
                        }
                        // A new computation under an id the definition
                        // already outputs: the ids would clash.
                        let o = output_of(e)?;
                        if !ids.insert(o.id) {
                            return None;
                        }
                        keys.push(k);
                        items.push(Arc::clone(e));
                    }
                }
            }
            if ids.insert(first.g.id) {
                items.push(column(&first.g));
            }
            Arc::new(
                LogicalPlan::new(PlanNode::Aggregate {
                    input: Arc::clone(input),
                    grouping: grouping.clone(),
                    aggregates: items,
                })
                .with_origin(lifted.origin),
            )
        }
        _ => lifted,
    };
    if !worth_sharing(&def) {
        return None;
    }
    let def_out = def.resolved_output()?;
    let mut readers = Vec::with_capacity(group.len());
    for &c in group {
        let cand = &candidates[c];
        let m = &members[cand.member];
        let top = core(&m.body);
        let out = top.resolved_output()?;
        let (Builtin::Eq, _, f) = builtin(&cand.conjunct)? else {
            return None;
        };
        let pred = recall(
            f,
            Builtin::Eq,
            vec![column(&cand.g), lit(cand.literal.clone())],
        );
        let filtered = Arc::new(LogicalPlan::new(PlanNode::Filter {
            input: Arc::clone(&def),
            predicate: pred,
        }));
        let same_output =
            out.len() == def_out.len() && out.iter().zip(&def_out).all(|(a, b)| a.id == b.id);
        let new_top = if same_output {
            filtered
        } else {
            Arc::new(LogicalPlan::new(PlanNode::Project {
                input: filtered,
                exprs: reader_exprs(top, &out, &def, &def_out)?,
            }))
        };
        let body = replace_core(&m.body, top, &new_top)?;
        let reader = if Arc::ptr_eq(&m.shift, &m.body) {
            body
        } else {
            Arc::new(
                LogicalPlan::new(PlanNode::Shift {
                    input: body,
                    offset: m.offset,
                })
                .with_origin(m.shift.origin),
            )
        };
        readers.push((Arc::as_ptr(&m.shift), reader));
    }
    Some((def, readers))
}

/// The first id [`canonical_states`] gives a state column: above any id a
/// resolved plan holds.
const CANONICAL_STATE_BASE: u64 = 1 << 60;

/// `plan` with the state columns of its partial aggregates renumbered in
/// walk order (children first) and every reference to them renamed, for
/// comparing instances only. Eager aggregation (K5) gives each instance's
/// `PartialAggregate` fresh state ids, so two instances equal apart from the
/// literal differ in those ids alone (TPC-DS q04, q11, q74, the store
/// channel). A comparison up to the renaming of ids a core defines, as
/// DuckDB's `CommonSubplanOptimizer` compares canonical plan signatures.
fn canonical_states(plan: &Arc<LogicalPlan>) -> (Arc<LogicalPlan>, HashMap<AttrId, AttrId>) {
    let mut map: HashMap<AttrId, AttrId> = HashMap::new();
    let mut next = CANONICAL_STATE_BASE;
    let out = Arc::clone(plan)
        .transform_up(&mut |p| {
            let p = if map.is_empty() {
                p
            } else {
                p.map_expressions(|e| Ok::<_, ()>(renamed(&e, &map)))?
            };
            if let PlanNode::PartialAggregate {
                input,
                keys,
                aggregates,
                states,
            } = &p.node
            {
                let states = states
                    .iter()
                    .map(|c| {
                        let id = *map.entry(c.id).or_insert_with(|| {
                            let id = AttrId(next);
                            next += 1;
                            id
                        });
                        BoundColumn { id, ..c.clone() }
                    })
                    .collect();
                return Ok(Arc::new(
                    LogicalPlan::new(PlanNode::PartialAggregate {
                        input: Arc::clone(input),
                        keys: keys.clone(),
                        aggregates: aggregates.clone(),
                        states,
                    })
                    .with_origin(p.origin),
                ));
            }
            Ok(p)
        })
        .unwrap_or_else(|()| Arc::clone(plan));
    (out, map)
}

/// Whether `e` reads a column whose id `ids` holds.
fn reads_any(e: &Arc<Expr>, ids: &HashMap<AttrId, AttrId>) -> bool {
    if ids.is_empty() {
        return false;
    }
    let mut found = false;
    let _ = Arc::clone(e).transform_up(&mut |x| {
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node {
            found |= ids.contains_key(&c.id);
        }
        Ok::<_, std::convert::Infallible>(x)
    });
    found
}

/// `e` with each column whose id `map` holds read under the mapped id.
fn renamed(e: &Arc<Expr>, map: &HashMap<AttrId, AttrId>) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |x| {
        Ok::<_, std::convert::Infallible>(match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => match map.get(&c.id) {
                Some(id) => Arc::new(Expr {
                    node: ExprNode::Resolved(ResolvedExpr::Column(BoundColumn {
                        id: *id,
                        ..c.clone()
                    })),
                    origin: x.origin,
                }),
                None => x,
            },
            _ => x,
        })
    });
    match out {
        Ok(e) => e,
        Err(never) => match never {},
    }
}

/// What an aggregate output item computes, for matching items across
/// instances: the item without its alias and without origins.
fn item_key(e: &Arc<Expr>, states: &HashMap<AttrId, AttrId>) -> Arc<Expr> {
    let e = expr_unorigin(strip_alias(e));
    if states.is_empty() {
        e
    } else {
        renamed(&e, states)
    }
}

/// The projection that gives an instance its own output over the
/// definition: a column the definition outputs under the same id is read as
/// it is; an aggregate item the definition computes under another id is read
/// from that column under the instance's id. `None` when an output is
/// neither.
fn reader_exprs(
    top: &Arc<LogicalPlan>,
    out: &[BoundColumn],
    def: &Arc<LogicalPlan>,
    def_out: &[BoundColumn],
) -> Option<Vec<Arc<Expr>>> {
    let (own, theirs) = match (&top.node, &def.node) {
        (PlanNode::Aggregate { aggregates: a, .. }, PlanNode::Aggregate { aggregates: d, .. }) => {
            (a.as_slice(), d.as_slice())
        }
        _ => (&[][..], &[][..]),
    };
    let own_states = canonical_states(top).1;
    let def_states = canonical_states(def).1;
    out.iter()
        .map(|o| {
            if def_out.iter().any(|d| d.id == o.id) {
                return Some(column(o));
            }
            let mine = own
                .iter()
                .find(|e| output_of(e).is_some_and(|c| c.id == o.id))?;
            let k = item_key(mine, &own_states);
            let source = theirs.iter().find(|e| item_key(e, &def_states) == k)?;
            let from = output_of(source)?;
            Some(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: column(from),
                    output: o.clone(),
                },
            ))))
        })
        .collect()
}

/// Whether an aggregate output item can be computed on groups its instance
/// did not compute without a new error: a column, or `count`, `min` or `max`
/// (the built-ins, by identity) whose arguments and `FILTER` run on any row.
/// `sum` and `avg` can overflow (ANSI) on another literal's rows.
fn runs_on_other_groups(e: &Arc<Expr>) -> bool {
    match &strip_alias(e).node {
        ExprNode::Resolved(ResolvedExpr::Column(_)) => true,
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            f.kind == FunctionKind::Aggregate
                && f.deterministic
                && matches!(f.id, FunctionId::Registry { .. })
                && matches!(f.id.name(), "count" | "min" | "max")
                && f.args
                    .iter()
                    .chain(f.filter.iter())
                    .all(|a| runs_on_any_row(a, true) && !has_outer_reference(a))
        }
        _ => false,
    }
}

/// `body` with its core `top` (below the filters and projections at its
/// top) replaced by `new_top`.
fn replace_core(
    body: &Arc<LogicalPlan>,
    top: &Arc<LogicalPlan>,
    new_top: &Arc<LogicalPlan>,
) -> Option<Arc<LogicalPlan>> {
    if Arc::ptr_eq(body, top) {
        return Some(Arc::clone(new_top));
    }
    match &body.node {
        PlanNode::Project { input, .. } | PlanNode::Filter { input, .. } => {
            let inner = replace_core(input, top, new_top)?;
            Arc::clone(body)
                .map_children(|_| Ok::<_, ()>(Arc::clone(&inner)))
                .ok()
        }
        _ => None,
    }
}

/// The number of `CteRef`s of each definition (by its source ids) in `plan`,
/// definitions included.
fn cte_refs(plan: &Arc<LogicalPlan>) -> HashMap<Vec<AttrId>, usize> {
    let mut out: HashMap<Vec<AttrId>, usize> = HashMap::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(plan)];
    while let Some(p) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        if let PlanNode::CteRef(r) = &p.node {
            *out.entry(r.source.clone()).or_default() += 1;
        }
        if let PlanNode::WithCte { ctes, .. } = &p.node {
            stack.extend(ctes.iter().map(|c| Arc::clone(&c.plan)));
        }
        p.for_each_expr(&mut |e| stack.extend(crate::restrict::subquery_plans(e)));
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    out
}

/// Every name a `WITH` definition or a `CteRef` in `plan` (and its subquery
/// plans) uses.
fn cte_names(plan: &Arc<LogicalPlan>) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(plan)];
    while let Some(p) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        match &p.node {
            PlanNode::CteRef(r) => {
                out.insert(r.name.clone());
            }
            PlanNode::WithCte { ctes, .. } => {
                for c in ctes {
                    out.insert(c.name.clone());
                    stack.push(Arc::clone(&c.plan));
                }
            }
            _ => {}
        }
        p.for_each_expr(&mut |e| stack.extend(crate::restrict::subquery_plans(e)));
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    out
}

/// A relation with every origin cleared (its expressions' too; subquery
/// plans are left as they are, which only makes keys compare unequal).
#[derive(Default)]
struct Unorigin {
    done: HashMap<*const LogicalPlan, Arc<LogicalPlan>>,
}

impl Unorigin {
    fn plan(&mut self, p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        if let Some(r) = self.done.get(&Arc::as_ptr(p)) {
            return Arc::clone(r);
        }
        let rebuilt = Arc::clone(p)
            .map_children(|c| Ok::<_, ()>(self.plan(&c)))
            .and_then(|r| r.map_expressions(|e| Ok(expr_unorigin(&e))))
            .unwrap_or_else(|()| Arc::clone(p));
        let out = if rebuilt.origin == Origin::UNKNOWN {
            rebuilt
        } else {
            Arc::new(LogicalPlan {
                node: rebuilt.node.clone(),
                origin: Origin::UNKNOWN,
            })
        };
        self.done.insert(Arc::as_ptr(p), Arc::clone(&out));
        out
    }
}

/// `e` with every origin cleared.
fn expr_unorigin(e: &Arc<Expr>) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |x| {
        Ok::<_, std::convert::Infallible>(if x.origin == Origin::UNKNOWN {
            x
        } else {
            Arc::new(Expr {
                node: x.node.clone(),
                origin: Origin::UNKNOWN,
            })
        })
    });
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

/// Whether `plan` (subquery plans included) holds the relation `target`.
fn reaches(plan: &Arc<LogicalPlan>, target: &Arc<LogicalPlan>) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(plan)];
    while let Some(p) = stack.pop() {
        if Arc::ptr_eq(&p, target) {
            return true;
        }
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        p.for_each_expr(&mut |e| stack.extend(crate::restrict::subquery_plans(e)));
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    false
}

fn equal(a: &Arc<LogicalPlan>, b: &Arc<LogicalPlan>) -> bool {
    Arc::ptr_eq(a, b) || **a == **b
}

/// What a literal conjunct on the key becomes.
enum Mode<'a> {
    /// A marker, for comparing instances: `in(g')` with no list.
    Mark,
    /// `g' IN (literals)`.
    In(&'a [Literal]),
    /// The reverse, for estimates: a definition's `g' IN (.., k, ..)` back to the instance's
    /// `g' = k` ([`instance_of`]).
    Narrow(&'a Literal),
}

/// The walk down a group key's lineage.
struct Lift<'a> {
    mode: Mode<'a>,
    /// The conjuncts found (each `g' = literal`).
    found: Vec<(Literal, Arc<Expr>)>,
    /// Rewritten relations by (relation, key, found above).
    memo: HashMap<(*const LogicalPlan, AttrId, bool), Option<Arc<LogicalPlan>>>,
}

impl<'a> Lift<'a> {
    fn new(mode: Mode<'a>) -> Self {
        Self {
            mode,
            found: Vec::new(),
            memo: HashMap::new(),
        }
    }

    /// The one literal every conjunct found compares with, and a conjunct.
    fn single(&self) -> Option<(Literal, Arc<Expr>)> {
        let (first, conjunct) = self.found.first()?;
        self.found
            .iter()
            .all(|(l, _)| l == first)
            .then(|| (first.clone(), Arc::clone(conjunct)))
    }

    /// The core `top` with the conjunct on key `g` rewritten: below an
    /// aggregate grouped by `g` (its output lists set aside in `Mark` mode),
    /// or through a `DISTINCT` / `UNION` with `g` in its output.
    fn core(&mut self, top: &Arc<LogicalPlan>, g: AttrId) -> Option<Arc<LogicalPlan>> {
        match &top.node {
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => {
                let input = self.walk(input, g, false)?;
                let aggregates = match self.mode {
                    // Instances whose aggregates output different columns
                    // still share: the definition outputs their union.
                    Mode::Mark => Vec::new(),
                    Mode::In(_) | Mode::Narrow(_) => aggregates.clone(),
                };
                Some(Arc::new(
                    LogicalPlan::new(PlanNode::Aggregate {
                        input,
                        grouping: grouping.clone(),
                        aggregates,
                    })
                    .with_origin(top.origin),
                ))
            }
            _ => self.walk(top, g, false),
        }
    }

    /// `plan` (which outputs `g`) with each conjunct `g' = literal` on the
    /// lineage of `g` rewritten; `None` when the lineage passes a node a
    /// filter on `g` does not commute with before a conjunct was found
    /// (`found`), or ends without one.
    fn walk(
        &mut self,
        plan: &Arc<LogicalPlan>,
        g: AttrId,
        found: bool,
    ) -> Option<Arc<LogicalPlan>> {
        let key = (Arc::as_ptr(plan), g, found);
        if let Some(r) = self.memo.get(&key) {
            return r.clone();
        }
        let r = self.walk_node(plan, g, found);
        self.memo.insert(key, r.clone());
        r
    }

    fn walk_node(
        &mut self,
        plan: &Arc<LogicalPlan>,
        g: AttrId,
        found: bool,
    ) -> Option<Arc<LogicalPlan>> {
        // Past the conjunct, a node the walk cannot follow stays as it is.
        let stop = || found.then(|| Arc::clone(plan));
        match &plan.node {
            PlanNode::Filter { input, predicate } => {
                let mut here = false;
                let parts: Vec<Arc<Expr>> = conjuncts(predicate)
                    .into_iter()
                    .map(|c| match self.key_conjunct(&c, g) {
                        Some((col, l)) => {
                            here = true;
                            self.found.push((l, Arc::clone(&c)));
                            self.rewrite(&c, &col)
                        }
                        None => c,
                    })
                    .collect();
                let input = self.walk(input, g, found || here)?;
                if !here {
                    return Some(with_child(plan, &input));
                }
                let predicate = and_all(parts)?;
                let rebuilt = with_child(plan, &input);
                Arc::clone(&rebuilt)
                    .map_expressions(|_| Ok::<_, ()>(Arc::clone(&predicate)))
                    .ok()
            }
            PlanNode::Project { input, exprs } => {
                let Some(e) = exprs
                    .iter()
                    .find(|e| output_of(e).is_some_and(|c| c.id == g))
                else {
                    return stop();
                };
                let ExprNode::Resolved(ResolvedExpr::Column(c)) = &strip_alias(e).node else {
                    return stop();
                };
                let input = self.walk(input, c.id, found)?;
                Some(with_child(plan, &input))
            }
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(keys),
                aggregates,
            } if is_bare(keys, g) && is_bare(aggregates, g) => {
                let input = self.walk(input, g, found)?;
                Some(with_child(plan, &input))
            }
            PlanNode::Distinct { input } => {
                let input = self.walk(input, g, found)?;
                Some(with_child(plan, &input))
            }
            // Eager aggregation's partial phase (K5) grouped by `g`: its keys
            // are input columns that keep their ids, so a filter on `g`
            // commutes with it as with a grouped aggregate.
            PlanNode::PartialAggregate { input, keys, .. } if is_bare(keys, g) => {
                let input = self.walk(input, g, found)?;
                Some(with_child(plan, &input))
            }
            PlanNode::Shift { input, offset } => {
                let Some(inner) = g.0.checked_sub(*offset) else {
                    return stop();
                };
                let input = self.walk(input, AttrId(inner), found)?;
                Some(with_child(plan, &input))
            }
            PlanNode::Union {
                inputs,
                by_name: false,
                ..
            } => {
                let Some(pos) = plan
                    .resolved_output()
                    .and_then(|o| o.iter().position(|c| c.id == g))
                else {
                    return stop();
                };
                let mut out = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let id = input.resolved_output()?.get(pos)?.id;
                    out.push(self.walk(input, id, found)?);
                }
                let mut it = out.into_iter();
                Arc::clone(plan)
                    .map_children(|c| Ok::<_, ()>(it.next().unwrap_or(c)))
                    .ok()
            }
            PlanNode::Join {
                left, right, kind, ..
            } => {
                if Arc::ptr_eq(left, right) {
                    return stop();
                }
                let in_left = outputs(left, g);
                let side_ok = match kind {
                    JoinKind::Inner | JoinKind::Cross => true,
                    JoinKind::LeftOuter
                    | JoinKind::LeftSemi
                    | JoinKind::LeftAnti
                    | JoinKind::Existence(_)
                    | JoinKind::Single => in_left,
                    JoinKind::RightOuter => !in_left,
                    _ => false,
                };
                if !side_ok || (!in_left && !outputs(right, g)) {
                    return stop();
                }
                let side = if in_left { left } else { right };
                let new = self.walk(side, g, found)?;
                Arc::clone(plan)
                    .map_children(|c| {
                        Ok::<_, ()>(if Arc::ptr_eq(&c, side) {
                            Arc::clone(&new)
                        } else {
                            c
                        })
                    })
                    .ok()
            }
            _ => stop(),
        }
    }

    /// The conjunct on key `g` this walk rewrites, with its column and literal: `g' = literal`,
    /// or in `Narrow(k)` mode `g' IN (..)` whose list holds `k`.
    fn key_conjunct(&self, c: &Arc<Expr>, g: AttrId) -> Option<(BoundColumn, Literal)> {
        match self.mode {
            Mode::Narrow(k) => {
                let (Builtin::In, [col, list @ ..], _) = builtin(c)? else {
                    return None;
                };
                let ExprNode::Resolved(ResolvedExpr::Column(col)) = &col.node else {
                    return None;
                };
                let holds_k = list
                    .iter()
                    .any(|e| matches!(&e.node, ExprNode::Literal(l) if l == k));
                (col.id == g && holds_k).then(|| (col.clone(), k.clone()))
            }
            Mode::Mark | Mode::In(_) => key_literal(c, g).map(|(col, l)| (col.clone(), l.clone())),
        }
    }

    /// The conjunct `c` (`g' = literal`) as this walk's mode writes it.
    fn rewrite(&self, c: &Arc<Expr>, col: &BoundColumn) -> Arc<Expr> {
        match self.mode {
            Mode::Narrow(k) => call(
                Builtin::Eq,
                vec![column(col), lit(k.clone())],
                DataType::Boolean,
            ),
            Mode::Mark => call(Builtin::In, vec![column(col)], DataType::Boolean),
            Mode::In(literals) => {
                let mut args = vec![column(col)];
                args.extend(literals.iter().cloned().map(lit));
                match builtin(c) {
                    Some((_, _, f)) => recall(f, Builtin::In, args),
                    None => call(Builtin::In, args, DataType::Boolean),
                }
            }
        }
    }
}

/// The instance a reader `Filter(g = k)` of the shared definition `def` replaced, for
/// estimates: `def` with the conjunct `g' IN (.., k, ..)` on the lineage of its output `g`
/// narrowed back to `g' = k`; `None` when no such conjunct is found. The join order of a
/// second optimization estimates the reader as this, as the first one estimated the instance,
/// so the plan stays a fixed point (read as `def`, a reader holds every literal's groups).
pub(crate) fn instance_of(
    def: &Arc<LogicalPlan>,
    g: AttrId,
    k: &Literal,
) -> Option<Arc<LogicalPlan>> {
    let mut lift = Lift::new(Mode::Narrow(k));
    let instance = lift.core(def, g)?;
    (!lift.found.is_empty()).then_some(instance)
}

/// The reader's own conjunct `g = k` on one of `outputs` (a column against a non-null literal):
/// the conjunct, the column and the literal.
pub(crate) fn reader_key(
    predicate: &Arc<Expr>,
    outputs: &[AttrId],
) -> Option<(Arc<Expr>, AttrId, Literal)> {
    conjuncts(predicate).into_iter().find_map(|c| {
        outputs
            .iter()
            .find_map(|&g| key_literal(&c, g).map(|(_, l)| (Arc::clone(&c), g, l.clone())))
    })
}

/// `plan` with its one child replaced.
fn with_child(plan: &Arc<LogicalPlan>, child: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    Arc::clone(plan)
        .map_children(|_| Ok::<_, ()>(Arc::clone(child)))
        .unwrap_or_else(|()| Arc::clone(plan))
}

/// Whether `plan` outputs attribute `g`.
fn outputs(plan: &Arc<LogicalPlan>, g: AttrId) -> bool {
    plan.resolved_output()
        .is_some_and(|o| o.iter().any(|c| c.id == g))
}

/// Whether `exprs` holds the bare column `g`.
fn is_bare(exprs: &[Arc<Expr>], g: AttrId) -> bool {
    exprs
        .iter()
        .any(|e| matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Column(c)) if c.id == g))
}

/// `c` as `g = literal` (either side): the column and the literal; a null
/// literal never (no row passes it).
fn key_literal(c: &Arc<Expr>, g: AttrId) -> Option<(&BoundColumn, &Literal)> {
    let (Builtin::Eq, [a, b], _) = builtin(c)? else {
        return None;
    };
    fn pick<'e>(x: &'e Expr, y: &'e Expr, g: AttrId) -> Option<(&'e BoundColumn, &'e Literal)> {
        match (&x.node, &y.node) {
            (ExprNode::Resolved(ResolvedExpr::Column(col)), ExprNode::Literal(l))
                if col.id == g && !matches!(l, Literal::Null(_)) =>
            {
                Some((col, l))
            }
            _ => None,
        }
    }
    pick(a, b, g).or_else(|| pick(b, a, g))
}

#[cfg(test)]
mod tests;

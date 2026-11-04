// SPDX-License-Identifier: Apache-2.0
//! The fragmenter (design P2-T176 §2.1, Trino's model): a physical plan cut
//! at **every** [`PhysNode::Exchange`] into stages. A stage's fragment is a
//! [`PhysicalPlan`] whose root is the exchange it writes (the root stage's
//! is the plan's root) and whose leaves include a
//! [`PhysNode::ExchangeSource`] in place of each exchange it reads. The
//! fragments are plan data: they travel as `krishiv.plan.v1` bytes like any
//! plan, and the same bytes run as every task of their stage.
//!
//! Not cut, and run whole by one task of their stage, as on one worker:
//! the bodies of shared plans (copied into each fragment that reads them)
//! and of recursive CTEs. A stage that holds one, a `range` source, a gather
//! read by its rows (not by a subquery) or a join that emits build rows
//! from a broadcast build runs as one task ([`Stage::single`]): several
//! tasks would each emit those rows.

use std::collections::{BTreeMap, BTreeSet};

use arrow_schema::SchemaRef;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::physical::{
    ExchangeId, ExchangePlan, HashScheme, JoinDistribution, Partitioning, PhysNode, PhysicalPlan,
    SharedId, SharedPlan,
};
use krishiv_io::FilterId;

/// A stage's read of an exchange another stage writes.
#[derive(Debug, Clone, PartialEq)]
pub struct StageInput {
    /// The exchange read.
    pub exchange: ExchangeId,
    /// The stage that writes it.
    pub stage: usize,
    /// How it places rows.
    pub partitioning: Partitioning,
    /// It is read by a subquery's plan, whose value each task computes
    /// over the whole exchange.
    pub subquery: bool,
}

/// One stage of a fragmented plan.
#[derive(Debug, Clone)]
pub struct Stage {
    /// Its index in [`fragment`]'s result: every stage it reads comes
    /// before it.
    pub id: usize,
    /// What each of its tasks runs.
    pub plan: PhysicalPlan,
    /// The exchange it writes and how; `None` for the root stage, whose
    /// rows are the query's result.
    pub output: Option<(ExchangeId, Partitioning)>,
    /// The exchanges it reads, in plan order.
    pub inputs: Vec<StageInput>,
    /// It must run as one task (see the module docs).
    pub single: bool,
    /// It scans tables: several tasks need the scans' splits.
    pub scans: bool,
}

type Result<T> = std::result::Result<T, KrishivError>;

/// `plan` cut at every exchange, the stages in dependency order (the root
/// stage last). `schemas` holds each exchange's columns
/// ([`krishiv_exec::physical::exchange_schemas`]).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a hash exchange without a routing
/// function (a plan written before it was named, design P2-T171 I4), or a
/// nondeterministic shared plan read by several stages (each would compute
/// it again); `INTERNAL_ERROR` for an exchange without a schema, or a plan
/// that already reads an exchange source (a fragment).
pub fn fragment(
    plan: &PhysicalPlan,
    schemas: &BTreeMap<ExchangeId, SchemaRef>,
) -> Result<Vec<Stage>> {
    let run = || {
        let mut cx = Cutter {
            plan,
            schemas,
            stages: Vec::new(),
            readers: BTreeMap::new(),
        };
        cx.stage(plan.root(), None)?;
        for (id, readers) in &cx.readers {
            let deterministic = plan
                .shared()
                .iter()
                .find(|s| s.id == *id)
                .is_none_or(|s| s.deterministic);
            if readers.len() > 1 && !deterministic {
                return Err(unsupported(
                    "a nondeterministic shared plan read by several stages",
                ));
            }
        }
        Ok(cx.stages)
    };
    krishiv_common::stack::on_big_stack(run).ok_or_else(|| {
        KrishivError::from(InternalError {
            message: "cannot start a fragmenter thread".to_owned(),
        })
    })?
}

struct Cutter<'a> {
    plan: &'a PhysicalPlan,
    schemas: &'a BTreeMap<ExchangeId, SchemaRef>,
    stages: Vec<Stage>,
    /// The stages reading each shared plan.
    readers: BTreeMap<SharedId, BTreeSet<usize>>,
}

/// What a fragment holds, gathered while it is cut.
#[derive(Default)]
struct Holds {
    inputs: Vec<StageInput>,
    single: bool,
    scans: bool,
    shared: BTreeSet<SharedId>,
}

impl Cutter<'_> {
    /// Cuts the stage whose root is `root` (`output` when it writes that
    /// exchange); its index.
    fn stage(&mut self, root: &PhysNode, output: Option<&ExchangePlan>) -> Result<usize> {
        let mut holds = Holds::default();
        let (cut, schema) = match output {
            Some(e) => {
                check_scheme(&e.partitioning)?;
                let input = self.cut(&e.input, &mut holds, false)?;
                (
                    PhysNode::Exchange(ExchangePlan {
                        id: e.id,
                        input: Box::new(input),
                        partitioning: e.partitioning.clone(),
                    }),
                    self.schema(e.id)?,
                )
            }
            None => (
                self.cut(root, &mut holds, false)?,
                SchemaRef::clone(self.plan.schema()),
            ),
        };
        // The shared plans the fragment reads, and those they read, in
        // the plan's order (a shared plan reads only earlier ones).
        let mut wanted = holds.shared.clone();
        for sp in self.plan.shared().iter().rev() {
            if wanted.contains(&sp.id) {
                let mut stack = vec![&sp.plan];
                while let Some(n) = stack.pop() {
                    if let PhysNode::SharedRead { id, .. } = n {
                        wanted.insert(*id);
                    }
                    if let PhysNode::Scan(_) = n {
                        holds.scans = true;
                    }
                    stack.extend(n.children());
                }
            }
        }
        let mut shared: Vec<SharedPlan> = self
            .plan
            .shared()
            .iter()
            .filter(|sp| wanted.contains(&sp.id))
            .cloned()
            .collect();
        let mut cut = cut;
        {
            let mut roots: Vec<&mut PhysNode> = vec![&mut cut];
            roots.extend(shared.iter_mut().map(|sp| &mut sp.plan));
            local_filters(&mut roots);
        }
        let fragment = PhysicalPlan::new(
            cut,
            schema,
            self.plan.explain().clone(),
            self.plan.env().clone(),
        )
        .with_shared(shared)?;
        let id = self.stages.len();
        for s in &wanted {
            self.readers.entry(*s).or_default().insert(id);
        }
        self.stages.push(Stage {
            id,
            plan: fragment,
            output: output.map(|e| (e.id, e.partitioning.clone())),
            inputs: holds.inputs,
            single: holds.single || !wanted.is_empty(),
            scans: holds.scans,
        });
        Ok(id)
    }

    fn schema(&self, id: ExchangeId) -> Result<SchemaRef> {
        self.schemas.get(&id).cloned().ok_or_else(|| {
            InternalError {
                message: format!("exchange {} has no schema", id.0),
            }
            .into()
        })
    }

    /// `n` with each exchange below it replaced by its source, cutting the
    /// stages that write them. `subquery`: `n` is (inside) a subquery's
    /// plan, whose value every task reads whole.
    fn cut(&mut self, n: &PhysNode, holds: &mut Holds, subquery: bool) -> Result<PhysNode> {
        match n {
            PhysNode::Exchange(e) => {
                let stage = self.stage(n, Some(e))?;
                if !subquery
                    && matches!(
                        e.partitioning,
                        Partitioning::Single | Partitioning::Range { .. }
                    )
                {
                    holds.single = true;
                }
                holds.inputs.push(StageInput {
                    exchange: e.id,
                    stage,
                    partitioning: e.partitioning.clone(),
                    subquery,
                });
                return Ok(PhysNode::ExchangeSource {
                    id: e.id,
                    partitioning: e.partitioning.clone(),
                    schema: self.schema(e.id)?,
                });
            }
            PhysNode::UnionLoop { .. } => {
                // Run whole, as on one worker: its exchanges are lowered.
                holds.single = true;
                let mut stack = vec![n];
                while let Some(m) = stack.pop() {
                    match m {
                        PhysNode::SharedRead { id, .. } => {
                            holds.shared.insert(*id);
                        }
                        PhysNode::Scan(_) => holds.scans = true,
                        _ => {}
                    }
                    stack.extend(m.children());
                }
                return Ok(n.clone());
            }
            PhysNode::SharedRead { id, .. } => {
                holds.shared.insert(*id);
            }
            PhysNode::ExchangeSource { id, .. } => {
                return Err(InternalError {
                    message: format!(
                        "exchange source {} in a plan being cut into stages (a fragment)",
                        id.0
                    ),
                }
                .into());
            }
            PhysNode::Range(_) => holds.single = true,
            // Its one row comes once: every task of the stage would emit it.
            PhysNode::OneRow if !subquery => holds.single = true,
            PhysNode::Scan(_) | PhysNode::LateFetch { .. } => holds.scans = true,
            PhysNode::NestedLoopJoin(j) if j.join_type.builds_preserved() => holds.single = true,
            PhysNode::HashJoin(j)
                if j.join_type.builds_preserved()
                    && j.distribution == JoinDistribution::Broadcast =>
            {
                holds.single = true;
            }
            _ => {}
        }
        // The children after the input of a node with subqueries are the
        // subqueries' plans.
        let first_subquery = match n {
            PhysNode::Filter { .. }
            | PhysNode::Project { .. }
            | PhysNode::ReplicateRows { .. }
            | PhysNode::Generate { .. }
            | PhysNode::Expand { .. } => 1,
            _ => usize::MAX,
        };
        let mut cut = n.clone();
        for (i, (child, slot)) in n.children().into_iter().zip(cut.children_mut()).enumerate() {
            *slot = self.cut(child, holds, subquery || i >= first_subquery)?;
        }
        Ok(cut)
    }
}

/// Keeps the runtime filters whose producer (a join's build, a top-N's
/// bound) and scan are both in the fragment of `roots`: a filter crossing
/// stages is dropped on both sides until its value crosses with it
/// (design P2-T177 unit 2). Dropping one never changes rows.
fn local_filters(roots: &mut [&mut PhysNode]) {
    let mut scanned = BTreeSet::new();
    let mut produced = BTreeSet::new();
    for root in roots.iter() {
        let mut stack: Vec<&PhysNode> = vec![&**root];
        while let Some(n) = stack.pop() {
            match n {
                PhysNode::Scan(s) => scanned.extend(s.runtime_filters.iter().map(|f| f.id)),
                PhysNode::HashJoin(j) => produced.extend(j.runtime_filters.iter().map(|f| f.id)),
                PhysNode::TopN { bound, .. } => produced.extend(*bound),
                _ => {}
            }
            stack.extend(n.children());
        }
    }
    let keep: BTreeSet<FilterId> = scanned.intersection(&produced).copied().collect();
    for root in roots.iter_mut() {
        let mut stack: Vec<&mut PhysNode> = vec![&mut **root];
        while let Some(n) = stack.pop() {
            match n {
                PhysNode::Scan(s) => s.runtime_filters.retain(|f| keep.contains(&f.id)),
                PhysNode::HashJoin(j) => j.runtime_filters.retain(|f| keep.contains(&f.id)),
                PhysNode::TopN { bound, .. } if bound.is_some_and(|b| !keep.contains(&b)) => {
                    *bound = None;
                }
                _ => {}
            }
            stack.extend(n.children_mut());
        }
    }
}

fn check_scheme(p: &Partitioning) -> Result<()> {
    match p {
        Partitioning::Hash {
            scheme: HashScheme::Unpinned,
            ..
        } => Err(unsupported(
            "a hash exchange without a routing function across stages",
        )),
        _ => Ok(()),
    }
}

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

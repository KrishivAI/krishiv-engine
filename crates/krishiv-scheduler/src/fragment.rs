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
//! tasks would each emit those rows. A nested-loop join split by
//! [`split_nested_loops`](crate::split_nested_loops) is not single: its
//! [`PhysNode::UnmatchedBuild`] half reads the join's broadcast exchange
//! (cut once, for the join) in a one-task stage of its own.

use std::collections::{BTreeMap, BTreeSet};

use arrow_schema::SchemaRef;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::physical::{
    ExchangeFilterPlan, ExchangeId, ExchangePlan, HashScheme, JoinDistribution, Partitioning,
    PhysNode, PhysicalPlan, ScanId, ScanNode, SharedId, SharedPlan,
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
    /// The scans its tasks may split among them: those of its streaming
    /// pipeline ([`streaming_scans`]). Every other scan is read whole by
    /// every task.
    pub dealable: BTreeSet<ScanId>,
}

/// The scans of the streaming pipeline of the tree at `n`, in plan order:
/// the probe side of each join, never a build side, a subquery's plan, a
/// shared plan's body, a recursive CTE or a late fetch. Only these may be
/// split among a stage's tasks: a dealt build side would join each task's
/// probe rows with part of the build and lose rows.
#[must_use]
pub fn streaming_scans(n: &PhysNode) -> Vec<&ScanNode> {
    fn walk<'a>(n: &'a PhysNode, out: &mut Vec<&'a ScanNode>) {
        match n {
            PhysNode::Scan(s) => out.push(s),
            PhysNode::HashJoin(j) => walk(&j.probe, out),
            PhysNode::SortMergeJoin(j) => walk(&j.probe, out),
            PhysNode::NestedLoopJoin(j) => walk(&j.probe, out),
            PhysNode::UnionLoop { .. } | PhysNode::LateFetch { .. } => {}
            // The children after the input are subqueries' plans.
            PhysNode::Filter { input, .. }
            | PhysNode::Project { input, .. }
            | PhysNode::ReplicateRows { input, .. }
            | PhysNode::Generate { input, .. }
            | PhysNode::Expand { input, .. } => walk(input, out),
            other => other.children().into_iter().for_each(|c| walk(c, out)),
        }
    }
    let mut out = Vec::new();
    walk(n, &mut out);
    out
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
        let mut root = plan.root().clone();
        let crossing = cross_stage_filters(&mut root);
        let mut cx = Cutter {
            plan,
            schemas,
            stages: Vec::new(),
            readers: BTreeMap::new(),
            crossing,
            writers: BTreeMap::new(),
        };
        cx.stage(&root, None)?;
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
    /// The filters an exchange collects for scans of other stages
    /// ([`cross_stage_filters`]): their scans keep them.
    crossing: BTreeSet<FilterId>,
    /// The stage writing each exchange cut so far.
    writers: BTreeMap<ExchangeId, usize>,
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
                        filters: e.filters.clone(),
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
            local_filters(&mut roots, &self.crossing);
        }
        let fragment = PhysicalPlan::new(
            cut,
            schema,
            self.plan.explain().clone(),
            self.plan.env().clone(),
        )
        .with_shared(shared)?;
        let id = self.stages.len();
        if let Some(e) = output {
            self.writers.insert(e.id, id);
        }
        for s in &wanted {
            self.readers.entry(*s).or_default().insert(id);
        }
        self.stages.push(Stage {
            id,
            output: output.map(|e| (e.id, e.partitioning.clone())),
            inputs: holds.inputs,
            dealable: streaming_scans(fragment.root())
                .iter()
                .map(|s| s.id)
                .collect(),
            single: holds.single || !wanted.is_empty(),
            scans: holds.scans,
            plan: fragment,
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
            PhysNode::NestedLoopJoin(j) if j.join_type.builds_preserved() && j.marks.is_none() => {
                holds.single = true;
            }
            PhysNode::UnmatchedBuild(u) => {
                // Every task would emit the build rows: one task does.
                holds.single = true;
                let stage = self.writers.get(&u.exchange).copied().ok_or_else(|| {
                    KrishivError::from(InternalError {
                        message: format!(
                            "the build rows of a split nested-loop join read exchange {}, \
                             which no earlier stage writes",
                            u.exchange.0
                        ),
                    })
                })?;
                holds.inputs.push(StageInput {
                    exchange: u.exchange,
                    stage,
                    partitioning: Partitioning::Broadcast,
                    subquery,
                });
                return Ok(n.clone());
            }
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
        let mut pairs: Vec<_> = n
            .children()
            .into_iter()
            .zip(cut.children_mut())
            .enumerate()
            .collect();
        // A join's build side is cut first, so its stages run before the
        // probe side's: a filter the build exchange collects is ready when
        // the probe side's scans run (see `cross_stage_filters`).
        if matches!(n, PhysNode::HashJoin(_)) {
            pairs.reverse();
        }
        for (i, (child, slot)) in pairs {
            *slot = self.cut(child, holds, subquery || i >= first_subquery)?;
        }
        Ok(cut)
    }
}

/// Moves each runtime filter of a hash join whose scan is in another stage
/// (below an exchange on the probe side) onto the exchange that feeds the
/// join's build, when the build input is that exchange: the exchange
/// collects the filter from the rows it writes, in the stage that writes
/// them, which runs before the scan's (design P2-T177, cross-stage
/// filters). Returns the moved filters; the others stay where they are.
fn cross_stage_filters(root: &mut PhysNode) -> BTreeSet<FilterId> {
    /// The filters of scans reachable from `n` without crossing an
    /// exchange, and the key type of every scan filter below `n`.
    fn scans(n: &PhysNode, local: bool, out: &mut BTreeSet<FilterId>) {
        if let PhysNode::Scan(s) = n
            && local
        {
            out.extend(s.runtime_filters.iter().map(|f| f.id));
        }
        let local = local && !matches!(n, PhysNode::Exchange(_));
        for c in n.children() {
            scans(c, local, out);
        }
    }
    let mut types = BTreeMap::new();
    let mut stack: Vec<&PhysNode> = vec![root];
    while let Some(n) = stack.pop() {
        if let PhysNode::Scan(s) = n {
            types.extend(s.runtime_filters.iter().map(|f| (f.id, f.key_type.clone())));
        }
        stack.extend(n.children());
    }
    let mut moved = BTreeSet::new();
    let mut stack: Vec<&mut PhysNode> = vec![root];
    while let Some(n) = stack.pop() {
        if let PhysNode::HashJoin(j) = n
            && !j.runtime_filters.is_empty()
            && let PhysNode::Exchange(build) = &mut *j.build
        {
            let mut local = BTreeSet::new();
            scans(&j.probe, true, &mut local);
            let build_keys = j.build_keys.clone();
            let null_safe = j.null_safe.clone();
            j.runtime_filters.retain(|f| {
                let (Some(key_type), Some(&column)) = (types.get(&f.id), build_keys.get(f.key))
                else {
                    return true;
                };
                if local.contains(&f.id) {
                    return true;
                }
                build.filters.push(ExchangeFilterPlan {
                    id: f.id,
                    column,
                    key_type: key_type.clone(),
                    options: f.options,
                    null_safe: null_safe.get(f.key).copied().unwrap_or(false),
                });
                moved.insert(f.id);
                false
            });
        }
        stack.extend(n.children_mut());
    }
    moved
}

/// Keeps the runtime filters whose producer (a join's build, a top-N's
/// bound) and scan are both in the fragment of `roots`, and the scans'
/// filters in `crossing` (collected by an exchange of an earlier stage,
/// whose value the task brings): any other filter crossing stages is
/// dropped on both sides. Dropping one never changes rows.
fn local_filters(roots: &mut [&mut PhysNode], crossing: &BTreeSet<FilterId>) {
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
    let mut keep: BTreeSet<FilterId> = scanned.intersection(&produced).copied().collect();
    keep.extend(scanned.intersection(crossing).copied());
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

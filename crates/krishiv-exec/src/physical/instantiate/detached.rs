// SPDX-License-Identifier: Apache-2.0
//! What a recursive CTE keeps to instantiate its anchor and each round of its recursive term long after the
//! instantiation that planned it ended ([`Detached`]).
//!
//! Spark fixes each unseeded random call's seed once, at analysis, and runs the anchor and every round as a plan of
//! its own: a non-recursive CTE the loop reads is computed again by each run, from the same seeds, round `k`'s at
//! partition `k - 1` (`EvalEnv::partition_shift`), so the anchor, round 1 and the outer query see the same rows and
//! later rounds others. Here a seed's salt is numbered per instantiation in compile order
//! (`BindOptions::seed_salt`), so a run reproduces the salts the instantiation that first built a shared plan gave
//! it ([`Instantiation::shared_body`]), and numbers its own calls from a range of its own (the anchor's and the
//! rounds' apart), the same in every round. The anchor is instantiated per run too: built once with the outer
//! plan, its random functions would continue the streams the outer query already drew.

use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

/// Where the unseeded salts of a loop's runs start, far past any instantiation's own count.
const RUN_SALTS: u64 = 1 << 40;

/// Which run of a loop an instantiation is for: its salts come from a range of their own.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LoopRun {
    /// The anchor.
    Anchor,
    /// A round of the recursive term.
    Round,
}

/// The first salt of the runs `run` of loop `id`.
fn run_salts(id: u32, run: LoopRun) -> u64 {
    let slot = u64::from(id) * 2 + u64::from(matches!(run, LoopRun::Round));
    RUN_SALTS.wrapping_add(slot << 20)
}

impl Instantiation<'_> {
    /// The body of shared plan `sp`, its unseeded random calls salted as where it was built first: a run of a loop
    /// builds again what the outer instantiation built, and must draw the same streams.
    pub(super) fn shared_body(&mut self, sp: &SharedPlan) -> Result<Built> {
        let salt = &self.env.seed_salt;
        let resume = match self.salts.get(&sp.id) {
            Some(&first) => Some(salt.swap(first, Ordering::Relaxed)),
            None => {
                self.salts.insert(sp.id, salt.load(Ordering::Relaxed));
                None
            }
        };
        let body = self.node(&sp.plan);
        if let Some(r) = resume {
            self.env.seed_salt.store(r, Ordering::Relaxed);
        }
        body
    }
}

/// What a recursive CTE needs to instantiate its anchor and its recursive
/// term once per run, long after the instantiation that built it ended.
pub(crate) struct Detached {
    pub(super) worker: WorkerContext,
    pub(super) task: TaskSpec,
    pub(super) spill: Option<SpillScope>,
    pub(super) env: EvalEnv,
    pub(super) shared_plans: Arc<[SharedPlan]>,
    /// The tables of the enclosing loops' current rounds.
    pub(super) loops: HashMap<u32, Arc<Output>>,
    /// The first salt of each shared plan the planning instantiation built.
    pub(super) salts: HashMap<super::super::SharedId, u64>,
}

impl Detached {
    /// The query's spill scope on this worker.
    pub(crate) fn spill(&self) -> Option<&SpillScope> {
        self.spill.as_ref()
    }

    /// `node` built with the loop tables `loops` (those of the enclosing
    /// loops, and this loop's previous round), as a run of loop `id` of its
    /// own: fresh values and random streams, and the plan's shared plans
    /// built again. `partition_shift` is the round's number less one (see
    /// `EvalEnv::partition_shift`; 0 for the anchor).
    ///
    /// # Errors
    /// As [`instantiate_task`].
    pub(crate) fn instantiate(
        &self,
        node: &PhysNode,
        mut loops: HashMap<u32, Arc<Output>>,
        (id, run, partition_shift): (u32, LoopRun, u32),
    ) -> Result<(Arc<dyn ExecNode>, SchemaRef)> {
        for (id, table) in &self.loops {
            loops.entry(*id).or_insert_with(|| Arc::clone(table));
        }
        let mut env = self.env.clone();
        // A loop inside a round of an outer loop keeps the outer round's
        // shift: the two compose instead of the inner one replacing it.
        env.partition_shift = env.partition_shift.saturating_add(partition_shift);
        env.seed_salt = Arc::new(AtomicU64::new(run_salts(id, run)));
        let run = || {
            let mut cx = Instantiation {
                worker: &self.worker,
                task: self.task.clone(),
                spill: self.spill.clone(),
                env,
                values: Arc::new(LocalValues::new()),
                ids: HashMap::new(),
                shared: HashMap::new(),
                shared_plans: Arc::clone(&self.shared_plans),
                loops,
                schemas: BTreeMap::new(),
                unit_weights: HashSet::new(),
                salts: self.salts.clone(),
            };
            cx.declare_filters(node)?;
            for sp in cx.shared_plans.iter() {
                cx.declare_filters(&sp.plan)?;
            }
            cx.build_shared()?;
            let built = cx.node(node)?;
            Ok((built.node, built.schema))
        };
        krishiv_common::stack::on_big_stack(run)
            .ok_or_else(|| internal("cannot start an instantiation thread".to_owned()))?
    }
}

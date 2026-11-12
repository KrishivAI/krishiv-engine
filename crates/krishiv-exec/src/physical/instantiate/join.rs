// SPDX-License-Identifier: Apache-2.0
//! The hash join's builder, and the selection fused into its probe (design
//! P2-E3 U2): the fused filter compiled over the probe input, and the row
//! stage a scan below hands the probe ([`RuntimeFilters::hand_to_consumer`]).

use super::*;
use std::collections::HashSet;

use krishiv_plan::{ExprNode, ResolvedExpr};

use crate::operators::select::Narrow;
use crate::physical::{ProbeColumn, ProbeFilter};

/// The hash join of `j` over inputs of `probe` and `build`, publishing its
/// runtime filters to `values` (which declares them) as `producer`.
pub(crate) fn hash_join(
    probe: &SchemaRef,
    build: &SchemaRef,
    j: &HashJoinPlan,
    env: &EvalEnv,
    spill: Option<SpillScope>,
    values: &Arc<dyn ValueService>,
    producer: Producer,
) -> Result<HashJoin> {
    if j.distinct_build && !j.distinct_build_allowed() {
        return Err(internal(format!(
            "a distinct build on a {:?} join{}",
            j.join_type,
            if j.residual.is_some() {
                " with a residual condition"
            } else {
                ""
            }
        )));
    }
    // The probe's input schema: the batches it receives, or the columns it
    // reads of them (a fused projection, design P2-E3 U2).
    let received = probe;
    let columns = j.select.as_ref().and_then(|s| s.columns.as_deref());
    let probe = &match columns {
        Some(c) => probe_columns(received, c)?,
        None => Arc::clone(received),
    };
    let join = HashJoin::try_new(JoinSpec {
        join_type: j.join_type,
        probe_schema: Arc::clone(probe),
        build_schema: Arc::clone(build),
        probe_keys: j.probe_keys.clone(),
        build_keys: j.build_keys.clone(),
        null_safe: j.null_safe.clone(),
        filter: residual(probe, build, j.residual.as_ref(), env)?,
        single_match: j.single_match,
        spill,
    })?;
    let join = match j.select.as_ref().and_then(|s| s.filter.as_ref()) {
        Some(f) => {
            let join = join.with_probe_filter(probe_filter(received, f, env)?)?;
            match narrow_probe_filter(received, f, env) {
                Some(narrow) => join.with_narrow_probe_filter(narrow)?,
                None => join,
            }
        }
        None => join,
    };
    let join = match columns {
        Some(c) => join.with_probe_columns(c.iter().map(|c| c.index).collect())?,
        None => join,
    };
    if j.runtime_filters.is_empty() {
        return Ok(join);
    }
    let filters = j
        .runtime_filters
        .iter()
        .map(|f| {
            // A planned Bloom size lets several build partitions' filters
            // OR; one producer sizes its Bloom for the keys it saw, as an
            // estimate can be far off.
            let mut options = f.options;
            if values.producers(ValueId::Filter(f.id)) == 1 {
                options.bloom_bytes = None;
            }
            Ok(JoinRuntimeFilter {
                key: f.key,
                filter: values.filter(f.id)?,
                options,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    join.with_filter_values(filters, Arc::clone(values), producer)
}

/// The probe input: columns `columns` of the batches of `received`, named
/// as the fused projection named them.
fn probe_columns(received: &SchemaRef, columns: &[ProbeColumn]) -> Result<SchemaRef> {
    let fields = columns
        .iter()
        .map(|c| {
            received
                .fields()
                .get(c.index)
                .map(|f| f.as_ref().clone().with_name(c.name.clone()))
                .ok_or_else(|| internal(format!("a fused probe column {} out of range", c.index)))
        })
        .collect::<Result<Vec<Field>>>()?;
    Ok(Arc::new(Schema::new_with_metadata(
        fields,
        received.metadata().clone(),
    )))
}

/// The fused filter `f` compiled over the probe input `probe`.
fn probe_filter(probe: &SchemaRef, f: &ProbeFilter, env: &EvalEnv) -> Result<Arc<dyn Predicate>> {
    if f.layout.len() != probe.fields().len() {
        return Err(internal(format!(
            "a fused probe filter over {} columns of a {}-column probe input",
            f.layout.len(),
            probe.fields().len()
        )));
    }
    let v = Values {
        env,
        subqueries: &[],
    };
    let e = compile_one(&f.predicate, &f.layout, probe, &v)?;
    Ok(Arc::new(Compiled(e)))
}

/// The fused filter `f` compiled over only the columns of `probe` it reads,
/// for the copy of a selective row stage's rows (design P2-E3 U2): `None`
/// when it reads every column or none, or does not compile over them (the
/// copy then holds every column; results are the same either way).
fn narrow_probe_filter(probe: &SchemaRef, f: &ProbeFilter, env: &EvalEnv) -> Option<Narrow> {
    let mut read = HashSet::new();
    read_columns(&f.predicate, &mut read);
    let columns: Vec<usize> = (0..f.layout.len())
        .filter(|&i| read.contains(&f.layout[i]))
        .collect();
    if columns.is_empty() || columns.len() >= f.layout.len() {
        return None;
    }
    let layout: Vec<AttrId> = columns.iter().map(|&i| f.layout[i]).collect();
    let schema = Arc::new(probe.project(&columns).ok()?);
    let v = Values {
        env,
        subqueries: &[],
    };
    let e = compile_one(&f.predicate, &layout, &schema, &v).ok()?;
    Some(Narrow {
        columns,
        predicate: Arc::new(Compiled(e)),
    })
}

/// The attributes `e` reads as columns, its sub-expressions' too.
fn read_columns(e: &Arc<Expr>, out: &mut HashSet<AttrId>) {
    if let ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::LambdaVar(c)) = &e.node {
        out.insert(c.id);
    }
    e.for_each_child(&mut |c| read_columns(c, out));
}

impl Instantiation<'_> {
    /// `join` running the row stage of the scan `j`'s selection names, when
    /// that scan was built in this instantiation and can hand it over (every
    /// filter it evaluates has a probe column); otherwise the scan keeps it
    /// and `join` is returned as it is.
    pub(super) fn hand_row_stage(&self, join: HashJoin, j: &HashJoinPlan) -> Result<HashJoin> {
        let Some(stage) = j.select.as_ref().and_then(|s| s.row_stage.as_ref()) else {
            return Ok(join);
        };
        let Some(filters) = self.stages.get(&stage.scan) else {
            return Ok(join);
        };
        let columns: HashMap<FilterId, usize> =
            stage.columns.iter().map(|c| (c.id, c.column)).collect();
        if !filters.hand_to_consumer(columns) {
            return Ok(join);
        }
        join.with_row_stage(filters.clone())
    }
}

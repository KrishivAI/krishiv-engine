// SPDX-License-Identifier: Apache-2.0
//! `PhysNode::Filter`: the operator over its subqueries, with the runtime
//! filters' row stage when the planner moved it here from the scan below
//! (design P2-E3 U1).

use super::*;
use crate::operators::RowStageSettings;
use crate::physical::ScanRuntimeFilter;

impl Instantiation<'_> {
    pub(super) fn filter(
        &mut self,
        input: &PhysNode,
        layout: &[AttrId],
        predicate: &Expr,
        subqueries: &[SubqueryPlan],
        (runtime, row_stage): (&[ScanRuntimeFilter], RowStageSettings),
    ) -> Result<Built> {
        let dynamic = RuntimeFilters::default();
        for f in runtime {
            dynamic.push(self.values.filter(f.id)?);
        }
        self.with_subqueries(input, subqueries, |input, v| {
            let op = compile_one(predicate, layout, input, v)?;
            Ok(Filter::with_row_stage(
                Arc::clone(input),
                Arc::new(Compiled(op)),
                dynamic.scan_filters(input),
                row_stage,
            ) as Arc<dyn Operator>)
        })
    }
}

pub(crate) fn filter_op(
    schema: &SchemaRef,
    layout: &[AttrId],
    predicate: &Expr,
    v: &Values<'_>,
) -> Result<Arc<dyn Operator>> {
    let e = compile_one(predicate, layout, schema, v)?;
    Ok(Filter::new(Arc::clone(schema), Arc::new(Compiled(e))))
}

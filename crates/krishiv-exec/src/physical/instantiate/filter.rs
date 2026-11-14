// SPDX-License-Identifier: Apache-2.0
//! `PhysNode::Filter` running the runtime-filter row stage of the scan
//! below it (design P2-E3 U1): the scan hands its drivers' stages over
//! ([`RuntimeFilters::hand_to_consumer`]) and the filter evaluates them
//! with its predicate, so the rows both keep are copied once.

use super::*;

use crate::physical::{ProbeFilter, ProbeRowStage};

use super::join::narrow_probe_filter;

impl Instantiation<'_> {
    /// The filter `predicate` over `input` (whose columns are the
    /// attributes `layout`), running the row stage `stage` names when that
    /// scan was built in this instantiation and can hand it over; otherwise
    /// the scan keeps its stage and the filter is a plain one.
    pub(super) fn filter_with_row_stage(
        &mut self,
        input: &PhysNode,
        layout: &[AttrId],
        predicate: &Arc<Expr>,
        stage: &ProbeRowStage,
    ) -> Result<Built> {
        let input = self.node(input)?;
        let v = Values {
            env: &self.env,
            subqueries: &[],
        };
        let compiled: Arc<dyn Predicate> =
            Arc::new(Compiled(compile_one(predicate, layout, &input.schema, &v)?));
        let columns: HashMap<FilterId, usize> =
            stage.columns.iter().map(|c| (c.id, c.column)).collect();
        let runtime = self
            .stages
            .get(&stage.scan)
            .filter(|filters| filters.hand_to_consumer(columns))
            .cloned();
        let op: Arc<dyn Operator> = match runtime {
            None => Filter::new(Arc::clone(&input.schema), compiled),
            Some(runtime) => {
                let narrow = narrow_probe_filter(
                    &input.schema,
                    &ProbeFilter {
                        layout: layout.to_vec(),
                        predicate: Arc::clone(predicate),
                    },
                    &self.env,
                );
                Filter::with_row_stage(Arc::clone(&input.schema), compiled, narrow, runtime)
            }
        };
        Ok(operator(input, op))
    }
}

// SPDX-License-Identifier: Apache-2.0
//! The instantiation of a [`PhysNode::Union`](crate::physical::PhysNode::Union): every input
//! built, their columns compared by physical type, and an input whose fields differ from the
//! first only in labels relabelled to the union's (`UnionRelabel`).

use std::sync::Arc;

use arrow_schema::SchemaRef;

use super::{Built, Instantiation, operator};
use crate::error::{Result, internal};
use crate::physical::PhysNode;
use crate::planner::nodes::FanInNode;

impl Instantiation<'_> {
    /// A [`PhysNode::Union`] of `inputs`: one fan-in over the built inputs, with the first
    /// input's schema.
    pub(super) fn union(&mut self, inputs: &[PhysNode]) -> Result<Built> {
        let mut nodes = Vec::with_capacity(inputs.len());
        let mut schema: Option<SchemaRef> = None;
        for i in inputs {
            let mut built = self.node(i)?;
            match &schema {
                None => schema = Some(Arc::clone(&built.schema)),
                Some(s) if !crate::physical::same_columns(s, &built.schema) => {
                    return Err(internal(
                        "union inputs of different physical types".to_owned(),
                    ));
                }
                // Fields that differ only in labels take the union's.
                Some(s) if s.fields() != built.schema.fields() => {
                    built = operator(
                        built,
                        crate::operators::union::UnionRelabel::new(Arc::clone(s)),
                    );
                }
                Some(_) => {}
            }
            nodes.push(built.node);
        }
        Ok(Built {
            schema: schema.ok_or_else(|| internal("a union without inputs".to_owned()))?,
            node: Arc::new(FanInNode { inputs: nodes }),
        })
    }
}

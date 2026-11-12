// SPDX-License-Identifier: Apache-2.0
//! A qualified or backquoted `timestampadd`/`timestampdiff` is not Spark's keyword form.
//!
//! Only the bare keyword form is Spark's grammar: `system.builtin.timestampadd(..)` is a call
//! Spark's catalog lacks. Its string unit is wrapped in a cast so the analyzer, which accepts a
//! bare string-literal unit, refuses it as `UNRESOLVED_ROUTINE`.

use std::sync::Arc;

use krishiv_plan::{CastMode, Expr as IrExpr, ExprNode, Literal as IrLiteral};

use super::Lowerer;
use crate::ast::Span;

impl Lowerer {
    /// Wrap the string unit of a non-keyword `timestampadd`/`timestampdiff` call in a cast.
    pub(super) fn mark_non_keyword_unit(
        &self,
        written_plain: bool,
        lname: &str,
        args: &mut [Arc<IrExpr>],
        sp: Span,
    ) {
        if !written_plain
            && matches!(lname, "timestampadd" | "timestampdiff")
            && let Some(first) = args.first_mut()
            && matches!(first.node, ExprNode::Literal(IrLiteral::String(_)))
        {
            *first = Arc::new(self.ex(
                ExprNode::Cast {
                    expr: Arc::clone(first),
                    to: krishiv_types::DataType::String,
                    mode: CastMode::Default,
                },
                sp,
            ));
        }
    }
}

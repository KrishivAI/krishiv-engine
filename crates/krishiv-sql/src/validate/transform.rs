// SPDX-License-Identifier: Apache-2.0
//! The checks Spark's `AstBuilder` makes on a `TRANSFORM` clause.

use krishiv_common::error::{
    LegacyErrorTemp0064, UnsupportedFeatureTransformDistinctAll, UnsupportedFeatureTransformNonHive,
};

use super::{Checker, R, at, data_type};
use crate::ast::{RowFormat, RowFormatKind, TransformOutput, TransformSelect};

impl Checker<'_> {
    /// A transform, in the order Spark builds it: the quantifier, the output
    /// column types, the clauses of the query specification, then the row
    /// formats (input first).
    pub(super) fn transform(&self, t: &TransformSelect) -> R {
        if t.quantifier.is_some() {
            return Err(at(UnsupportedFeatureTransformDistinctAll {}, t.span));
        }
        if let Some(TransformOutput::Columns(columns)) = &t.output {
            columns.iter().try_for_each(|c| data_type(&c.data_type))?;
        }
        self.select(&t.select)?;
        t.input_format
            .iter()
            .chain(&t.output_format)
            .try_for_each(|f| self.script_row_format(f, t))
    }

    fn script_row_format(&self, f: &RowFormat, t: &TransformSelect) -> R {
        match &f.kind {
            RowFormatKind::Serde(_) => Err(at(UnsupportedFeatureTransformNonHive {}, t.span)),
            RowFormatKind::Delimited(d) => match d.lines_terminated_by.as_deref() {
                Some(lines) if lines != "\n" => Err(at(
                    LegacyErrorTemp0064 {
                        msg: format!(
                            "LINES TERMINATED BY only supports newline '\\n' right now: {lines}"
                        ),
                    },
                    t.span,
                )),
                _ => Ok(()),
            },
        }
    }
}

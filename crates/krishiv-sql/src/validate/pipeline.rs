// SPDX-License-Identifier: Apache-2.0
//! The checks Spark's `AstBuilder` makes on pipeline datasets (`CREATE MATERIALIZED VIEW`, `CREATE STREAMING
//! TABLE`, `CREATE FLOW`) and on `INSERT OVERWRITE DIRECTORY`.

use krishiv_common::error::{DuplicateClauses, LegacyErrorTemp0035, LegacyErrorTemp0049};

use super::{Checker, R, at};
use crate::ast::{
    CreateFlow, CreatePipelineDataset, DirectoryFormat, InsertDirectory, PipelineDatasetKind,
    TableElement,
};

impl Checker<'_> {
    /// A pipeline dataset: the table clauses, then (for a materialized view) the query it needs.
    pub(super) fn pipeline_dataset(&self, c: &CreatePipelineDataset) -> R {
        let t = &c.table;
        self.ddl_name(&t.name)?;
        for element in &t.elements {
            if let TableElement::Column(column) = element {
                self.column_def(column, t.span)?;
            }
        }
        if let Some(clause) = t.duplicate_clause {
            return Err(at(
                DuplicateClauses {
                    clause_name: clause.to_owned(),
                },
                t.span,
            ));
        }
        self.property_list(&t.tblproperties, true, t.span)?;
        self.option_values(&t.tblproperties)?;
        match &t.as_query {
            Some(q) => self.query(q),
            None if c.kind == PipelineDatasetKind::MaterializedView => Err(at(
                LegacyErrorTemp0035 {
                    message: "Unable to find query for CREATE MATERIALIZED VIEW statement"
                        .to_owned(),
                },
                c.span,
            )),
            None => Ok(()),
        }
    }

    /// A flow: its name, then its insert.
    pub(super) fn create_flow(&self, f: &CreateFlow) -> R {
        self.ddl_name(&f.name)?;
        self.insert(&f.insert)
    }

    /// `INSERT OVERWRITE DIRECTORY`: the query, then the one-path rule.
    pub(super) fn insert_directory(&self, d: &InsertDirectory) -> R {
        self.query(&d.query)?;
        if let DirectoryFormat::Provider { options, .. } = &d.format {
            self.option_values(options)?;
        }
        if d.path.is_some() == d.has_path_option() {
            return Err(at(LegacyErrorTemp0049 {}, d.span));
        }
        Ok(())
    }
}

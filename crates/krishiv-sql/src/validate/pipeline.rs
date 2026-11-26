// SPDX-License-Identifier: Apache-2.0
//! The checks Spark's `AstBuilder` makes on pipeline datasets (`CREATE MATERIALIZED VIEW`, `CREATE STREAMING
//! TABLE`, `CREATE FLOW`) and on `INSERT OVERWRITE DIRECTORY`.

use krishiv_common::error::{
    DuplicateClauses, LegacyErrorTemp0035, LegacyErrorTemp0049, LocalMustWithSchemaFile,
};

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
                    message: "Unable to find query for CREATE MATERIALIZED VIEW statement."
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
        if d.has_path() == d.has_path_option() {
            return Err(at(LegacyErrorTemp0049 {}, d.header_span));
        }
        // Spark checks the scheme in `visitInsertOverwriteDir` (the USING form), not in the Hive form.
        if d.local && matches!(d.format, DirectoryFormat::Provider { .. }) {
            let path = d
                .option_path()
                .filter(|p| !p.is_empty())
                .or(d.path.as_deref());
            if let Some(scheme) = path.and_then(uri_scheme).filter(|s| *s != "file") {
                return Err(at(
                    LocalMustWithSchemaFile {
                        actual_schema: scheme.to_owned(),
                    },
                    d.header_span,
                ));
            }
        }
        Ok(())
    }
}

/// The scheme of a path as Hadoop's `new Path(s).toUri` reads it (`CatalogUtils.stringToURI`): the text before a
/// colon, when it is a valid scheme (`s3` of `s3://bucket/p`); none for a plain path.
fn uri_scheme(path: &str) -> Option<&str> {
    let (scheme, rest) = path.split_once(':')?;
    // On Windows Hadoop's `Path` reads `C:` or `C:/x` as a drive, not a scheme.
    if cfg!(windows)
        && scheme.len() == 1
        && scheme.as_bytes()[0].is_ascii_alphabetic()
        && (rest.is_empty() || rest.starts_with(['/', '\\']))
    {
        return None;
    }
    let mut chars = scheme.chars();
    let first = chars.next()?;
    (first.is_ascii_alphabetic()
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')))
    .then_some(scheme)
}

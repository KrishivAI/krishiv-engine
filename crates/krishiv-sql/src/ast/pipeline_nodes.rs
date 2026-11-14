// SPDX-License-Identifier: Apache-2.0
//! AST nodes of the pipeline-dataset statements (`CREATE MATERIALIZED VIEW`, `CREATE STREAMING TABLE`,
//! `CREATE FLOW`), `LOAD DATA` and `INSERT OVERWRITE [LOCAL] DIRECTORY`.

use super::*;

/// What a [`CreatePipelineDataset`] creates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PipelineDatasetKind {
    /// `CREATE MATERIALIZED VIEW`.
    MaterializedView,
    /// `CREATE STREAMING TABLE`.
    StreamingTable,
}

impl PipelineDatasetKind {
    /// The words Spark shows for it (`<pipelineDatasetType>`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MaterializedView => "MATERIALIZED VIEW",
            Self::StreamingTable => "STREAMING TABLE",
        }
    }
}

/// `CREATE MATERIALIZED VIEW | STREAMING TABLE [IF NOT EXISTS] name [(columns)] [clauses] [AS query]`:
/// the `CREATE TABLE` clauses over a pipeline dataset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreatePipelineDataset {
    /// Which dataset.
    pub kind: PipelineDatasetKind,
    /// The name, columns, clauses and query, as a `CREATE TABLE` reads them.
    pub table: CreateTable,
    /// Source text, from `CREATE` to the end of the statement.
    pub span: Span,
}

/// `CREATE FLOW name [COMMENT '...'] AS INSERT INTO|OVERWRITE target query`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreateFlow {
    /// The flow name.
    pub name: DdlName,
    /// `COMMENT '...'`.
    pub comment: Option<String>,
    /// The insert the flow runs.
    pub insert: Insert,
    /// Source text, from `CREATE` to the end of the statement.
    pub span: Span,
}

/// `LOAD DATA [LOCAL] INPATH 'path' [OVERWRITE] INTO TABLE name [PARTITION (...)]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoadData {
    /// `LOCAL`.
    pub local: bool,
    /// The path.
    pub path: String,
    /// `OVERWRITE`.
    pub overwrite: bool,
    /// The target table.
    pub table: DdlName,
    /// `PARTITION (...)`.
    pub partition: Option<PartitionSpec>,
    /// Source text, from `LOAD` to the end of the statement.
    pub span: Span,
}

/// How an [`InsertDirectory`] writes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DirectoryFormat {
    /// `USING provider [OPTIONS (...)]`.
    Provider {
        /// The data source.
        provider: ObjectName,
        /// The options.
        options: Vec<TableOption>,
    },
    /// `[ROW FORMAT ...] [STORED AS ...]` (the Hive form).
    Hive {
        /// `ROW FORMAT ...`.
        row_format: Option<RowFormat>,
        /// `STORED AS ...`.
        file_format: Option<FileFormat>,
    },
}

/// `INSERT OVERWRITE [LOCAL] DIRECTORY ['path'] format query`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InsertDirectory {
    /// `LOCAL`.
    pub local: bool,
    /// The directory, when written outside `OPTIONS`.
    pub path: Option<String>,
    /// The format.
    pub format: DirectoryFormat,
    /// The rows to write.
    pub query: Box<Query>,
    /// Source text of the insert header, from `INSERT` to the end of the format (Spark's `insertInto` context).
    pub header_span: Span,
    /// Source text, from `INSERT` to the end of the statement.
    pub span: Span,
}

impl InsertDirectory {
    /// The `path` key of `OPTIONS`, if written: its string value (`""` when the value is not a string
    /// literal).
    #[must_use]
    pub fn option_path(&self) -> Option<&str> {
        let DirectoryFormat::Provider { options, .. } = &self.format else {
            return None;
        };
        let option = options.iter().find(|o| {
            o.key.parts.len() == 1
                && o.key
                    .parts
                    .first()
                    .is_some_and(|p| p.value.eq_ignore_ascii_case("path"))
        })?;
        Some(match option.value.as_ref().map(|e| &e.kind) {
            Some(ExprKind::Literal(Literal::String(s))) => s,
            _ => "",
        })
    }

    /// Whether the directory is written outside `OPTIONS` (Spark counts an empty literal as none).
    #[must_use]
    pub fn has_path(&self) -> bool {
        self.path.as_deref().is_some_and(|p| !p.is_empty())
    }

    /// Whether `OPTIONS` carries a `path` (an empty value counts as none, as in Spark).
    #[must_use]
    pub fn has_path_option(&self) -> bool {
        self.option_path().is_some_and(|p| !p.is_empty())
    }
}

// SPDX-License-Identifier: Apache-2.0
//! `CALL cat.system.rewrite_data_files(table => ..)` on an Iceberg table (design
//! `P3-T027-36e-maintenance.md` §4.3): the table plans its file groups
//! ([`RowLevelTable::plan_rewrite`]) and each group is one commit, run as a row-level rewrite of
//! the group's files, rows unchanged, through the table's
//! [`group_rewriter`](RowLevelTable::group_rewriter) at the group's output size ceiling.
//!
//! Each group reads a fresh snapshot of the table (the previous group's commit is its base).
//! Not here yet: `strategy => 'sort'`, `sort_order`, `where`, `branch`, partial progress and
//! `remove-dangling-deletes` (refused or ignored by name, see [`refuse_unsupported`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::StreamExt;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::physical::TableWriteRequest;
use krishiv_exec::planner::{
    REWRITE_GROUP_OPTION, REWRITE_WRITE_MAX_OPTION, RewriteGroup, RowLevelOperation, RowLevelTable,
    RowLevelWrite, TableProviderResolver as _,
};
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;

use super::rewrite_scan::GroupTarget;
use super::{first_scan, ident};
use crate::session::{Session, internal};

/// The arguments of the call, as the procedure's parameters hold them.
#[derive(Debug, Default)]
pub(crate) struct RewriteArgs {
    pub(crate) strategy: Option<String>,
    pub(crate) sort_order: Option<String>,
    pub(crate) options: BTreeMap<String, String>,
    pub(crate) filter: Option<String>,
    pub(crate) branch: Option<String>,
}

/// What the call does not run yet, refused before any file is read.
fn refuse_unsupported(args: &RewriteArgs) -> Result<(), KrishivError> {
    let unsupported = |what: &str| -> Result<(), KrishivError> {
        Err(KrishivUnsupportedFeature {
            feature: format!("rewrite_data_files with {what}"),
        }
        .into())
    };
    if args
        .strategy
        .as_deref()
        .is_some_and(|s| !s.eq_ignore_ascii_case("binpack"))
    {
        return unsupported("a strategy other than binpack");
    }
    if args.sort_order.is_some() {
        return unsupported("sort_order");
    }
    if args.filter.is_some() {
        return unsupported("where");
    }
    if args.branch.is_some() {
        return unsupported("branch");
    }
    Ok(())
}

/// A result of `names` and their values, one row.
fn result_row(values: [i64; 5]) -> Result<RecordBatch, KrishivError> {
    let names = [
        "rewritten_data_files_count",
        "added_data_files_count",
        "rewritten_bytes_count",
        "failed_data_files_count",
        "removed_delete_files_count",
    ];
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .map(|n| Field::new(*n, ArrowType::Int64, true))
            .collect::<Vec<_>>(),
    ));
    let columns: Vec<ArrayRef> = values
        .iter()
        .map(|v| Arc::new(Int64Array::from(vec![*v])) as ArrayRef)
        .collect();
    RecordBatch::try_new(schema, columns)
        .map_err(|e| internal(format!("the rewrite_data_files result: {e}")))
}

impl Session {
    /// The row-level table of `quoted` and its columns, read at the table's current snapshot.
    async fn rewrite_target(
        &self,
        quoted: &str,
    ) -> Result<(Arc<dyn RowLevelTable>, Vec<TableColumn>), KrishivError> {
        let plain = self.with_locals(Vec::new());
        let probe = plain
            .analyze_variable_query(&format!("SELECT * FROM {quoted}"))
            .await?;
        let (_, physical) = plain.physical_probe(&probe.analyzed, &probe.env).await?;
        let scan = first_scan(physical.root())
            .ok_or_else(|| internal("a rewritten table that reads no table".to_owned()))?;
        let provider = probe.env.catalog.provider(&scan)?;
        let table = provider
            .row_level(RowLevelOperation::Delete)
            .ok_or_else(|| KrishivUnsupportedFeature {
                feature: "rewrite_data_files on a table that is not an Iceberg table".to_owned(),
            })?;
        let columns = probe
            .analyzed
            .output
            .iter()
            .zip(&probe.analyzed.output_declared)
            .map(|(c, declared)| TableColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
                char_varchar: declared.clone(),
            })
            .collect();
        Ok((table, columns))
    }

    /// Runs the call: plans the groups once, then commits each over a fresh snapshot.
    pub(crate) async fn run_rewrite_data_files(
        &self,
        quoted: &str,
        args: &RewriteArgs,
    ) -> Result<RecordBatch, KrishivError> {
        refuse_unsupported(args)?;
        let (table, _) = self.rewrite_target(quoted).await?;
        let groups = table.plan_rewrite(&args.options)?;
        let before = i64::try_from(table.live_data_files()?).unwrap_or(i64::MAX);
        let (mut rewritten, mut bytes) = (0_i64, 0_i64);
        for group in &groups {
            // The previous group's commit is this group's base.
            let (table, columns) = self.rewrite_target(quoted).await?;
            self.rewrite_group(&table, quoted, &columns, group).await?;
            rewritten += i64::try_from(group.files.len()).unwrap_or(i64::MAX);
            bytes += group.input_bytes;
        }
        self.refresh_files();
        let after = if groups.is_empty() {
            before
        } else {
            let (table, _) = self.rewrite_target(quoted).await?;
            i64::try_from(table.live_data_files()?).unwrap_or(i64::MAX)
        };
        // Files after, less files before, plus the files the groups replaced.
        let added = if groups.is_empty() {
            0
        } else {
            (after - before + rewritten).max(0)
        };
        result_row([rewritten, added, bytes, 0, 0])
    }

    /// One group: its files read as they are, written at the group's size ceiling in one commit.
    async fn rewrite_group(
        &self,
        table: &Arc<dyn RowLevelTable>,
        quoted: &str,
        columns: &[TableColumn],
        group: &RewriteGroup,
    ) -> Result<(), KrishivError> {
        let mut with_row_id = columns.to_vec();
        with_row_id.push(TableColumn {
            name: ROW_FILE_COLUMN.to_owned(),
            data_type: DataType::Binary,
            nullable: false,
            char_varchar: None,
        });
        with_row_id.push(TableColumn {
            name: ROW_POSITION_COLUMN.to_owned(),
            data_type: DataType::Long,
            nullable: false,
            char_varchar: None,
        });
        let (part, entry) = self.state.catalog.put_hidden_provider(
            with_row_id.clone(),
            Arc::new(GroupTarget {
                table: Arc::clone(table),
                files: group.files.clone(),
                tasks: group.expected_output_files,
            }),
        );
        let select = columns
            .iter()
            .map(|c| {
                crate::functions::expr(&format!("{}.{}", ident("t"), ident(&c.name)))
                    .map(|e| e.alias(&c.name))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let frame = self
            .hidden_frame(&part, entry, &with_row_id)
            .alias("t")
            .select(select);
        let analysis = frame.action_analysis().await?;
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        let target = first_scan(physical.root())
            .ok_or_else(|| internal("a rewrite group that reads no table".to_owned()))?;
        let mut request = RowLevelWrite::new(
            RowLevelOperation::Delete,
            quoted,
            Arc::clone(physical.schema()),
            group.files.clone(),
            None,
        );
        request.write_id = Some(crate::session::job_uuid());
        request
            .options
            .insert(REWRITE_GROUP_OPTION.to_owned(), String::new());
        request.options.insert(
            REWRITE_WRITE_MAX_OPTION.to_owned(),
            group.write_max_file_size.to_string(),
        );
        let plan = physical.into_table_write(target, TableWriteRequest::RowLevel(request));
        let mut stream = self.run(&plan, &analysis.env).await?;
        while let Some(batch) = stream.next().await {
            batch?;
        }
        Ok(())
    }
}

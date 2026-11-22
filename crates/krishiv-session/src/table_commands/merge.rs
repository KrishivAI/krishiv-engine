// SPDX-License-Identifier: Apache-2.0
//! Copy-on-write `MERGE INTO` on a table whose provider offers a row-level table for
//! [`RowLevelOperation::Merge`] (design note P3-T021 sections 2.3 and 6.1, `P3-PR-DML-row-level.md`
//! §3 item 5). The plan is ordinary operators over the one pinned snapshot of the target that
//! [`RowLevelTable::scan`] reads (hash join, aggregate, project, `PhysNode::TableWrite`):
//!
//! 1. **Source.** The source is read twice (find, then rewrite), so both reads must see the same
//!    rows: an insert-only `MERGE` reads it once as it is, a source that is one scan of a pinned
//!    table reads that snapshot, anything else is first streamed into a spill file
//!    (`merge_source.rs`, delta-spark's `MergeIntoMaterializeSource` `auto` rule).
//! 2. **Find** (not for an insert-only `MERGE`): the target with its row id joined to the source
//!    on the `ON` condition, inner without `NOT MATCHED [BY TARGET]` and `NOT MATCHED BY SOURCE`
//!    clauses, keeping the source rows with the former and the target rows with the latter
//!    (delta-spark's `findTouchedFiles`, `ClassicMergeExecutor.scala:72-220`). Each row gets the
//!    clause that fires for it, in clause order within its kind; the rows are grouped per target
//!    row (the source rows that match it) and then per file. The result: the touched files
//!    (files with a matched target row for which some `MATCHED` clause holds when every clause is
//!    `MATCHED`; files with any matched target row otherwise; every file with a `NOT MATCHED BY
//!    SOURCE` clause), the rows of each clause and the target rows more than one source row
//!    matches.
//! 3. **Multiple matches** per the table's [`CardinalityRule`]: Delta's
//!    `DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW_IN_MERGE` unless the only `MATCHED` clause is
//!    one unconditional `DELETE` (`MergeIntoCommandBase.scala:265-282`).
//! 4. **Rewrite**: the touched files joined to the source (keeping the target rows, and the
//!    source rows when there are `NOT MATCHED` clauses: `writeAllChanges`), the rows the clauses
//!    delete or no clause inserts dropped, every column the first firing clause's value (an
//!    `UPDATE`'s new value or an `INSERT`'s, stored with Spark's ANSI store assignment), else the
//!    target's own, through `PhysNode::TableWrite` into the table's
//!    [`rewriter`](RowLevelTable::rewriter). An insert-only `MERGE` (delta-spark's
//!    `writeOnlyInserts`) writes the source rows no target row matches (a left anti join) and
//!    touches no file.
//!
//! The counts of each kind of change go to the rewriter as options ([`COUNTS_OPTION`]), with the
//! clauses ([`CLAUSES_OPTION`]) and the `ON` condition as written; the result is the table's
//! [`output`](RowLevelTable::output) of them.
//!
//! Analysis follows delta-spark's `ResolveDeltaMergeInto` and `PreprocessTableMerge`: the `ON`
//! condition and `MATCHED` clauses see the target and the source, `NOT MATCHED` clauses only the
//! source and `NOT MATCHED BY SOURCE` clauses only the target; `UPDATE SET *` and `INSERT *` assign
//! every target column the source column of its name; an `INSERT` without a column gives it
//! `NULL`; conditions must be deterministic and hold no subquery.
//!
//! The table raises its own error classes through [`RowLevelTable::error`]: the planner holds no
//! table format's code beyond the [`CardinalityRule`] it reads.
//!
//! Not here yet: schema evolution (`WITH SCHEMA EVOLUTION` does not parse; under
//! `spark.databricks.delta.schema.autoMerge.enabled` a `MERGE` that would add a column is
//! refused), Spark's own `MERGE_CARDINALITY_VIOLATION` rule (a table with
//! [`CardinalityRule::SparkMergeCardinality`] is refused until its oracle exists),
//! `spark.sql.caseSensitive = true` (refused), nested assignment keys and column defaults for an
//! omitted `INSERT` column (`NULL` is written).

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::StreamExt;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::physical::{PhysNode, TableWriteRequest};
use krishiv_exec::planner::{
    CardinalityRule, RowLevelCounts, RowLevelOperation, RowLevelWrite, TableProviderResolver as _,
};
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;

use super::row_level::{
    ALIAS_OPTION, Assignment, CONDITION_OPTION, SCAN_MS_OPTION, Target, first_scan, ident,
};
use crate::Column;
use crate::dataframe::{DataFrame, JoinType};
use crate::functions::{call_function, col, expr, lit};
use crate::session::{Session, internal};

/// The rewrite's option holding the statement's counts, a JSON object: `source_rows`,
/// `matched_updated`, `matched_deleted`, `not_matched_by_source_updated`,
/// `not_matched_by_source_deleted`, `inserted` (`null`: every written row is an insert) and
/// `materialize_ms`.
pub(crate) const COUNTS_OPTION: &str = "__krishiv_row_level_merge_counts";

/// The rewrite's option holding the clauses, a JSON array in statement order of objects
/// `{"kind": "matched" | "notMatched" | "notMatchedBySource", "action": "update" | "delete" |
/// "insert", "condition": text | null}`.
pub(crate) const CLAUSES_OPTION: &str = "__krishiv_row_level_merge_clauses";

/// delta-spark's insert-only path (`merge.optimizeInsertOnlyMerge.enabled`, default true): an
/// insert-only `MERGE` skips the multiple-match check.
const INSERT_ONLY_CONF: &str = "spark.databricks.delta.merge.optimizeInsertOnlyMerge.enabled";

/// delta-spark's automatic schema evolution (`schema.autoMerge.enabled`, default false).
const AUTO_MERGE_CONF: &str = "spark.databricks.delta.schema.autoMerge.enabled";

/// The column marking a target row of the join (a suffix is added if a column has the name).
const TARGET_PRESENT: &str = "__krishiv_merge_target_present";

/// The column marking a source row of the join (a suffix is added if a column has the name).
const SOURCE_PRESENT: &str = "__krishiv_merge_source_present";

/// Which rows a clause fires on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClauseKind {
    /// `WHEN MATCHED`.
    Matched,
    /// `WHEN NOT MATCHED [BY TARGET]`.
    NotMatched,
    /// `WHEN NOT MATCHED BY SOURCE`.
    NotMatchedBySource,
}

impl ClauseKind {
    fn json(self) -> &'static str {
        match self {
            Self::Matched => "matched",
            Self::NotMatched => "notMatched",
            Self::NotMatchedBySource => "notMatchedBySource",
        }
    }
}

/// What a clause does; values are SQL text of the statement.
#[derive(Debug, Clone)]
pub(crate) enum ClauseAction {
    /// `UPDATE SET c = v, ...`.
    Update(Vec<Assignment>),
    /// `UPDATE SET *`.
    UpdateStar,
    /// `DELETE`.
    Delete,
    /// `INSERT *`.
    InsertStar,
    /// `INSERT (c, ...) VALUES (v, ...)`.
    Insert(Vec<Assignment>),
}

impl ClauseAction {
    fn json(&self) -> &'static str {
        match self {
            Self::Update(_) | Self::UpdateStar => "update",
            Self::Delete => "delete",
            Self::InsertStar | Self::Insert(_) => "insert",
        }
    }
}

/// One `WHEN ... THEN ...`.
#[derive(Debug, Clone)]
pub(crate) struct MergeClause {
    pub(crate) kind: ClauseKind,
    /// `AND ...` as written.
    pub(crate) condition: Option<String>,
    pub(crate) action: ClauseAction,
}

/// A `MERGE INTO` statement, its parts as SQL text.
#[derive(Debug, Clone)]
pub(crate) struct MergeStatement {
    /// The target as Spark's errors print it (a name the session reads).
    pub(crate) target: String,
    /// The name the statement reads the target under.
    pub(crate) target_alias: String,
    /// The source: a query.
    pub(crate) source: String,
    /// The name the statement reads the source under.
    pub(crate) source_alias: String,
    /// The `ON` condition.
    pub(crate) on: String,
    pub(crate) clauses: Vec<MergeClause>,
}

/// A clause after analysis: per target column the value it stores (`None`: the target's own for
/// an `UPDATE`, `NULL` for an `INSERT`).
#[derive(Debug, Clone)]
pub(super) struct Resolved {
    pub(super) kind: ClauseKind,
    pub(super) condition: Option<String>,
    /// The condition parsed, a one-sided clause's references qualified by its side's alias.
    pub(super) cond: Option<krishiv_plan::Expr>,
    pub(super) delete: bool,
    pub(super) insert: bool,
    /// Per target column the value a clause assigns, parsed, qualified as `cond`.
    pub(super) exprs: Vec<Option<krishiv_plan::Expr>>,
}

/// What the find query found.
#[derive(Debug, Default)]
struct Found {
    touched: Vec<Vec<u8>>,
    /// Target rows more than one source row matched.
    multiple: u64,
    /// Target rows of each clause.
    per_clause: Vec<u64>,
    /// Source rows an `INSERT` clause fires for.
    inserted: u64,
}

/// `base`, or `base_<n>` for the first `n` that no name of `taken` spells, case-insensitively.
fn fresh(base: &str, taken: &[&str]) -> String {
    let clash = |name: &str| taken.iter().any(|t| t.eq_ignore_ascii_case(name));
    let mut name = base.to_owned();
    let mut n = 0u32;
    while clash(&name) {
        n += 1;
        name = format!("{base}_{n}");
    }
    name
}

fn qualified(alias: &str, column: &str) -> String {
    format!("{}.{}", ident(alias), ident(column))
}

/// Whether a clause's condition holds (`NULL` does not; no condition always does).
fn hit(c: &Resolved) -> Column {
    c.cond.as_ref().map_or_else(
        || lit(true),
        |e| call_function("coalesce", [Column::new(e.clone()), lit(false)]),
    )
}

/// `if(c1, v1, if(c2, v2, .. otherwise))`: the first branch whose condition holds.
fn first_of(branches: Vec<(Column, Column)>, otherwise: Column) -> Column {
    branches
        .into_iter()
        .rev()
        .fold(otherwise, |acc, (c, v)| call_function("if", [c, v, acc]))
}

impl Session {
    /// Runs `m` if its target's provider offers a row-level table for `MERGE`; `None` leaves the
    /// statement to the caller's refusal.
    #[expect(
        clippy::too_many_lines,
        reason = "the statement's steps in order: analysis, source, find, check, rewrite"
    )]
    pub(crate) async fn run_merge(
        &self,
        m: &MergeStatement,
    ) -> Result<Option<DataFrame>, KrishivError> {
        let plain = self.with_locals(Vec::new());
        let Ok(probe) = plain
            .analyze_variable_query(&format!("SELECT * FROM {}", m.target))
            .await
        else {
            return Ok(None);
        };
        let Ok((_, physical)) = plain.physical(&probe.analyzed, &probe.env).await else {
            return Ok(None);
        };
        let Some(scan) = first_scan(physical.root()) else {
            return Ok(None);
        };
        let Ok(provider) = probe.env.catalog.provider(&scan) else {
            return Ok(None);
        };
        let Some(table) = provider.row_level(RowLevelOperation::Merge) else {
            return Ok(None);
        };
        // A Delta table's root, to find a reread of the target in the rewrite (`row_level.rs`).
        #[cfg(feature = "delta")]
        let delta_root = probe.env.catalog.delta_root(&scan).transpose()?;
        let rule = table.capabilities().cardinality;
        if rule != CardinalityRule::DeltaWithUnconditionalDeleteExemption {
            return Err(KrishivUnsupportedFeature {
                feature: "MERGE INTO a table with Spark's own multiple-match (cardinality) rule"
                    .to_owned(),
            }
            .into());
        }
        let conf = self.state.conf.snapshot();
        let flag =
            |key: &str| crate::conf::get(&conf, key).map(|v| v.trim().eq_ignore_ascii_case("true"));
        super::merge_checks::clause_order(&m.clauses)?;
        // Names resolve case-insensitively here: Spark's case-sensitive resolution is refused.
        if flag("spark.sql.caseSensitive") == Some(true) {
            return Err(KrishivUnsupportedFeature {
                feature: "MERGE INTO with spark.sql.caseSensitive = true".to_owned(),
            }
            .into());
        }
        let evolve = flag(AUTO_MERGE_CONF) == Some(true);
        let columns: Vec<TableColumn> = probe
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
        let mut with_row_id = columns.clone();
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
        let source = plain.nested_frame(&m.source).await?;
        let source_columns: Vec<String> = source
            .schema()
            .await?
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        // The presence markers take names that no target or source column spells.
        let taken: Vec<&str> = columns
            .iter()
            .map(|c| c.name.as_str())
            .chain(source_columns.iter().map(String::as_str))
            .collect();
        let target_present = fresh(TARGET_PRESENT, &taken);
        let source_present = fresh(SOURCE_PRESENT, &taken);
        let target_frame = |files: Option<Vec<Vec<u8>>>| {
            let (name, entry) = self.state.catalog.put_hidden_provider(
                with_row_id.clone(),
                Arc::new(Target {
                    table: Arc::clone(&table),
                    op: RowLevelOperation::Merge,
                    files,
                }),
            );
            let frame = self
                .hidden_frame(&name, entry, &with_row_id)
                .with_column(&target_present, lit(true))
                .alias(&m.target_alias);
            (name, frame)
        };
        let source_frame = |df: &DataFrame| {
            df.with_column(&source_present, lit(true))
                .alias(&m.source_alias)
        };

        // Analysis, on frames that read nothing.
        let (_, empty_target) = target_frame(Some(Vec::new()));
        let probe_source = source_frame(&source);
        let checks = super::merge_checks::Checks {
            target: &m.target,
            source_alias: &m.source_alias,
            target_alias: &m.target_alias,
            columns: &columns,
            source_columns: &source_columns,
            table: &table,
            evolve,
        };
        let probe_join = checks.on(&empty_target, &probe_source, &m.on).await?;
        let (resolved, value_types) = checks
            .clauses(&m.clauses, &empty_target, &probe_source, &probe_join)
            .await?;

        let has = |k: ClauseKind| resolved.iter().any(|c| c.kind == k);
        let (matched, inserts, by_source) = (
            has(ClauseKind::Matched),
            has(ClauseKind::NotMatched),
            has(ClauseKind::NotMatchedBySource),
        );
        let insert_only = !matched && !by_source && flag(INSERT_ONLY_CONF) != Some(false);

        // 1. The source: as it is (insert-only), a pinned snapshot, or a spilled copy
        // (`merge_source.rs`).
        let materialize_started = std::time::Instant::now();
        let held = self.merge_source(&source, insert_only).await?;
        let source_rows = match held.rows {
            Some(n) => n,
            // Only the metric needs it: an insert-only MERGE reads its rows once below.
            None => held.frame.count().await?,
        };
        let materialize_ms = if held.kind == super::merge_source::SourceKind::Spilled {
            materialize_started.elapsed().as_millis()
        } else {
            0
        };
        let s = source_frame(&held.frame);
        let on = expr(&m.on)?;
        let tp = qualified(&m.target_alias, &target_present);
        let sp = qualified(&m.source_alias, &source_present);
        let guards: Vec<Column> = resolved
            .iter()
            .map(|c| {
                let kind = match c.kind {
                    ClauseKind::Matched => format!("{tp} IS NOT NULL AND {sp} IS NOT NULL"),
                    ClauseKind::NotMatched => format!("{tp} IS NULL"),
                    ClauseKind::NotMatchedBySource => {
                        format!("{tp} IS NOT NULL AND {sp} IS NULL")
                    }
                };
                Ok(expr(&kind)?.and(hit(c)))
            })
            .collect::<Result<_, KrishivError>>()?;
        // The clause that fires for a joined row (its index), -1 to copy a target row, -2 to drop
        // a source row.
        let action = first_of(
            guards
                .iter()
                .zip(0_i64..)
                .map(|(g, i)| (g.clone(), lit(i)))
                .collect(),
            call_function(
                "if",
                [
                    expr(&format!("{tp} IS NOT NULL"))?,
                    lit(-1_i64),
                    lit(-2_i64),
                ],
            ),
        );

        // 2. The touched files and the counts.
        let find_started = std::time::Instant::now();
        let found = if insert_only {
            Found {
                per_clause: vec![0; resolved.len()],
                ..Found::default()
            }
        } else {
            let (_, whole) = target_frame(None);
            let how = match (inserts, by_source) {
                (true, true) => JoinType::FullOuter,
                (true, false) => JoinType::RightOuter,
                (false, true) => JoinType::LeftOuter,
                (false, false) => JoinType::Inner,
            };
            let touch = if !inserts && !by_source {
                // Every clause is `MATCHED`: a file is touched by a row some clause holds for.
                let any = resolved
                    .iter()
                    .map(hit)
                    .reduce(Column::or)
                    .unwrap_or_else(|| lit(false));
                expr(&format!("{sp} IS NOT NULL"))?.and(any)
            } else {
                expr(&format!("{tp} IS NOT NULL"))?
            };
            let insert_codes: Vec<String> = resolved
                .iter()
                .zip(0_i64..)
                .filter(|(c, _)| c.insert)
                .map(|(_, i)| i.to_string())
                .collect();
            let rows = whole.join(&s, on.clone(), how).select([
                expr(&qualified(&m.target_alias, ROW_FILE_COLUMN))?.alias("f"),
                expr(&qualified(&m.target_alias, ROW_POSITION_COLUMN))?.alias("i"),
                expr(&sp)?.alias("sp"),
                action.clone().alias("act"),
                call_function("if", [touch, lit(1_i64), lit(0_i64)]).alias("touch"),
            ]);
            let ins = if insert_codes.is_empty() {
                "0".to_owned()
            } else {
                format!("if(act IN ({}), 1, 0)", insert_codes.join(", "))
            };
            let per_row = rows.group_by([col("f"), col("i")]).agg([
                crate::functions::count(col("sp")).alias("pairs"),
                crate::functions::max(col("touch")).alias("touch"),
                crate::functions::max(col("act")).alias("act"),
                crate::functions::sum(expr(&ins)?).alias("ins"),
            ]);
            let per_file = per_row.group_by([col("f"), col("act")]).agg([
                expr("count(*)")?.alias("rows"),
                crate::functions::max(col("touch")).alias("touch"),
                crate::functions::sum(expr("if(f IS NOT NULL AND pairs > 1, 1, 0)")?)
                    .alias("multiple"),
                crate::functions::sum(col("ins")).alias("ins"),
            ]);
            found_of(&per_file.collect().await?, resolved.len())?
        };
        let scan_ms = find_started.elapsed().as_millis();

        // 3. The multiple-match check (Delta's rule; an insert-only `MERGE` skips it).
        let exempt = {
            let mut m_clauses = resolved.iter().filter(|c| c.kind == ClauseKind::Matched);
            matches!(
                (m_clauses.next(), m_clauses.next()),
                (Some(c), None) if c.delete && c.condition.is_none()
            )
        };
        if found.multiple > 0 && !exempt {
            return Err(
                krishiv_common::error::DeltaMultipleSourceRowMatchingTargetRowInMerge {
                    usage_reference: super::merge_checks::USAGE_REFERENCE.to_owned(),
                }
                .into(),
            );
        }

        // 4. The rewrite.
        let (target_name, rewrite) = if insert_only {
            // Every row of the anti join is a source row no target row matches: the clauses'
            // own conditions decide.
            let (name, whole) = target_frame(None);
            let rows = s.join(&whole, on, JoinType::LeftAnti);
            let guards: Vec<Column> = resolved.iter().map(hit).collect();
            let keep = first_of(
                guards.iter().map(|g| (g.clone(), lit(true))).collect(),
                lit(false),
            );
            (
                name,
                merge_project(&rows.filter(keep), &resolved, &guards, &columns, None)?,
            )
        } else {
            let (name, touched) = target_frame(Some(found.touched.clone()));
            let how = if inserts {
                JoinType::FullOuter
            } else {
                JoinType::LeftOuter
            };
            let rows = touched.join(&s, on, how);
            let keep = first_of(
                resolved
                    .iter()
                    .zip(&guards)
                    .map(|(c, g)| (g.clone(), lit(!c.delete)))
                    .collect(),
                expr(&format!("{tp} IS NOT NULL"))?,
            );
            (
                name,
                merge_project(
                    &rows.filter(keep),
                    &resolved,
                    &guards,
                    &columns,
                    Some(&m.target_alias),
                )?,
            )
        };
        let analysis = rewrite.action_analysis().await?;
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        let target = find_scan(physical.root(), &target_name)
            .ok_or_else(|| internal("a MERGE rewrite that reads no target".to_owned()))?;
        // The rewrite's own read of the target is the hidden provider: a scan of the target in it
        // is an assignment subquery's read of the table. The source is read before the rewrite (a
        // snapshot or a spilled copy), so its own plan is asked too.
        let source_analysis = source.action_analysis().await?;
        let rereads = |plan: &krishiv_plan::LogicalPlan, catalog: &crate::catalog::QueryCatalog| {
            #[cfg(feature = "delta")]
            if let Some(root) = &delta_root {
                return crate::catalog::reads_delta_root(plan, root, catalog);
            }
            let _ = catalog;
            crate::catalog::reads_source(plan, &scan.source)
        };
        let rereads_target = rereads(analysis.analyzed.plan.root(), &analysis.env.catalog)
            || rereads(
                source_analysis.analyzed.plan.root(),
                &source_analysis.env.catalog,
            );
        // The conflict check reads the table by the `ON` condition's target conjuncts; a `NOT
        // MATCHED BY SOURCE` clause reads every row, and so does a rewrite that reads the target
        // again (as an `UPDATE` does: `row_level.rs` `read_filter`).
        let read_filter = if by_source || rereads_target {
            None
        } else {
            Some(Arc::new(on_of(&m.on)?))
        };
        let mut request = RowLevelWrite::new(
            RowLevelOperation::Merge,
            m.target.clone(),
            Arc::clone(physical.schema()),
            found.touched.clone(),
            read_filter,
        );
        request.write_id = Some(crate::session::job_uuid());
        let count_of = |pick: &dyn Fn(&Resolved) -> bool| -> u64 {
            resolved
                .iter()
                .zip(&found.per_clause)
                .filter(|(c, _)| pick(c))
                .map(|(_, n)| *n)
                .sum()
        };
        let matched_updated =
            count_of(&|c| c.kind == ClauseKind::Matched && !c.delete && !c.insert);
        let matched_deleted = count_of(&|c| c.kind == ClauseKind::Matched && c.delete);
        let by_source_updated =
            count_of(&|c| c.kind == ClauseKind::NotMatchedBySource && !c.delete);
        let by_source_deleted = count_of(&|c| c.kind == ClauseKind::NotMatchedBySource && c.delete);
        let counts = serde_json::json!({
            "source_rows": source_rows,
            "matched_updated": matched_updated,
            "matched_deleted": matched_deleted,
            "not_matched_by_source_updated": by_source_updated,
            "not_matched_by_source_deleted": by_source_deleted,
            "inserted": (!insert_only).then_some(found.inserted),
            "materialize_ms": u64::try_from(materialize_ms).unwrap_or(u64::MAX),
        });
        let clauses: Vec<serde_json::Value> = m
            .clauses
            .iter()
            .map(|c| {
                serde_json::json!({
                    "kind": c.kind.json(),
                    "action": c.action.json(),
                    "condition": c.condition,
                })
            })
            .collect();
        let options = [
            (COUNTS_OPTION, counts.to_string()),
            (
                CLAUSES_OPTION,
                serde_json::Value::Array(clauses).to_string(),
            ),
            (CONDITION_OPTION, m.on.clone()),
            (SCAN_MS_OPTION, scan_ms.to_string()),
            (ALIAS_OPTION, m.target_alias.clone()),
        ];
        for (k, v) in options {
            request.options.insert(k.to_owned(), v);
        }
        let plan = physical.into_table_write(target, TableWriteRequest::RowLevel(request));
        let written = async {
            let mut stream = self.run(&plan, &analysis.env).await?;
            while let Some(batch) = stream.next().await {
                batch?;
            }
            Ok::<(), KrishivError>(())
        }
        .await;
        if let Err(e) = written {
            return Err(super::row_level::write_failed(
                &table,
                RowLevelOperation::Merge,
                e,
                &columns,
                &value_types,
            ));
        }
        self.refresh_files();

        let mut out = RowLevelCounts::affected(0);
        out.updated = matched_updated + by_source_updated;
        out.deleted = matched_deleted + by_source_deleted;
        out.inserted = if insert_only { 0 } else { found.inserted };
        out.affected = out.updated + out.deleted + out.inserted;
        match table.output(&out) {
            Some(batch) => self
                .script_result(batch.schema(), vec![batch])
                .await
                .map(Some),
            None => self.empty_frame().await.map(Some),
        }
    }
}

/// The written rows of `rows` (the join, filtered): each target column the value of the first
/// clause that fires (an `UPDATE`'s value or the target's own, an `INSERT`'s value or `NULL`),
/// else the target's own (`target`: the target's alias; `None`: no target row is read, an
/// insert-only `MERGE`).
fn merge_project(
    rows: &DataFrame,
    resolved: &[Resolved],
    guards: &[Column],
    columns: &[TableColumn],
    target: Option<&str>,
) -> Result<DataFrame, KrishivError> {
    let mut select = Vec::with_capacity(columns.len());
    for (i, c) in columns.iter().enumerate() {
        let null = || {
            Column::new(krishiv_plan::Expr::literal(krishiv_plan::Literal::Null(
                Some(c.data_type.clone()),
            )))
        };
        let own = match target {
            Some(t) => expr(&qualified(t, &c.name))?,
            None => null(),
        };
        let mut branches = Vec::new();
        for (clause, guard) in resolved.iter().zip(guards) {
            if clause.delete {
                continue;
            }
            let value = match clause.exprs.get(i).cloned().flatten() {
                Some(v) => Column::new(v).cast_ansi(c.data_type.clone()),
                None if clause.insert => null(),
                None => own.clone(),
            };
            branches.push((guard.clone(), value));
        }
        select.push(first_of(branches, own).alias(&c.name));
    }
    Ok(rows.select(select))
}

impl Session {
    /// The frame of `query`, analysed from inside a statement `sql` is already running. Boxed
    /// as `dyn` because `sql` awaits statements that await this.
    fn nested_frame<'a>(
        &'a self,
        query: &'a str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<DataFrame, KrishivError>> + Send + 'a>> {
        Box::pin(async move { self.sql(query).await })
    }
}

/// The scan of the hidden table `name` in `node`.
fn find_scan(node: &PhysNode, name: &str) -> Option<krishiv_plan::Scan> {
    match node {
        PhysNode::Scan(s) => match &s.scan.source {
            krishiv_plan::ReadSource::Table { name: parts, .. }
                if parts.last().is_some_and(|p| p == name) =>
            {
                Some(s.scan.clone())
            }
            _ => None,
        },
        _ => node.children().into_iter().find_map(|c| find_scan(c, name)),
    }
}

/// The `ON` condition as the table's conflict check takes it (parsed, unresolved).
fn on_of(on: &str) -> Result<krishiv_plan::Expr, KrishivError> {
    Ok(expr(on)?.expr)
}

/// The find query's rows (`f`, `act`, `rows`, `touch`, `multiple`, `ins`) folded.
fn found_of(batches: &[RecordBatch], clauses: usize) -> Result<Found, KrishivError> {
    let mut found = Found {
        per_clause: vec![0; clauses],
        ..Found::default()
    };
    let mut touched: BTreeSet<Vec<u8>> = BTreeSet::new();
    let long = |b: &RecordBatch, i: usize| -> Result<Vec<Option<i64>>, KrishivError> {
        let c = b
            .column(i)
            .as_primitive_opt::<Int64Type>()
            .ok_or_else(|| internal("a MERGE count that is not a long".to_owned()))?;
        Ok((0..c.len())
            .map(|r| c.is_valid(r).then(|| c.value(r)))
            .collect())
    };
    let n = |v: Option<i64>| u64::try_from(v.unwrap_or(0)).unwrap_or(0);
    for b in batches {
        let file = b.column(0);
        let files: Vec<Option<Vec<u8>>> = match file.data_type() {
            ArrowType::Binary => file
                .as_binary::<i32>()
                .iter()
                .map(|v| v.map(<[u8]>::to_vec))
                .collect(),
            ArrowType::BinaryView => file
                .as_binary_view()
                .iter()
                .map(|v| v.map(<[u8]>::to_vec))
                .collect(),
            other => return Err(internal(format!("a touched file column of type {other}"))),
        };
        let (act, rows, touch, multiple, ins) = (
            long(b, 1)?,
            long(b, 2)?,
            long(b, 3)?,
            long(b, 4)?,
            long(b, 5)?,
        );
        for (r, f) in files.into_iter().enumerate() {
            let get = |v: &[Option<i64>]| v.get(r).copied().flatten();
            match f {
                None => found.inserted += n(get(&ins)),
                Some(f) => {
                    if get(&touch).unwrap_or(0) > 0 {
                        touched.insert(f);
                    }
                    found.multiple += n(get(&multiple));
                    if let Some(a) = get(&act).and_then(|a| usize::try_from(a).ok())
                        && let Some(slot) = found.per_clause.get_mut(a)
                    {
                        *slot += n(get(&rows));
                    }
                }
            }
        }
    }
    found.touched = touched.into_iter().collect();
    Ok(found)
}

#[cfg(all(test, feature = "delta"))]
#[test]
fn the_delta_connector_reads_the_same_merge_option_keys() {
    use krishiv_connectors_delta::write::merge as d;
    assert_eq!(COUNTS_OPTION, d::MERGE_COUNTS_OPTION);
    assert_eq!(CLAUSES_OPTION, d::MERGE_CLAUSES_OPTION);
}

impl Session {
    /// `MERGE INTO`: run through the target's row-level table when its provider offers one for
    /// `MERGE`, else Spark's v1 refusal after the table resolves.
    pub(crate) async fn exec_merge(
        &self,
        m: &krishiv_sql::ast::Merge,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        use krishiv_sql::ast::{MergeAction, MergeInsert, MergeMatched, MergeSource};
        let env = self.query_env();
        // What Spark analyzes before the target: the `ON` and clause expressions' subqueries and a
        // source that is a subquery (a time travel clause there is the error).
        let more: Vec<&krishiv_sql::ast::Expr> = std::iter::once(&m.on)
            .chain(m.clauses.iter().flat_map(|c| {
                let action: Vec<&krishiv_sql::ast::Expr> = match &c.action {
                    MergeAction::Update(assignments) => {
                        assignments.iter().map(|a| &a.value).collect()
                    }
                    MergeAction::Insert(MergeInsert::Values { values, .. }) => {
                        values.iter().collect()
                    }
                    _ => Vec::new(),
                };
                c.condition.iter().chain(action)
            }))
            .collect();
        let sources: Vec<&krishiv_sql::ast::Query> = match &m.source {
            MergeSource::Query { query, .. } => vec![&**query],
            _ => Vec::new(),
        };
        let changes = super::RowChanges {
            targets: &[],
            values: &[],
            condition: None,
            options: &m.options,
            more: &more,
            sources: &sources,
        };
        // The target's `WITH (...)` options reach no table of the session catalog: refused there.
        if !m.options.is_empty() {
            return self
                .exec_row_level_change(&m.target, None, "MERGE INTO TABLE", &changes, original)
                .await;
        }
        let target = self
            .row_level_target_checked(&m.target, "MERGE INTO TABLE", &env, &changes, original)
            .await?;
        let (name, quoted) = target.clone();
        let slice = |span: krishiv_sql::ast::Span| {
            original
                .get(span.start..span.end)
                .unwrap_or_default()
                .to_owned()
        };
        let target_alias = m
            .target_alias
            .as_ref()
            .map_or_else(|| name.name.clone(), |a| a.value.clone());
        let (source, source_alias) = match &m.source {
            MergeSource::Table {
                name,
                options,
                alias,
                ..
            } => {
                let last = match name {
                    krishiv_sql::ast::DdlName::Parts(p) => {
                        p.parts.last().map(|i| i.value.clone()).unwrap_or_default()
                    }
                    other => slice(other.span()),
                };
                // The source is a plain read with its own `WITH (...)` options, as Spark reads it
                // (AstBuilder.scala 1286-1290 at v4.3.0): each option as written.
                let with = if options.is_empty() {
                    String::new()
                } else {
                    let written: Vec<String> = options.iter().map(|o| slice(o.span)).collect();
                    format!(" WITH ({})", written.join(", "))
                };
                (
                    format!("SELECT * FROM {}{with}", slice(name.span())),
                    alias.as_ref().map_or(last, |a| a.value.clone()),
                )
            }
            MergeSource::Query { query, alias, .. } => (
                slice(query.span),
                alias.as_ref().map_or_else(
                    || "__auto_generated_subquery_name".to_owned(),
                    |a| a.value.clone(),
                ),
            ),
            _ => return Err(internal("a MERGE source of an unknown form".to_owned())),
        };
        let key = |target: &krishiv_sql::ast::ObjectName| {
            // `SET t.c = ..`: the qualifier names the target (its alias or table name).
            let mut parts: Vec<&str> = target.parts.iter().map(|p| p.value.as_str()).collect();
            if parts.len() > 1
                && (parts[0].eq_ignore_ascii_case(&target_alias)
                    || parts[0].eq_ignore_ascii_case(&name.name))
            {
                parts.remove(0);
            }
            parts.join(".")
        };
        let mut clauses = Vec::with_capacity(m.clauses.len());
        for c in &m.clauses {
            let kind = match c.matched {
                MergeMatched::Matched => ClauseKind::Matched,
                MergeMatched::NotMatchedBySource => ClauseKind::NotMatchedBySource,
                _ => ClauseKind::NotMatched,
            };
            let action = match &c.action {
                MergeAction::Update(assignments) => ClauseAction::Update(
                    assignments
                        .iter()
                        .map(|a| Assignment {
                            column: key(&a.target),
                            value: (!crate::command::is_bare_default(&a.value))
                                .then(|| slice(a.value.span)),
                        })
                        .collect(),
                ),
                MergeAction::UpdateStar => ClauseAction::UpdateStar,
                MergeAction::Delete => ClauseAction::Delete,
                MergeAction::Insert(MergeInsert::Star) => ClauseAction::InsertStar,
                MergeAction::Insert(MergeInsert::Values {
                    columns, values, ..
                }) => ClauseAction::Insert(
                    columns
                        .iter()
                        .zip(values)
                        .map(|(c, v)| Assignment {
                            column: c
                                .parts
                                .iter()
                                .map(|p| p.value.as_str())
                                .collect::<Vec<_>>()
                                .join("."),
                            value: (!crate::command::is_bare_default(v)).then(|| slice(v.span)),
                        })
                        .collect(),
                ),
                _ => return Err(internal("a MERGE action of an unknown form".to_owned())),
            };
            clauses.push(MergeClause {
                kind,
                condition: c.condition.as_ref().map(|e| slice(e.span)),
                action,
            });
        }
        let statement = MergeStatement {
            target: quoted.clone(),
            target_alias,
            source,
            source_alias,
            on: slice(m.on.span),
            clauses,
        };
        // The statement's hidden tables (the target's scans, the source's copy) go at its end,
        // a cancelled one's too.
        let ran = super::row_level::Swept::new(
            Arc::clone(&self.state.catalog),
            self.run_merge(&statement),
        )
        .await;
        if let Some(done) = ran? {
            // The table's data changed: what read it is stale (as after an `INSERT`).
            self.state.catalog.store.bump();
            self.forget_column_stats(&name);
            self.forget_cached(&name);
            return Ok(done);
        }
        // No MERGE-capable row-level table: the refusal path, on the target already checked.
        self.exec_row_level_change_on(target, None, "MERGE INTO TABLE", &changes, original)
            .await
    }
}

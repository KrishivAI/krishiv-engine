// SPDX-License-Identifier: Apache-2.0
//! Late fetch (design P2-T063 §4): an `ORDER BY ... LIMIT` over a file scan
//! that ranks a narrow scan and reads the other columns of the rows it keeps.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::config::{ConfigValue, LATE_FETCH_MAX_ROWS};
use krishiv_io::ColumnPath;
use krishiv_plan::{AttrId, Expr, ExprNode, LogicalPlan, PlanNode, ResolvedExpr, SortOrder};

use super::build::{Planned, Planner, columns, has_subquery, internal, select, source_name};
use crate::error::Result;
use crate::operator::Source;
use crate::operators::{FetchSlot, LateFetch};
use crate::physical::{PhysNode, ScanNode};

impl Planner<'_> {
    /// `ORDER BY order LIMIT n OFFSET skip` over a scan that reads only the
    /// sort keys and the filter's columns, and fetches the other columns for
    /// the rows kept, when that's cheaper (design P2-T063 §4).
    pub(super) fn late_fetch(
        &mut self,
        sorted: &Arc<LogicalPlan>,
        order: &[SortOrder],
        skip: u64,
        n: u64,
        deterministic: bool,
    ) -> Result<Option<Planned>> {
        // A projection of plain columns only picks columns of the fetch's output.
        let (below, picked) = match &sorted.node {
            PlanNode::Project { input, exprs }
                if exprs
                    .iter()
                    .all(|e| matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Column(_)))) =>
            {
                (input, sorted.resolved_output())
            }
            _ => (sorted, None),
        };
        let (scan, filter) = match &below.node {
            PlanNode::Scan(scan) => (scan, None),
            PlanNode::Filter { input, predicate } => match &input.node {
                PlanNode::Scan(scan) if !has_subquery(predicate) => (scan, Some(predicate)),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        let max_rows = self
            .cx
            .config
            .get(&LATE_FETCH_MAX_ROWS)
            .and_then(ConfigValue::as_int)
            .and_then(|m| u64::try_from(m).ok())
            .unwrap_or(0);
        let total = n.saturating_add(skip);
        if n == 0 || total > max_rows || order.iter().any(|o| has_subquery(&o.expr)) {
            return Ok(None);
        }
        // The hidden `_metadata` column is made by the file scan, which a
        // late fetch of the other columns would bypass.
        if scan.metadata_column().is_some() {
            return Ok(None);
        }
        let provider = self.cx.tables.provider(scan)?;
        let factory = provider.scan()?;
        if !factory.supports_take() {
            return Ok(None);
        }
        // A fetch costs every deferred chunk of each unit it touches, a plain
        // scan at most those of all units.
        let Some(units) = provider.scan_units() else {
            return Ok(None);
        };
        if n.min(units) > units / 4 {
            return Ok(None);
        }
        let mut read = HashSet::new();
        for e in order.iter().map(|o| &o.expr).chain(filter) {
            columns_read(e, &mut read);
        }
        let (kept, deferred): (Vec<_>, Vec<_>) = scan
            .output
            .iter()
            .cloned()
            .partition(|c| read.contains(&c.id) || !factory.in_files(&c.name));
        if deferred.is_empty() {
            return Ok(None);
        }
        let narrow = krishiv_plan::Scan {
            output: kept,
            ..scan.clone()
        };
        let input = self.scan_reading(below, &narrow, filter, true)?;
        let width = narrow.output.len();
        let (Some(&file_attr), Some(&row_attr)) =
            (input.layout.get(width), input.layout.get(width + 1))
        else {
            return Err(internal("a scan without its row id"));
        };
        let mut top = self.top_n(input, order, total, deterministic)?;
        if skip != 0 {
            top = Self::skip_rows(top, skip, n);
        }
        let file_col = top.column(file_attr)?;
        let row_col = top.column(row_attr)?;
        let mut fetched = 0;
        let output = scan
            .output
            .iter()
            .map(|c| {
                if narrow.output.iter().any(|k| k.id == c.id) {
                    return Ok(FetchSlot::Input(top.column(c.id)?));
                }
                fetched += 1;
                Ok(FetchSlot::Fetched(fetched - 1))
            })
            .collect::<Result<Vec<_>>>()?;
        let projection: Vec<ColumnPath> = deferred
            .iter()
            .map(|c| ColumnPath::new(c.name.clone()))
            .collect();
        let schema = LateFetch::new(
            &top.schema,
            factory,
            projection.clone(),
            file_col,
            row_col,
            output.clone(),
        )?
        .schema();
        let detail = format!("{} [{}]", source_name(scan), columns(&deferred));
        let layout = scan.output.iter().map(|c| c.id).collect();
        let fetch = ScanNode {
            id: self.next_scan_id()?,
            scan: krishiv_plan::Scan {
                output: deferred,
                ..scan.clone()
            },
            projection,
            filter: None,
            runtime_filters: Vec::new(),
            row_stage: self.rf.row_stage(),
            row_id: false,
        };
        let fetched = top.wrap(
            |input| PhysNode::LateFetch {
                input,
                scan: fetch,
                file_col,
                row_col,
                output,
            },
            schema,
            layout,
            ("LateFetch", detail),
        );
        Ok(Some(match picked {
            Some(columns) => select(fetched, &columns, "Project")?,
            None => fetched,
        }))
    }
}

fn columns_read(e: &Expr, out: &mut HashSet<AttrId>) {
    if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node {
        out.insert(c.id);
    }
    e.for_each_child(&mut |c| columns_read(c, out));
}

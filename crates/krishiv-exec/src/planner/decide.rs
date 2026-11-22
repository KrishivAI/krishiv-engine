// SPDX-License-Identifier: Apache-2.0
//! Offering a scan's filter to a source that decides once (`ScanPushdown::decide`), and the residual left over.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_io::ColumnPath;
use krishiv_plan::{AttrId, Expr};

use super::build::*;
use crate::error::Result;
use crate::physical::scan_filter;

impl<'a> Planner<'a> {
    /// Offers every conjunct of `predicate` to a scan that decides once, and binds the read it planned.
    /// The residual is Spark's: the conjuncts the scan did not take that translate, then those that do
    /// not, then those holding a subquery.
    pub(super) fn decide_scan(
        &mut self,
        factory: &dyn crate::operators::ScanPushdown,
        table: &SchemaRef,
        scan: &krishiv_plan::Scan,
        predicate: &Arc<Expr>,
        projection: &[ColumnPath],
    ) -> Result<Decision> {
        use crate::operators::{DecidedScan, OfferedConjunct, SourceConjunct};
        let layout: Vec<AttrId> = table
            .fields()
            .iter()
            .map(|f| match scan.output.iter().find(|c| c.name == *f.name()) {
                Some(c) => c.id,
                None => self.fresh_id(),
            })
            .collect();
        let mut offered = Vec::new();
        let mut exprs: Vec<Expr> = Vec::new();
        // The conjunct each offered one comes from: a split `LIKE` offers three.
        let mut group: Vec<usize> = Vec::new();
        let mut groups = 0usize;
        let mut subqueries = Vec::new();
        let written: Vec<&Expr> = krishiv_plan::conjuncts(predicate)
            .into_iter()
            .filter(|c| !krishiv_plan::has_subquery(c))
            .collect();
        // Spark infers `IsNotNull` of the columns the comparisons reject nulls of, all ahead of the
        // written conjuncts (the oracle's `four_conjunct`).
        for c in &written {
            for inferred in super::source_filter::inferred_not_null(c, &scan.output) {
                let known = |e: &Expr| {
                    exprs
                        .iter()
                        .chain(written.iter().copied())
                        .any(|k| e.to_string() == k.to_string())
                };
                if !known(&inferred) {
                    offered.push(OfferedConjunct {
                        filter: scan_filter(table, &layout, &inferred, scan, &self.env)?,
                        source: super::source_filter::classify(&inferred, &scan.output),
                    });
                    exprs.push(inferred);
                    group.push(groups);
                    groups += 1;
                }
            }
        }
        for c in krishiv_plan::conjuncts(predicate) {
            if krishiv_plan::has_subquery(c) {
                subqueries.push(c);
                continue;
            }
            let filter = scan_filter(table, &layout, c, scan, &self.env)?;
            // A split `LIKE` is two leaves plus Spark's `length(x) >= n` guard, which is no leaf: the
            // `LIKE` itself stays above the scan as that guard.
            let mut sources = match super::source_filter::split_like(c, &scan.output) {
                Some(two) => two.to_vec(),
                None => vec![super::source_filter::classify(c, &scan.output)],
            };
            if sources.len() == 2 {
                sources.push(SourceConjunct::Untranslated);
            }
            let this = groups;
            groups += 1;
            for source in sources {
                offered.push(OfferedConjunct {
                    filter: filter.clone(),
                    source,
                });
                exprs.push(c.clone());
                group.push(this);
            }
        }
        let decision = factory.decide(&offered, projection)?;
        if decision.pushed.len() != offered.len() {
            return Err(internal(
                "a scan decided on a different number of conjuncts",
            ));
        }
        let id = self.cx.tables.bind_scan(Arc::clone(&decision.bound))?;
        let taken = |i: usize| decision.pushed.get(i).copied().unwrap_or(false);
        let (mut kept, mut untranslated): (Vec<&Expr>, Vec<&Expr>) = (Vec::new(), Vec::new());
        let mut placed: Vec<usize> = Vec::new();
        for (i, (o, e)) in offered.iter().zip(&exprs).enumerate() {
            if taken(i) {
                continue;
            }
            // A split `LIKE` is offered three times; it stays once.
            if placed.contains(&group[i]) {
                continue;
            }
            placed.push(group[i]);
            match o.source {
                SourceConjunct::Untranslated => untranslated.push(e),
                _ => kept.push(e),
            }
        }
        kept.extend(untranslated);
        kept.extend(subqueries);
        let pushed: Vec<u32> = (0..offered.len())
            .filter(|i| taken(*i))
            .filter_map(|i| u32::try_from(i).ok())
            .collect();
        Ok(Decision {
            shown: pushed
                .iter()
                .filter_map(|i| exprs.get(*i as usize))
                .map(ToString::to_string)
                .collect(),
            decided: DecidedScan { id, pushed },
            residual: super::source_filter::and_all(kept),
            bound: decision.bound,
        })
    }
}

/// What a scan that decides once answered for one filter: see `Builder::decide_scan`.
pub(super) struct Decision {
    pub(super) decided: crate::operators::DecidedScan,
    pub(super) bound: Arc<dyn crate::operators::BoundScan>,
    /// The pushed conjuncts as `EXPLAIN` shows them.
    pub(super) shown: Vec<String>,
    pub(super) residual: Option<Expr>,
}

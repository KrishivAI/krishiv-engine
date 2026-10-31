// SPDX-License-Identifier: Apache-2.0
//! The planner reads one `PlanStats` per query: a relation's statistics are
//! asked of its table once, however many joins and runtime filters cost it
//! (before, each join estimated both inputs again and `has_statistics`
//! walked the subtree once more).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use krishiv_exec::planner::{MemoryTable, MemoryTables, TableProvider};
use krishiv_optimizer::TableStats;
use krishiv_plan::analyzer::testing::context;
use krishiv_plan::analyzer::{MemoryCatalog, ResolutionConf, TableColumn};

use super::{Env, table};

/// A table that counts how often its statistics are read.
#[derive(Debug)]
struct Counted {
    inner: MemoryTable,
    reads: Arc<AtomicUsize>,
}

impl TableProvider for Counted {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn krishiv_exec::ScanPushdown>> {
        self.inner.scan()
    }

    fn statistics(&self) -> Option<TableStats> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.statistics()
    }
}

/// Tables `t1`, `t2`, `t3` (the rows of [`table`], statistics of 1,000,
/// 20,000 and 10 million rows) and the counters of their statistics reads.
fn env() -> (Env, Vec<Arc<AtomicUsize>>) {
    let (schema, batches) = table(10);
    let columns: Vec<TableColumn> = schema
        .fields()
        .iter()
        .map(|f| TableColumn {
            name: f.name().clone(),
            data_type: krishiv_types::DataType::from_arrow(f.data_type()).unwrap(),
            nullable: f.is_nullable(),
        })
        .collect();
    let mut catalog = MemoryCatalog::new("spark_catalog");
    let mut tables = MemoryTables::new();
    let mut counters = Vec::new();
    for (name, rows) in [("t1", 1_000), ("t2", 20_000), ("t3", 10_000_000)] {
        let reads = Arc::new(AtomicUsize::new(0));
        catalog = catalog.with_table(&["default"], name, columns.clone(), None);
        let inner = MemoryTable::new(Arc::clone(&schema), batches.clone()).with_statistics(Some(
            TableStats {
                rows,
                columns: HashMap::new(),
            },
        ));
        let counted = Counted {
            inner,
            reads: Arc::clone(&reads),
        };
        tables = tables.with_table(&["spark_catalog", "default", name], Arc::new(counted));
        counters.push(reads);
    }
    let env = Env {
        ctx: context(catalog, ResolutionConf::default()),
        tables: Arc::new(tables),
    };
    (env, counters)
}

fn reads(counters: &[Arc<AtomicUsize>]) -> Vec<usize> {
    counters.iter().map(|c| c.load(Ordering::SeqCst)).collect()
}

#[test]
fn a_multi_join_query_reads_each_tables_statistics_once() {
    let (env, counters) = env();
    let plan = env
        .plan(
            "SELECT t1.a FROM t1 JOIN t2 ON t1.a = t2.a JOIN t3 ON t2.a = t3.a \
             WHERE t3.b > 30 AND t1.b < 50",
        )
        .unwrap();
    let text = plan.to_string();
    assert_eq!(text.matches("HashJoin").count(), 2, "{text}");
    assert_eq!(reads(&counters), [1, 1, 1], "{text}");
}

#[test]
fn a_subquery_and_a_self_join_read_once_per_relation() {
    let (env, counters) = env();
    // The self join reads one shared scan of `t2` through two instances:
    // one relation, one read. The subquery's plan is read once too.
    let plan = env
        .plan(
            "SELECT x.a FROM t2 x JOIN t2 y ON x.a = y.a \
             WHERE x.a IN (SELECT a FROM t1 WHERE b > 20)",
        )
        .unwrap();
    let text = plan.to_string();
    assert!(text.contains("HashJoin"), "{text}");
    assert_eq!(reads(&counters), [1, 1, 0], "{text}");
}

#[test]
fn the_estimates_still_choose_how_a_join_runs() {
    // The estimated size of the build decides broadcast or shuffle: the
    // 1,000-row table broadcasts, the 10-million-row one is partitioned.
    let (env, _) = env();
    let small = env
        .plan("SELECT t1.a FROM t3 JOIN t1 ON t1.a = t3.a")
        .unwrap()
        .to_string();
    assert!(small.contains("BroadcastExchange"), "{small}");
    let big = env
        .plan("SELECT t1.a FROM t1 JOIN t3 ON t1.a = t3.a")
        .unwrap()
        .to_string();
    assert!(!big.contains("BroadcastExchange"), "{big}");
    assert!(big.contains("hashpartitioning"), "{big}");
}

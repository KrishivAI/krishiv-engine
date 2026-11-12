// SPDX-License-Identifier: Apache-2.0
//! P3 W1 shape (a): a correlated aggregate subquery over the outer block's own relation runs as
//! a window, with the same rows either way (duplicated dimension keys included: form M).

use krishiv_common::config::Config;

use super::*;

fn env_tp() -> Env {
    let t_schema = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("b", ArrowType::Int64, false),
    ]));
    let a: Int32Array = (0..40).map(|i| (i % 9 != 0).then_some(i % 5)).collect();
    let b: Int64Array = (0..40).map(|i| i64::from(i * 7 % 11)).collect();
    let t = RecordBatch::try_new(Arc::clone(&t_schema), vec![Arc::new(a), Arc::new(b)]).unwrap();
    let p_schema = Arc::new(Schema::new(vec![
        Field::new("pk", ArrowType::Int32, false),
        Field::new("flag", ArrowType::Int32, false),
    ]));
    // Keys 0..4 each once, 1 and 3 twice (the multiplicity form M must survive).
    let pk: Int32Array = [0, 1, 1, 2, 3, 3, 4].into_iter().collect();
    let flag: Int32Array = [1, 1, 1, 0, 1, 1, 1].into_iter().collect();
    let p =
        RecordBatch::try_new(Arc::clone(&p_schema), vec![Arc::new(pk), Arc::new(flag)]).unwrap();
    let cols = |s: &SchemaRef| -> Vec<TableColumn> {
        s.fields()
            .iter()
            .map(|f| TableColumn {
                name: f.name().clone(),
                data_type: krishiv_types::DataType::from_arrow(f.data_type()).unwrap(),
                nullable: f.is_nullable(),
                char_varchar: None,
            })
            .collect()
    };
    let catalog = MemoryCatalog::new("spark_catalog")
        .with_table(&["default"], "t", cols(&t_schema), None)
        .with_table(&["default"], "p", cols(&p_schema), None);
    let tables = MemoryTables::new()
        .with_table(
            &["spark_catalog", "default", "t"],
            Arc::new(MemoryTable::new(t_schema, vec![Batch::new(t)])),
        )
        .with_table(
            &["spark_catalog", "default", "p"],
            Arc::new(MemoryTable::new(p_schema, vec![Batch::new(p)])),
        );
    Env {
        ctx: context(catalog, ResolutionConf::default()),
        tables: Arc::new(tables),
    }
}

/// Row counts for the two tables (the rule fires only on estimates from statistics).
#[derive(Debug)]
struct Counts;

impl krishiv_optimizer::Statistics for Counts {
    fn table(&self, scan: &krishiv_plan::Scan) -> Option<krishiv_optimizer::TableStats> {
        let krishiv_plan::ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let rows = match name.last()?.as_str() {
            "t" => 40,
            "p" => 7,
            _ => return None,
        };
        Some(krishiv_optimizer::TableStats {
            rows,
            columns: std::collections::HashMap::new(),
        })
    }
}

fn rows_under(env: &Env, sql: &str, on: bool) -> (Vec<String>, bool) {
    let mut config = Config::new();
    if on {
        config
            .set("krishiv.optimizer.subqueryAsWindow", "true")
            .unwrap();
    }
    let analyzed = env.analyze(sql);
    let plan = krishiv_optimizer::optimize_with_udfs(
        &analyzed,
        &Counts,
        &krishiv_udf_sdk::UdfLibrary::new(),
        &config,
    )
    .unwrap();
    let windowed = plan.root().to_string().contains("Window [");
    let cx = PlanningContext::new(Arc::clone(&env.tables) as _, Arc::new(config));
    let runnable = env.runnable(DefaultPlanner.plan(&plan, &cx).unwrap());
    let batches = run(&runnable).unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    (rows, windowed)
}

fn same_rows(sql: &str) {
    let env = env_tp();
    let (off, w_off) = rows_under(&env, sql, false);
    let (on, w_on) = rows_under(&env, sql, true);
    assert!(!w_off, "off must not window");
    assert!(w_on, "on must window: {sql}");
    assert!(!off.is_empty());
    assert_eq!(off, on, "{sql}");
}

#[test]
fn min_over_the_dimension_s_keys_keeps_the_rows() {
    same_rows(
        "SELECT t.a, t.b FROM t, p WHERE p.pk = t.a AND p.flag = 1 \
         AND t.b <= (SELECT min(t2.b) + 3 FROM t t2 WHERE t2.a = p.pk)",
    );
}

#[test]
fn max_keeps_the_rows() {
    same_rows(
        "SELECT t.a, t.b FROM t, p WHERE p.pk = t.a AND p.flag = 1 \
         AND t.b >= (SELECT max(t2.b) - 2 FROM t t2 WHERE t2.a = p.pk)",
    );
}

#[test]
fn null_keys_of_a_self_correlated_subquery_are_dropped() {
    // The null partition must not be aggregated and kept (design 2.3).
    same_rows(
        "SELECT t.a, t.b FROM t WHERE t.b > 5 AND t.b <= (SELECT min(t2.b) + 3 FROM t t2 WHERE t2.a = t.a AND t2.b > 5)",
    );
}

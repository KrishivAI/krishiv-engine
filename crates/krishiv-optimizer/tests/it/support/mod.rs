// SPDX-License-Identifier: Apache-2.0
//! Shared test support: the TPC-H catalog and queries, resolved by the
//! analyzer.

use std::sync::Arc;

use krishiv_plan::analyzer::testing::{TestFunctions, block_on, context};
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, ArgType, DefaultAnalyzer, FunctionEntry, FunctionLookup,
    FunctionRegistry, Inputs, MemoryCatalog, ResolutionConf, ResultType, SparkProfile, TableColumn,
};
use krishiv_plan::{FunctionId, FunctionKind, Plan, SourceId};
use krishiv_types::DataType;
use serde_json::Value;

/// The workspace root.
pub(crate) fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The TPC-H tables (and the analyzer fixture's others) as Spark 4.2 typed them.
pub(crate) fn catalog() -> MemoryCatalog {
    let path = root().join("tests/fixtures/spark-4.2.0/analyzer-goldens.json");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut cat = MemoryCatalog::new("spark_catalog");
    for (name, cols) in g["tables"].as_object().unwrap() {
        let columns = cols
            .as_array()
            .unwrap()
            .iter()
            .map(|f| TableColumn {
                name: f["name"].as_str().unwrap().to_owned(),
                data_type: krishiv_types::parse_data_type(f["type"].as_str().unwrap()).unwrap(),
                nullable: f["nullable"].as_bool().unwrap(),
            })
            .collect();
        cat = cat.with_table(&["default"], name, columns, None);
    }
    cat
}

/// The test registry, plus `rand` and `first` (nondeterministic).
#[derive(Debug)]
struct Functions;

impl FunctionRegistry for Functions {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup {
        let (kind, result): (_, ResultType) = match name {
            [n] if n == "rand" => (
                FunctionKind::Scalar,
                Arc::new(|_| {
                    Ok(ArgType {
                        data_type: Some(DataType::Double),
                        nullable: false,
                    })
                }),
            ),
            // `first(expr)`: the first value of `expr`, of its type.
            [n] if n == "first" => (
                FunctionKind::Aggregate,
                Arc::new(|args| {
                    Ok(ArgType {
                        data_type: args.first().and_then(|a| a.data_type.clone()),
                        nullable: true,
                    })
                }),
            ),
            _ => return TestFunctions.lookup(name, profile),
        };
        FunctionLookup::Found(FunctionEntry {
            id: FunctionId::Registry {
                name: name[0].clone(),
                id: 1000,
            },
            kind,
            inputs: Inputs::Any,
            result,
        })
    }

    fn version(&self) -> u64 {
        0
    }
}

/// An analysis context over [`catalog`].
pub(crate) fn ctx() -> AnalysisContext {
    let mut ctx = context(catalog(), ResolutionConf::default());
    ctx.functions = Arc::new(Functions);
    ctx
}

/// Parses, lowers and resolves `sql`.
pub(crate) fn resolve(sql: &str, ctx: &AnalysisContext) -> Plan {
    let plan =
        krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    block_on(DefaultAnalyzer.analyze(&plan, ctx))
        .unwrap_or_else(|e| panic!("{sql}: {e:?}"))
        .plan
}

/// The text of TPC-H query `q` (1-based).
pub(crate) fn tpch_sql(q: usize) -> String {
    std::fs::read_to_string(root().join(format!("bench/queries/tpch/q{q:02}.sql"))).unwrap()
}

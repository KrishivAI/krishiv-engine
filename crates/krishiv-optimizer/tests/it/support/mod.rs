// SPDX-License-Identifier: Apache-2.0
//! Shared test support: the TPC-H catalog and queries, resolved by the
//! analyzer, a reference interpreter, generated data and a random number
//! generator.

pub(crate) mod data;
pub(crate) mod eval;

use std::sync::Arc;

use krishiv_plan::analyzer::testing::{TestFunctions, block_on, context};
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, ArgType, DefaultAnalyzer, FunctionEntry, FunctionLookup,
    FunctionRegistry, MemoryCatalog, ResolutionConf, SparkProfile, TableColumn,
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
        let (kind, data_type) = match name {
            [n] if n == "rand" => (FunctionKind::Scalar, DataType::Double),
            [n] if n == "first" => (FunctionKind::Aggregate, DataType::String),
            _ => return TestFunctions.lookup(name, profile),
        };
        FunctionLookup::Found(FunctionEntry {
            id: FunctionId::Registry {
                name: name[0].clone(),
                id: 1000,
            },
            kind,
            result: Arc::new(move |_| {
                Ok(ArgType {
                    data_type: Some(data_type.clone()),
                    nullable: true,
                })
            }),
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

/// A xorshift generator.
#[derive(Debug, Clone)]
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    /// The next value.
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `lo..=hi`.
    pub(crate) fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + i64::try_from(self.next() % u64::try_from(hi - lo + 1).unwrap()).unwrap()
    }

    /// One of `items`.
    pub(crate) fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[usize::try_from(self.range(0, i64::try_from(items.len()).unwrap() - 1)).unwrap()]
    }
}

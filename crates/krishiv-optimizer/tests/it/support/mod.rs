// SPDX-License-Identifier: Apache-2.0
//! Shared test support: the TPC-H catalog and queries, resolved by the
//! analyzer, a reference interpreter, generated data and a random number
//! generator.

pub(crate) mod data;
pub(crate) mod eval;

use std::sync::Arc;

use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    AnalysisContext, Analyzer as _, ArgType, Arity, BuiltinFunctions, CallTypes, DefaultAnalyzer,
    FunctionEntry, FunctionLookup, FunctionRegistry, Inputs, MemoryCatalog, ResolutionConf,
    ResultType, SparkProfile, TableColumn,
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

/// The built-in functions, plus two nondeterministic test doubles: `rand()`
/// (Spark's, not implemented yet) and the aggregate
/// `nondeterministic_first(expr)` (Spark 4.2's `first` is deterministic,
/// so no built-in aggregate is not).
#[derive(Debug)]
struct Functions;

impl FunctionRegistry for Functions {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup {
        let (kind, result): (_, ResultType) = match name {
            [n] if n == "rand" => (
                FunctionKind::Scalar,
                Arc::new(|_: &CallTypes<'_>| Ok(ArgType::new(DataType::Double, false))),
            ),
            [n] if n == "nondeterministic_first" => (
                FunctionKind::Aggregate,
                Arc::new(|c: &CallTypes<'_>| {
                    Ok(ArgType {
                        data_type: c.data_type(0).cloned(),
                        nullable: true,
                    })
                }),
            ),
            _ => return BuiltinFunctions::new().lookup(name, profile),
        };
        FunctionLookup::Found(FunctionEntry {
            id: FunctionId::Registry {
                name: name[0].clone(),
                id: 1000,
            },
            kind,
            arity: Arity::any(),
            inputs: Inputs::Any,
            result,
            deterministic: false,
        })
    }

    fn version(&self) -> u64 {
        0
    }
}

/// An analysis context over [`catalog`].
pub(crate) fn ctx() -> AnalysisContext {
    ctx_over(catalog())
}

/// An analysis context over `catalog`.
pub(crate) fn ctx_over(catalog: MemoryCatalog) -> AnalysisContext {
    let mut ctx = context(catalog, ResolutionConf::default());
    ctx.functions = Arc::new(Functions);
    ctx
}

/// Parses, lowers and resolves `sql`, or `None` when it does not resolve
/// (used to list queries a harness cannot cover yet).
pub(crate) fn try_resolve(sql: &str, ctx: &AnalysisContext) -> Option<Plan> {
    let plan =
        krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .ok()?;
    block_on(DefaultAnalyzer.analyze(&plan, ctx))
        .ok()
        .map(|r| r.plan)
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

/// Default row counts with every column's distinct count known, all values
/// distinct: the estimates `optimize` made before a literal equality on a
/// column without a distinct count took the default selectivity. Rule tests
/// that need a selective filter on a table without statistics plan under
/// them.
#[derive(Debug)]
pub(crate) struct AllDistinct;

impl krishiv_optimizer::Statistics for AllDistinct {
    fn table(&self, scan: &krishiv_plan::Scan) -> Option<krishiv_optimizer::TableStats> {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a constant"
        )]
        let rows = krishiv_optimizer::DEFAULT_ROWS as u64;
        let columns = scan.output.iter().map(|c| {
            let stats = krishiv_optimizer::ColumnStats {
                distinct: Some(rows),
                ..krishiv_optimizer::ColumnStats::default()
            };
            (c.name.clone(), stats)
        });
        Some(krishiv_optimizer::TableStats {
            rows,
            columns: columns.collect(),
        })
    }
}

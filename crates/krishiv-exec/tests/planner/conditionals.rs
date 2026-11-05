// SPDX-License-Identifier: Apache-2.0
//! Conditional functions bound in physical plans: `coalesce`, `if`, `nvl`,
//! `nvl2`, `nullif`, `ifnull`, `zeroifnull` and `nullifzero`, in ANSI and
//! legacy mode.
//!
//! Each test is a statement of the Spark 4.2.0 golden files this fix covers
//! (`tests/spark-golden/v4.2.0/results/*.sql.out`), with Spark's recorded
//! rows or error class: `conditional-functions` [5, 6, 8, 9, 13-16],
//! `nonansi/conditional-functions` [5, 6, 8, 9, 13-16], `having` [48],
//! `postgreSQL/join` [134, 135] and `sql-compatibility-functions` [0-7].
//! `sql-compatibility-functions` [13, 14] need a temporary view and
//! `range(5)`; they run here in the same shape over a memory table.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal64Type, Decimal128Type, Float32Type, Float64Type};
use arrow_array::{Array, Float64Array, Int32Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};

use super::{Runnable, run};
use krishiv_exec::planner::{
    DefaultPlanner, MemoryTable, MemoryTables, PhysicalPlanner, PlanningContext,
    TableProviderResolver,
};
use krishiv_types::batch::Batch;

/// An analysis and planning context with name resolution for `tables`.
struct Cx {
    ctx: krishiv_plan::analyzer::AnalysisContext,
    tables: Arc<dyn TableProviderResolver>,
    config: Arc<Config>,
}

/// `Cx` without tables, with `spark.sql.ansi.enabled = ansi`.
fn cx(ansi: bool) -> Cx {
    let mut config = Config::new();
    config
        .set(
            "spark.sql.ansi.enabled",
            if ansi { "true" } else { "false" },
        )
        .unwrap();
    Cx {
        ctx: context(
            MemoryCatalog::new("spark_catalog"),
            ResolutionConf {
                ansi,
                ..ResolutionConf::default()
            },
        ),
        tables: Arc::new(MemoryTables::new()),
        config: Arc::new(config),
    }
}

/// `Cx` with the golden `conditional_t(c1 DOUBLE, c2 INT)`:
/// `(1d, 0), (2d, 1), (NULL, 1), (NaN, 0)`.
fn conditional_cx(ansi: bool) -> Cx {
    let mut base = cx(ansi);
    let schema = Arc::new(Schema::new(vec![
        Field::new("c1", ArrowType::Float64, true),
        Field::new("c2", ArrowType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                Some(2.0),
                None,
                Some(f64::NAN),
            ])),
            Arc::new(Int32Array::from(vec![Some(0), Some(1), Some(1), Some(0)])),
        ],
    )
    .unwrap();
    let columns = vec![
        TableColumn {
            name: "c1".to_owned(),
            data_type: krishiv_types::DataType::Double,
            nullable: true,
        },
        TableColumn {
            name: "c2".to_owned(),
            data_type: krishiv_types::DataType::Integer,
            nullable: true,
        },
    ];
    let catalog = MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "conditional_t",
        columns,
        None,
    );
    base.ctx = context(
        catalog,
        ResolutionConf {
            ansi,
            ..ResolutionConf::default()
        },
    );
    base.tables = Arc::new(MemoryTables::new().with_table(
        &["spark_catalog", "default", "conditional_t"],
        Arc::new(MemoryTable::new(schema, vec![Batch::new(batch)])),
    ));
    base
}

/// Parses, analyzes, optimizes and plans `sql`.
fn plan(cx: &Cx, sql: &str) -> Result<Runnable, KrishivError> {
    let plan =
        krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .map_err(KrishivError::from)?;
    let plan = block_on(DefaultAnalyzer.analyze(&plan, &cx.ctx))?.plan;
    let plan = krishiv_optimizer::optimize(&plan)?;
    Ok(Runnable::new(
        DefaultPlanner.plan(
            &plan,
            &PlanningContext::new(Arc::clone(&cx.tables), Arc::clone(&cx.config)),
        )?,
        Arc::clone(&cx.tables),
    ))
}

/// One cell as Spark's golden files render it (`NULL` for null).
fn cell(a: &dyn Array, row: usize) -> String {
    if a.is_null(row) || a.data_type() == &ArrowType::Null {
        return "NULL".to_owned();
    }
    match a.data_type() {
        ArrowType::Boolean => a.as_boolean().value(row).to_string(),
        ArrowType::Int8 => a
            .as_primitive::<arrow_array::types::Int8Type>()
            .value(row)
            .to_string(),
        ArrowType::Int16 => a
            .as_primitive::<arrow_array::types::Int16Type>()
            .value(row)
            .to_string(),
        ArrowType::Int32 => a
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(row)
            .to_string(),
        ArrowType::Int64 => a
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(row)
            .to_string(),
        ArrowType::Float32 => {
            let v = a.as_primitive::<Float32Type>().value(row);
            float_cell(f64::from(v))
        }
        ArrowType::Float64 => float_cell(a.as_primitive::<Float64Type>().value(row)),
        ArrowType::Decimal64(_, scale) => decimal_cell(
            i128::from(a.as_primitive::<Decimal64Type>().value(row)),
            *scale,
        ),
        ArrowType::Decimal128(..) => {
            let v = a.as_primitive::<Decimal128Type>().value(row);
            let ArrowType::Decimal128(_, scale) = a.data_type() else {
                unreachable!()
            };
            decimal_cell(v, *scale)
        }
        ArrowType::Utf8View => a.as_string_view().value(row).to_owned(),
        ArrowType::Utf8 => a.as_string::<i32>().value(row).to_owned(),
        ArrowType::LargeUtf8 => a.as_string::<i64>().value(row).to_owned(),
        other => panic!("no test formatting for {other}"),
    }
}

/// A float as Spark renders it: `1.0`, `2.1`, `NaN`, `Infinity`.
fn float_cell(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_owned()
    } else if v.is_infinite() {
        if v > 0.0 {
            "Infinity".to_owned()
        } else {
            "-Infinity".to_owned()
        }
    } else {
        format!("{v:?}")
    }
}

/// A decimal unscaled value with `scale` digits after the point.
fn decimal_cell(unscaled: i128, scale: i8) -> String {
    let scale = usize::try_from(scale).unwrap();
    let digits = unscaled.abs().to_string();
    let sign = if unscaled < 0 { "-" } else { "" };
    if scale == 0 {
        format!("{sign}{digits}")
    } else if digits.len() > scale {
        let (int, frac) = digits.split_at(digits.len() - scale);
        format!("{sign}{int}.{frac}")
    } else {
        format!("{sign}0.{:0>width$}", digits, width = scale)
    }
}

/// The rows of `sql` in `cx`, each row tab-joined, sorted as the golden
/// runner sorts unordered results.
fn rows(cx: &Cx, sql: &str) -> Vec<String> {
    let runnable = plan(cx, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = run(&runnable).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("\t"),
            );
        }
    }
    out.sort();
    out
}

/// `sql` fails in `cx` with `class` (and `sql_state` when `Some`).
fn fails(cx: &Cx, sql: &str, class: ErrorClass, sql_state: Option<&str>) {
    let e = match plan(cx, sql) {
        Ok(runnable) => run(&runnable).unwrap_err(),
        Err(e) => e,
    };
    assert_eq!(e.error_class(), class, "{sql}: {e}");
    assert_eq!(e.sql_state(), sql_state, "{sql}: {e}");
}

/// The golden `conditional-functions` statements that only need literals, in
/// ANSI mode: `if` and `coalesce` skip the untaken branch, so `1/0` never
/// raises.
#[test]
fn conditional_if_and_coalesce_skip_untaken_branches() {
    let cx = cx(true);
    // [5, 6].
    assert_eq!(rows(&cx, "SELECT if(1 == 1, 1, 1/0)"), ["1.0"]);
    assert_eq!(rows(&cx, "SELECT if(1 != 1, 1/0, 1)"), ["1.0"]);
    // [8, 9].
    assert_eq!(rows(&cx, "SELECT coalesce(1, 1/0)"), ["1.0"]);
    assert_eq!(rows(&cx, "SELECT coalesce(null, 1, 1/0)"), ["1.0"]);
}

/// The golden `conditional-functions` `nullifzero`/`zeroifnull` statements in
/// ANSI mode: string zeros cast to `BIGINT`, so `'abc'` raises
/// `CAST_INVALID_INPUT`.
#[test]
fn conditional_nullifzero_and_zeroifnull_ansi() {
    let cx = cx(true);
    // [13].
    assert_eq!(
        rows(
            &cx,
            "SELECT nullifzero(0), nullifzero(cast(0 as tinyint)), \
             nullifzero(cast(0 as bigint)), nullifzero('0'), nullifzero(0.0), \
             nullifzero(1), nullifzero(null)"
        ),
        ["NULL\tNULL\tNULL\tNULL\tNULL\t1\tNULL"]
    );
    // [14].
    fails(
        &cx,
        "SELECT nullifzero('abc')",
        ErrorClass::CastInvalidInput,
        Some("22018"),
    );
    // [15].
    assert_eq!(
        rows(
            &cx,
            "SELECT zeroifnull(null), zeroifnull(1), zeroifnull(cast(1 as tinyint)), \
             zeroifnull(cast(1 as bigint))"
        ),
        ["0\t1\t1\t1"]
    );
    // [16].
    fails(
        &cx,
        "SELECT zeroifnull('abc')",
        ErrorClass::CastInvalidInput,
        Some("22018"),
    );
}

/// The golden `nonansi/conditional-functions` statements: without ANSI a
/// division by zero is `NULL` (still untaken here) and a malformed string
/// cast is `NULL`, so the functions return their first argument.
#[test]
fn conditional_functions_legacy() {
    let cx = cx(false);
    // [5, 6, 8, 9].
    assert_eq!(rows(&cx, "SELECT if(1 == 1, 1, 1/0)"), ["1.0"]);
    assert_eq!(rows(&cx, "SELECT if(1 != 1, 1/0, 1)"), ["1.0"]);
    assert_eq!(rows(&cx, "SELECT coalesce(1, 1/0)"), ["1.0"]);
    assert_eq!(rows(&cx, "SELECT coalesce(null, 1, 1/0)"), ["1.0"]);
    // [13].
    assert_eq!(
        rows(
            &cx,
            "SELECT nullifzero(0), nullifzero(cast(0 as tinyint)), \
             nullifzero(cast(0 as bigint)), nullifzero('0'), nullifzero(0.0), \
             nullifzero(1), nullifzero(null)"
        ),
        ["NULL\tNULL\tNULL\tNULL\tNULL\t1\tNULL"]
    );
    // [14, 16]: the `NULL` cast result never equals zero.
    assert_eq!(rows(&cx, "SELECT nullifzero('abc')"), ["abc"]);
    // A boolean against a number: the number casts to boolean (non-zero is
    // true), so `true = 1` holds and `nullif` gives NULL.
    assert_eq!(rows(&cx, "SELECT nullif(true, 1)"), ["NULL"]);
    // [15].
    assert_eq!(
        rows(
            &cx,
            "SELECT zeroifnull(null), zeroifnull(1), zeroifnull(cast(1 as tinyint)), \
             zeroifnull(cast(1 as bigint))"
        ),
        ["0\t1\t1\t1"]
    );
    assert_eq!(rows(&cx, "SELECT zeroifnull('abc')"), ["abc"]);
}

/// The golden `conditional-functions` statements over the golden table: `if`
/// and `coalesce` evaluate per row, lazily.
#[test]
fn conditional_if_and_coalesce_over_rows() {
    let cx = conditional_cx(true);
    assert_eq!(
        rows(&cx, "SELECT if(c2 >= 0, 1-0, 1/0) from conditional_t"),
        ["1.0", "1.0", "1.0", "1.0"]
    );
    assert_eq!(
        rows(&cx, "SELECT coalesce(c2, 1/0) from conditional_t"),
        ["0.0", "0.0", "1.0", "1.0"]
    );
    // A taken branch still raises: `if` takes `1 / c2` where `c2 = 0`,
    // and `coalesce` takes `10 / c2` where `nullif` nulled the row out.
    fails(
        &cx,
        "SELECT if(c2 = 0, 1 / c2, c2) from conditional_t",
        ErrorClass::DivideByZero,
        Some("22012"),
    );
    fails(
        &cx,
        "SELECT coalesce(nullif(c2, 0), 10 / c2) from conditional_t",
        ErrorClass::DivideByZero,
        Some("22012"),
    );
}

/// The golden `having` [48]: `COALESCE` in `GROUP BY` and `HAVING`.
#[test]
fn conditional_coalesce_in_group_by_and_having() {
    let cx = cx(true);
    assert_eq!(
        rows(
            &cx,
            "SELECT 1 GROUP BY COALESCE(1, 1) HAVING COALESCE(1, 1) = 1 \
             OR COALESCE(1, 1) IS NOT NULL"
        ),
        ["1"]
    );
}

/// The golden `postgreSQL/join` [134, 135]: `COALESCE` over nullable join
/// outputs.
#[test]
fn conditional_coalesce_over_joins() {
    let cx = cx(true);
    for n in [134, 135] {
        let (outer, inner) = if n == 134 {
            ("sub4.value2", "sub6.value1")
        } else {
            ("value2", "sub6.value1")
        };
        let sql = format!(
            "SELECT * FROM ( SELECT 1 as key1 ) sub1 LEFT JOIN \
             ( SELECT sub3.key3, {outer}, COALESCE({outer}, 66) as value3 FROM \
             ( SELECT 1 as key3 ) sub3 LEFT JOIN \
             ( SELECT sub5.key5, COALESCE({inner}, 1) as value2 FROM \
             ( SELECT 1 as key5 ) sub5 LEFT JOIN \
             ( SELECT 2 as key6, 42 as value1 ) sub6 ON sub5.key5 = sub6.key6 \
             ) sub4 ON sub4.key5 = sub3.key3 \
             ) sub2 ON sub1.key1 = sub2.key3"
        );
        assert_eq!(rows(&cx, &sql), ["1\t1\t1\t1"], "statement {n}");
    }
}

/// The golden `sql-compatibility-functions` [0-7]: `ifnull`, `nullif`, `nvl`
/// and `nvl2` over strings and mixed numerics.
#[test]
fn sql_compatibility_conditionals() {
    let cx = cx(true);
    // [0-3].
    assert_eq!(
        rows(
            &cx,
            "SELECT ifnull(null, 'x'), ifnull('y', 'x'), ifnull(null, null)"
        ),
        ["x\ty\tNULL"]
    );
    assert_eq!(
        rows(&cx, "SELECT nullif('x', 'x'), nullif('x', 'y')"),
        ["NULL\tx"]
    );
    assert_eq!(
        rows(&cx, "SELECT nvl(null, 'x'), nvl('y', 'x'), nvl(null, null)"),
        ["x\ty\tNULL"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT nvl2(null, 'x', 'y'), nvl2('n', 'x', 'y'), nvl2(null, null, null)"
        ),
        ["y\tx\tNULL"]
    );
    // [4-7]: `2.1d` is a double; `nullif` keeps the first argument's type.
    assert_eq!(
        rows(&cx, "SELECT ifnull(1, 2.1d), ifnull(null, 2.1d)"),
        ["1.0\t2.1"]
    );
    assert_eq!(
        rows(&cx, "SELECT nullif(1, 2.1d), nullif(1, 1.0d)"),
        ["1\tNULL"]
    );
    assert_eq!(
        rows(&cx, "SELECT nvl(1, 2.1d), nvl(null, 2.1d)"),
        ["1.0\t2.1"]
    );
    assert_eq!(
        rows(&cx, "SELECT nvl2(null, 1, 2.1d), nvl2('n', 1, 2.1d)"),
        ["2.1\t1.0"]
    );
}

/// `sql-compatibility-functions` [13, 14] in the same shape over memory
/// tables: `nvl` grouping and `nullif` over an aggregate.
#[test]
fn sql_compatibility_conditionals_over_aggregates() {
    let mut base = cx(true);
    let schema = Arc::new(Schema::new(vec![
        Field::new("s", ArrowType::Utf8View, true),
        Field::new("a", ArrowType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringViewArray::from(vec![Some("gamma"), None])),
            Arc::new(Int32Array::from(vec![Some(3), Some(4)])),
        ],
    )
    .unwrap();
    let catalog = MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "compat",
        vec![
            TableColumn {
                name: "s".to_owned(),
                data_type: krishiv_types::DataType::String,
                nullable: true,
            },
            TableColumn {
                name: "a".to_owned(),
                data_type: krishiv_types::DataType::Integer,
                nullable: true,
            },
        ],
        None,
    );
    base.ctx = context(
        catalog,
        ResolutionConf {
            ansi: true,
            ..ResolutionConf::default()
        },
    );
    base.tables = Arc::new(MemoryTables::new().with_table(
        &["spark_catalog", "default", "compat"],
        Arc::new(MemoryTable::new(schema, vec![Batch::new(batch)])),
    ));
    let cx = base;
    assert_eq!(
        rows(
            &cx,
            "SELECT nvl(s, 'value'), count(*) FROM compat GROUP BY nvl(s, 'value')"
        ),
        ["gamma\t1", "value\t1"]
    );
    assert_eq!(rows(&cx, "SELECT nullif(SUM(a), 0) FROM compat"), ["7"]);
    assert_eq!(rows(&cx, "SELECT nullif(SUM(a), 7) FROM compat"), ["NULL"]);
}

/// `zeroifnull` widens decimals as the analyzer types them: a
/// `decimal(10,2)` meets the hidden `0` at `decimal(12,2)`.
#[test]
fn conditional_zeroifnull_widens_decimals() {
    let cx = cx(true);
    assert_eq!(
        rows(
            &cx,
            "SELECT zeroifnull(cast(1.5 as decimal(10,2))), zeroifnull(cast(NULL as decimal(10,2)))"
        ),
        ["1.50\t0.00"]
    );
}

/// `nullif` with an untyped `NULL` side: `NULL` never equals, so the first
/// argument's value is kept.
#[test]
fn conditional_nullif_with_null() {
    let cx = cx(true);
    assert_eq!(
        rows(&cx, "SELECT nullif(NULL, 1), nullif(1, NULL)"),
        ["NULL\t1"]
    );
}

/// The golden `conditional-functions` `nanvl` statements over the golden
/// table: `b` is evaluated only where `a` is NaN, so `1 / 0` never raises
/// for the other rows.
#[test]
fn conditional_nanvl_over_rows() {
    for ansi in [true, false] {
        let cx = conditional_cx(ansi);
        // [0-2].
        assert_eq!(
            rows(&cx, "SELECT nanvl(c2, c1/c2 + c1/c2) FROM conditional_t"),
            ["0.0", "0.0", "1.0", "1.0"],
            "ansi = {ansi}"
        );
        assert_eq!(
            rows(&cx, "SELECT nanvl(c2, 1/0) FROM conditional_t"),
            ["0.0", "0.0", "1.0", "1.0"],
            "ansi = {ansi}"
        );
        assert_eq!(rows(&cx, "SELECT nanvl(1-0, 1/0)"), ["1.0"]);
    }
}

/// `nanvl(a, b)`: `b` where `a` is NaN, else `a`; a malformed `a` raises
/// with ANSI on and is NULL without.
#[test]
fn conditional_nanvl_takes_the_second_argument_for_nan_only() {
    let ansi = cx(true);
    assert_eq!(
        rows(&ansi, "SELECT nanvl(cast('NaN' as double), 123)"),
        ["123.0"]
    );
    assert_eq!(rows(&ansi, "SELECT nanvl(1, 123)"), ["1.0"]);
    fails(
        &ansi,
        "SELECT nanvl('abc', 123)",
        ErrorClass::CastInvalidInput,
        Some("22018"),
    );
    let legacy = cx(false);
    assert_eq!(rows(&legacy, "SELECT nanvl('abc', 123)"), ["NULL"]);
    assert_eq!(rows(&legacy, "SELECT nanvl(1, 123)"), ["1.0"]);
}

/// The output column names keep the call as written, never the rewritten
/// form (`CASE WHEN`, `coalesce`).
#[test]
fn conditional_names_keep_the_call_as_written() {
    let cx = conditional_cx(true);
    let names_of = |sql: &str| {
        let runnable = plan(&cx, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        super::names(runnable.schema())
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        names_of("SELECT nullif(c2, 0) FROM conditional_t"),
        ["nullif(c2, 0)"]
    );
    assert_eq!(
        names_of("SELECT nullifzero(c2) FROM conditional_t"),
        ["nullifzero(c2)"]
    );
    assert_eq!(
        names_of("SELECT if(c2 >= 0, 1-0, 1/0) FROM conditional_t"),
        ["(IF((c2 >= 0), (1 - 0), (1 / 0)))"]
    );
    assert_eq!(names_of("SELECT zeroifnull(1)"), ["zeroifnull(1)"]);
    assert_eq!(
        names_of("SELECT nanvl(c2, 1/0) FROM conditional_t"),
        ["nanvl(c2, (1 / 0))"]
    );
    assert_eq!(
        names_of("SELECT nvl(c2, 0) FROM conditional_t"),
        ["nvl(c2, 0)"]
    );
}

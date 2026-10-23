// SPDX-License-Identifier: Apache-2.0
//! Hand-built plans that together use every `PhysNode` variant and every
//! kind of value a plan holds. They do not depend on the planner, so the
//! golden bytes pinned from them change only when the format does.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef, TimeUnit};
use krishiv_common::secret::OptionValue;
use krishiv_exec::JoinType;
use krishiv_exec::aggregate::{AggInput, AggregateExpr, AggregateFunction, AggregateOutput};
use krishiv_exec::physical::{
    AggregatePlan, ExchangeId, ExchangePlan, ExprEnv, HashJoinPlan, JoinDistribution,
    JoinFilterPlan, NestedLoopPlan, Partitioning, PhysNode, PhysicalPlan, ProjectItemPlan,
    Residual, ScanFilter, ScanId, ScanNode, ScanRuntimeFilter, SubqueryId, SubqueryPlan,
    SubqueryValueKind,
};
use krishiv_exec::planner::ExplainNode;
use krishiv_exec::window::{
    FrameBound, FrameUnits, WindowFrame, WindowFunction, WindowFunctionPlan, WindowPlan,
};
use krishiv_exec::{RowStageSettings, SortKey};
use krishiv_io::{ColumnPath, FilterId, KeyFilterOptions};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, CastMode, DayTimeField, Expr, ExprNode, FunctionId, FunctionKind,
    Literal, Origin, PlanId, ReadSource, ResolvedExpr, ResolvedFunction, Scan, SourceId, Span,
    TableVersion, YearMonthField,
};
use krishiv_types::{DataType, DecimalType};

pub(crate) fn env() -> ExprEnv {
    ExprEnv {
        ansi: true,
        zone: Some(-28_800),
        query_start_micros: 1_760_000_000_123_456,
        user: Arc::from("analyst"),
    }
}

pub(crate) fn column(id: u64, name: &str, data_type: DataType, nullable: bool) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type,
        nullable,
    }
}

pub(crate) fn col(id: u64, name: &str, data_type: DataType) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(column(
        id, name, data_type, true,
    )))))
}

pub(crate) fn lit(l: Literal) -> Arc<Expr> {
    Arc::new(Expr::literal(l))
}

pub(crate) fn call(op: Builtin, args: Vec<Arc<Expr>>, data_type: DataType) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(op),
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            data_type,
            nullable: true,
            deterministic: true,
        },
    ))))
}

/// `a > 10 AND b IS NOT NULL`, with `a` shared by both conjuncts.
pub(crate) fn predicate() -> Arc<Expr> {
    let a = col(1, "a", DataType::Integer);
    let gt = call(
        Builtin::Gt,
        vec![Arc::clone(&a), lit(Literal::Integer(10))],
        DataType::Boolean,
    );
    let not_null = call(Builtin::IsNotNull, vec![a], DataType::Boolean);
    call(Builtin::And, vec![gt, not_null], DataType::Boolean)
}

pub(crate) fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("s", ArrowType::Utf8View, false),
        Field::new("d", ArrowType::Decimal128(12, 2), true),
        Field::new("t", ArrowType::Timestamp(TimeUnit::Microsecond, None), true),
    ]))
}

pub(crate) fn scan_columns() -> Vec<BoundColumn> {
    vec![
        column(1, "a", DataType::Integer, true),
        column(2, "s", DataType::String, false),
    ]
}

/// A scan of a table, with a pushed filter and both runtime-filter specs.
pub(crate) fn table_scan() -> ScanNode {
    let options = BTreeMap::from([
        ("mode".to_owned(), OptionValue::Plain("strict".to_owned())),
        (
            "token".to_owned(),
            OptionValue::reference("prod", "warehouse-token"),
        ),
    ]);
    ScanNode {
        id: ScanId(7),
        scan: Scan {
            source: ReadSource::Table {
                name: vec!["spark_catalog".into(), "default".into(), "t".into()],
                options,
            },
            output: scan_columns(),
            version: Some(TableVersion(42)),
        },
        projection: vec![ColumnPath::new("a"), ColumnPath::nested(["s", "inner"])],
        filter: Some(ScanFilter {
            layout: vec![AttrId(1), AttrId(2)],
            predicate: predicate(),
            exact: true,
        }),
        runtime_filters: vec![ScanRuntimeFilter {
            id: FilterId(3),
            column: ColumnPath::new("a"),
            key_type: ArrowType::Int32,
        }],
        row_stage: RowStageSettings {
            min_rows: 1000,
            max_pass_rate: 0.75,
        },
    }
}

/// A scan of files, without a version or a filter.
pub(crate) fn file_scan() -> ScanNode {
    ScanNode {
        id: ScanId(8),
        scan: Scan {
            source: ReadSource::Files {
                format: "parquet".to_owned(),
                paths: vec![
                    OptionValue::Plain("s3://bucket/a/*.parquet".to_owned()),
                    OptionValue::reference("prod", "signed-path"),
                ],
                options: BTreeMap::new(),
            },
            output: scan_columns(),
            version: None,
        },
        projection: vec![ColumnPath::new("a")],
        filter: None,
        runtime_filters: Vec::new(),
        row_stage: RowStageSettings::default(),
    }
}

pub(crate) fn scan() -> PhysNode {
    PhysNode::Scan(table_scan())
}

fn boxed(node: PhysNode) -> Box<PhysNode> {
    Box::new(node)
}

pub(crate) fn subquery() -> SubqueryPlan {
    SubqueryPlan {
        id: SubqueryId(1_000_000),
        kind: SubqueryValueKind::Scalar,
        data_type: ArrowType::Int64,
        plan: PhysNode::Limit {
            input: boxed(PhysNode::OneRow),
            offset: 0,
            limit: 1,
        },
    }
}

pub(crate) fn aggregate_exprs() -> Vec<AggregateExpr> {
    let a = |function, arg_type| AggregateExpr {
        function,
        args: vec![0],
        input: AggInput::Raw,
        arg_type,
        name: "x".to_owned(),
    };
    vec![
        AggregateExpr::count_star("n"),
        a(AggregateFunction::Count, ArrowType::Int32),
        a(AggregateFunction::Sum, ArrowType::Decimal128(12, 2)),
        a(AggregateFunction::Avg, ArrowType::Float64),
        a(AggregateFunction::Min, ArrowType::Utf8View),
        a(AggregateFunction::Max, ArrowType::Date32),
        a(
            AggregateFunction::First { ignore_nulls: true },
            ArrowType::Int32,
        ),
        a(
            AggregateFunction::Last {
                ignore_nulls: false,
            },
            ArrowType::Int32,
        ),
        AggregateExpr {
            function: AggregateFunction::Sum,
            args: vec![3, 4],
            input: AggInput::State,
            arg_type: ArrowType::Int32,
            name: "merged".to_owned(),
        },
    ]
}

fn window_function(function: WindowFunction, frame: WindowFrame, name: &str) -> WindowFunctionPlan {
    WindowFunctionPlan {
        function,
        frame,
        filter: None,
        name: name.to_owned(),
    }
}

/// Every ranking function, partitioned and ordered.
pub(crate) fn window_ranking() -> WindowPlan {
    let frame = WindowFrame::running(FrameUnits::Range);
    WindowPlan {
        partition_by: vec![1, 0],
        order_by: vec![SortKey::asc(0), SortKey::desc(2)],
        functions: vec![
            window_function(WindowFunction::RowNumber, frame, "rn"),
            window_function(WindowFunction::Rank, frame, "rk"),
            window_function(WindowFunction::DenseRank, frame, "drk"),
            window_function(WindowFunction::PercentRank, frame, "prk"),
            window_function(WindowFunction::CumeDist, frame, "cd"),
            window_function(WindowFunction::Ntile(4), frame, "nt"),
        ],
    }
}

/// Aggregates over every frame the operator supports (both units, the
/// whole partition and the running frame), a bound that is the current
/// row at both ends, a `FILTER` column and the aggregate input kinds.
pub(crate) fn window_aggregates() -> WindowPlan {
    let aggs = aggregate_exprs();
    let over = |i: usize, frame, name: &str| {
        window_function(WindowFunction::Aggregate(aggs[i].clone()), frame, name)
    };
    let current = |units| WindowFrame {
        units,
        lower: FrameBound::CurrentRow,
        upper: FrameBound::CurrentRow,
    };
    WindowPlan {
        partition_by: Vec::new(),
        order_by: Vec::new(),
        functions: vec![
            over(0, WindowFrame::whole(FrameUnits::Rows), "whole_rows"),
            over(2, WindowFrame::whole(FrameUnits::Range), "whole_range"),
            over(3, WindowFrame::running(FrameUnits::Rows), "run_rows"),
            over(4, WindowFrame::running(FrameUnits::Range), "run_range"),
            over(5, current(FrameUnits::Rows), "cur_rows"),
            over(6, current(FrameUnits::Range), "cur_range"),
            WindowFunctionPlan {
                filter: Some(3),
                ..over(7, WindowFrame::whole(FrameUnits::Rows), "filtered")
            },
            over(8, WindowFrame::running(FrameUnits::Rows), "merged"),
        ],
    }
}

pub(crate) fn residual() -> Residual {
    Residual {
        layout: vec![AttrId(1), AttrId(2), AttrId(3)],
        conjuncts: vec![predicate(), lit(Literal::Boolean(true))],
    }
}

fn join(join_type: JoinType) -> HashJoinPlan {
    HashJoinPlan {
        probe: boxed(scan()),
        build: boxed(PhysNode::Scan(file_scan())),
        join_type,
        probe_keys: vec![0, 1],
        build_keys: vec![1, 0],
        null_safe: vec![false, true],
        residual: Some(residual()),
        runtime_filters: vec![JoinFilterPlan {
            id: FilterId(3),
            key: 0,
            options: KeyFilterOptions {
                max_values: 50,
                exact_max_values: 4096,
                bloom: true,
                bloom_max_items: 4_000_000,
                bloom_max_bytes: 8 << 20,
                bloom_fpp: 0.01,
                bloom_max_fpp: 0.05,
            },
        }],
        distribution: JoinDistribution::Broadcast,
    }
}

/// One node of every variant, each as the root of its own plan.
pub(crate) fn nodes() -> Vec<(&'static str, PhysNode)> {
    let keys = vec![SortKey::asc(0), SortKey::desc(1)];
    let project = PhysNode::Project {
        input: boxed(scan()),
        layout: vec![AttrId(1), AttrId(2)],
        items: vec![
            ProjectItemPlan::Column {
                index: 1,
                name: None,
            },
            ProjectItemPlan::Column {
                index: 0,
                name: Some("renamed".to_owned()),
            },
            ProjectItemPlan::Expr {
                name: "sum".to_owned(),
                expr: call(
                    Builtin::Add,
                    vec![col(1, "a", DataType::Integer), lit(Literal::Integer(1))],
                    DataType::Integer,
                ),
            },
        ],
        nullable: Some(vec![true, false, true]),
        subqueries: vec![subquery()],
    };
    vec![
        ("scan", scan()),
        ("file_scan", PhysNode::Scan(file_scan())),
        ("one_row", PhysNode::OneRow),
        (
            "filter",
            PhysNode::Filter {
                input: boxed(scan()),
                layout: vec![AttrId(1), AttrId(2)],
                predicate: predicate(),
                subqueries: vec![subquery()],
            },
        ),
        ("project", project),
        (
            "project_without_nullable",
            PhysNode::Project {
                input: boxed(PhysNode::OneRow),
                layout: Vec::new(),
                items: Vec::new(),
                nullable: None,
                subqueries: Vec::new(),
            },
        ),
        (
            "limit",
            PhysNode::Limit {
                input: boxed(scan()),
                offset: 0,
                limit: u64::MAX,
            },
        ),
        (
            "limit_offset",
            PhysNode::Limit {
                input: boxed(scan()),
                offset: 7,
                limit: 5,
            },
        ),
        (
            "replicate_rows",
            PhysNode::ReplicateRows {
                input: boxed(scan()),
                layout: vec![AttrId(1)],
                count: lit(Literal::Long(2)),
                subqueries: vec![subquery()],
            },
        ),
        (
            "sort",
            PhysNode::Sort {
                input: boxed(scan()),
                keys: keys.clone(),
            },
        ),
        (
            "top_n",
            PhysNode::TopN {
                input: boxed(scan()),
                keys,
                limit: 10,
            },
        ),
        (
            "aggregate",
            PhysNode::Aggregate {
                input: boxed(scan()),
                aggregate: AggregatePlan {
                    group_by: vec![0, 2],
                    distinct: vec![false; aggregate_exprs().len()],
                    aggregates: aggregate_exprs(),
                    output: AggregateOutput::Partial,
                },
            },
        ),
        (
            "window_ranking",
            PhysNode::Window {
                input: boxed(scan()),
                window: window_ranking(),
            },
        ),
        (
            "window_aggregates",
            PhysNode::Window {
                input: boxed(PhysNode::Sort {
                    input: boxed(scan()),
                    keys: vec![SortKey::asc(0), SortKey::desc(1)],
                }),
                window: window_aggregates(),
            },
        ),
        (
            "aggregate_final",
            PhysNode::Aggregate {
                input: boxed(scan()),
                aggregate: AggregatePlan {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr::count_star("n")],
                    distinct: vec![true],
                    output: AggregateOutput::Final,
                },
            },
        ),
        ("hash_join", PhysNode::HashJoin(join(JoinType::Inner))),
        (
            "hash_join_anti",
            PhysNode::HashJoin(HashJoinPlan {
                residual: None,
                runtime_filters: Vec::new(),
                ..join(JoinType::NullAwareAnti)
            }),
        ),
        (
            "hash_join_partitioned",
            PhysNode::HashJoin(HashJoinPlan {
                distribution: JoinDistribution::Partitioned,
                ..join(JoinType::Inner)
            }),
        ),
        (
            "nested_loop_join",
            PhysNode::NestedLoopJoin(NestedLoopPlan {
                probe: boxed(scan()),
                build: boxed(PhysNode::OneRow),
                join_type: JoinType::LeftSemi,
                residual: Some(residual()),
            }),
        ),
        (
            "union",
            PhysNode::Union {
                inputs: vec![scan(), PhysNode::OneRow, PhysNode::Scan(file_scan())],
            },
        ),
        (
            "write_files",
            PhysNode::WriteFiles {
                input: boxed(scan()),
            },
        ),
        (
            "exchange_hash",
            PhysNode::Exchange(ExchangePlan {
                id: ExchangeId(11),
                input: boxed(scan()),
                partitioning: Partitioning::Hash { keys: vec![0, 1] },
            }),
        ),
        (
            "exchange_single",
            PhysNode::Exchange(ExchangePlan {
                id: ExchangeId(12),
                input: boxed(scan()),
                partitioning: Partitioning::Single,
            }),
        ),
        (
            "exchange_broadcast",
            PhysNode::Exchange(ExchangePlan {
                id: ExchangeId(13),
                input: boxed(scan()),
                partitioning: Partitioning::Broadcast,
            }),
        ),
        (
            "exchange_round_robin",
            PhysNode::Exchange(ExchangePlan {
                id: ExchangeId(14),
                input: boxed(scan()),
                partitioning: Partitioning::RoundRobin,
            }),
        ),
        (
            "exchange_range",
            PhysNode::Exchange(ExchangePlan {
                id: ExchangeId(15),
                input: boxed(scan()),
                partitioning: Partitioning::Range {
                    keys: vec![SortKey::asc(0), SortKey::desc(1)],
                },
            }),
        ),
    ]
}

/// A plan of `root`, with a two-level explain tree and a warning.
pub(crate) fn plan_of(root: PhysNode) -> PhysicalPlan {
    let explain = ExplainNode::new(
        "Root",
        "details",
        vec![
            ExplainNode::leaf("Left", "l"),
            ExplainNode::new("Right", "", vec![ExplainNode::leaf("Leaf", "x")]),
        ],
    );
    PhysicalPlan::new(root, schema(), explain, env())
        .with_warnings(vec!["a hint was ignored".into()])
}

/// Every kind of expression and literal, in one projection.
pub(crate) fn expressions() -> PhysNode {
    let decimal = DecimalType::try_new(38, 10).unwrap();
    let literals = vec![
        Literal::Null(None),
        Literal::Null(Some(DataType::Decimal(decimal))),
        Literal::Boolean(true),
        Literal::Byte(i8::MIN),
        Literal::Short(i16::MAX),
        Literal::Integer(i32::MIN),
        Literal::Long(i64::MAX),
        Literal::Float(-0.0),
        Literal::Float(f32::NAN),
        Literal::Double(f64::INFINITY),
        Literal::Decimal {
            unscaled: -(10_i128.pow(38) - 1),
            data_type: decimal,
        },
        Literal::String("héllo \u{1F600}".to_owned()),
        Literal::Binary(vec![0, 255, 7]),
        Literal::Date(-719_162),
        Literal::Timestamp(1_760_000_000_000_000),
        Literal::TimestampNtz(-1),
        Literal::YearMonthInterval {
            months: -14,
            start: YearMonthField::Year,
            end: YearMonthField::Month,
        },
        Literal::DayTimeInterval {
            micros: 93_784_000_000,
            start: DayTimeField::Day,
            end: DayTimeField::Second,
        },
    ];
    let mut items: Vec<ProjectItemPlan> = literals
        .into_iter()
        .enumerate()
        .map(|(i, l)| ProjectItemPlan::Expr {
            name: format!("l{i}"),
            expr: lit(l),
        })
        .collect();
    let a = col(1, "a", DataType::Integer);
    let origin = Origin::sql(SourceId(3), Span::new(5, 9)).with_plan_id(PlanId(-4));
    let cast = |mode| {
        Arc::new(Expr::new(ExprNode::Cast {
            expr: Arc::clone(&a),
            to: DataType::Long,
            mode,
        }))
    };
    let registry = Arc::new(
        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Registry {
                    name: "sum".to_owned(),
                    id: 17,
                },
                kind: FunctionKind::Aggregate,
                args: vec![Arc::clone(&a)],
                distinct: true,
                filter: Some(predicate()),
                data_type: DataType::Long,
                nullable: false,
                deterministic: false,
            },
        )))
        .with_origin(origin),
    );
    let exprs = vec![
        cast(CastMode::Legacy),
        cast(CastMode::Ansi),
        cast(CastMode::Try),
        registry,
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::OuterColumn(
            column(9, "outer", DataType::Date, false),
        )))),
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
            expr: Arc::clone(&a),
            output: column(10, "aliased", DataType::Timestamp, true),
        }))),
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Builtin(Builtin::Concat),
                kind: FunctionKind::Window,
                args: Vec::new(),
                distinct: false,
                filter: None,
                data_type: DataType::TimestampNtz,
                nullable: true,
                deterministic: true,
            },
        )))),
    ];
    items.extend(
        exprs
            .into_iter()
            .enumerate()
            .map(|(i, expr)| ProjectItemPlan::Expr {
                name: format!("e{i}"),
                expr,
            }),
    );
    PhysNode::Project {
        input: boxed(scan()),
        layout: vec![AttrId(1), AttrId(2)],
        items,
        nullable: None,
        subqueries: Vec::new(),
    }
}

/// The named plans the inventory test, the error tests and the golden
/// files share.
pub(crate) fn cases() -> Vec<(String, PhysicalPlan)> {
    let mut out: Vec<(String, PhysicalPlan)> = nodes()
        .into_iter()
        .map(|(name, node)| (name.to_owned(), plan_of(node)))
        .collect();
    out.push(("expressions".to_owned(), plan_of(expressions())));
    out
}

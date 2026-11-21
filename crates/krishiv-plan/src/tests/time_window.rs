// SPDX-License-Identifier: Apache-2.0
//! `window(..)` and `session_window(..)` on hand-built plans (the rewrite of
//! `analyzer/time_window.rs`): output types, the plan shape and every error class.
//! Results are in the `time-window.sql` golden (`golden-report --only time-window`).

use krishiv_types::DataType;

use super::dsl::{c, count_star, f, int, s, star};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisError, AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf,
    TableColumn,
};
use crate::{Expr, ExprNode, Grouping, LogicalPlan, Origin, Plan, PlanNode};

fn catalog() -> MemoryCatalog {
    let col = |name: &str, data_type: DataType| TableColumn {
        name: name.to_owned(),
        data_type,
        nullable: true,
        char_varchar: None,
    };
    MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "t",
        vec![
            col("a", DataType::String),
            col("ts", DataType::Timestamp),
            col("n", DataType::TimestampNtz),
            col("x", DataType::Integer),
        ],
        None,
    )
}

fn run(plan: LogicalPlan) -> Result<AnalyzedPlan, AnalysisError> {
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &context(catalog(), ResolutionConf::default())))
}

fn class(r: &Result<AnalyzedPlan, AnalysisError>) -> &'static str {
    r.as_ref()
        .map_or_else(|e| e.error.error_class().name(), |_| "OK")
}

/// `window(time, args..)` over string literals.
fn window(time: &str, args: &[&str]) -> Expr {
    let mut v = vec![c(time)];
    v.extend(args.iter().map(|a| s(a)));
    f("window", v)
}

/// `SELECT window.start, window.end, count(*) FROM t GROUP BY <window>`.
fn grouped(window: Expr) -> LogicalPlan {
    LogicalPlan::table(["t"]).aggregate(
        vec![window],
        vec![c("window.start"), c("window.end"), count_star()],
    )
}

fn types(r: &Result<AnalyzedPlan, AnalysisError>) -> Vec<DataType> {
    r.as_ref()
        .unwrap()
        .output
        .iter()
        .map(|c| c.data_type.clone())
        .collect()
}

#[test]
fn a_tumbling_window_groups_by_the_window_struct() {
    let r = run(grouped(window("ts", &["5 minutes"])));
    assert_eq!(class(&r), "OK", "{:?}", r.as_ref().err().map(|e| &e.error));
    assert_eq!(
        types(&r),
        vec![DataType::Timestamp, DataType::Timestamp, DataType::Long]
    );
    let plan = r.unwrap().plan.root().to_string();
    assert!(plan.contains("__time_windows"), "{plan}");
}

#[test]
fn a_sliding_window_with_an_offset_and_timestamp_ntz() {
    let r = run(grouped(window(
        "n",
        &["10 minutes", "5 minutes", "2 minutes"],
    )));
    assert_eq!(class(&r), "OK", "{:?}", r.as_ref().err().map(|e| &e.error));
    assert_eq!(
        types(&r),
        vec![
            DataType::TimestampNtz,
            DataType::TimestampNtz,
            DataType::Long
        ]
    );
}

#[test]
fn a_window_in_a_select_item_and_in_a_where() {
    let select = LogicalPlan::table(["t"]).project(vec![
        Expr::new(ExprNode::Field {
            base: window("ts", &["5 minutes"]).into(),
            name: "start".to_owned(),
        })
        .alias("bucket_start"),
    ]);
    assert_eq!(class(&run(select)), "OK");
    let filtered = LogicalPlan::table(["t"])
        .filter(Expr::function(
            "isnotnull",
            vec![Expr::new(ExprNode::Field {
                base: window("ts", &["5 minutes"]).into(),
                name: "start".to_owned(),
            })],
        ))
        .aggregate(vec![], vec![count_star()]);
    assert_eq!(class(&run(filtered)), "OK");
}

#[test]
fn durations_are_checked_as_spark_checks_them() {
    let class_of = |args: &[&str]| class(&run(grouped(window("ts", args))));
    assert_eq!(
        class_of(&["-5 minutes"]),
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE"
    );
    assert_eq!(
        class_of(&["0 minutes"]),
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE"
    );
    assert_eq!(
        class_of(&["10 minutes", "-5 minutes"]),
        "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE"
    );
    assert_eq!(
        class_of(&["10 minutes", "20 minutes"]),
        "DATATYPE_MISMATCH.PARAMETER_CONSTRAINT_VIOLATION"
    );
    assert_eq!(
        class_of(&["10 minutes", "5 minutes", "6 minutes"]),
        "DATATYPE_MISMATCH.PARAMETER_CONSTRAINT_VIOLATION"
    );
    assert_eq!(class_of(&["not_an_interval"]), "CANNOT_PARSE_INTERVAL");
    assert_eq!(class_of(&["2 months"]), "_LEGACY_ERROR_TEMP_3231");
    assert_eq!(class_of(&[]), "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    // A negative start time below the slide is valid.
    assert_eq!(class_of(&["10 minutes", "10 minutes", "-5 minutes"]), "OK");
}

#[test]
fn other_argument_errors() {
    // Not a string, integer or long literal.
    let by_column = f("window", vec![c("ts"), c("a")]);
    assert_eq!(class(&run(grouped(by_column))), "_LEGACY_ERROR_TEMP_1207");
    // A time column that is not a timestamp.
    let by_int = f("window", vec![c("x"), s("5 minutes")]);
    assert_eq!(
        class(&run(grouped(by_int))),
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    // Integer durations are microseconds.
    let micros = f("window", vec![c("ts"), int(5_000_000)]);
    assert_eq!(class(&run(grouped(micros))), "OK");
}

#[test]
fn an_inner_window_without_arguments_is_an_arity_error_not_a_panic() {
    // `window(window(), '5 minutes')`: Spark raises WRONG_NUM_ARGS for the inner call.
    let inner = f("window", vec![]);
    let nested = f("window", vec![inner, s("5 minutes")]);
    assert_eq!(
        class(&run(grouped(nested))),
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"
    );
}

#[test]
fn a_star_next_to_a_window_returns_the_window_column_once() {
    // `SELECT *, window(ts, '5 minutes') FROM t`: the table's columns plus one `window`.
    let plan = LogicalPlan::table(["t"]).project(vec![star(), window("ts", &["5 minutes"])]);
    let r = run(plan);
    assert_eq!(class(&r), "OK");
    assert_eq!(types(&r).len(), 5, "a, ts, n, x and one window column");
}

#[test]
fn a_star_with_qualify_over_a_window_returns_the_window_column_once() {
    // `SELECT *, row_number() OVER (ORDER BY ts) AS rn FROM t QUALIFY window(ts, '5 minutes').start IS NOT NULL`
    let qualified = LogicalPlan::table(["t"])
        .project(vec![
            star(),
            Expr::new(ExprNode::Window(crate::WindowExpr {
                function: f("row_number", vec![]).into(),
                window: crate::WindowSpecRef::Spec(crate::WindowSpec {
                    partition_by: vec![],
                    order_by: vec![crate::SortOrder::asc(c("ts"))],
                    frame: None,
                }),
            }))
            .alias("rn"),
        ])
        .filter(f(
            "qualify",
            vec![f(
                "isnotnull",
                vec![Expr::new(ExprNode::Field {
                    base: window("ts", &["5 minutes"]).into(),
                    name: "start".to_owned(),
                })],
            )],
        ))
        .with_origin(Origin::sql(
            crate::SourceId(0),
            crate::Span { start: 0, end: 1 },
        ));
    let r = run(qualified);
    assert_eq!(class(&r), "OK", "{:?}", r.as_ref().err().map(|e| &e.error));
    // The window column comes first, once, as Spark's rewrite puts it; the star has the table's columns.
    let names: Vec<String> = r.unwrap().output.iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, ["window", "a", "ts", "n", "x", "rn"]);
}

#[test]
fn two_distinct_windows_in_one_operator_are_refused() {
    let two = LogicalPlan::table(["t"]).project(vec![
        window("ts", &["5 minutes"]).alias("w1"),
        window("ts", &["10 minutes"]).alias("w2"),
    ]);
    assert_eq!(class(&run(two)), "_LEGACY_ERROR_TEMP_1039");
    // The same window twice is one window.
    let same = LogicalPlan::table(["t"]).project(vec![
        window("ts", &["5 minutes"]).alias("w1"),
        window("ts", &["5 minutes"]).alias("w2"),
    ]);
    assert_eq!(class(&run(same)), "OK");
    let mixed = LogicalPlan::table(["t"]).project(vec![
        window("ts", &["5 minutes"]).alias("w1"),
        f("session_window", vec![c("ts"), s("5 minutes")]).alias("w2"),
    ]);
    assert_eq!(class(&run(mixed)), "_LEGACY_ERROR_TEMP_1039");
}

#[test]
fn a_window_in_having_or_order_by_over_an_aggregate_has_no_input() {
    let having = LogicalPlan::table(["t"])
        .aggregate(vec![c("ts")], vec![count_star().alias("cnt")])
        .filter(Expr::function(
            "isnotnull",
            vec![window("ts", &["5 minutes"])],
        ));
    assert_eq!(
        class(&run(having)),
        "MISSING_ATTRIBUTES.RESOLVED_ATTRIBUTE_MISSING_FROM_INPUT"
    );
}

/// `SELECT a, session_window.start, session_window.end, count(*) FROM t GROUP BY a, <session>`.
fn sessions(gap: Expr) -> LogicalPlan {
    LogicalPlan::table(["t"]).aggregate(
        vec![c("a"), f("session_window", vec![c("ts"), gap])],
        vec![
            c("a"),
            c("session_window.start"),
            c("session_window.end"),
            count_star(),
        ],
    )
}

#[test]
fn a_session_window_groups_by_the_merged_range() {
    let r = run(sessions(s("5 minutes")));
    assert_eq!(class(&r), "OK", "{:?}", r.as_ref().err().map(|e| &e.error));
    assert_eq!(
        types(&r),
        vec![
            DataType::String,
            DataType::Timestamp,
            DataType::Timestamp,
            DataType::Long
        ]
    );
    // A gap that is a column, and the timestamp_ntz kind.
    let by_column = LogicalPlan::table(["t"]).aggregate(
        vec![f("session_window", vec![c("n"), c("a")])],
        vec![c("session_window.start"), count_star()],
    );
    let r = run(by_column);
    assert_eq!(class(&r), "OK", "{:?}", r.as_ref().err().map(|e| &e.error));
    assert_eq!(types(&r), vec![DataType::TimestampNtz, DataType::Long]);
}

#[test]
fn session_window_errors() {
    let wrong_count = LogicalPlan::table(["t"])
        .aggregate(vec![f("session_window", vec![c("ts")])], vec![count_star()]);
    assert_eq!(
        class(&run(wrong_count)),
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"
    );
    // ROLLUP over the session window is Spark's internal error.
    let rollup = LogicalPlan::new(PlanNode::Aggregate {
        input: std::sync::Arc::new(LogicalPlan::table(["t"])),
        grouping: Grouping::Rollup(vec![
            c("a").into(),
            f("session_window", vec![c("ts"), s("5 minutes")]).into(),
        ]),
        aggregates: vec![c("a").into(), count_star().into()],
    });
    assert_eq!(class(&run(rollup)), "INTERNAL_ERROR");
    // In a WHERE: one range per row.
    let start = Expr::new(ExprNode::Field {
        base: f("session_window", vec![c("ts"), s("5 minutes")]).into(),
        name: "start".to_owned(),
    });
    let filtered = LogicalPlan::table(["t"])
        .filter(Expr::function("isnotnull", vec![start]))
        .aggregate(vec![], vec![count_star()]);
    assert_eq!(class(&run(filtered)), "OK");
}

#[test]
fn nested_windows_and_window_time() {
    // `window(window(ts, '1 minute'), '5 minutes')` windows the inner window's `window_time`.
    let nested = f("window", vec![window("ts", &["1 minute"]), s("5 minutes")]);
    let r = run(grouped(nested));
    assert_eq!(class(&r), "OK", "{:?}", r.as_ref().err().map(|e| &e.error));
    // `window_time` over a grouped window is neither a key nor an aggregate.
    let grouped_time = LogicalPlan::table(["t"]).aggregate(
        vec![window("ts", &["5 minutes"])],
        vec![f("window_time", vec![c("window")]), count_star()],
    );
    assert_eq!(class(&run(grouped_time)), "MISSING_AGGREGATION");
    // A struct that is not a window column, and a type that is not a struct.
    let bad_struct = LogicalPlan::table(["t"]).project(vec![f(
        "window_time",
        vec![f(
            "named_struct",
            vec![s("start"), c("ts"), s("end"), c("ts")],
        )],
    )]);
    assert_eq!(class(&run(bad_struct)), "_LEGACY_ERROR_TEMP_3101");
    let not_struct = LogicalPlan::table(["t"]).project(vec![f("window_time", vec![c("ts")])]);
    assert_eq!(
        class(&run(not_struct)),
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    let arity = LogicalPlan::table(["t"]).project(vec![f("window_time", vec![c("ts"), c("ts")])]);
    assert_eq!(class(&run(arity)), "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    // A window column read as `window_time`.
    let over_subquery = LogicalPlan::table(["t"])
        .aggregate(
            vec![window("ts", &["5 minutes"])],
            vec![c("window"), count_star().alias("cnt")],
        )
        .project(vec![
            f("window_time", vec![c("window")]).alias("event_time"),
        ]);
    let r = run(over_subquery);
    assert_eq!(class(&r), "OK", "{:?}", r.as_ref().err().map(|e| &e.error));
    assert_eq!(types(&r), vec![DataType::Timestamp]);
}

#[test]
fn a_window_over_a_window_column_of_a_derived_table_reads_its_window_time() {
    // `SELECT window(inner_window, '5 minutes') AS w, sum(cnt) FROM (SELECT window(ts, '1 minute') AS
    // inner_window, count(*) AS cnt FROM t GROUP BY window(ts, '1 minute')) GROUP BY window(inner_window, ..)`.
    let inner = LogicalPlan::table(["t"]).aggregate(
        vec![window("ts", &["1 minute"])],
        vec![
            window("ts", &["1 minute"]).alias("inner_window"),
            count_star().alias("cnt"),
        ],
    );
    let outer = inner.aggregate(
        vec![f("window", vec![c("inner_window"), s("5 minutes")])],
        vec![c("window.start"), f("sum", vec![c("cnt")])],
    );
    let r = run(outer);
    assert_eq!(class(&r), "OK", "{:?}", r.as_ref().err().map(|e| &e.error));
    // A struct that no `window(..)` made is not a window column.
    let plain = LogicalPlan::table(["t"])
        .project(vec![
            f("named_struct", vec![s("start"), c("ts"), s("end"), c("ts")]).alias("inner_window"),
        ])
        .aggregate(
            vec![f("window", vec![c("inner_window"), s("5 minutes")])],
            vec![c("window.start")],
        );
    assert_ne!(class(&run(plain)), "OK");
}

#[test]
fn qualify_reads_the_window_below_the_select_list() {
    let qualified = LogicalPlan::table(["t"])
        .project(vec![
            c("ts"),
            Expr::new(ExprNode::Window(crate::WindowExpr {
                function: f("row_number", vec![]).into(),
                window: crate::WindowSpecRef::Spec(crate::WindowSpec {
                    partition_by: vec![],
                    order_by: vec![crate::SortOrder::asc(c("ts"))],
                    frame: None,
                }),
            }))
            .alias("rn"),
        ])
        .filter(f(
            "qualify",
            vec![f(
                "isnotnull",
                vec![Expr::new(ExprNode::Field {
                    base: window("ts", &["5 minutes"]).into(),
                    name: "start".to_owned(),
                })],
            )],
        ))
        .with_origin(Origin::sql(
            crate::SourceId(0),
            crate::Span { start: 0, end: 1 },
        ));
    let r = run(qualified);
    assert_eq!(class(&r), "OK", "{:?}", r.as_ref().err().map(|e| &e.error));
    // The window column comes first, as Spark's rewrite puts it.
    let names: Vec<String> = r.unwrap().output.iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, ["window", "ts", "rn"]);
}

#[test]
fn a_window_in_a_pivot_is_no_aggregate_and_window_time_takes_one_argument() {
    let wrong = LogicalPlan::table(["t"]).aggregate(
        vec![window("ts", &["5 minutes"])],
        vec![
            f("window_time", vec![c("window"), c("window")]),
            count_star(),
        ],
    );
    assert_eq!(class(&run(wrong)), "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
}

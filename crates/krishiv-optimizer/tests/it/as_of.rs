// SPDX-License-Identifier: Apache-2.0
//! `PushDownAsOfCondition` and the as-of arm of the filter pushdown: one plan snapshot per case,
//! over `t(a int, b string, c bigint)` and `u(a int, d string)`. Right-only `ON` conjuncts move
//! into the right input (inner and left), left-only ones into the left input for inner only;
//! a left-only filter from above moves into the left input (both kinds); anything that reads the
//! right input from above, or both inputs, or is nondeterministic, stays.

use krishiv_optimizer::optimize_with;
use krishiv_plan::analyzer::testing::block_on;
use krishiv_plan::analyzer::{Analyzer as _, DefaultAnalyzer};
use krishiv_plan::{Plan, SourceId};

use crate::support;

const CASES: &[(&str, &str)] = &[
    (
        "as_of_right_conjunct_inner",
        "SELECT * FROM t ASOF JOIN u MATCH_CONDITION (t.a >= u.a) ON t.b = u.d AND u.d = 'x'",
    ),
    (
        "as_of_right_conjunct_left",
        "SELECT * FROM t LEFT ASOF JOIN u MATCH_CONDITION (t.a >= u.a) ON t.b = u.d AND u.d = 'x'",
    ),
    (
        "as_of_left_conjunct_inner",
        "SELECT * FROM t ASOF JOIN u MATCH_CONDITION (t.a >= u.a) ON t.b = u.d AND t.c > 3",
    ),
    (
        "as_of_left_conjunct_left_stays",
        "SELECT * FROM t LEFT ASOF JOIN u MATCH_CONDITION (t.a >= u.a) ON t.b = u.d AND t.c > 3",
    ),
    (
        "as_of_left_filter_above_inner",
        "SELECT * FROM t ASOF JOIN u MATCH_CONDITION (t.a >= u.a) ON t.b = u.d WHERE t.c > 3",
    ),
    (
        "as_of_left_filter_above_left",
        "SELECT * FROM t LEFT ASOF JOIN u MATCH_CONDITION (t.a >= u.a) ON t.b = u.d WHERE t.c > 3",
    ),
    (
        "as_of_right_filter_above_stays",
        "SELECT * FROM t ASOF JOIN u MATCH_CONDITION (t.a >= u.a) ON t.b = u.d WHERE u.d = 'x'",
    ),
    (
        "as_of_both_sides_filter_above_stays",
        "SELECT * FROM t ASOF JOIN u MATCH_CONDITION (t.a >= u.a) ON t.b = u.d WHERE t.c > length(u.d)",
    ),
    (
        "as_of_nondeterministic_filter_above_stays",
        // A nondeterministic `ON` conjunct is refused by the analyzer, as in Spark.
        "SELECT * FROM t ASOF JOIN u MATCH_CONDITION (t.a >= u.a) ON t.b = u.d WHERE t.c > 3 AND rand() > 0.5",
    ),
];

fn resolve(sql: &str) -> Plan {
    let options = krishiv_sql::ParserOptions {
        asof_join: true,
        ..krishiv_sql::ParserOptions::default()
    };
    let plan = krishiv_sql::parse_and_lower(sql, &options, SourceId(0))
        .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    block_on(DefaultAnalyzer.analyze(&plan, &support::ctx()))
        .unwrap_or_else(|e| panic!("{sql}: {e:?}"))
        .plan
}

#[test]
fn as_of_conditions_and_filters_move_where_they_may() {
    for (name, sql) in CASES {
        let plan = resolve(sql);
        let optimized = optimize_with(&plan, &support::AllDistinct).unwrap();
        assert_eq!(
            optimized.root().resolved_output(),
            plan.root().resolved_output(),
            "{name} keeps its output"
        );
        let again = optimize_with(&optimized, &support::AllDistinct).unwrap();
        assert!(
            again.structurally_eq(&optimized),
            "{name} is a fixed point:\nFIRST\n{}\nAGAIN\n{}",
            optimized.root(),
            again.root()
        );
        insta::assert_snapshot!(*name, optimized.root().to_string(), sql);
    }
}

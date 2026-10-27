// SPDX-License-Identifier: Apache-2.0
//! The 22 TPC-H queries (`bench/queries/tpch`) built by hand as unresolved
//! plans, shaped as Spark's parser shapes them, so they can serve as the
//! oracle for SQL lowering: comma joins are inner joins without a condition,
//! folded left; `AND`/`OR` chains are balanced trees; `<>` is `not(==)`; a
//! `SELECT` with aggregates but no `GROUP BY` is a `Project`; `HAVING` is a
//! `Filter` over the `Aggregate`.

use std::sync::Arc;

use super::dsl::*;
use crate::{
    Cte, Expr, ExprNode, FunctionCall, JoinCondition, JoinKind, Literal, LogicalPlan, PlanNode,
};

fn q01() -> LogicalPlan {
    from(&["lineitem"])
        .filter(le(c("l_shipdate"), date("1998-09-02")))
        .aggregate(
            cols(&["l_returnflag", "l_linestatus"]),
            vec![
                c("l_returnflag"),
                c("l_linestatus"),
                sum(c("l_quantity")).alias("sum_qty"),
                sum(c("l_extendedprice")).alias("sum_base_price"),
                sum(disc_price()).alias("sum_disc_price"),
                sum(mul(disc_price(), add(int(1), c("l_tax")))).alias("sum_charge"),
                avg(c("l_quantity")).alias("avg_qty"),
                avg(c("l_extendedprice")).alias("avg_price"),
                avg(c("l_discount")).alias("avg_disc"),
                count_star().alias("count_order"),
            ],
        )
        .sort(vec![asc(c("l_returnflag")), asc(c("l_linestatus"))])
}

fn q02() -> LogicalPlan {
    let min_cost = from(&["partsupp", "supplier", "nation", "region"])
        .filter(and(vec![
            eq(c("p_partkey"), c("ps_partkey")),
            eq(c("s_suppkey"), c("ps_suppkey")),
            eq(c("s_nationkey"), c("n_nationkey")),
            eq(c("n_regionkey"), c("r_regionkey")),
            eq(c("r_name"), s("EUROPE")),
        ]))
        .project(vec![f("min", vec![c("ps_supplycost")])]);
    from(&["part", "supplier", "partsupp", "nation", "region"])
        .filter(and(vec![
            eq(c("p_partkey"), c("ps_partkey")),
            eq(c("s_suppkey"), c("ps_suppkey")),
            eq(c("p_size"), int(15)),
            like(c("p_type"), "%BRASS"),
            eq(c("s_nationkey"), c("n_nationkey")),
            eq(c("n_regionkey"), c("r_regionkey")),
            eq(c("r_name"), s("EUROPE")),
            eq(c("ps_supplycost"), scalar(min_cost)),
        ]))
        .project(cols(&[
            "s_acctbal",
            "s_name",
            "n_name",
            "p_partkey",
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        ]))
        .sort(vec![
            desc(c("s_acctbal")),
            asc(c("n_name")),
            asc(c("s_name")),
            asc(c("p_partkey")),
        ])
        .limit(int(100))
}

fn q03() -> LogicalPlan {
    from(&["customer", "orders", "lineitem"])
        .filter(and(vec![
            eq(c("c_mktsegment"), s("BUILDING")),
            eq(c("c_custkey"), c("o_custkey")),
            eq(c("l_orderkey"), c("o_orderkey")),
            lt(c("o_orderdate"), date("1995-03-15")),
            gt(c("l_shipdate"), date("1995-03-15")),
        ]))
        .aggregate(
            cols(&["l_orderkey", "o_orderdate", "o_shippriority"]),
            vec![
                c("l_orderkey"),
                sum(disc_price()).alias("revenue"),
                c("o_orderdate"),
                c("o_shippriority"),
            ],
        )
        .sort(vec![desc(c("revenue")), asc(c("o_orderdate"))])
        .limit(int(10))
}

fn q04() -> LogicalPlan {
    let late = from(&["lineitem"])
        .filter(and(vec![
            eq(c("l_orderkey"), c("o_orderkey")),
            lt(c("l_commitdate"), c("l_receiptdate")),
        ]))
        .project(vec![star()]);
    from(&["orders"])
        .filter(and(vec![
            ge(c("o_orderdate"), date("1993-07-01")),
            lt(c("o_orderdate"), date("1993-10-01")),
            exists(late),
        ]))
        .aggregate(
            cols(&["o_orderpriority"]),
            vec![c("o_orderpriority"), count_star().alias("order_count")],
        )
        .sort(vec![asc(c("o_orderpriority"))])
}

fn q05() -> LogicalPlan {
    from(&[
        "customer", "orders", "lineitem", "supplier", "nation", "region",
    ])
    .filter(and(vec![
        eq(c("c_custkey"), c("o_custkey")),
        eq(c("l_orderkey"), c("o_orderkey")),
        eq(c("l_suppkey"), c("s_suppkey")),
        eq(c("c_nationkey"), c("s_nationkey")),
        eq(c("s_nationkey"), c("n_nationkey")),
        eq(c("n_regionkey"), c("r_regionkey")),
        eq(c("r_name"), s("ASIA")),
        ge(c("o_orderdate"), date("1994-01-01")),
        lt(c("o_orderdate"), date("1995-01-01")),
    ]))
    .aggregate(
        cols(&["n_name"]),
        vec![c("n_name"), sum(disc_price()).alias("revenue")],
    )
    .sort(vec![desc(c("revenue"))])
}

fn q06() -> LogicalPlan {
    from(&["lineitem"])
        .filter(and(vec![
            ge(c("l_shipdate"), date("1994-01-01")),
            lt(c("l_shipdate"), date("1995-01-01")),
            between(c("l_discount"), dec("0.05"), dec("0.07")),
            lt(c("l_quantity"), int(24)),
        ]))
        .project(vec![
            sum(mul(c("l_extendedprice"), c("l_discount"))).alias("revenue"),
        ])
}

fn q07() -> LogicalPlan {
    let shipping = from(&[
        "supplier",
        "lineitem",
        "orders",
        "customer",
        "nation n1",
        "nation n2",
    ])
    .filter(and(vec![
        eq(c("s_suppkey"), c("l_suppkey")),
        eq(c("o_orderkey"), c("l_orderkey")),
        eq(c("c_custkey"), c("o_custkey")),
        eq(c("s_nationkey"), c("n1.n_nationkey")),
        eq(c("c_nationkey"), c("n2.n_nationkey")),
        or(
            and(vec![
                eq(c("n1.n_name"), s("FRANCE")),
                eq(c("n2.n_name"), s("GERMANY")),
            ]),
            and(vec![
                eq(c("n1.n_name"), s("GERMANY")),
                eq(c("n2.n_name"), s("FRANCE")),
            ]),
        ),
        between(c("l_shipdate"), date("1995-01-01"), date("1996-12-31")),
    ]))
    .project(vec![
        c("n1.n_name").alias("supp_nation"),
        c("n2.n_name").alias("cust_nation"),
        year(c("l_shipdate")).alias("l_year"),
        disc_price().alias("volume"),
    ])
    .alias("shipping");
    shipping
        .aggregate(
            cols(&["supp_nation", "cust_nation", "l_year"]),
            vec![
                c("supp_nation"),
                c("cust_nation"),
                c("l_year"),
                sum(c("volume")).alias("revenue"),
            ],
        )
        .sort(vec![
            asc(c("supp_nation")),
            asc(c("cust_nation")),
            asc(c("l_year")),
        ])
}

fn q08() -> LogicalPlan {
    let all_nations = from(&[
        "part",
        "supplier",
        "lineitem",
        "orders",
        "customer",
        "nation n1",
        "nation n2",
        "region",
    ])
    .filter(and(vec![
        eq(c("p_partkey"), c("l_partkey")),
        eq(c("s_suppkey"), c("l_suppkey")),
        eq(c("l_orderkey"), c("o_orderkey")),
        eq(c("o_custkey"), c("c_custkey")),
        eq(c("c_nationkey"), c("n1.n_nationkey")),
        eq(c("n1.n_regionkey"), c("r_regionkey")),
        eq(c("r_name"), s("AMERICA")),
        eq(c("s_nationkey"), c("n2.n_nationkey")),
        between(c("o_orderdate"), date("1995-01-01"), date("1996-12-31")),
        eq(c("p_type"), s("ECONOMY ANODIZED STEEL")),
    ]))
    .project(vec![
        year(c("o_orderdate")).alias("o_year"),
        disc_price().alias("volume"),
        c("n2.n_name").alias("nation"),
    ])
    .alias("all_nations");
    let brazil = sum(when(eq(c("nation"), s("BRAZIL")), c("volume"), int(0)));
    all_nations
        .aggregate(
            cols(&["o_year"]),
            vec![
                c("o_year"),
                div(brazil, sum(c("volume"))).alias("mkt_share"),
            ],
        )
        .sort(vec![asc(c("o_year"))])
}

fn q09() -> LogicalPlan {
    let profit = from(&[
        "part", "supplier", "lineitem", "partsupp", "orders", "nation",
    ])
    .filter(and(vec![
        eq(c("s_suppkey"), c("l_suppkey")),
        eq(c("ps_suppkey"), c("l_suppkey")),
        eq(c("ps_partkey"), c("l_partkey")),
        eq(c("p_partkey"), c("l_partkey")),
        eq(c("o_orderkey"), c("l_orderkey")),
        eq(c("s_nationkey"), c("n_nationkey")),
        like(c("p_name"), "%green%"),
    ]))
    .project(vec![
        c("n_name").alias("nation"),
        year(c("o_orderdate")).alias("o_year"),
        sub(disc_price(), mul(c("ps_supplycost"), c("l_quantity"))).alias("amount"),
    ])
    .alias("profit");
    profit
        .aggregate(
            cols(&["nation", "o_year"]),
            vec![
                c("nation"),
                c("o_year"),
                sum(c("amount")).alias("sum_profit"),
            ],
        )
        .sort(vec![asc(c("nation")), desc(c("o_year"))])
}

fn q10() -> LogicalPlan {
    from(&["customer", "orders", "lineitem", "nation"])
        .filter(and(vec![
            eq(c("c_custkey"), c("o_custkey")),
            eq(c("l_orderkey"), c("o_orderkey")),
            ge(c("o_orderdate"), date("1993-10-01")),
            lt(c("o_orderdate"), date("1994-01-01")),
            eq(c("l_returnflag"), s("R")),
            eq(c("c_nationkey"), c("n_nationkey")),
        ]))
        .aggregate(
            cols(&[
                "c_custkey",
                "c_name",
                "c_acctbal",
                "c_phone",
                "n_name",
                "c_address",
                "c_comment",
            ]),
            vec![
                c("c_custkey"),
                c("c_name"),
                sum(disc_price()).alias("revenue"),
                c("c_acctbal"),
                c("n_name"),
                c("c_address"),
                c("c_phone"),
                c("c_comment"),
            ],
        )
        .sort(vec![desc(c("revenue"))])
        .limit(int(20))
}

fn q11() -> LogicalPlan {
    let value = || sum(mul(c("ps_supplycost"), c("ps_availqty")));
    let german = || {
        from(&["partsupp", "supplier", "nation"]).filter(and(vec![
            eq(c("ps_suppkey"), c("s_suppkey")),
            eq(c("s_nationkey"), c("n_nationkey")),
            eq(c("n_name"), s("GERMANY")),
        ]))
    };
    let threshold = german().project(vec![mul(value(), dec("0.0001000000"))]);
    german()
        .aggregate(
            cols(&["ps_partkey"]),
            vec![c("ps_partkey"), value().alias("value")],
        )
        .filter(gt(value(), scalar(threshold)))
        .sort(vec![desc(c("value"))])
}

fn q12() -> LogicalPlan {
    let urgent = || {
        or(
            eq(c("o_orderpriority"), s("1-URGENT")),
            eq(c("o_orderpriority"), s("2-HIGH")),
        )
    };
    let not_urgent = and(vec![
        ne(c("o_orderpriority"), s("1-URGENT")),
        ne(c("o_orderpriority"), s("2-HIGH")),
    ]);
    from(&["orders", "lineitem"])
        .filter(and(vec![
            eq(c("o_orderkey"), c("l_orderkey")),
            in_list(c("l_shipmode"), vec![s("MAIL"), s("SHIP")]),
            lt(c("l_commitdate"), c("l_receiptdate")),
            lt(c("l_shipdate"), c("l_commitdate")),
            ge(c("l_receiptdate"), date("1994-01-01")),
            lt(c("l_receiptdate"), date("1995-01-01")),
        ]))
        .aggregate(
            cols(&["l_shipmode"]),
            vec![
                c("l_shipmode"),
                sum(when(urgent(), int(1), int(0))).alias("high_line_count"),
                sum(when(not_urgent, int(1), int(0))).alias("low_line_count"),
            ],
        )
        .sort(vec![asc(c("l_shipmode"))])
}

fn q13() -> LogicalPlan {
    let c_orders = LogicalPlan::table(["customer"])
        .join(
            LogicalPlan::table(["orders"]),
            JoinKind::LeftOuter,
            JoinCondition::On(Arc::new(and(vec![
                eq(c("c_custkey"), c("o_custkey")),
                not(like(c("o_comment"), "%special%requests%")),
            ]))),
        )
        .aggregate(
            cols(&["c_custkey"]),
            vec![c("c_custkey"), f("count", vec![c("o_orderkey")])],
        );
    let c_orders = LogicalPlan::new(PlanNode::SubqueryAlias {
        input: Arc::new(c_orders),
        alias: "c_orders".to_owned(),
        columns: vec!["c_custkey".to_owned(), "c_count".to_owned()],
    });
    c_orders
        .aggregate(
            cols(&["c_count"]),
            vec![c("c_count"), count_star().alias("custdist")],
        )
        .sort(vec![desc(c("custdist")), desc(c("c_count"))])
}

fn q14() -> LogicalPlan {
    let promo = sum(when(like(c("p_type"), "PROMO%"), disc_price(), int(0)));
    from(&["lineitem", "part"])
        .filter(and(vec![
            eq(c("l_partkey"), c("p_partkey")),
            ge(
                c("l_shipdate"),
                Literal::Date(9374).into(), // date '1995-09-01'
            ),
            lt(c("l_shipdate"), date("1995-10-01")),
        ]))
        .project(vec![
            div(mul(dec("100.00"), promo), sum(disc_price())).alias("promo_revenue"),
        ])
}

fn q15() -> LogicalPlan {
    let revenue = from(&["lineitem"])
        .filter(and(vec![
            ge(c("l_shipdate"), date("1996-01-01")),
            lt(c("l_shipdate"), date("1996-04-01")),
        ]))
        .aggregate(
            cols(&["supplier_no"]),
            vec![
                c("l_suppkey").alias("supplier_no"),
                sum(disc_price()).alias("total_revenue"),
            ],
        );
    let max_revenue = from(&["revenue"]).project(vec![f("max", vec![c("total_revenue")])]);
    let main = from(&["supplier", "revenue"])
        .filter(and(vec![
            eq(c("s_suppkey"), c("supplier_no")),
            eq(c("total_revenue"), scalar(max_revenue)),
        ]))
        .project(cols(&[
            "s_suppkey",
            "s_name",
            "s_address",
            "s_phone",
            "total_revenue",
        ]))
        .sort(vec![asc(c("s_suppkey"))]);
    LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(main),
        ctes: vec![Cte {
            name: "revenue".to_owned(),
            plan: Arc::new(revenue),
        }],
    })
}

fn q16() -> LogicalPlan {
    let complaints = from(&["supplier"])
        .filter(like(c("s_comment"), "%Customer%Complaints%"))
        .project(cols(&["s_suppkey"]));
    let supplier_cnt = Expr::new(ExprNode::Function(FunctionCall {
        name: "count".to_owned(),
        qualifier: Vec::new(),
        arg_names: Vec::new(),
        args: vec![Arc::new(c("ps_suppkey"))],
        distinct: true,
        filter: None,
    }));
    from(&["partsupp", "part"])
        .filter(and(vec![
            eq(c("p_partkey"), c("ps_partkey")),
            ne(c("p_brand"), s("Brand#45")),
            not(like(c("p_type"), "MEDIUM POLISHED%")),
            in_list(
                c("p_size"),
                [49, 14, 23, 45, 19, 3, 36, 9]
                    .into_iter()
                    .map(int)
                    .collect(),
            ),
            not(in_subquery(c("ps_suppkey"), complaints)),
        ]))
        .aggregate(
            cols(&["p_brand", "p_type", "p_size"]),
            vec![
                c("p_brand"),
                c("p_type"),
                c("p_size"),
                supplier_cnt.alias("supplier_cnt"),
            ],
        )
        .sort(vec![
            desc(c("supplier_cnt")),
            asc(c("p_brand")),
            asc(c("p_type")),
            asc(c("p_size")),
        ])
}

fn q17() -> LogicalPlan {
    let avg_qty = from(&["lineitem"])
        .filter(eq(c("l_partkey"), c("p_partkey")))
        .project(vec![mul(dec("0.2"), avg(c("l_quantity")))]);
    from(&["lineitem", "part"])
        .filter(and(vec![
            eq(c("p_partkey"), c("l_partkey")),
            eq(c("p_brand"), s("Brand#23")),
            eq(c("p_container"), s("MED BOX")),
            lt(c("l_quantity"), scalar(avg_qty)),
        ]))
        .project(vec![
            div(sum(c("l_extendedprice")), dec("7.0")).alias("avg_yearly"),
        ])
}

fn q18() -> LogicalPlan {
    let large = from(&["lineitem"])
        .aggregate(cols(&["l_orderkey"]), cols(&["l_orderkey"]))
        .filter(gt(sum(c("l_quantity")), int(300)));
    from(&["customer", "orders", "lineitem"])
        .filter(and(vec![
            in_subquery(c("o_orderkey"), large),
            eq(c("c_custkey"), c("o_custkey")),
            eq(c("o_orderkey"), c("l_orderkey")),
        ]))
        .aggregate(
            cols(&[
                "c_name",
                "c_custkey",
                "o_orderkey",
                "o_orderdate",
                "o_totalprice",
            ]),
            vec![
                c("c_name"),
                c("c_custkey"),
                c("o_orderkey"),
                c("o_orderdate"),
                c("o_totalprice"),
                sum(c("l_quantity")),
            ],
        )
        .sort(vec![desc(c("o_totalprice")), asc(c("o_orderdate"))])
        .limit(int(100))
}

fn q19() -> LogicalPlan {
    let branch = |brand: &str, containers: [&str; 4], qty: i32, size: i32| {
        and(vec![
            eq(c("p_partkey"), c("l_partkey")),
            eq(c("p_brand"), s(brand)),
            in_list(c("p_container"), containers.iter().map(|v| s(v)).collect()),
            ge(c("l_quantity"), int(qty)),
            le(c("l_quantity"), add(int(qty), int(10))),
            between(c("p_size"), int(1), int(size)),
            in_list(c("l_shipmode"), vec![s("AIR"), s("AIR REG")]),
            eq(c("l_shipinstruct"), s("DELIVER IN PERSON")),
        ])
    };
    from(&["lineitem", "part"])
        .filter(any(vec![
            branch("Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1, 5),
            branch(
                "Brand#23",
                ["MED BAG", "MED BOX", "MED PKG", "MED PACK"],
                10,
                10,
            ),
            branch(
                "Brand#34",
                ["LG CASE", "LG BOX", "LG PACK", "LG PKG"],
                20,
                15,
            ),
        ]))
        .project(vec![sum(disc_price()).alias("revenue")])
}

fn q20() -> LogicalPlan {
    let forest = from(&["part"])
        .filter(like(c("p_name"), "forest%"))
        .project(cols(&["p_partkey"]));
    let half_qty = from(&["lineitem"])
        .filter(and(vec![
            eq(c("l_partkey"), c("ps_partkey")),
            eq(c("l_suppkey"), c("ps_suppkey")),
            ge(c("l_shipdate"), date("1994-01-01")),
            lt(c("l_shipdate"), date("1995-01-01")),
        ]))
        .project(vec![mul(dec("0.5"), sum(c("l_quantity")))]);
    let excess = from(&["partsupp"])
        .filter(and(vec![
            in_subquery(c("ps_partkey"), forest),
            gt(c("ps_availqty"), scalar(half_qty)),
        ]))
        .project(cols(&["ps_suppkey"]));
    from(&["supplier", "nation"])
        .filter(and(vec![
            in_subquery(c("s_suppkey"), excess),
            eq(c("s_nationkey"), c("n_nationkey")),
            eq(c("n_name"), s("CANADA")),
        ]))
        .project(cols(&["s_name", "s_address"]))
        .sort(vec![asc(c("s_name"))])
}

fn q21() -> LogicalPlan {
    let other_supplier = |alias: &str, late: bool| {
        let mut terms = vec![
            eq(c(&format!("{alias}.l_orderkey")), c("l1.l_orderkey")),
            ne(c(&format!("{alias}.l_suppkey")), c("l1.l_suppkey")),
        ];
        if late {
            terms.push(gt(
                c(&format!("{alias}.l_receiptdate")),
                c(&format!("{alias}.l_commitdate")),
            ));
        }
        from(&[&format!("lineitem {alias}")])
            .filter(and(terms))
            .project(vec![star()])
    };
    from(&["supplier", "lineitem l1", "orders", "nation"])
        .filter(and(vec![
            eq(c("s_suppkey"), c("l1.l_suppkey")),
            eq(c("o_orderkey"), c("l1.l_orderkey")),
            eq(c("o_orderstatus"), s("F")),
            gt(c("l1.l_receiptdate"), c("l1.l_commitdate")),
            exists(other_supplier("l2", false)),
            not(exists(other_supplier("l3", true))),
            eq(c("s_nationkey"), c("n_nationkey")),
            eq(c("n_name"), s("SAUDI ARABIA")),
        ]))
        .aggregate(
            cols(&["s_name"]),
            vec![c("s_name"), count_star().alias("numwait")],
        )
        .sort(vec![desc(c("numwait")), asc(c("s_name"))])
        .limit(int(100))
}

fn q22() -> LogicalPlan {
    let avg_balance = from(&["customer"])
        .filter(and(vec![gt(c("c_acctbal"), dec("0.00")), cntrycode_in()]))
        .project(vec![avg(c("c_acctbal"))]);
    let has_orders = from(&["orders"])
        .filter(eq(c("o_custkey"), c("c_custkey")))
        .project(vec![star()]);
    let custsale = from(&["customer"])
        .filter(and(vec![
            cntrycode_in(),
            gt(c("c_acctbal"), scalar(avg_balance)),
            not(exists(has_orders)),
        ]))
        .project(vec![
            f("substring", vec![c("c_phone"), int(1), int(2)]).alias("cntrycode"),
            c("c_acctbal"),
        ])
        .alias("custsale");
    custsale
        .aggregate(
            cols(&["cntrycode"]),
            vec![
                c("cntrycode"),
                count_star().alias("numcust"),
                sum(c("c_acctbal")).alias("totacctbal"),
            ],
        )
        .sort(vec![asc(c("cntrycode"))])
}

pub(super) const QUERIES: [fn() -> LogicalPlan; 22] = [
    q01, q02, q03, q04, q05, q06, q07, q08, q09, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19,
    q20, q21, q22,
];

#[test]
fn pretty_printer_snapshots() {
    for (i, q) in QUERIES.iter().enumerate() {
        let plan = q();
        assert!(!plan.is_resolved(), "q{:02} is unresolved", i + 1);
        insta::assert_snapshot!(format!("q{:02}", i + 1), plan.to_string());
    }
}

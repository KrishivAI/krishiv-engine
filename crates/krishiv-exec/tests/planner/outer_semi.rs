// SPDX-License-Identifier: Apache-2.0
//! The semi join below an outer join (optimizer `outer_semi`, TPC-DS q40's
//! shape): where it reduces an outer join's preserved side, the executed plan
//! returns the rows of the same plan with the rule off.
//!
//! The data set is the committed TPC-H SF0.01 Parquet fixture.

use std::path::Path;

use super::partial_aggregate::{physical, runtime, sorted};
use super::run;
use super::tpch::parquet_tables;

/// `lineitem LEFT JOIN partsupp` joined to a filtered `part`, a right outer
/// form, a NULL preserved key, and a key that repeats on the inner join's
/// other side (five nations a region), with the rule on and off.
#[test]
fn semi_joins_below_outer_joins_return_the_rows_of_the_rule_off() {
    let rt = runtime();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let t = parquet_tables(&dir, &rt);
    let queries = [
        "SELECT p_brand, count(*), count(ps_supplycost), sum(l_quantity) \
         FROM lineitem LEFT JOIN partsupp ON l_partkey = ps_partkey \
         AND l_suppkey = ps_suppkey, part \
         WHERE l_partkey = p_partkey AND p_size = 3 GROUP BY p_brand",
        "SELECT p_brand, count(*), count(ps_supplycost) FROM partsupp \
         RIGHT JOIN lineitem ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey, part \
         WHERE l_partkey = p_partkey AND p_size = 3 GROUP BY p_brand",
        "SELECT p_brand, count(*), count(ps_supplycost) FROM (SELECT l_suppkey, \
         CASE WHEN l_orderkey % 3 = 0 THEN NULL ELSE l_partkey END AS pk FROM lineitem) l \
         LEFT JOIN partsupp ON pk = ps_partkey AND l_suppkey = ps_suppkey, part \
         WHERE pk = p_partkey AND p_size = 3 GROUP BY p_brand",
        "SELECT n_name, count(*), count(c_custkey) FROM supplier \
         LEFT JOIN customer ON s_nationkey = c_nationkey AND c_acctbal > 9000, nation \
         WHERE s_nationkey = n_regionkey AND n_name = 'FRANCE' GROUP BY n_name",
    ];
    let off = [("krishiv.optimizer.semiBelowOuterJoin", "false")];
    for q in queries {
        let (with, steps) = t.optimized_in(q, &[]).unwrap();
        let (without, _) = t.optimized_in(q, &off).unwrap();
        let notes: Vec<&String> = steps
            .iter()
            .filter(|s| s.batch == "SemiBelowOuterJoin")
            .map(|s| &s.note)
            .collect();
        assert!(
            notes.iter().any(|n| n.starts_with("added")) && !with.structurally_eq(&without),
            "{q}\n{notes:?}\n{}",
            with.root()
        );
        let got = sorted(&run(&physical(&t, &with).unwrap()).unwrap());
        let want = sorted(&run(&physical(&t, &without).unwrap()).unwrap());
        assert!(!want.is_empty(), "{q}");
        assert_eq!(got, want, "{q}\n{}", with.root());
    }
}

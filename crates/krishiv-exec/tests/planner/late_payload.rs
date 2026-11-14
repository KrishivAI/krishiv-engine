// SPDX-License-Identifier: Apache-2.0
//! The late payload join rule (P2b S5 lever (2), optimizer `late_payload`):
//! where it moves a dimension's payload above the semi and anti joins, the
//! executed plan returns the rows of the same plan with the rule off.
//!
//! The data set is the committed TPC-H SF0.01 Parquet fixture (the design note
//! asked for SF0.1, which is not committed).

use std::path::Path;

use super::partial_aggregate::{physical, runtime, sorted};
use super::run;
use super::tpch::{parquet_tables, sql};

/// TPC-H q21 and q18, and queries whose dimension key repeats (five nations a
/// region), with the rule on and off over the same statistics.
#[test]
fn moved_payload_joins_return_the_rows_of_the_rule_off() {
    let rt = runtime();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let t = parquet_tables(&dir, &rt);
    let queries = [
        sql(21),
        sql(18),
        "SELECT n_name, count(*) FROM supplier, nation WHERE s_nationkey = n_regionkey \
         AND EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey) GROUP BY n_name"
            .to_owned(),
        "SELECT n_name, count(*), sum(s_acctbal) FROM supplier, nation \
         WHERE s_nationkey = n_regionkey \
         AND NOT EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey \
         AND l_comment LIKE '%xyzzy%') GROUP BY n_name"
            .to_owned(),
    ];
    // A large query's semi and anti joins are not broadcast; at SF0.01 every
    // one would be, and then none builds the path.
    let on = [("spark.sql.autoBroadcastJoinThreshold", "-1")];
    let off = [
        ("spark.sql.autoBroadcastJoinThreshold", "-1"),
        ("krishiv.optimizer.latePayloadJoin", "false"),
    ];
    for q in &queries {
        let (with, steps) = t.optimized_in(q, &on).unwrap();
        let (without, _) = t.optimized_in(q, &off).unwrap();
        let notes: Vec<&String> = steps
            .iter()
            .filter(|s| s.batch == "LatePayloadJoin")
            .map(|s| &s.note)
            .collect();
        // The rule fired: it is the only difference between the two plans.
        assert!(
            notes.iter().any(|n| n.starts_with("moved")) && !with.structurally_eq(&without),
            "{q}\n{notes:?}\n{}",
            with.root()
        );
        let got = sorted(&run(&physical(&t, &with).unwrap()).unwrap());
        let want = sorted(&run(&physical(&t, &without).unwrap()).unwrap());
        assert!(!want.is_empty(), "{q}");
        assert_eq!(got, want, "{q}\n{}", with.root());
    }
}

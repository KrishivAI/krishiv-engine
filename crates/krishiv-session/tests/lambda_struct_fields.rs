// SPDX-License-Identifier: Apache-2.0
//! A field of a struct-typed lambda variable (`acc.sum`), as in the Spark
//! 4.2.0 `higher-order-functions` golden.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use futures::executor::block_on;
use krishiv_session::Session;

#[test]
fn aggregate_reads_the_fields_of_its_accumulator() {
    let s = Session::new().unwrap();
    for f in ["aggregate", "reduce"] {
        let df = block_on(s.sql(&format!(
            "SELECT {f}(array(1, 2, 3), (0 AS sum, 0 AS n), (acc, x) -> (acc.sum + x, acc.n + 1), acc -> acc.sum / acc.n)"
        )))
        .unwrap();
        let b = block_on(df.collect()).unwrap();
        assert_eq!(b[0].column(0).as_primitive::<Float64Type>().value(0), 2.0);
    }
}

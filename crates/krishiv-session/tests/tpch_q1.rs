// SPDX-License-Identifier: Apache-2.0
//! TPC-H Q1 through `Session::sql` and through the DataFrame API: both
//! give the same rows. A small checked-in `lineitem` runs in CI; set
//! `KRISHIV_TPCH_SF1` to a directory holding `lineitem.parquet` to run the
//! same check at scale factor 1. The date bound is written as
//! `date '1998-09-02'` (`date '1998-12-01' - interval '90' day` folded):
//! interval literals are not supported yet.
#![cfg(test)]

use arrow_array::RecordBatch;
use arrow_select::concat::concat_batches;
use futures::executor::block_on;
use krishiv_session::functions::{avg, col, count_star, lit, sum};
use krishiv_session::{DataFrame, Session};
use krishiv_types::DataType;

const Q1: &str = "\
SELECT l_returnflag, l_linestatus,
       sum(l_quantity) AS sum_qty,
       sum(l_extendedprice) AS sum_base_price,
       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
       avg(l_quantity) AS avg_qty,
       avg(l_extendedprice) AS avg_price,
       avg(l_discount) AS avg_disc,
       count(*) AS count_order
FROM lineitem
WHERE l_shipdate <= date '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus";

/// Q1 in the DataFrame API, over the read `lineitem`.
fn q1(lineitem: &DataFrame) -> DataFrame {
    let disc_price = || col("l_extendedprice") * (lit(1) - col("l_discount"));
    lineitem
        .filter(col("l_shipdate").lt_eq(lit("1998-09-02").cast(DataType::Date)))
        .group_by([col("l_returnflag"), col("l_linestatus")])
        .agg([
            sum(col("l_quantity")).alias("sum_qty"),
            sum(col("l_extendedprice")).alias("sum_base_price"),
            sum(disc_price()).alias("sum_disc_price"),
            sum(disc_price() * (lit(1) + col("l_tax"))).alias("sum_charge"),
            avg(col("l_quantity")).alias("avg_qty"),
            avg(col("l_extendedprice")).alias("avg_price"),
            avg(col("l_discount")).alias("avg_disc"),
            count_star().alias("count_order"),
        ])
        .order_by([col("l_returnflag"), col("l_linestatus")])
}

/// The result as one batch.
fn one(batches: &[RecordBatch]) -> RecordBatch {
    concat_batches(&batches[0].schema(), batches).unwrap()
}

/// The rows as `|`-separated text (strings, decimals and longs: Q1's
/// types).
fn rows(batch: &RecordBatch) -> Vec<String> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal128Type, Int64Type};
    use arrow_schema::DataType as A;
    (0..batch.num_rows())
        .map(|r| {
            batch
                .columns()
                .iter()
                .map(|c| match c.data_type() {
                    A::Utf8 => c.as_string::<i32>().value(r).to_owned(),
                    A::Utf8View => c.as_string_view().value(r).to_owned(),
                    A::Decimal128(..) => c.as_primitive::<Decimal128Type>().value_as_string(r),
                    A::Int64 => c.as_primitive::<Int64Type>().value(r).to_string(),
                    other => panic!("unexpected type {other}"),
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

/// Q1 both ways over `path`; the rows.
fn both_ways(path: &str) -> RecordBatch {
    let s = Session::new().unwrap();
    let lineitem = s.read().parquet([path]);
    block_on(lineitem.create_or_replace_temp_view("lineitem")).unwrap();
    let sql = block_on(s.sql(Q1)).unwrap();
    let df = q1(&lineitem);
    assert_eq!(
        block_on(sql.schema()).unwrap(),
        block_on(df.schema()).unwrap()
    );
    let a = one(&block_on(sql.collect()).unwrap());
    let b = one(&block_on(df.collect()).unwrap());
    assert_eq!(a, b);
    a
}

#[test]
fn q1_sql_and_dataframe_agree_at_sf1() {
    let Ok(dir) = std::env::var("KRISHIV_TPCH_SF1") else {
        return;
    };
    // The TPC-H answer set for Q1 at SF1, with Spark's result scales. The
    // `N|O` `sum_disc_price` ends in 6, not the specification's 5: the
    // files come from DuckDB's generator, and DuckDB computes the same
    // value over them.
    assert_eq!(
        rows(&both_ways(&format!("{dir}/lineitem.parquet"))),
        [
            "A|F|37734107.00|56586554400.73|53758257134.8700|55909065222.827692|25.522006|38273.129735|0.049985|1478493",
            "N|F|991417.00|1487504710.38|1413082168.0541|1469649223.194375|25.516472|38284.467761|0.050093|38854",
            "N|O|74476040.00|111701729697.74|106118230307.6056|110367043872.497010|25.502227|38249.117989|0.049997|2920374",
            "R|F|37719753.00|56568041380.90|53741292684.6040|55889619119.831932|25.505794|38250.854626|0.050009|1478870",
        ]
    );
}

#[test]
fn q1_sql_and_dataframe_agree_on_the_fixture() {
    // 5,955 SF1 rows (`l_orderkey % 4000 = 7`, 87 shipped after the
    // cutoff); expected values computed by DuckDB over the same file,
    // averages rounded half-up to Spark's scale 6.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tpch/q1-lineitem.parquet"
    );
    assert_eq!(
        rows(&both_ways(path)),
        [
            "A|F|38726.00|58248853.52|55351018.3989|57501389.798720|25.990604|39093.190282|0.049779|1490",
            "N|F|780.00|1250893.64|1196162.6245|1236223.665904|23.636364|37905.867879|0.042121|33",
            "N|O|74790.00|112099053.24|106417493.3379|110726631.475009|25.656947|38455.935931|0.049822|2915",
            "R|F|36245.00|53924848.79|51310184.5452|53388426.326085|25.346154|37709.684469|0.049315|1430",
        ]
    );
}

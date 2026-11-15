# Legacy Parquet layouts

- `users.parquet`: copied unchanged from apache/spark v4.2.0
  (`examples/src/main/resources/users.parquet`, also
  `sql/connect/common/src/test/resources/query-tests/test-data/users.parquet`; Apache-2.0).
  Written by parquet-mr 1.4.3 through Avro: `favorite_numbers` is the legacy two-level list
  `required group favorite_numbers (LIST) { repeated int32 array; }`.
  Spark 4.2.0 (`spark.read.load`) reads the schema
  `name string, favorite_color string, favorite_numbers array<int>` (all nullable) and the rows
  `[Alyssa,null,[3, 9, 15, 20]]`, `[Ben,red,[]]`. Checked by
  `crates/krishiv-testkit/tests/parquet_interop.rs` (`spark_users_two_level_list`, PY-043).

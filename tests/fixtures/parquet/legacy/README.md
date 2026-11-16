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
- `required_nested.parquet`: written by pyarrow with every field required: `id int`,
  `m map<struct<a int>, int>`, `l array<struct<x int>>`; rows `(1, {{a: 1} -> 10}, [{x: 5}])`,
  `(2, {}, [])`. Spark 4.2.0 reads every field at every depth as nullable (the map key's struct
  field too) and the same rows. Generator:
  `pa.map_(pa.field('key', pa.struct([pa.field('a', pa.int32(), nullable=False)]), nullable=False),
  pa.field('value', pa.int32(), nullable=False))`, the list's element and its `x` likewise
  `nullable=False`, written with `pq.write_table`. Checked by
  `crates/krishiv-session/tests/variant_parquet.rs` (`required_nested_fields_read_nullable`).

Spark 3.5 to 4.2 read both files the same way (`DataSource.resolveRelation` over
`dataSchema.asNullable`).

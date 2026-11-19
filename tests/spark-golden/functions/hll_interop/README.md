# HLL interop with Spark (P3-T086, design §4)

- `fixture.json`: `gen.py` on Spark 4.2.0: `hll_sketch_agg` (lgK 12) over 100,000 distinct `int`, `bigint`,
  `string` and `binary` values (estimation mode), with Spark's printed `hll_sketch_estimate`.
- `krishiv-compact.json`: Krishiv's compact sketch of the same data (written by the test with
  `KRISHIV_HLL_INTEROP_DUMP=<path>`). `reverse.json`: Spark 4.2.0 reading those bytes: the same estimate alone and
  unioned with its own sketch. Regenerate: run the test with the dump variable, then
  `spark-submit gen.py reverse.json krishiv-compact.json` in `apache/spark:4.2.0`.
- Test: `crates/krishiv-session/tests/aggregate_hll_sketches.rs` `spark_written_estimation_mode_sketches_agree_with_krishivs`.
- `hll.sql` #10 stays a known failure: `CAST(X'C1' AS STRING)` makes invalid UTF-8, which Krishiv's string carrier
  cannot hold.

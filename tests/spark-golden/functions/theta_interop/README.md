# Theta interop with Spark (P3 PR-49 part 2, design P3-T086 §4)

- `fixture.json`: `gen.py` on Spark 4.3.0: `theta_sketch_agg` (lgNomEntries 12) over 100,000 distinct `int`, `bigint`,
  `float`, `double`, `string`, `binary`, `array<int>` and `array<bigint>` values (estimation mode), with Spark's
  printed `theta_sketch_estimate`, and `intersection_of_nothing`: Spark 4.3.0's `theta_sketch_estimate(
  theta_intersection_agg(NULL binary))` (0). The 4.2.1 and 4.1.4 results of that statement are not recorded: no
  oracle of those versions was at hand.
- `krishiv-compact.json`: Krishiv's compact sketch of the same data (the test writes it with
  `KRISHIV_THETA_INTEROP_DUMP=<path>`). `reverse.json`: Spark 4.3.0 reading those bytes: the same estimate alone and
  unioned with its own sketch. Regenerate: run the test with the dump variable, then
  `spark-submit gen.py reverse.json krishiv-compact.json` (Spark 4.3.0 bin distribution, JDK 21).
- Test: `crates/krishiv-session/tests/aggregate_theta_sketches.rs`
  `spark_written_estimation_mode_sketches_agree_with_krishivs`.

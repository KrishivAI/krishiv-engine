-- query 22: nope(1)
SELECT percentile_approx(a, b) AS c1 FROM t22 WHERE x IN (1, 2) AND y = 'x(percentile_approx)';
SELECT kll_merge_agg_bigint(a, b) AS c2 FROM t22 WHERE x IN (1, 2) AND y = 'x(kll_merge_agg_bigint)';
SELECT encode(a, b) AS c3 FROM t22 WHERE x IN (1, 2) AND y = 'x(encode)';
SELECT isnotnull(a, b) AS c4 FROM t22 WHERE x IN (1, 2) AND y = 'x(isnotnull)';
SELECT from_utc_timestamp(a, b) AS c5 FROM t22 WHERE x IN (1, 2) AND y = 'x(from_utc_timestamp)';
SELECT kll_sketch_to_string_float(a, b) AS c6 FROM t22 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_float)';

-- query 29: nope(1)
SELECT kll_sketch_agg_bigint(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_bigint)';
SELECT approx_top_k_accumulate(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT kll_sketch_get_quantile_bigint(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_bigint)';
SELECT split_part(a, b) AS c4 FROM t29 WHERE x IN (1, 2) AND y = 'x(split_part)';
SELECT inline(a, b) AS c5 FROM t29 WHERE x IN (1, 2) AND y = 'x(inline)';

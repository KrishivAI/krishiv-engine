-- query 29: nope(1)
SELECT kll_sketch_agg_double(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_double)';
SELECT approx_top_k_accumulate(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT kll_sketch_get_quantile_double(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_double)';
SELECT split(a, b) AS c4 FROM t29 WHERE x IN (1, 2) AND y = 'x(split)';
SELECT initcap(a, b) AS c5 FROM t29 WHERE x IN (1, 2) AND y = 'x(initcap)';

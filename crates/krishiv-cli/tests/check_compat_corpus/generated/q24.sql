-- query 24: nope(1)
SELECT count_min_sketch(a, b) AS c1 FROM t24 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT tuple_sketch_agg_double(a, b) AS c2 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_double)';
SELECT approx_top_k_accumulate(a, b) AS c3 FROM t24 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT tinyint(a, b) AS c4 FROM t24 WHERE x IN (1, 2) AND y = 'x(tinyint)';
SELECT try_reflect(a, b) AS c5 FROM t24 WHERE x IN (1, 2) AND y = 'x(try_reflect)';
SELECT regr_slope(a, b) AS c6 FROM t24 WHERE x IN (1, 2) AND y = 'x(regr_slope)';

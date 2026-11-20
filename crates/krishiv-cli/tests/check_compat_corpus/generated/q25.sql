-- query 25: nope(1)
SELECT count_min_sketch(a, b) AS c1 FROM t25 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT tuple_sketch_estimate_double(a, b) AS c2 FROM t25 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_estimate_double)';
SELECT approx_top_k_accumulate(a, b) AS c3 FROM t25 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT timestamp_seconds(a, b) AS c4 FROM t25 WHERE x IN (1, 2) AND y = 'x(timestamp_seconds)';
SELECT tuple_difference_integer(a, b) AS c5 FROM t25 WHERE x IN (1, 2) AND y = 'x(tuple_difference_integer)';
SELECT regr_r2(a, b) AS c6 FROM t25 WHERE x IN (1, 2) AND y = 'x(regr_r2)';

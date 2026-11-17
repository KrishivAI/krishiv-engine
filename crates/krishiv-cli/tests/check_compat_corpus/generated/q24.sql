-- query 24: nope(1)
SELECT count_min_sketch(a, b) AS c1 FROM t24 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT tuple_intersection_theta_double(a, b) AS c2 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_double)';
SELECT approx_top_k_accumulate(a, b) AS c3 FROM t24 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT to_binary(a, b) AS c4 FROM t24 WHERE x IN (1, 2) AND y = 'x(to_binary)';
SELECT xpath_long(a, b) AS c5 FROM t24 WHERE x IN (1, 2) AND y = 'x(xpath_long)';
SELECT tuple_intersection_agg_double(a, b) AS c6 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_agg_double)';

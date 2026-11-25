-- query 24: nope(1)
SELECT tuple_sketch_theta_integer(a, b) AS c1 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_theta_integer)';
SELECT approx_top_k_combine(a, b) AS c2 FROM t24 WHERE x IN (1, 2) AND y = 'x(approx_top_k_combine)';
SELECT tuple_sketch_summary_double(a, b) AS c3 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_summary_double)';
SELECT approx_top_k(a, b) AS c4 FROM t24 WHERE x IN (1, 2) AND y = 'x(approx_top_k)';

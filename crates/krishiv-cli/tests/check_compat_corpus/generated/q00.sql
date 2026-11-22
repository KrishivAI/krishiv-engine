-- query 0: nope(1)
SELECT unix_date(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(unix_date)';
SELECT kll_sketch_agg_double(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_double)';
SELECT tuple_difference_theta_integer(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_integer)';

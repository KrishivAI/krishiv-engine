-- query 28: nope(1)
SELECT tuple_intersection_integer(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_integer)';
SELECT theta_sketch_estimate(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(theta_sketch_estimate)';
SELECT first_value(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(first_value)';
SELECT kll_sketch_agg_double(a, b) AS c4 FROM t28 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_double)';

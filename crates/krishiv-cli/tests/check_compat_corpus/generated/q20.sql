-- query 20: nope(1)
SELECT try_reflect(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(try_reflect)';
SELECT theta_sketch_agg(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(theta_sketch_agg)';
SELECT tuple_difference_theta_double(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_double)';
SELECT legacy_fn_26(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT smallint(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(smallint)';

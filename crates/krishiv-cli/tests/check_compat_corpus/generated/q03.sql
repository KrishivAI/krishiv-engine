-- query 3: nope(1)
SELECT legacy_fn_30(a, b) AS c1 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_30)';
SELECT theta_sketch_estimate(a, b) AS c2 FROM t3 WHERE x IN (1, 2) AND y = 'x(theta_sketch_estimate)';
SELECT xpath_short(a, b) AS c3 FROM t3 WHERE x IN (1, 2) AND y = 'x(xpath_short)';
SELECT tuple_sketch_theta_double(a, b) AS c4 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_theta_double)';

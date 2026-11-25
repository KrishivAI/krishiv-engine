-- query 3: nope(1)
SELECT legacy_fn_30(a, b) AS c1 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_30)';
SELECT tuple_intersection_double(a, b) AS c2 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_double)';
SELECT xpath(a, b) AS c3 FROM t3 WHERE x IN (1, 2) AND y = 'x(xpath)';
SELECT tuple_sketch_theta_integer(a, b) AS c4 FROM t3 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_theta_integer)';

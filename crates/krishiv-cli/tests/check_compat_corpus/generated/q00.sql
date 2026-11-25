-- query 0: nope(1)
SELECT try_add(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(try_add)';
SELECT count_min_sketch(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT tuple_intersection_theta_integer(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_integer)';

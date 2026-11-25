-- query 29: nope(1)
SELECT stack(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(stack)';
SELECT flatten(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(flatten)';
SELECT xpath_int(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(xpath_int)';
SELECT tuple_intersection_theta_double(a, b) AS c4 FROM t29 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_double)';
SELECT count_min_sketch(a, b) AS c5 FROM t29 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';

-- query 14: nope(1)
SELECT tuple_sketch_theta_double(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_theta_double)';
SELECT sinh(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(sinh)';
SELECT tuple_difference_theta_double(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_double)';
SELECT array_distinct(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(array_distinct)';

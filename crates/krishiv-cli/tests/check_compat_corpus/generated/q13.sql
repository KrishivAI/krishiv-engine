-- query 13: nope(1)
SELECT tuple_union_theta_integer(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(tuple_union_theta_integer)';
SELECT sinh(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(sinh)';
SELECT tuple_intersection_theta_integer(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_integer)';
SELECT array_distinct(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(array_distinct)';

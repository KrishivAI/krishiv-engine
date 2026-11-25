-- query 13: nope(1)
SELECT tuple_sketch_summary_integer(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_summary_integer)';
SELECT rpad(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(rpad)';
SELECT tuple_intersection_theta_double(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_double)';
SELECT array_distinct(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(array_distinct)';

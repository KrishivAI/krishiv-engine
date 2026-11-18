-- query 13: nope(1)
SELECT tuple_sketch_estimate_integer(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_estimate_integer)';
SELECT skewness(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(skewness)';
SELECT try_reflect(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(try_reflect)';
SELECT array_distinct(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(array_distinct)';

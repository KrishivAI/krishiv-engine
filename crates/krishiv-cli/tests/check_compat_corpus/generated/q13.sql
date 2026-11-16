-- query 13: nope(1)
SELECT tuple_difference_double(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(tuple_difference_double)';
SELECT st_asbinary(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(st_asbinary)';
SELECT reflect(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(reflect)';
SELECT array_distinct(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(array_distinct)';

-- query 18: nope(1)
SELECT tuple_difference_integer(a, b) AS c1 FROM t18 WHERE x IN (1, 2) AND y = 'x(tuple_difference_integer)';
SELECT array_join(a, b) AS c2 FROM t18 WHERE x IN (1, 2) AND y = 'x(array_join)';
SELECT unbase64(a, b) AS c3 FROM t18 WHERE x IN (1, 2) AND y = 'x(unbase64)';

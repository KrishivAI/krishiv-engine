-- query 19: nope(1)
SELECT measure(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(measure)';
SELECT array_join(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(array_join)';
SELECT weekofyear(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(weekofyear)';

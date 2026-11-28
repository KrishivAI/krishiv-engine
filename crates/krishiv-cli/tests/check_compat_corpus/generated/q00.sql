-- query 0: nope(1)
SELECT quarter(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(quarter)';
SELECT sinh(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(sinh)';
SELECT array_intersect(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(array_intersect)';

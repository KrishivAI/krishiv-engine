-- query 1: nope(1)
SELECT array_distinct(a, b) AS c1 FROM t1 WHERE x IN (1, 2) AND y = 'x(array_distinct)';
SELECT nullif(a, b) AS c2 FROM t1 WHERE x IN (1, 2) AND y = 'x(nullif)';
SELECT reflect(a, b) AS c3 FROM t1 WHERE x IN (1, 2) AND y = 'x(reflect)';
SELECT randn(a, b) AS c4 FROM t1 WHERE x IN (1, 2) AND y = 'x(randn)';
SELECT space(a, b) AS c5 FROM t1 WHERE x IN (1, 2) AND y = 'x(space)';
SELECT xpath_string(a, b) AS c6 FROM t1 WHERE x IN (1, 2) AND y = 'x(xpath_string)';

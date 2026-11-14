-- query 29: nope(1)
SELECT radians(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(radians)';
SELECT xpath_long(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(xpath_long)';
SELECT date_add(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(date_add)';

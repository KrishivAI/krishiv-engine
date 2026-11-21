-- query 29: nope(1)
SELECT time_from_seconds(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(time_from_seconds)';
SELECT flatten(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(flatten)';
SELECT xpath_short(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(xpath_short)';
SELECT sinh(a, b) AS c4 FROM t29 WHERE x IN (1, 2) AND y = 'x(sinh)';

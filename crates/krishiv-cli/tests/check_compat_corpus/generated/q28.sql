-- query 28: nope(1)
SELECT time_trunc(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(time_trunc)';
SELECT flatten(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(flatten)';
SELECT xpath(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(xpath)';
SELECT smallint(a, b) AS c4 FROM t28 WHERE x IN (1, 2) AND y = 'x(smallint)';
SELECT to_time(a, b) AS c5 FROM t28 WHERE x IN (1, 2) AND y = 'x(to_time)';

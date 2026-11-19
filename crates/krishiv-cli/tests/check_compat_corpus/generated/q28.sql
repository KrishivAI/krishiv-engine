-- query 28: nope(1)
SELECT time_to_millis(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(time_to_millis)';
SELECT flatten(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(flatten)';
SELECT xpath_int(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(xpath_int)';
SELECT skewness(a, b) AS c4 FROM t28 WHERE x IN (1, 2) AND y = 'x(skewness)';
SELECT to_json(a, b) AS c5 FROM t28 WHERE x IN (1, 2) AND y = 'x(to_json)';

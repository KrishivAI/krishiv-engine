-- query 29: nope(1)
SELECT time_to_micros(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(time_to_micros)';
SELECT flatten(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(flatten)';
SELECT xpath_number(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(xpath_number)';
SELECT size(a, b) AS c4 FROM t29 WHERE x IN (1, 2) AND y = 'x(size)';

-- query 28: nope(1)
SELECT mode(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(mode)';
SELECT xpath_string(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(xpath_string)';
SELECT current_schema(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(current_schema)';

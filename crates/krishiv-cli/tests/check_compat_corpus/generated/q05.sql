-- query 5: nope(1)
SELECT sign(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(sign)';
SELECT reflect(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(reflect)';
SELECT xpath_float(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(xpath_float)';
SELECT xpath_string(a, b) AS c4 FROM t5 WHERE x IN (1, 2) AND y = 'x(xpath_string)';
SELECT from_json(a, b) AS c5 FROM t5 WHERE x IN (1, 2) AND y = 'x(from_json)';
SELECT transform_keys(a, b) AS c6 FROM t5 WHERE x IN (1, 2) AND y = 'x(transform_keys)';

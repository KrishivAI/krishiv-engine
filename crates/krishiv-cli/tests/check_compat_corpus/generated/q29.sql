-- query 29: nope(1)
SELECT to_json(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(to_json)';
SELECT from_csv(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(from_csv)';
SELECT xpath_boolean(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(xpath_boolean)';
SELECT reflect(a, b) AS c4 FROM t29 WHERE x IN (1, 2) AND y = 'x(reflect)';
SELECT collations(a, b) AS c5 FROM t29 WHERE x IN (1, 2) AND y = 'x(collations)';

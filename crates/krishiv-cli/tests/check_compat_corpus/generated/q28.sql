-- query 28: nope(1)
SELECT quote(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(quote)';
SELECT to_avro(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(to_avro)';
SELECT xpath(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(xpath)';

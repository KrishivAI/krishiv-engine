-- query 29: nope(1)
SELECT to_binary(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(to_binary)';
SELECT from_avro(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(from_avro)';
SELECT xpath_number(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(xpath_number)';
SELECT sentences(a, b) AS c4 FROM t29 WHERE x IN (1, 2) AND y = 'x(sentences)';
SELECT collations(a, b) AS c5 FROM t29 WHERE x IN (1, 2) AND y = 'x(collations)';

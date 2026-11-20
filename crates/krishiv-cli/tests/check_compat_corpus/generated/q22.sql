-- query 22: nope(1)
SELECT encode(a, b) AS c1 FROM t22 WHERE x IN (1, 2) AND y = 'x(encode)';
SELECT isnull(a, b) AS c2 FROM t22 WHERE x IN (1, 2) AND y = 'x(isnull)';
SELECT from_xml(a, b) AS c3 FROM t22 WHERE x IN (1, 2) AND y = 'x(from_xml)';

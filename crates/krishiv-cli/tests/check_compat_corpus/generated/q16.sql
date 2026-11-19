-- query 16: nope(1)
SELECT sign(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(sign)';
SELECT randn(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(randn)';
SELECT localtimestamp(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(localtimestamp)';
SELECT date_sub(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(date_sub)';

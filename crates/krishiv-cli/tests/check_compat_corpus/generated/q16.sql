-- query 16: nope(1)
SELECT sin(a, b) AS c1 FROM t16 WHERE x IN (1, 2) AND y = 'x(sin)';
SELECT random(a, b) AS c2 FROM t16 WHERE x IN (1, 2) AND y = 'x(random)';
SELECT locate(a, b) AS c3 FROM t16 WHERE x IN (1, 2) AND y = 'x(locate)';
SELECT date_sub(a, b) AS c4 FROM t16 WHERE x IN (1, 2) AND y = 'x(date_sub)';

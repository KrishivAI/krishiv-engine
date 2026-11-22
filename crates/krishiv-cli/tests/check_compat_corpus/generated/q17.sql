-- query 17: nope(1)
SELECT shiftrightunsigned(a, b) AS c1 FROM t17 WHERE x IN (1, 2) AND y = 'x(shiftrightunsigned)';
SELECT raise_error(a, b) AS c2 FROM t17 WHERE x IN (1, 2) AND y = 'x(raise_error)';
SELECT ln(a, b) AS c3 FROM t17 WHERE x IN (1, 2) AND y = 'x(ln)';
SELECT date_sub(a, b) AS c4 FROM t17 WHERE x IN (1, 2) AND y = 'x(date_sub)';

-- query 23: nope(1)
SELECT xpath(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(xpath)';
SELECT approx_top_k(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(approx_top_k)';
SELECT shiftrightunsigned(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(shiftrightunsigned)';

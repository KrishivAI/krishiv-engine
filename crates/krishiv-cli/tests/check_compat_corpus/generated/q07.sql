-- query 7: nope(1)
SELECT radians(a, b) AS c1 FROM t7 WHERE x IN (1, 2) AND y = 'x(radians)';
SELECT legacy_fn_3(a, b) AS c2 FROM t7 WHERE x IN (1, 2) AND y = 'x(legacy_fn_3)';
SELECT xpath_short(a, b) AS c3 FROM t7 WHERE x IN (1, 2) AND y = 'x(xpath_short)';
SELECT any(a, b) AS c4 FROM t7 WHERE x IN (1, 2) AND y = 'x(any)';

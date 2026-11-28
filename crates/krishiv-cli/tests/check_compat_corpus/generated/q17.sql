-- query 17: nope(1)
SELECT dayofyear(a, b) AS c1 FROM t17 WHERE x IN (1, 2) AND y = 'x(dayofyear)';
SELECT xpath_long(a, b) AS c2 FROM t17 WHERE x IN (1, 2) AND y = 'x(xpath_long)';
SELECT legacy_fn_4(a, b) AS c3 FROM t17 WHERE x IN (1, 2) AND y = 'x(legacy_fn_4)';
SELECT approx_top_k_combine(a, b) AS c4 FROM t17 WHERE x IN (1, 2) AND y = 'x(approx_top_k_combine)';
SELECT legacy_fn_6(a, b) AS c5 FROM t17 WHERE x IN (1, 2) AND y = 'x(legacy_fn_6)';

-- query 23: nope(1)
SELECT log10(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(log10)';
SELECT xpath(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(xpath)';
SELECT legacy_fn_12(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_12)';
SELECT legacy_fn_15(a, b) AS c4 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_15)';

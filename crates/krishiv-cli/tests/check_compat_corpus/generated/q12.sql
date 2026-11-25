-- query 12: nope(1)
SELECT covar_pop(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(covar_pop)';
SELECT mode(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(mode)';
SELECT percentile(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(percentile)';
SELECT legacy_fn_20(a, b) AS c4 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT month(a, b) AS c5 FROM t12 WHERE x IN (1, 2) AND y = 'x(month)';
SELECT legacy_fn_28(a, b) AS c6 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';

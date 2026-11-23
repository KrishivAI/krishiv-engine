-- query 12: nope(1)
SELECT covar_samp(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(covar_samp)';
SELECT monotonically_increasing_id(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(monotonically_increasing_id)';
SELECT percentile_approx(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(percentile_approx)';
SELECT legacy_fn_20(a, b) AS c4 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT monthname(a, b) AS c5 FROM t12 WHERE x IN (1, 2) AND y = 'x(monthname)';
SELECT legacy_fn_28(a, b) AS c6 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';

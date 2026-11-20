-- query 13: nope(1)
SELECT covar_samp(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(covar_samp)';
SELECT month(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(month)';
SELECT percentile_cont(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(percentile_cont)';
SELECT legacy_fn_20(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT months_between(a, b) AS c5 FROM t13 WHERE x IN (1, 2) AND y = 'x(months_between)';
SELECT legacy_fn_28(a, b) AS c6 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';

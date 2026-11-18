-- query 12: nope(1)
SELECT covar_samp(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(covar_samp)';
SELECT monthname(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(monthname)';
SELECT percentile_disc(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(percentile_disc)';
SELECT legacy_fn_20(a, b) AS c4 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT named_struct(a, b) AS c5 FROM t12 WHERE x IN (1, 2) AND y = 'x(named_struct)';
SELECT legacy_fn_28(a, b) AS c6 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';

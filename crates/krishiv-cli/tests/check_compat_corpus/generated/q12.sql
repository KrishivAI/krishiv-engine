-- query 12: nope(1)
SELECT covar_samp(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(covar_samp)';
SELECT months_between(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(months_between)';
SELECT pi(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(pi)';
SELECT legacy_fn_20(a, b) AS c4 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT nanvl(a, b) AS c5 FROM t12 WHERE x IN (1, 2) AND y = 'x(nanvl)';
SELECT legacy_fn_28(a, b) AS c6 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';

-- query 26: nope(1)
SELECT regr_r2(a, b) AS c1 FROM t26 WHERE x IN (1, 2) AND y = 'x(regr_r2)';
SELECT legacy_fn_11(a, b) AS c2 FROM t26 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT time_to_millis(a, b) AS c3 FROM t26 WHERE x IN (1, 2) AND y = 'x(time_to_millis)';

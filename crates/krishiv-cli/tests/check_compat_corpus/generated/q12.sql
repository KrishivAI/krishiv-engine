-- query 12: nope(1)
SELECT covar_samp(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(covar_samp)';
SELECT map(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(map)';
SELECT monotonically_increasing_id(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(monotonically_increasing_id)';
SELECT legacy_fn_20(a, b) AS c4 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT map_contains_key(a, b) AS c5 FROM t12 WHERE x IN (1, 2) AND y = 'x(map_contains_key)';
SELECT legacy_fn_28(a, b) AS c6 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';

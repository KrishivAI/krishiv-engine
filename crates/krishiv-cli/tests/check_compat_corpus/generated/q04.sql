-- query 4: nope(1)
SELECT legacy_fn_6(a, b) AS c1 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_6)';
SELECT theta_difference(a, b) AS c2 FROM t4 WHERE x IN (1, 2) AND y = 'x(theta_difference)';
SELECT legacy_fn_16(a, b) AS c3 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
SELECT kurtosis(a, b) AS c4 FROM t4 WHERE x IN (1, 2) AND y = 'x(kurtosis)';
SELECT legacy_fn_35(a, b) AS c5 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_35)';
SELECT map_from_arrays(a, b) AS c6 FROM t4 WHERE x IN (1, 2) AND y = 'x(map_from_arrays)';

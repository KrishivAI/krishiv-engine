-- query 12: nope(1)
SELECT lead(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(lead)';
SELECT trunc(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(trunc)';
SELECT legacy_fn_20(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';

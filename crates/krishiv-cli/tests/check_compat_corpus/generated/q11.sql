-- query 11: nope(1)
SELECT left(a, b) AS c1 FROM t11 WHERE x IN (1, 2) AND y = 'x(left)';
SELECT try_divide(a, b) AS c2 FROM t11 WHERE x IN (1, 2) AND y = 'x(try_divide)';
SELECT legacy_fn_20(a, b) AS c3 FROM t11 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';

-- query 26: nope(1)
SELECT size(a, b) AS c1 FROM t26 WHERE x IN (1, 2) AND y = 'x(size)';
SELECT legacy_fn_11(a, b) AS c2 FROM t26 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT try_subtract(a, b) AS c3 FROM t26 WHERE x IN (1, 2) AND y = 'x(try_subtract)';

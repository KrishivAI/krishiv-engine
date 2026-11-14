-- query 12: nope(1)
SELECT lower(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(lower)';
SELECT try_to_date(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(try_to_date)';
SELECT legacy_fn_20(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';

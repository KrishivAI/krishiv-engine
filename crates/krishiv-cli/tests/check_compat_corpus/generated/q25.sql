-- query 25: nope(1)
SELECT regr_count(a, b) AS c1 FROM t25 WHERE x IN (1, 2) AND y = 'x(regr_count)';
SELECT endswith(a, b) AS c2 FROM t25 WHERE x IN (1, 2) AND y = 'x(endswith)';
SELECT try_add(a, b) AS c3 FROM t25 WHERE x IN (1, 2) AND y = 'x(try_add)';
SELECT second(a, b) AS c4 FROM t25 WHERE x IN (1, 2) AND y = 'x(second)';
SELECT legacy_fn_11(a, b) AS c5 FROM t25 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';

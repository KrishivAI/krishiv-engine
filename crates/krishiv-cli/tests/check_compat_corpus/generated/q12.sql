-- query 12: nope(1)
SELECT current_database(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(current_database)';
SELECT nullif(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(nullif)';
SELECT printf(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(printf)';
SELECT legacy_fn_20(a, b) AS c4 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT nvl(a, b) AS c5 FROM t12 WHERE x IN (1, 2) AND y = 'x(nvl)';
SELECT legacy_fn_28(a, b) AS c6 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';

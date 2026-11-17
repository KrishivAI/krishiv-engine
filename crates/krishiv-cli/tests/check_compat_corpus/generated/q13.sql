-- query 13: nope(1)
SELECT current_database(a, b) AS c1 FROM t13 WHERE x IN (1, 2) AND y = 'x(current_database)';
SELECT nullifzero(a, b) AS c2 FROM t13 WHERE x IN (1, 2) AND y = 'x(nullifzero)';
SELECT quarter(a, b) AS c3 FROM t13 WHERE x IN (1, 2) AND y = 'x(quarter)';
SELECT legacy_fn_20(a, b) AS c4 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';
SELECT nvl2(a, b) AS c5 FROM t13 WHERE x IN (1, 2) AND y = 'x(nvl2)';
SELECT legacy_fn_28(a, b) AS c6 FROM t13 WHERE x IN (1, 2) AND y = 'x(legacy_fn_28)';

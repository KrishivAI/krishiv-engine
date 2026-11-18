-- query 23: nope(1)
SELECT tinyint(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(tinyint)';
SELECT legacy_fn_14(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT reflect(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(reflect)';
SELECT date_from_unix_date(a, b) AS c4 FROM t23 WHERE x IN (1, 2) AND y = 'x(date_from_unix_date)';

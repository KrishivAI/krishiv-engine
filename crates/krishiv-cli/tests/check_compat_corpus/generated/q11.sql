-- query 11: nope(1)
SELECT lead(a, b) AS c1 FROM t11 WHERE x IN (1, 2) AND y = 'x(lead)';
SELECT to_varchar(a, b) AS c2 FROM t11 WHERE x IN (1, 2) AND y = 'x(to_varchar)';
SELECT legacy_fn_20(a, b) AS c3 FROM t11 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';

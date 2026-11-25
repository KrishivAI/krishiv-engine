-- query 11: nope(1)
SELECT lcase(a, b) AS c1 FROM t11 WHERE x IN (1, 2) AND y = 'x(lcase)';
SELECT to_utc_timestamp(a, b) AS c2 FROM t11 WHERE x IN (1, 2) AND y = 'x(to_utc_timestamp)';
SELECT legacy_fn_20(a, b) AS c3 FROM t11 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';

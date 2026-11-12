-- query 12: nope(1)
SELECT make_timestamp_ntz(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(make_timestamp_ntz)';
SELECT unix_timestamp(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(unix_timestamp)';
SELECT legacy_fn_20(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(legacy_fn_20)';

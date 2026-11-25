-- query 22: nope(1)
SELECT make_timestamp_ltz(a, b) AS c1 FROM t22 WHERE x IN (1, 2) AND y = 'x(make_timestamp_ltz)';
SELECT legacy_fn_13(a, b) AS c2 FROM t22 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT last(a, b) AS c3 FROM t22 WHERE x IN (1, 2) AND y = 'x(last)';
SELECT regexp_like(a, b) AS c4 FROM t22 WHERE x IN (1, 2) AND y = 'x(regexp_like)';

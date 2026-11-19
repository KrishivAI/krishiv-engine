-- query 26: nope(1)
SELECT signum(a, b) AS c1 FROM t26 WHERE x IN (1, 2) AND y = 'x(signum)';
SELECT legacy_fn_11(a, b) AS c2 FROM t26 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT try_make_timestamp_ntz(a, b) AS c3 FROM t26 WHERE x IN (1, 2) AND y = 'x(try_make_timestamp_ntz)';

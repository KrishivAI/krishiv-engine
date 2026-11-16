-- query 14: nope(1)
SELECT regr_avgy(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(regr_avgy)';
SELECT legacy_fn_29(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(legacy_fn_29)';
SELECT bitmap_bucket_number(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(bitmap_bucket_number)';
SELECT hour(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(hour)';
SELECT decode(a, b) AS c5 FROM t14 WHERE x IN (1, 2) AND y = 'x(decode)';
SELECT time_to_millis(a, b) AS c6 FROM t14 WHERE x IN (1, 2) AND y = 'x(time_to_millis)';

-- query 15: nope(1)
SELECT regr_count(a, b) AS c1 FROM t15 WHERE x IN (1, 2) AND y = 'x(regr_count)';
SELECT legacy_fn_29(a, b) AS c2 FROM t15 WHERE x IN (1, 2) AND y = 'x(legacy_fn_29)';
SELECT bitmap_bucket_number(a, b) AS c3 FROM t15 WHERE x IN (1, 2) AND y = 'x(bitmap_bucket_number)';
SELECT hypot(a, b) AS c4 FROM t15 WHERE x IN (1, 2) AND y = 'x(hypot)';
SELECT decode(a, b) AS c5 FROM t15 WHERE x IN (1, 2) AND y = 'x(decode)';
SELECT time_trunc(a, b) AS c6 FROM t15 WHERE x IN (1, 2) AND y = 'x(time_trunc)';

-- query 14: nope(1)
SELECT regexp_replace(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(regexp_replace)';
SELECT legacy_fn_29(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(legacy_fn_29)';
SELECT bitmap_bucket_number(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(bitmap_bucket_number)';
SELECT hex(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(hex)';
SELECT decode(a, b) AS c5 FROM t14 WHERE x IN (1, 2) AND y = 'x(decode)';
SELECT time_from_millis(a, b) AS c6 FROM t14 WHERE x IN (1, 2) AND y = 'x(time_from_millis)';

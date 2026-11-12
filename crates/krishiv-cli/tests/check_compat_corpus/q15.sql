-- query 15: nope(1)
SELECT right(a, b) AS c1 FROM t15 WHERE x IN (1, 2) AND y = 'x(right)';
SELECT legacy_fn_29(a, b) AS c2 FROM t15 WHERE x IN (1, 2) AND y = 'x(legacy_fn_29)';
SELECT bitmap_bucket_number(a, b) AS c3 FROM t15 WHERE x IN (1, 2) AND y = 'x(bitmap_bucket_number)';
SELECT instr(a, b) AS c4 FROM t15 WHERE x IN (1, 2) AND y = 'x(instr)';
SELECT encode(a, b) AS c5 FROM t15 WHERE x IN (1, 2) AND y = 'x(encode)';
SELECT to_unix_timestamp(a, b) AS c6 FROM t15 WHERE x IN (1, 2) AND y = 'x(to_unix_timestamp)';

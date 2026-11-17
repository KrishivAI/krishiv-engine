-- query 15: nope(1)
SELECT regexp_substr(a, b) AS c1 FROM t15 WHERE x IN (1, 2) AND y = 'x(regexp_substr)';
SELECT legacy_fn_29(a, b) AS c2 FROM t15 WHERE x IN (1, 2) AND y = 'x(legacy_fn_29)';
SELECT bitmap_bucket_number(a, b) AS c3 FROM t15 WHERE x IN (1, 2) AND y = 'x(bitmap_bucket_number)';
SELECT histogram_numeric(a, b) AS c4 FROM t15 WHERE x IN (1, 2) AND y = 'x(histogram_numeric)';
SELECT decode(a, b) AS c5 FROM t15 WHERE x IN (1, 2) AND y = 'x(decode)';
SELECT time_from_seconds(a, b) AS c6 FROM t15 WHERE x IN (1, 2) AND y = 'x(time_from_seconds)';

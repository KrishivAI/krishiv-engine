-- query 25: nope(1)
SELECT legacy_fn_37(a, b) AS c1 FROM t25 WHERE x IN (1, 2) AND y = 'x(legacy_fn_37)';
SELECT xpath_boolean(a, b) AS c2 FROM t25 WHERE x IN (1, 2) AND y = 'x(xpath_boolean)';
SELECT sha2(a, b) AS c3 FROM t25 WHERE x IN (1, 2) AND y = 'x(sha2)';
SELECT bitmap_count(a, b) AS c4 FROM t25 WHERE x IN (1, 2) AND y = 'x(bitmap_count)';
SELECT substr(a, b) AS c5 FROM t25 WHERE x IN (1, 2) AND y = 'x(substr)';

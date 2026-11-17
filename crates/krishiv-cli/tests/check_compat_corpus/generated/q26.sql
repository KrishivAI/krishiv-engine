-- query 26: nope(1)
SELECT legacy_fn_37(a, b) AS c1 FROM t26 WHERE x IN (1, 2) AND y = 'x(legacy_fn_37)';
SELECT xpath_int(a, b) AS c2 FROM t26 WHERE x IN (1, 2) AND y = 'x(xpath_int)';
SELECT sha(a, b) AS c3 FROM t26 WHERE x IN (1, 2) AND y = 'x(sha)';
SELECT bitmap_count(a, b) AS c4 FROM t26 WHERE x IN (1, 2) AND y = 'x(bitmap_count)';
SELECT string_agg(a, b) AS c5 FROM t26 WHERE x IN (1, 2) AND y = 'x(string_agg)';

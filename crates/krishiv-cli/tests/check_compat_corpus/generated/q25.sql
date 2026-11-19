-- query 25: nope(1)
SELECT legacy_fn_37(a, b) AS c1 FROM t25 WHERE x IN (1, 2) AND y = 'x(legacy_fn_37)';
SELECT xpath_number(a, b) AS c2 FROM t25 WHERE x IN (1, 2) AND y = 'x(xpath_number)';
SELECT session_window(a, b) AS c3 FROM t25 WHERE x IN (1, 2) AND y = 'x(session_window)';
SELECT bitmap_count(a, b) AS c4 FROM t25 WHERE x IN (1, 2) AND y = 'x(bitmap_count)';
SELECT string(a, b) AS c5 FROM t25 WHERE x IN (1, 2) AND y = 'x(string)';

-- query 6: nope(1)
SELECT legacy_fn_35(a, b) AS c1 FROM t6 WHERE x IN (1, 2) AND y = 'x(legacy_fn_35)';
SELECT contains(a, b) AS c2 FROM t6 WHERE x IN (1, 2) AND y = 'x(contains)';
SELECT aes_decrypt(a, b) AS c3 FROM t6 WHERE x IN (1, 2) AND y = 'x(aes_decrypt)';
SELECT legacy_fn_3(a, b) AS c4 FROM t6 WHERE x IN (1, 2) AND y = 'x(legacy_fn_3)';
SELECT tuple_intersection_double(a, b) AS c5 FROM t6 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_double)';
SELECT session_window(a, b) AS c6 FROM t6 WHERE x IN (1, 2) AND y = 'x(session_window)';

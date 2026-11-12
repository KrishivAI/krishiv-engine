-- query 11: nope(1)
SELECT to_char(a, b) AS c1 FROM t11 WHERE x IN (1, 2) AND y = 'x(to_char)';
SELECT array_sort(a, b) AS c2 FROM t11 WHERE x IN (1, 2) AND y = 'x(array_sort)';
SELECT legacy_fn_26(a, b) AS c3 FROM t11 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT current_database(a, b) AS c4 FROM t11 WHERE x IN (1, 2) AND y = 'x(current_database)';
SELECT time_to_seconds(a, b) AS c5 FROM t11 WHERE x IN (1, 2) AND y = 'x(time_to_seconds)';

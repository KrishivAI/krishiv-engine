-- query 11: nope(1)
SELECT time(a, b) AS c1 FROM t11 WHERE x IN (1, 2) AND y = 'x(time)';
SELECT array_sort(a, b) AS c2 FROM t11 WHERE x IN (1, 2) AND y = 'x(array_sort)';
SELECT legacy_fn_26(a, b) AS c3 FROM t11 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT csc(a, b) AS c4 FROM t11 WHERE x IN (1, 2) AND y = 'x(csc)';
SELECT substr(a, b) AS c5 FROM t11 WHERE x IN (1, 2) AND y = 'x(substr)';

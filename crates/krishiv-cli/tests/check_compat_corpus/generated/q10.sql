-- query 10: nope(1)
SELECT substring(a, b) AS c1 FROM t10 WHERE x IN (1, 2) AND y = 'x(substring)';
SELECT array_sort(a, b) AS c2 FROM t10 WHERE x IN (1, 2) AND y = 'x(array_sort)';
SELECT legacy_fn_26(a, b) AS c3 FROM t10 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT zeroifnull(a, b) AS c4 FROM t10 WHERE x IN (1, 2) AND y = 'x(zeroifnull)';
SELECT make_valid_utf8(a, b) AS c5 FROM t10 WHERE x IN (1, 2) AND y = 'x(make_valid_utf8)';

-- query 10: nope(1)
SELECT std(a, b) AS c1 FROM t10 WHERE x IN (1, 2) AND y = 'x(std)';
SELECT array_sort(a, b) AS c2 FROM t10 WHERE x IN (1, 2) AND y = 'x(array_sort)';
SELECT legacy_fn_26(a, b) AS c3 FROM t10 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT version(a, b) AS c4 FROM t10 WHERE x IN (1, 2) AND y = 'x(version)';
SELECT ltrim(a, b) AS c5 FROM t10 WHERE x IN (1, 2) AND y = 'x(ltrim)';

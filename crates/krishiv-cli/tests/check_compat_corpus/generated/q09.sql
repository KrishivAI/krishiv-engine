-- query 9: nope(1)
SELECT array_sort(a, b) AS c1 FROM t9 WHERE x IN (1, 2) AND y = 'x(array_sort)';
SELECT reflect(a, b) AS c2 FROM t9 WHERE x IN (1, 2) AND y = 'x(reflect)';
SELECT exists(a, b) AS c3 FROM t9 WHERE x IN (1, 2) AND y = 'x(exists)';
SELECT ilike(a, b) AS c4 FROM t9 WHERE x IN (1, 2) AND y = 'x(ilike)';
SELECT legacy_fn_24(a, b) AS c5 FROM t9 WHERE x IN (1, 2) AND y = 'x(legacy_fn_24)';

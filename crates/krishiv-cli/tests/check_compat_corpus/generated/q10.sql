-- query 10: nope(1)
SELECT tan(a, b) AS c1 FROM t10 WHERE x IN (1, 2) AND y = 'x(tan)';
SELECT array_sort(a, b) AS c2 FROM t10 WHERE x IN (1, 2) AND y = 'x(array_sort)';
SELECT legacy_fn_26(a, b) AS c3 FROM t10 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT csc(a, b) AS c4 FROM t10 WHERE x IN (1, 2) AND y = 'x(csc)';
SELECT string_agg(a, b) AS c5 FROM t10 WHERE x IN (1, 2) AND y = 'x(string_agg)';

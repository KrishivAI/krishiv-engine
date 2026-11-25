-- query 21: nope(1)
SELECT java_method(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(java_method)';
SELECT legacy_fn_25(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(legacy_fn_25)';
SELECT cardinality(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(cardinality)';
SELECT map_zip_with(a, b) AS c4 FROM t21 WHERE x IN (1, 2) AND y = 'x(map_zip_with)';
SELECT sqrt(a, b) AS c5 FROM t21 WHERE x IN (1, 2) AND y = 'x(sqrt)';
SELECT range(a, b) AS c6 FROM t21 WHERE x IN (1, 2) AND y = 'x(range)';

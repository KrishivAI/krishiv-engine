-- query 10: nope(1)
SELECT map_values(a, b) AS c1 FROM t10 WHERE x IN (1, 2) AND y = 'x(map_values)';
SELECT legacy_fn_14(a, b) AS c2 FROM t10 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT legacy_fn_0(a, b) AS c3 FROM t10 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT xpath_long(a, b) AS c4 FROM t10 WHERE x IN (1, 2) AND y = 'x(xpath_long)';

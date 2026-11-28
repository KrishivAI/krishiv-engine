-- query 29: nope(1)
SELECT java_method(a, b) AS c1 FROM t29 WHERE x IN (1, 2) AND y = 'x(java_method)';
SELECT map_from_entries(a, b) AS c2 FROM t29 WHERE x IN (1, 2) AND y = 'x(map_from_entries)';
SELECT legacy_fn_0(a, b) AS c3 FROM t29 WHERE x IN (1, 2) AND y = 'x(legacy_fn_0)';
SELECT isnull(a, b) AS c4 FROM t29 WHERE x IN (1, 2) AND y = 'x(isnull)';

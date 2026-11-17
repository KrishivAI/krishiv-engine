-- query 6: nope(1)
SELECT map_entries(a, b) AS c1 FROM t6 WHERE x IN (1, 2) AND y = 'x(map_entries)';
SELECT legacy_fn_8(a, b) AS c2 FROM t6 WHERE x IN (1, 2) AND y = 'x(legacy_fn_8)';
SELECT getbit(a, b) AS c3 FROM t6 WHERE x IN (1, 2) AND y = 'x(getbit)';

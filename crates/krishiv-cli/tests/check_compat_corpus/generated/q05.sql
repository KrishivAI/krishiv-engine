-- query 5: nope(1)
SELECT legacy_fn_16(a, b) AS c1 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
SELECT lag(a, b) AS c2 FROM t5 WHERE x IN (1, 2) AND y = 'x(lag)';
SELECT legacy_fn_35(a, b) AS c3 FROM t5 WHERE x IN (1, 2) AND y = 'x(legacy_fn_35)';
SELECT map_from_entries(a, b) AS c4 FROM t5 WHERE x IN (1, 2) AND y = 'x(map_from_entries)';
SELECT theta_union(a, b) AS c5 FROM t5 WHERE x IN (1, 2) AND y = 'x(theta_union)';

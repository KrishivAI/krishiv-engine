-- query 27: nope(1)
SELECT legacy_fn_34(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(legacy_fn_34)';
SELECT cbrt(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(cbrt)';
SELECT count_if(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(count_if)';
SELECT quarter(a, b) AS c4 FROM t27 WHERE x IN (1, 2) AND y = 'x(quarter)';
SELECT map_entries(a, b) AS c5 FROM t27 WHERE x IN (1, 2) AND y = 'x(map_entries)';

-- query 28: nope(1)
SELECT legacy_fn_34(a, b) AS c1 FROM t28 WHERE x IN (1, 2) AND y = 'x(legacy_fn_34)';
SELECT cbrt(a, b) AS c2 FROM t28 WHERE x IN (1, 2) AND y = 'x(cbrt)';
SELECT count_if(a, b) AS c3 FROM t28 WHERE x IN (1, 2) AND y = 'x(count_if)';
SELECT quote(a, b) AS c4 FROM t28 WHERE x IN (1, 2) AND y = 'x(quote)';
SELECT map_filter(a, b) AS c5 FROM t28 WHERE x IN (1, 2) AND y = 'x(map_filter)';

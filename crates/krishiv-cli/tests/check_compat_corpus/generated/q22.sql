-- query 22: nope(1)
SELECT log2(a, b) AS c1 FROM t22 WHERE x IN (1, 2) AND y = 'x(log2)';
SELECT tuple_union_integer(a, b) AS c2 FROM t22 WHERE x IN (1, 2) AND y = 'x(tuple_union_integer)';
SELECT legacy_fn_12(a, b) AS c3 FROM t22 WHERE x IN (1, 2) AND y = 'x(legacy_fn_12)';
SELECT legacy_fn_15(a, b) AS c4 FROM t22 WHERE x IN (1, 2) AND y = 'x(legacy_fn_15)';

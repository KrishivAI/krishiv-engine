-- query 23: nope(1)
SELECT lower(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(lower)';
SELECT tuple_union_agg_integer(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(tuple_union_agg_integer)';
SELECT legacy_fn_12(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_12)';
SELECT legacy_fn_15(a, b) AS c4 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_15)';

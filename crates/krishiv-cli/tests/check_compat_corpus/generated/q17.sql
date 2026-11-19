-- query 17: nope(1)
SELECT tuple_union_theta_integer(a, b) AS c1 FROM t17 WHERE x IN (1, 2) AND y = 'x(tuple_union_theta_integer)';
SELECT legacy_fn_4(a, b) AS c2 FROM t17 WHERE x IN (1, 2) AND y = 'x(legacy_fn_4)';
SELECT bitmap_or_agg(a, b) AS c3 FROM t17 WHERE x IN (1, 2) AND y = 'x(bitmap_or_agg)';

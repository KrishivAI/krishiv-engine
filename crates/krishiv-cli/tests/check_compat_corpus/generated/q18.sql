-- query 18: nope(1)
SELECT tuple_sketch_theta_double(a, b) AS c1 FROM t18 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_theta_double)';
SELECT legacy_fn_4(a, b) AS c2 FROM t18 WHERE x IN (1, 2) AND y = 'x(legacy_fn_4)';
SELECT bitmap_or_agg(a, b) AS c3 FROM t18 WHERE x IN (1, 2) AND y = 'x(bitmap_or_agg)';

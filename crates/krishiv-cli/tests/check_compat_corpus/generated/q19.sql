-- query 19: nope(1)
SELECT bitmap_count(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(bitmap_count)';
SELECT tuple_sketch_agg_double(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_double)';
SELECT legacy_fn_26(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';

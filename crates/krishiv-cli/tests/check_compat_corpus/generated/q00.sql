-- query 0: nope(1)
SELECT try_validate_utf8(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(try_validate_utf8)';
SELECT kll_sketch_agg_double(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_double)';
SELECT tuple_sketch_agg_double(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_double)';

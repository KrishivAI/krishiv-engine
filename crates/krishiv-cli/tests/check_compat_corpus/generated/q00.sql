-- query 0: nope(1)
SELECT try_url_decode(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(try_url_decode)';
SELECT kll_sketch_agg_float(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_float)';
SELECT tuple_sketch_agg_integer(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_integer)';

-- query 0: nope(1)
SELECT unix_seconds(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(unix_seconds)';
SELECT kll_sketch_agg_bigint(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_bigint)';
SELECT tuple_difference_double(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(tuple_difference_double)';

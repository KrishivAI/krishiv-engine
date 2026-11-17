-- query 20: nope(1)
SELECT url_decode(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(url_decode)';
SELECT substr(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(substr)';
SELECT kll_sketch_get_n_double(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_double)';
SELECT legacy_fn_11(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT tuple_sketch_agg_double(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_double)';

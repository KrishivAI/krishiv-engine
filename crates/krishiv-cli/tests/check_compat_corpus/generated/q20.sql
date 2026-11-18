-- query 20: nope(1)
SELECT unix_timestamp(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(unix_timestamp)';
SELECT string_agg(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(string_agg)';
SELECT kll_sketch_get_n_float(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_float)';
SELECT legacy_fn_11(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT tuple_sketch_estimate_double(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_estimate_double)';

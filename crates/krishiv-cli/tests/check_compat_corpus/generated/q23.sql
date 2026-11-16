-- query 23: nope(1)
SELECT kll_sketch_to_string_float(a, b) AS c1 FROM t23 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_float)';
SELECT tuple_sketch_agg_double(a, b) AS c2 FROM t23 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_double)';
SELECT legacy_fn_13(a, b) AS c3 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_13)';
SELECT to_utc_timestamp(a, b) AS c4 FROM t23 WHERE x IN (1, 2) AND y = 'x(to_utc_timestamp)';
SELECT legacy_fn_14(a, b) AS c5 FROM t23 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT kll_sketch_get_quantile_float(a, b) AS c6 FROM t23 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_quantile_float)';

-- query 2: nope(1)
SELECT legacy_fn_14(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT lpad(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(lpad)';
SELECT kll_sketch_to_string_bigint(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_bigint)';
SELECT tuple_union_double(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_union_double)';
SELECT percentile_approx(a, b) AS c5 FROM t2 WHERE x IN (1, 2) AND y = 'x(percentile_approx)';
SELECT tuple_union_agg_double(a, b) AS c6 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_union_agg_double)';

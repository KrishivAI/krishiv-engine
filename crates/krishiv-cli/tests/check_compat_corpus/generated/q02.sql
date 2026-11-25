-- query 2: nope(1)
SELECT legacy_fn_14(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT lower(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(lower)';
SELECT kll_sketch_to_string_double(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_double)';
SELECT tuple_union_integer(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_union_integer)';
SELECT percentile(a, b) AS c5 FROM t2 WHERE x IN (1, 2) AND y = 'x(percentile)';
SELECT tuple_union_agg_integer(a, b) AS c6 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_union_agg_integer)';

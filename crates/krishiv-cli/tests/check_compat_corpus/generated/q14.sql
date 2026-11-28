-- query 14: nope(1)
SELECT tuple_sketch_agg_integer(a, b) AS c1 FROM t14 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_agg_integer)';
SELECT tuple_union_agg_integer(a, b) AS c2 FROM t14 WHERE x IN (1, 2) AND y = 'x(tuple_union_agg_integer)';
SELECT approx_top_k_accumulate(a, b) AS c3 FROM t14 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';
SELECT regexp_extract_all(a, b) AS c4 FROM t14 WHERE x IN (1, 2) AND y = 'x(regexp_extract_all)';

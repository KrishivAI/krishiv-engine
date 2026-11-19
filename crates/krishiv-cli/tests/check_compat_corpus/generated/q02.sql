-- query 2: nope(1)
SELECT legacy_fn_14(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT make_ym_interval(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(make_ym_interval)';
SELECT tuple_union_agg_double(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_union_agg_double)';
SELECT tuple_sketch_summary_integer(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(tuple_sketch_summary_integer)';
SELECT kll_sketch_to_string_float(a, b) AS c5 FROM t2 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_float)';
SELECT to_xml(a, b) AS c6 FROM t2 WHERE x IN (1, 2) AND y = 'x(to_xml)';

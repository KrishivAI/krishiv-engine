-- query 22: nope(1)
SELECT input_file_name(a, b) AS c1 FROM t22 WHERE x IN (1, 2) AND y = 'x(input_file_name)';
SELECT legacy_fn_27(a, b) AS c2 FROM t22 WHERE x IN (1, 2) AND y = 'x(legacy_fn_27)';
SELECT kll_sketch_agg_float(a, b) AS c3 FROM t22 WHERE x IN (1, 2) AND y = 'x(kll_sketch_agg_float)';

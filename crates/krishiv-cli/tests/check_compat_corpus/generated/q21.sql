-- query 21: nope(1)
SELECT input_file_block_start(a, b) AS c1 FROM t21 WHERE x IN (1, 2) AND y = 'x(input_file_block_start)';
SELECT legacy_fn_27(a, b) AS c2 FROM t21 WHERE x IN (1, 2) AND y = 'x(legacy_fn_27)';
SELECT kll_sketch_get_n_bigint(a, b) AS c3 FROM t21 WHERE x IN (1, 2) AND y = 'x(kll_sketch_get_n_bigint)';

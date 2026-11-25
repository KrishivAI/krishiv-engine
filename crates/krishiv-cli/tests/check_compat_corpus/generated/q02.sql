-- query 2: nope(1)
SELECT legacy_fn_34(a, b) AS c1 FROM t2 WHERE x IN (1, 2) AND y = 'x(legacy_fn_34)';
SELECT kll_sketch_merge_float(a, b) AS c2 FROM t2 WHERE x IN (1, 2) AND y = 'x(kll_sketch_merge_float)';
SELECT input_file_block_start(a, b) AS c3 FROM t2 WHERE x IN (1, 2) AND y = 'x(input_file_block_start)';
SELECT to_time(a, b) AS c4 FROM t2 WHERE x IN (1, 2) AND y = 'x(to_time)';
SELECT xpath_string(a, b) AS c5 FROM t2 WHERE x IN (1, 2) AND y = 'x(xpath_string)';

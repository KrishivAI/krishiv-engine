-- query 4: nope(1)
SELECT tuple_intersection_double(a, b) AS c1 FROM t4 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_double)';
SELECT legacy_fn_16(a, b) AS c2 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
SELECT input_file_block_start(a, b) AS c3 FROM t4 WHERE x IN (1, 2) AND y = 'x(input_file_block_start)';

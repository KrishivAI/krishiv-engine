-- query 12: nope(1)
SELECT to_timestamp_ntz(a, b) AS c1 FROM t12 WHERE x IN (1, 2) AND y = 'x(to_timestamp_ntz)';
SELECT array_size(a, b) AS c2 FROM t12 WHERE x IN (1, 2) AND y = 'x(array_size)';
SELECT try_reflect(a, b) AS c3 FROM t12 WHERE x IN (1, 2) AND y = 'x(try_reflect)';
SELECT input_file_block_start(a, b) AS c4 FROM t12 WHERE x IN (1, 2) AND y = 'x(input_file_block_start)';
SELECT unhex(a, b) AS c5 FROM t12 WHERE x IN (1, 2) AND y = 'x(unhex)';
SELECT crc32(a, b) AS c6 FROM t12 WHERE x IN (1, 2) AND y = 'x(crc32)';

-- query 25: nope(1)
SELECT tuple_difference_integer(a, b) AS c1 FROM t25 WHERE x IN (1, 2) AND y = 'x(tuple_difference_integer)';
SELECT bitmap_bit_position(a, b) AS c2 FROM t25 WHERE x IN (1, 2) AND y = 'x(bitmap_bit_position)';
SELECT theta_union(a, b) AS c3 FROM t25 WHERE x IN (1, 2) AND y = 'x(theta_union)';
SELECT approx_top_k_accumulate(a, b) AS c4 FROM t25 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';

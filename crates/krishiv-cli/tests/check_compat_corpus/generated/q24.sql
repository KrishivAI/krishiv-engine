-- query 24: nope(1)
SELECT tuple_intersection_theta_double(a, b) AS c1 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_intersection_theta_double)';
SELECT bitmap_bit_position(a, b) AS c2 FROM t24 WHERE x IN (1, 2) AND y = 'x(bitmap_bit_position)';
SELECT tuple_difference_theta_double(a, b) AS c3 FROM t24 WHERE x IN (1, 2) AND y = 'x(tuple_difference_theta_double)';
SELECT approx_top_k_accumulate(a, b) AS c4 FROM t24 WHERE x IN (1, 2) AND y = 'x(approx_top_k_accumulate)';

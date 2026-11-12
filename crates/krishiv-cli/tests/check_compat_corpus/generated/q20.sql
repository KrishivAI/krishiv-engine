-- query 20: nope(1)
SELECT kll_sketch_to_string_double(a, b) AS c1 FROM t20 WHERE x IN (1, 2) AND y = 'x(kll_sketch_to_string_double)';
SELECT kll_sketch_merge_bigint(a, b) AS c2 FROM t20 WHERE x IN (1, 2) AND y = 'x(kll_sketch_merge_bigint)';
SELECT tuple_union_theta_double(a, b) AS c3 FROM t20 WHERE x IN (1, 2) AND y = 'x(tuple_union_theta_double)';
SELECT legacy_fn_21(a, b) AS c4 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_21)';
SELECT legacy_fn_26(a, b) AS c5 FROM t20 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';

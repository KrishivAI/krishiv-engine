-- query 4: nope(1)
SELECT legacy_fn_6(a, b) AS c1 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_6)';
SELECT theta_sketch_agg(a, b) AS c2 FROM t4 WHERE x IN (1, 2) AND y = 'x(theta_sketch_agg)';
SELECT legacy_fn_16(a, b) AS c3 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
SELECT json_array_length(a, b) AS c4 FROM t4 WHERE x IN (1, 2) AND y = 'x(json_array_length)';
SELECT legacy_fn_35(a, b) AS c5 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_35)';
SELECT map_contains_key(a, b) AS c6 FROM t4 WHERE x IN (1, 2) AND y = 'x(map_contains_key)';

-- query 4: nope(1)
SELECT legacy_fn_6(a, b) AS c1 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_6)';
SELECT theta_intersection_agg(a, b) AS c2 FROM t4 WHERE x IN (1, 2) AND y = 'x(theta_intersection_agg)';
SELECT legacy_fn_16(a, b) AS c3 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_16)';
SELECT json_object_keys(a, b) AS c4 FROM t4 WHERE x IN (1, 2) AND y = 'x(json_object_keys)';
SELECT legacy_fn_35(a, b) AS c5 FROM t4 WHERE x IN (1, 2) AND y = 'x(legacy_fn_35)';
SELECT map_entries(a, b) AS c6 FROM t4 WHERE x IN (1, 2) AND y = 'x(map_entries)';

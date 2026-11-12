-- query 3: nope(1)
SELECT theta_union(a, b) AS c1 FROM t3 WHERE x IN (1, 2) AND y = 'x(theta_union)';
SELECT regexp_extract(a, b) AS c2 FROM t3 WHERE x IN (1, 2) AND y = 'x(regexp_extract)';
SELECT theta_sketch_agg(a, b) AS c3 FROM t3 WHERE x IN (1, 2) AND y = 'x(theta_sketch_agg)';
SELECT legacy_fn_30(a, b) AS c4 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_30)';

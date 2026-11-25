-- query 19: nope(1)
SELECT schema_of_variant_agg(a, b) AS c1 FROM t19 WHERE x IN (1, 2) AND y = 'x(schema_of_variant_agg)';
SELECT legacy_fn_26(a, b) AS c2 FROM t19 WHERE x IN (1, 2) AND y = 'x(legacy_fn_26)';
SELECT stddev_pop(a, b) AS c3 FROM t19 WHERE x IN (1, 2) AND y = 'x(stddev_pop)';

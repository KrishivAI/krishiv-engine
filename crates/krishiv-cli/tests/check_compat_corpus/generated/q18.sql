-- query 18: nope(1)
SELECT bigint(a, b) AS c1 FROM t18 WHERE x IN (1, 2) AND y = 'x(bigint)';
SELECT stddev_pop(a, b) AS c2 FROM t18 WHERE x IN (1, 2) AND y = 'x(stddev_pop)';
SELECT schema_of_variant_agg(a, b) AS c3 FROM t18 WHERE x IN (1, 2) AND y = 'x(schema_of_variant_agg)';
SELECT schema_of_avro(a, b) AS c4 FROM t18 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';

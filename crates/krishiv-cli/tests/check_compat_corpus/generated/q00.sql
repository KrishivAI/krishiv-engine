-- query 0: nope(1)
SELECT variant_get(a, b) AS c1 FROM t0 WHERE x IN (1, 2) AND y = 'x(variant_get)';
SELECT count_min_sketch(a, b) AS c2 FROM t0 WHERE x IN (1, 2) AND y = 'x(count_min_sketch)';
SELECT schema_of_avro(a, b) AS c3 FROM t0 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';

-- query 7: nope(1)
SELECT legacy_fn_32(a, b) AS c1 FROM t7 WHERE x IN (1, 2) AND y = 'x(legacy_fn_32)';
SELECT schema_of_avro(a, b) AS c2 FROM t7 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';
SELECT lpad(a, b) AS c3 FROM t7 WHERE x IN (1, 2) AND y = 'x(lpad)';

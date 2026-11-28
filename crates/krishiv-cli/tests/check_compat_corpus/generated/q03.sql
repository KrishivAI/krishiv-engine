-- query 3: nope(1)
SELECT xpath(a, b) AS c1 FROM t3 WHERE x IN (1, 2) AND y = 'x(xpath)';
SELECT log10(a, b) AS c2 FROM t3 WHERE x IN (1, 2) AND y = 'x(log10)';
SELECT legacy_fn_30(a, b) AS c3 FROM t3 WHERE x IN (1, 2) AND y = 'x(legacy_fn_30)';
SELECT schema_of_avro(a, b) AS c4 FROM t3 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';

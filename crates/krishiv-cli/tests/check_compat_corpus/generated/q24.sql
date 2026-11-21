-- query 24: nope(1)
SELECT timestamp_seconds(a, b) AS c1 FROM t24 WHERE x IN (1, 2) AND y = 'x(timestamp_seconds)';
SELECT legacy_fn_14(a, b) AS c2 FROM t24 WHERE x IN (1, 2) AND y = 'x(legacy_fn_14)';
SELECT schema_of_avro(a, b) AS c3 FROM t24 WHERE x IN (1, 2) AND y = 'x(schema_of_avro)';
SELECT date_from_unix_date(a, b) AS c4 FROM t24 WHERE x IN (1, 2) AND y = 'x(date_from_unix_date)';

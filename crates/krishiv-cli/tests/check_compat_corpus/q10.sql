-- query 10: nope(1)
SELECT legacy_fn_12(a, b) AS c1 FROM t10 WHERE x IN (1, 2) AND y = 'x(legacy_fn_12)';
SELECT regr_r2(a, b) AS c2 FROM t10 WHERE x IN (1, 2) AND y = 'x(regr_r2)';
SELECT least(a, b) AS c3 FROM t10 WHERE x IN (1, 2) AND y = 'x(least)';
SELECT version(a, b) AS c4 FROM t10 WHERE x IN (1, 2) AND y = 'x(version)';
SELECT tuple_union_double(a, b) AS c5 FROM t10 WHERE x IN (1, 2) AND y = 'x(tuple_union_double)';
SELECT to_utc_timestamp(a, b) AS c6 FROM t10 WHERE x IN (1, 2) AND y = 'x(to_utc_timestamp)';

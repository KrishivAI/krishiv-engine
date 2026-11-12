-- query 27: nope(1)
SELECT st_geogfromwkb(a, b) AS c1 FROM t27 WHERE x IN (1, 2) AND y = 'x(st_geogfromwkb)';
SELECT legacy_fn_11(a, b) AS c2 FROM t27 WHERE x IN (1, 2) AND y = 'x(legacy_fn_11)';
SELECT unix_seconds(a, b) AS c3 FROM t27 WHERE x IN (1, 2) AND y = 'x(unix_seconds)';

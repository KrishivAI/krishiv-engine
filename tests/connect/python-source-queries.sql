-- P3-T040: the same data as a Python data source (`src`) and as Parquet (`pq`); each statement
-- is run on both with `{t}` replaced by the table name and the results must be equal.
SELECT * FROM {t}
SELECT count(*) FROM {t}
SELECT id, name FROM {t} WHERE id > 20
SELECT id FROM {t} WHERE name LIKE 'n1%'
SELECT grp, count(*), sum(v), min(v), max(v) FROM {t} GROUP BY grp
SELECT grp, avg(v) FROM {t} GROUP BY grp HAVING count(*) > 5
SELECT count(DISTINCT grp) FROM {t}
SELECT id, v * 2 AS w FROM {t} WHERE v IS NOT NULL
SELECT id FROM {t} WHERE v IS NULL
SELECT id, name FROM {t} ORDER BY name DESC, id LIMIT 7
SELECT grp, array_sort(collect_list(id)) FROM {t} GROUP BY grp
SELECT a.id, b.name FROM {t} a JOIN {t} b ON a.id = b.id + 1
SELECT a.grp, count(*) FROM {t} a LEFT JOIN {t} b ON a.id = b.id * 2 GROUP BY a.grp
SELECT id, row_number() OVER (PARTITION BY grp ORDER BY id) FROM {t}
SELECT id, sum(v) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM {t}
SELECT grp FROM {t} GROUP BY grp ORDER BY grp
SELECT id FROM {t} WHERE id IN (SELECT id FROM {t} WHERE v > 50)
SELECT id, CASE WHEN v IS NULL THEN 'none' WHEN v > 50 THEN 'hi' ELSE 'lo' END FROM {t}
SELECT * FROM {t} UNION ALL SELECT * FROM {t} WHERE id < 5
SELECT d, count(*) FROM {t} GROUP BY d

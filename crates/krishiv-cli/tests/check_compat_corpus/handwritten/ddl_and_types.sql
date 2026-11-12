CREATE TABLE IF NOT EXISTS events (
  id BIGINT,
  name VARCHAR(64),
  price NUMERIC(10, 2),
  tags ARRAY<STRING>,
  ts TIMESTAMP
) USING parquet PARTITIONED BY (ts);

ALTER TABLE events ADD COLUMNS (source STRING, score DOUBLE);

INSERT INTO events(id, name, price)
SELECT id, upper(name), try_cast(raw_price AS DECIMAL(10, 2))
FROM staging
WHERE length(name) > 0;

SELECT id, java_method('java.lang.Math', 'abs', id) AS a,
       coalesce(price, 0), if(price > 10, 'hi', 'lo')
FROM events;

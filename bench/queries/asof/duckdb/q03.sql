SELECT count(*) AS n, count(q.bid_cents) AS matched, sum(q.bid_cents) AS bid
FROM trades t ASOF LEFT JOIN (SELECT * FROM quotes WHERE symbol < 'S0900') q
  ON t.symbol = q.symbol AND t.ts >= q.ts

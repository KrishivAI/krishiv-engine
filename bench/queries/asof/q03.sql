SELECT count(*) AS n, count(q.bid_cents) AS matched, sum(q.bid_cents) AS bid
FROM trades t LEFT ASOF JOIN (SELECT * FROM quotes WHERE symbol < 'S0900') q
  MATCH_CONDITION (t.ts >= q.ts)
  ON t.symbol = q.symbol

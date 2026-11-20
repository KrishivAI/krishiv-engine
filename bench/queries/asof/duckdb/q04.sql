SELECT count(*) AS n, sum(q.bid_cents) AS bid
FROM (SELECT * FROM trades WHERE symbol < 'S0100') t
  ASOF JOIN (SELECT * FROM quotes WHERE symbol < 'S0100') q
  ON t.ts >= q.ts

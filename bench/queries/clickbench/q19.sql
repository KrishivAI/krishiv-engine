SELECT UserID, CAST(FLOOR(EventTime % 3600 / 60) AS BIGINT) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC, UserID, m, SearchPhrase LIMIT 10;

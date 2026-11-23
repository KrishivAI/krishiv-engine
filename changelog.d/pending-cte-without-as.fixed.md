A CTE written without `AS` (`WITH c (SELECT 1) SELECT * FROM c`) now parses; the parenthesis is the body, not a column list.

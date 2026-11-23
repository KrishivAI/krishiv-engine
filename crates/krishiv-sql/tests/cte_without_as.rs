// SPDX-License-Identifier: Apache-2.0
//! `AS` is optional in a CTE: Spark's `namedQuery` is
//! `name (columnAliases)? AS? (query)` (Spark 4.2.1 run 2026-11-21:
//! `WITH c (SELECT 1) SELECT * FROM c` parses and returns 1).
use krishiv_sql::ParserOptions;
use krishiv_sql::parse_query;

fn ctes(sql: &str) -> Vec<(String, usize)> {
    let Ok(q) = parse_query(sql, &ParserOptions::default()) else {
        return vec![("parse error".into(), 0)];
    };
    q.with
        .map(|w| {
            w.ctes
                .iter()
                .map(|c| (c.name.value.clone(), c.columns.len()))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn a_cte_body_without_as_is_not_a_column_list() {
    assert_eq!(ctes("WITH c (SELECT 1) SELECT * FROM c"), [("c".into(), 0)]);
    assert_eq!(ctes("WITH c (VALUES 1) SELECT * FROM c"), [("c".into(), 0)]);
    assert_eq!(
        ctes("WITH c ((SELECT 1)) SELECT * FROM c"),
        [("c".into(), 0)]
    );
    assert_eq!(
        ctes("WITH c (SELECT 1), d (SELECT 2) SELECT * FROM c, d"),
        [("c".into(), 0), ("d".into(), 0)]
    );
}

#[test]
fn a_column_list_still_reads_as_one() {
    assert_eq!(
        ctes("WITH c (a, b) AS (SELECT 1, 2) SELECT * FROM c"),
        [("c".into(), 2)]
    );
    assert_eq!(
        ctes("WITH c (a) (SELECT 1) SELECT * FROM c"),
        [("c".into(), 1)]
    );
}

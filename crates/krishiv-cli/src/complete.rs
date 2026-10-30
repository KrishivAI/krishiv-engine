// SPDX-License-Identifier: Apache-2.0
//! Tab-completion sources for the REPL.
//!
//! Candidates come from existing public APIs only: the function registry
//! (`krishiv-functions`), the session catalog (`Session::temp_table_names`
//! plus `DataFrame::schema` for columns) and the config registry
//! (`krishiv-common`). The keyword list is the one exception: `krishiv-sql`
//! exposes no keyword-list API (only [`krishiv_sql::lexer::is_keyword`] for
//! a single word), so the REPL keeps its own copy of the reserved words
//! from `crates/krishiv-sql/src/keywords.rs`. If the engine ever publishes
//! the list, this copy should go.
//!
//! The interactive wiring (a line editor calling [`complete`]) is a
//! follow-up: it needs the workspace's `reedline` crate wired into
//! `krishiv-cli`, which this unit does not add (no new dependencies).
//! Until then the same function backs the REPL's dumb-terminal loop.

use std::collections::BTreeMap;

/// Reserved and clause words, copied from `krishiv-sql/src/keywords.rs`
/// (`KEYWORDS`, upper case, sorted for binary search). Non-reserved words
/// (`year`, ...) are identifiers, so they are not offered.
pub(crate) static KEYWORDS: &[&str] = &[
    "ALL",
    "AND",
    "ANTI",
    "ANY",
    "AS",
    "AUTHORIZATION",
    "BETWEEN",
    "BOTH",
    "CALL",
    "CASE",
    "CAST",
    "CHECK",
    "CLUSTER",
    "COLLATE",
    "COLLATION",
    "COLLATIONS",
    "COLUMN",
    "CONSTRAINT",
    "CREATE",
    "CROSS",
    "CURRENT_DATE",
    "CURRENT_PATH",
    "CURRENT_SCHEMA",
    "CURRENT_TIME",
    "CURRENT_TIMESTAMP",
    "CURRENT_USER",
    "DATEADD",
    "DATEDIFF",
    "DATE_ADD",
    "DATE_DIFF",
    "DISTINCT",
    "DISTRIBUTE",
    "ELSE",
    "END",
    "ESCAPE",
    "EXCEPT",
    "EXECUTE",
    "EXISTS",
    "EXTRACT",
    "FALSE",
    "FETCH",
    "FILTER",
    "FOR",
    "FOREIGN",
    "FROM",
    "FULL",
    "GRANT",
    "GROUP",
    "HAVING",
    "IN",
    "INNER",
    "INTERSECT",
    "INTERVAL",
    "INTO",
    "IS",
    "JOIN",
    "LATERAL",
    "LEADING",
    "LEFT",
    "LIMIT",
    "MINUS",
    "NATURAL",
    "NOT",
    "NULL",
    "OFFSET",
    "ON",
    "ONLY",
    "OR",
    "ORDER",
    "OUTER",
    "OVERLAPS",
    "OVERLAY",
    "PIVOT",
    "POSITION",
    "PRIMARY",
    "QUALIFY",
    "RECURSIVE",
    "REFERENCES",
    "RIGHT",
    "SELECT",
    "SEMI",
    "SESSION_USER",
    "SOME",
    "SORT",
    "SQL",
    "STRUCT",
    "SUBSTR",
    "SUBSTRING",
    "TABLE",
    "TABLESAMPLE",
    "THEN",
    "TIME",
    "TIMEDIFF",
    "TIMESTAMPADD",
    "TIMESTAMPDIFF",
    "TO",
    "TRAILING",
    "TRIM",
    "TRUE",
    "TRY_CAST",
    "UNION",
    "UNIQUE",
    "UNKNOWN",
    "UNPIVOT",
    "USER",
    "USING",
    "WATERMARK",
    "WHEN",
    "WHERE",
    "WINDOW",
    "WITH",
    "WITHIN",
];

/// Whether `word` is a keyword offered as a completion (ASCII
/// case-insensitive, as the parser folds).
pub(crate) fn is_keyword_word(word: &str) -> bool {
    let upper = word.to_ascii_uppercase();
    KEYWORDS
        .binary_search_by(|k| k.as_bytes().cmp(upper.as_bytes()))
        .is_ok()
}

/// Function names from the registry, lower case, sorted.
pub(crate) fn function_names() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = krishiv_functions::registry::builtins()
        .functions()
        .iter()
        .map(|d| d.name)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Config keys from the config registry: every key and its Spark alias.
pub(crate) fn config_keys() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = krishiv_common::config::REGISTRY
        .iter()
        .flat_map(|e| std::iter::once(e.key).chain(e.spark_alias))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Table names of the session catalog (temp views).
pub(crate) fn table_names(session: &krishiv::Session) -> Vec<String> {
    session.temp_table_names()
}

/// Column names of `table` (empty when the table is unknown: completion
/// must never fail).
pub(crate) fn column_names(session: &krishiv::Session, table: &str) -> Vec<String> {
    futures::executor::block_on(session.table(table).schema())
        .map(|s| s.fields().iter().map(|f| f.name().clone()).collect())
        .unwrap_or_default()
}

/// What the completer may offer: tables and, per table, columns.
#[derive(Debug, Default)]
pub(crate) struct CompletionContext {
    /// Table names of the session catalog.
    pub(crate) tables: Vec<String>,
    /// Column names by table.
    pub(crate) columns: BTreeMap<String, Vec<String>>,
}

impl CompletionContext {
    /// Gathers tables and their columns from the session catalog with the
    /// existing public APIs. Unknown tables keep no columns.
    pub(crate) fn gather(session: &krishiv::Session) -> Self {
        let tables = table_names(session);
        let columns = tables
            .iter()
            .map(|t| (t.clone(), column_names(session, t)))
            .collect();
        Self { tables, columns }
    }
}

/// Candidates for the word under `cursor` in `line`: keywords, function
/// names, catalog tables/columns and `SET` keys, filtered by the current
/// word prefix (case-insensitive) and sorted. Context-aware: after `SET`
/// only config keys; after `FROM`/`JOIN` (and friends) tables first;
/// after `SELECT ... FROM <table>` that table's columns.
///
/// `cursor` is a byte offset; past-the-end and mid-character values are
/// clamped to a character boundary, so an unbalanced quote elsewhere in
/// the line cannot break completion.
pub(crate) fn complete(line: &str, cursor: usize, ctx: &CompletionContext) -> Vec<String> {
    let end = line.floor_char_boundary(cursor.min(line.len()));
    let prefix = &line[..end];
    if in_set_context(prefix) {
        // Config keys are dotted: match the whole `a.b.c` run, not just
        // the tail segment.
        return filter_by_prefix(config_keys(), dotted_word(prefix));
    }
    let word = tail_word(prefix);
    let mut out = if in_table_context(prefix) {
        let mut out = filter_by_prefix(ctx.tables.iter().map(String::as_str), word);
        out.extend(keyword_candidates(word));
        out
    } else if let Some(table) = selected_table(line) {
        // The table may come after the cursor (`SELECT l_<cursor> FROM t`),
        // so it is read from the whole line while the word is not.
        let cols = ctx.columns.get(&table).cloned().unwrap_or_default();
        let mut out = filter_by_prefix(cols.iter().map(String::as_str), word);
        out.extend(filter_by_prefix(function_names(), word));
        out.extend(keyword_candidates(word));
        out
    } else {
        let mut out = keyword_candidates(word);
        out.extend(filter_by_prefix(function_names(), word));
        out.extend(filter_by_prefix(
            ctx.tables.iter().map(String::as_str),
            word,
        ));
        out
    };
    out.sort();
    out.dedup();
    out
}

/// Candidates from the keyword list matching `word`.
fn keyword_candidates(word: &str) -> Vec<String> {
    filter_by_prefix(KEYWORDS.iter().copied(), word)
}

/// `candidates` starting with `word` (case-insensitive), as owned strings.
fn filter_by_prefix<'a>(candidates: impl IntoIterator<Item = &'a str>, word: &str) -> Vec<String> {
    let lower = word.to_ascii_lowercase();
    candidates
        .into_iter()
        .filter(|c| c.to_ascii_lowercase().starts_with(&lower))
        .map(ToString::to_string)
        .collect()
}

/// The byte index where the word under the cursor starts: past the
/// last character that cannot appear in a word. Char-boundary safe for
/// any input.
fn word_start(prefix: &str) -> usize {
    prefix
        .char_indices()
        .rev()
        .find(|(_, c)| !(c.is_alphanumeric() || *c == '_' || *c == '.'))
        .map_or(0, |(i, c)| i + c.len_utf8())
}

/// The identifier (or dotted path tail) right before the cursor.
fn tail_word(prefix: &str) -> &str {
    let word = &prefix[word_start(prefix)..];
    word.rsplit('.').next().unwrap_or(word)
}

/// The dotted run right before the cursor (`spark.sql.ansi`), for `SET`
/// key completion.
fn dotted_word(prefix: &str) -> &str {
    prefix[word_start(prefix)..].trim_start_matches('.')
}

/// Whether `prefix` is inside a `SET <key>` assignment: the current
/// statement starts with `SET` and carries no assigned value yet.
fn in_set_context(prefix: &str) -> bool {
    let stmt = prefix.rsplit(';').next().unwrap_or(prefix);
    let trimmed = stmt.trim_start();
    let Some(head) = trimmed.get(..3) else {
        return false;
    };
    if !head.eq_ignore_ascii_case("SET") {
        return false;
    }
    let rest = trimmed.get(3..).unwrap_or("");
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return false;
    }
    // `SET`, `SET key<cursor>` and `SET key =<cursor>` complete keys; once
    // a value follows the `=`, key completion no longer applies.
    match rest.find('=') {
        None => true,
        Some(eq) => rest[eq + 1..].trim().is_empty(),
    }
}

/// Whether a table name is expected: the last whole word before the
/// cursor is one that takes a table (`FROM`, `JOIN`, ...).
fn in_table_context(prefix: &str) -> bool {
    const TABLE_WORDS: &[&str] = &["FROM", "JOIN", "INTO", "TABLE", "UPDATE", "DESCRIBE"];
    // Drop the partial word under the cursor (byte index into `prefix`).
    let last = prefix[..word_start(prefix)]
        .split_whitespace()
        .next_back()
        .unwrap_or("");
    TABLE_WORDS.iter().any(|w| last.eq_ignore_ascii_case(w))
}

/// The table of `SELECT ... FROM <table>` when the line names exactly
/// one, for column completion.
fn selected_table(line: &str) -> Option<String> {
    // The last standalone `FROM` word (byte indices into `line` over an
    // ASCII-only scan, so no case-folded copy is ever sliced).
    let mut from_end = None;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            if line[start..i].eq_ignore_ascii_case("FROM") {
                from_end = Some(i);
            }
        } else {
            i += 1;
        }
    }
    let after = line[from_end?..].trim_start();
    let table = after
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
        .next()
        .unwrap_or("")
        .trim();
    if table.is_empty() || table.contains([';', '(', ')']) {
        return None;
    }
    Some(table.to_owned())
}

#[cfg(test)]
mod tests {
    #[test]
    fn keywords_are_upper_case_sorted_and_unique() {
        // `is_keyword_word` binary-searches the table: a misplaced entry would silently vanish.
        for pair in KEYWORDS.windows(2) {
            assert!(pair[0] < pair[1], "{} before {}", pair[0], pair[1]);
        }
        for k in KEYWORDS {
            assert_eq!(*k, k.to_ascii_uppercase());
        }
        assert!(is_keyword_word("select"));
        assert!(!is_keyword_word("zzzword"));
    }

    use super::*;

    fn ctx() -> CompletionContext {
        CompletionContext {
            tables: vec!["lineitem".to_owned(), "orders".to_owned()],
            columns: BTreeMap::from([
                (
                    "lineitem".to_owned(),
                    vec!["l_orderkey".to_owned(), "l_partkey".to_owned()],
                ),
                ("orders".to_owned(), vec!["o_orderkey".to_owned()]),
            ]),
        }
    }

    #[test]
    fn keywords_complete_case_insensitively() {
        assert!(is_keyword_word("select"));
        assert!(is_keyword_word("SELECT"));
        assert!(!is_keyword_word("year"));
        // A strict prefix of a keyword is not a keyword itself.
        assert!(!is_keyword_word("sel"));
        let got = complete("SEL", 3, &ctx());
        assert!(got.contains(&"SELECT".to_owned()), "{got:?}");
    }

    #[test]
    fn functions_come_from_the_registry() {
        let names = function_names();
        assert!(names.len() > 50, "{} functions", names.len());
        assert!(names.contains(&"abs"), "{names:?}");
        assert!(names.contains(&"count"), "{names:?}");
        let got = complete("select ab", 9, &ctx());
        assert!(got.contains(&"abs".to_owned()), "{got:?}");
    }

    #[test]
    fn tables_complete_after_from() {
        let got = complete("select * from lin", 17, &ctx());
        assert!(got.contains(&"lineitem".to_owned()), "{got:?}");
        assert!(!got.contains(&"l_orderkey".to_owned()), "{got:?}");
    }

    #[test]
    fn columns_complete_after_select_from_table() {
        let got = complete("select l_ from lineitem", 9, &ctx());
        assert!(got.contains(&"l_orderkey".to_owned()), "{got:?}");
        assert!(got.contains(&"l_partkey".to_owned()), "{got:?}");
        assert!(!got.contains(&"o_orderkey".to_owned()), "{got:?}");
    }

    #[test]
    fn set_completes_config_keys_only() {
        assert!(!config_keys().is_empty());
        assert!(config_keys().contains(&"spark.sql.ansi.enabled"));
        let got = complete("SET spark.sql.ansi", 18, &ctx());
        assert!(
            got.contains(&"spark.sql.ansi.enabled".to_owned()),
            "{got:?}"
        );
        assert!(!got.contains(&"SELECT".to_owned()), "{got:?}");
        assert!(!got.contains(&"lineitem".to_owned()), "{got:?}");
    }

    #[test]
    fn cursor_is_clamped_and_quotes_do_not_break_it() {
        let line = "select 'unbalanced from lineitem";
        let got = complete(line, line.len() + 99, &ctx());
        assert!(!got.is_empty());
        let mid = complete("SEL", 1, &ctx());
        assert!(mid.contains(&"SELECT".to_owned()), "{mid:?}");
        // Multibyte separators and words never panic the slicer.
        let uni = "select héllo。SEL";
        let got = complete(uni, uni.len(), &ctx());
        assert!(got.contains(&"SELECT".to_owned()), "{got:?}");
        assert!(!is_keyword_word("sél"));
    }

    #[test]
    fn context_comes_from_the_session_catalog() {
        let session = krishiv::Session::new().expect("session");
        let flown =
            futures::executor::block_on(session.sql("select 1 as a, 2 as b")).expect("query");
        futures::executor::block_on(flown.create_or_replace_temp_view("compl_t")).expect("view");
        let gathered = CompletionContext::gather(&session);
        assert!(
            gathered.tables.contains(&"compl_t".to_owned()),
            "{gathered:?}"
        );
        let cols = gathered.columns.get("compl_t").cloned().unwrap_or_default();
        assert_eq!(cols, vec!["a".to_owned(), "b".to_owned()]);
        let got = complete("select * from compl", 17, &gathered);
        assert!(got.contains(&"compl_t".to_owned()), "{got:?}");
    }
}

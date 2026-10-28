// SPDX-License-Identifier: Apache-2.0
//! Keyword classes of Spark 4.2's grammar (`SqlBaseParser.g4`).
//!
//! With `spark.sql.ansi.enabled` and `spark.sql.ansi.enforceReservedKeywords`
//! both on, the reserved words (every keyword outside
//! `ansiNonReserved`) are never identifiers. Otherwise every keyword may be
//! an identifier and ANTLR's prediction picks a reading; this parser is
//! deterministic, so it treats the same words as reserved where an
//! identifier and a keyword reading compete (a column reference, a table
//! name, an alias without `AS`), and accepts any word where only an
//! identifier fits (after `AS`, after `.`, in a column list).

/// Keyword classes by upper-case word, sorted: `R` reserved in ANSI mode
/// with enforced keywords (every keyword outside `ansiNonReserved`), `C`
/// starts a clause, `N` never names a function before `(` (structural or
/// strict-non-reserved), `P` may start a keyword primary, `S`
/// strict-non-reserved.
const KEYWORDS: &[(&str, u8)] = &[
    ("ALL", R),
    ("AND", R | N),
    ("ANTI", C | N | S),
    ("ANY", R),
    ("AS", R | N),
    ("AUTHORIZATION", R),
    ("BETWEEN", N),
    ("BOTH", R),
    ("CALL", R),
    ("CASE", R | P),
    ("CAST", R | P),
    ("CHECK", R),
    ("CLUSTER", C),
    ("COLLATE", R),
    ("COLLATION", R),
    ("COLLATIONS", R),
    ("COLUMN", R),
    ("CONSTRAINT", R),
    ("CREATE", R),
    ("CROSS", R | N | S),
    ("CURRENT_DATE", R | P),
    ("CURRENT_PATH", R | P),
    ("CURRENT_SCHEMA", R),
    ("CURRENT_TIME", R | P),
    ("CURRENT_TIMESTAMP", R | P),
    ("CURRENT_USER", R | P),
    ("DATEADD", P),
    ("DATEDIFF", P),
    ("DATE_ADD", P),
    ("DATE_DIFF", P),
    ("DISTINCT", R),
    ("DISTRIBUTE", C),
    ("ELSE", R | N),
    ("END", R | N),
    ("ESCAPE", R),
    ("EXCEPT", R | N | S),
    ("EXECUTE", R),
    ("EXISTS", P),
    ("EXTRACT", P),
    ("FALSE", R | P),
    ("FETCH", R),
    ("FILTER", R),
    ("FOR", R),
    ("FOREIGN", R),
    ("FROM", R | N),
    ("FULL", R | N | S),
    ("GRANT", R),
    ("GROUP", R),
    ("HAVING", R | N),
    ("IN", R | N),
    ("INNER", R | N | S),
    ("INTERSECT", R | N | S),
    ("INTERVAL", P),
    ("INTO", R),
    ("IS", R | N),
    ("JOIN", R | N | S),
    ("LATERAL", R | N | S),
    ("LEADING", R),
    ("LEFT", R | S),
    ("LIMIT", C),
    ("MINUS", C | N | S),
    ("NATURAL", R | N | S),
    ("NOT", R),
    ("NULL", R | P),
    ("OFFSET", R),
    ("ON", R | N | S),
    ("ONLY", R),
    ("OR", R | N),
    ("ORDER", R),
    ("OUTER", R),
    ("OVERLAPS", R),
    ("OVERLAY", P),
    ("PIVOT", C),
    ("POSITION", P),
    ("PRIMARY", R),
    ("QUALIFY", C),
    ("RECURSIVE", R),
    ("REFERENCES", R),
    ("RIGHT", R | S),
    ("SELECT", R),
    ("SEMI", C | N | S),
    ("SESSION_USER", R | P),
    ("SOME", R),
    ("SORT", C),
    ("SQL", R),
    ("STRUCT", P),
    ("SUBSTR", P),
    ("SUBSTRING", P),
    ("TABLE", R),
    ("TABLESAMPLE", C),
    ("THEN", R | N),
    ("TIME", R),
    ("TIMEDIFF", P),
    ("TIMESTAMPADD", P),
    ("TIMESTAMPDIFF", P),
    ("TO", R),
    ("TRAILING", R),
    ("TRIM", P),
    ("TRUE", P),
    ("TRY_CAST", P),
    ("UNION", R | N | S),
    ("UNIQUE", R),
    ("UNKNOWN", R),
    ("UNPIVOT", C),
    ("USER", R | P),
    ("USING", R | N | S),
    ("WATERMARK", C),
    ("WHEN", R | N),
    ("WHERE", R | N),
    ("WINDOW", C),
    ("WITH", R),
    ("WITHIN", R),
];

/// `word`, upper-cased the way Spark's case-insensitive stream does
/// (`Character.toUpperCase` per character), into `buf`; `None` if it does
/// not fit (no keyword is that long).
fn upper<'b>(word: &str, buf: &'b mut [u8; 24]) -> Option<&'b str> {
    if word.is_ascii() {
        let out = buf.get_mut(..word.len())?;
        out.copy_from_slice(word.as_bytes());
        out.make_ascii_uppercase();
        return std::str::from_utf8(out).ok();
    }
    let mut len = 0;
    for c in word.chars() {
        let mut up = c.to_uppercase();
        let u = match (up.next(), up.next()) {
            (Some(u), None) if u.is_ascii() => u,
            _ => return None,
        };
        *buf.get_mut(len)? = u8::try_from(u).ok()?;
        len += 1;
    }
    std::str::from_utf8(buf.get(..len)?).ok()
}

/// Whether `word` is the keyword `kw` (upper-case).
pub(crate) fn is_kw(word: &str, kw: &str) -> bool {
    // A non-ASCII character folds to at most one ASCII letter, so only a
    // longer non-ASCII word can match a keyword of another length.
    if word.len() == kw.len() || word.len() < kw.len() || word.is_ascii() {
        return word.eq_ignore_ascii_case(kw);
    }
    let mut buf = [0; 24];
    upper(word, &mut buf) == Some(kw)
}

/// [`classify`] bit: reserved in ANSI mode with enforced keywords.
pub(crate) const RESERVED_WORD: u8 = 1;
/// [`classify`] bit: starts a clause, so it is not an alias without `AS`.
pub(crate) const CLAUSE_WORD: u8 = 2;
/// [`classify`] bit: never names a function before `(`.
pub(crate) const NOT_FUNCTION_WORD: u8 = 4;
/// [`classify`] bit: may start a keyword primary (`CASE`, `CAST`, ...).
pub(crate) const PRIMARY_WORD: u8 = 8;
/// [`classify`] bit: strict-non-reserved (never a table alias).
pub(crate) const STRICT_WORD: u8 = 16;
const R: u8 = RESERVED_WORD;
const C: u8 = CLAUSE_WORD;
const N: u8 = NOT_FUNCTION_WORD;
const P: u8 = PRIMARY_WORD;
const S: u8 = STRICT_WORD;

/// The keyword classes of `word`, as bits; computed once per token.
pub(crate) fn classify(word: &str) -> u8 {
    let mut buf = [0; 24];
    upper(word, &mut buf)
        .and_then(|up| KEYWORDS.binary_search_by(|(k, _)| k.cmp(&up)).ok())
        .and_then(|i| KEYWORDS.get(i))
        .map_or(0, |(_, bits)| *bits)
}

/// Every Spark 4.2 keyword (`tests/fixtures/spark-4.2.0/sql-keywords.json`,
/// generated from `SqlBaseLexer.g4` / `SqlBaseParser.g4`) with whether it
/// is reserved in ANSI mode with enforced keywords. Sorted for binary
/// search; the single source for `sql_keywords()` (passed through the SQL
/// front end, never duplicated).
pub const SQL_KEYWORDS: &[(&str, bool)] = &[
    ("ADD", false),
    ("AFTER", false),
    ("AGGREGATE", false),
    ("ALL", true),
    ("ALTER", false),
    ("ALWAYS", false),
    ("ANALYZE", false),
    ("AND", true),
    ("ANTI", false),
    ("ANY", true),
    ("ANY_VALUE", false),
    ("APPROX", false),
    ("ARCHIVE", false),
    ("ARRAY", false),
    ("AS", true),
    ("ASC", false),
    ("ASENSITIVE", false),
    ("AT", false),
    ("ATOMIC", false),
    ("AUTHORIZATION", true),
    ("BEGIN", false),
    ("BERNOULLI", false),
    ("BETWEEN", false),
    ("BIGINT", false),
    ("BINARY", false),
    ("BINDING", false),
    ("BOOLEAN", false),
    ("BOTH", true),
    ("BUCKET", false),
    ("BUCKETS", false),
    ("BY", false),
    ("BYTE", false),
    ("CACHE", false),
    ("CALL", true),
    ("CALLED", false),
    ("CASCADE", false),
    ("CASE", true),
    ("CAST", true),
    ("CATALOG", false),
    ("CATALOGS", false),
    ("CHANGE", false),
    ("CHANGES", false),
    ("CHAR", false),
    ("CHARACTER", false),
    ("CHECK", true),
    ("CLEAR", false),
    ("CLOSE", false),
    ("CLUSTER", false),
    ("CLUSTERED", false),
    ("CODEGEN", false),
    ("COLLATE", true),
    ("COLLATION", true),
    ("COLLATIONS", true),
    ("COLLECTION", false),
    ("COLUMN", true),
    ("COLUMNS", false),
    ("COMMENT", false),
    ("COMMIT", false),
    ("COMPACT", false),
    ("COMPACTIONS", false),
    ("COMPENSATION", false),
    ("COMPUTE", false),
    ("CONCATENATE", false),
    ("CONDITION", false),
    ("CONSTRAINT", true),
    ("CONTAINS", false),
    ("CONTINUE", false),
    ("COST", false),
    ("CREATE", true),
    ("CROSS", true),
    ("CUBE", false),
    ("CURRENT", false),
    ("CURRENT_DATABASE", false),
    ("CURRENT_DATE", true),
    ("CURRENT_PATH", true),
    ("CURRENT_SCHEMA", true),
    ("CURRENT_TIME", true),
    ("CURRENT_TIMESTAMP", true),
    ("CURRENT_USER", true),
    ("CURSOR", false),
    ("DATA", false),
    ("DATABASE", false),
    ("DATABASES", false),
    ("DATE", false),
    ("DATEADD", false),
    ("DATEDIFF", false),
    ("DATE_ADD", false),
    ("DATE_DIFF", false),
    ("DAY", false),
    ("DAYOFYEAR", false),
    ("DAYS", false),
    ("DBPROPERTIES", false),
    ("DEC", false),
    ("DECIMAL", false),
    ("DECLARE", false),
    ("DEFAULT", false),
    ("DEFAULT_PATH", false),
    ("DEFINED", false),
    ("DEFINER", false),
    ("DELAY", false),
    ("DELETE", false),
    ("DELIMITED", false),
    ("DESC", false),
    ("DESCRIBE", false),
    ("DETERMINISTIC", false),
    ("DFS", false),
    ("DIRECTORIES", false),
    ("DIRECTORY", false),
    ("DISTANCE", false),
    ("DISTINCT", true),
    ("DISTRIBUTE", false),
    ("DIV", false),
    ("DO", false),
    ("DOUBLE", false),
    ("DROP", false),
    ("ELSE", true),
    ("ELSEIF", false),
    ("END", true),
    ("ENFORCED", false),
    ("ESCAPE", true),
    ("ESCAPED", false),
    ("EVOLUTION", false),
    ("EXACT", false),
    ("EXCEPT", true),
    ("EXCHANGE", false),
    ("EXCLUDE", false),
    ("EXCLUSIVE", false),
    ("EXECUTE", true),
    ("EXISTS", false),
    ("EXIT", false),
    ("EXPLAIN", false),
    ("EXPORT", false),
    ("EXTEND", false),
    ("EXTENDED", false),
    ("EXTERNAL", false),
    ("EXTRACT", false),
    ("FALSE", true),
    ("FETCH", true),
    ("FIELDS", false),
    ("FILEFORMAT", false),
    ("FILTER", true),
    ("FIRST", false),
    ("FLOAT", false),
    ("FLOW", false),
    ("FOLLOWING", false),
    ("FOR", true),
    ("FOREIGN", true),
    ("FORMAT", false),
    ("FORMATTED", false),
    ("FOUND", false),
    ("FROM", true),
    ("FULL", true),
    ("FUNCTION", false),
    ("FUNCTIONS", false),
    ("GENERATED", false),
    ("GEOGRAPHY", false),
    ("GEOMETRY", false),
    ("GLOBAL", false),
    ("GRANT", true),
    ("GROUP", true),
    ("GROUPING", false),
    ("HANDLER", false),
    ("HAVING", true),
    ("HOUR", false),
    ("HOURS", false),
    ("IDENTIFIED", false),
    ("IDENTIFIER", false),
    ("IDENTITY", false),
    ("IF", false),
    ("IGNORE", false),
    ("ILIKE", false),
    ("IMMEDIATE", false),
    ("IMPORT", false),
    ("IN", true),
    ("INCLUDE", false),
    ("INCLUSIVE", false),
    ("INCREMENT", false),
    ("INDEX", false),
    ("INDEXES", false),
    ("INNER", true),
    ("INPATH", false),
    ("INPUT", false),
    ("INPUTFORMAT", false),
    ("INSENSITIVE", false),
    ("INSERT", false),
    ("INT", false),
    ("INTEGER", false),
    ("INTERSECT", true),
    ("INTERVAL", false),
    ("INTO", true),
    ("INVOKER", false),
    ("IS", true),
    ("ITEMS", false),
    ("ITERATE", false),
    ("JOIN", true),
    ("JSON", false),
    ("KEY", false),
    ("KEYS", false),
    ("LANGUAGE", false),
    ("LAST", false),
    ("LATERAL", true),
    ("LAZY", false),
    ("LEADING", true),
    ("LEAVE", false),
    ("LEFT", true),
    ("LEVEL", false),
    ("LIKE", false),
    ("LIMIT", false),
    ("LINES", false),
    ("LIST", false),
    ("LOAD", false),
    ("LOCAL", false),
    ("LOCATION", false),
    ("LOCK", false),
    ("LOCKS", false),
    ("LOGICAL", false),
    ("LONG", false),
    ("LOOP", false),
    ("MACRO", false),
    ("MAP", false),
    ("MATCHED", false),
    ("MATERIALIZED", false),
    ("MAX", false),
    ("MEASURE", false),
    ("MERGE", false),
    ("METRICS", false),
    ("MICROSECOND", false),
    ("MICROSECONDS", false),
    ("MILLISECOND", false),
    ("MILLISECONDS", false),
    ("MINUS", false),
    ("MINUTE", false),
    ("MINUTES", false),
    ("MODIFIES", false),
    ("MONTH", false),
    ("MONTHS", false),
    ("MSCK", false),
    ("NAME", false),
    ("NAMESPACE", false),
    ("NAMESPACES", false),
    ("NANOSECOND", false),
    ("NANOSECONDS", false),
    ("NATURAL", true),
    ("NEAREST", false),
    ("NEXT", false),
    ("NO", false),
    ("NONE", false),
    ("NORELY", false),
    ("NOT", true),
    ("NULL", true),
    ("NULLS", false),
    ("NUMERIC", false),
    ("OF", false),
    ("OFFSET", true),
    ("ON", true),
    ("ONLY", true),
    ("OPEN", false),
    ("OPTION", false),
    ("OPTIONS", false),
    ("OR", true),
    ("ORDER", true),
    ("OUT", false),
    ("OUTER", true),
    ("OUTPUTFORMAT", false),
    ("OVER", false),
    ("OVERLAPS", true),
    ("OVERLAY", false),
    ("OVERWRITE", false),
    ("PARTITION", false),
    ("PARTITIONED", false),
    ("PARTITIONS", false),
    ("PATH", false),
    ("PERCENT", false),
    ("PIVOT", false),
    ("PLACING", false),
    ("POSITION", false),
    ("PRECEDING", false),
    ("PRIMARY", true),
    ("PRINCIPALS", false),
    ("PROCEDURE", false),
    ("PROCEDURES", false),
    ("PROPERTIES", false),
    ("PURGE", false),
    ("QUALIFY", false),
    ("QUARTER", false),
    ("QUERY", false),
    ("RANGE", false),
    ("READ", false),
    ("READS", false),
    ("REAL", false),
    ("RECORDREADER", false),
    ("RECORDWRITER", false),
    ("RECOVER", false),
    ("RECURSION", false),
    ("RECURSIVE", true),
    ("REDUCE", false),
    ("REFERENCES", true),
    ("REFRESH", false),
    ("REGEXP", false),
    ("RELY", false),
    ("RENAME", false),
    ("REPAIR", false),
    ("REPEAT", false),
    ("REPEATABLE", false),
    ("REPLACE", false),
    ("RESET", false),
    ("RESPECT", false),
    ("RESTRICT", false),
    ("RETURN", false),
    ("RETURNS", false),
    ("REVOKE", false),
    ("RIGHT", true),
    ("RLIKE", false),
    ("ROLE", false),
    ("ROLES", false),
    ("ROLLBACK", false),
    ("ROLLUP", false),
    ("ROW", false),
    ("ROWS", false),
    ("SCHEMA", false),
    ("SCHEMAS", false),
    ("SECOND", false),
    ("SECONDS", false),
    ("SECURITY", false),
    ("SELECT", true),
    ("SEMI", false),
    ("SEPARATED", false),
    ("SERDE", false),
    ("SERDEPROPERTIES", false),
    ("SESSION_USER", true),
    ("SET", false),
    ("SETS", false),
    ("SHORT", false),
    ("SHOW", false),
    ("SIMILARITY", false),
    ("SINGLE", false),
    ("SKEWED", false),
    ("SMALLINT", false),
    ("SOME", true),
    ("SORT", false),
    ("SORTED", false),
    ("SOURCE", false),
    ("SPECIFIC", false),
    ("SQL", true),
    ("SQLEXCEPTION", false),
    ("SQLSTATE", false),
    ("START", false),
    ("STATISTICS", false),
    ("STORED", false),
    ("STRATIFY", false),
    ("STREAM", false),
    ("STREAMING", false),
    ("STRING", false),
    ("STRUCT", false),
    ("SUBSTR", false),
    ("SUBSTRING", false),
    ("SYNC", false),
    ("SYSTEM", false),
    ("SYSTEM_PATH", false),
    ("SYSTEM_TIME", false),
    ("SYSTEM_VERSION", false),
    ("TABLE", true),
    ("TABLES", false),
    ("TABLESAMPLE", false),
    ("TARGET", false),
    ("TBLPROPERTIES", false),
    ("TEMP", false),
    ("TEMPORARY", false),
    ("TERMINATED", false),
    ("THEN", true),
    ("TIME", true),
    ("TIMEDIFF", false),
    ("TIMESTAMP", false),
    ("TIMESTAMPADD", false),
    ("TIMESTAMPDIFF", false),
    ("TIMESTAMP_LTZ", false),
    ("TIMESTAMP_NTZ", false),
    ("TINYINT", false),
    ("TO", true),
    ("TOUCH", false),
    ("TRAILING", true),
    ("TRANSACTION", false),
    ("TRANSACTIONS", false),
    ("TRANSFORM", false),
    ("TRIM", false),
    ("TRUE", false),
    ("TRUNCATE", false),
    ("TRY_CAST", false),
    ("TYPE", false),
    ("UNARCHIVE", false),
    ("UNBOUNDED", false),
    ("UNCACHE", false),
    ("UNION", true),
    ("UNIQUE", true),
    ("UNKNOWN", true),
    ("UNLOCK", false),
    ("UNPIVOT", false),
    ("UNSET", false),
    ("UNTIL", false),
    ("UPDATE", false),
    ("USE", false),
    ("USER", true),
    ("USING", true),
    ("VALUE", false),
    ("VALUES", false),
    ("VAR", false),
    ("VARCHAR", false),
    ("VARIABLE", false),
    ("VARIANT", false),
    ("VERSION", false),
    ("VIEW", false),
    ("VIEWS", false),
    ("VOID", false),
    ("WATERMARK", false),
    ("WEEK", false),
    ("WEEKS", false),
    ("WHEN", true),
    ("WHERE", true),
    ("WHILE", false),
    ("WINDOW", false),
    ("WITH", true),
    ("WITHIN", true),
    ("WITHOUT", false),
    ("X", false),
    ("YEAR", false),
    ("YEARS", false),
    ("ZONE", false),
];

/// The rows of `sql_keywords()`: `(keyword, reserved)` with `reserved`
/// honoring `enforce_reserved` (as Spark's `SqlKeywords` does: reserved
/// only when ANSI reserved keywords are enforced).
#[must_use]
pub fn sql_keywords(enforce_reserved: bool) -> Vec<(String, bool)> {
    SQL_KEYWORDS
        .iter()
        .map(|(k, r)| ((*k).to_owned(), *r && enforce_reserved))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_sorted_for_binary_search() {
        assert!(KEYWORDS.windows(2).all(|w| w[0].0 < w[1].0));
    }

    /// The reserved and strict-non-reserved classes are Spark's grammar's
    /// (fixture generated from `SqlBaseLexer.g4` / `SqlBaseParser.g4`).
    #[test]
    fn classes_match_spark_grammar() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/spark-4.2.0/sql-keywords.json"
        );
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let keywords = doc["keywords"].as_array().unwrap();
        assert!(keywords.len() > 400);
        for k in keywords {
            let text = k["text"].as_str().unwrap();
            let bits = classify(text);
            assert_eq!(
                bits & R != 0,
                k["ansi_reserved"].as_bool().unwrap(),
                "{text}"
            );
            assert_eq!(
                bits & S != 0,
                k["strict_non_reserved"].as_bool().unwrap(),
                "{text}"
            );
        }
    }

    #[test]
    fn keywords_fold_like_java() {
        assert!(is_kw("select", "SELECT"));
        // Dotless i and long s upper-case to ASCII in Java.
        assert!(is_kw("\u{131}n", "IN"));
        assert!(is_kw("\u{17F}elect", "SELECT"));
        assert!(!is_kw("selectx", "SELECT"));
        assert!(!is_kw("ß", "SS"));
        assert_eq!(classify("From"), RESERVED_WORD | NOT_FUNCTION_WORD);
        assert_eq!(classify("year"), 0);
        assert_eq!(classify("cast"), RESERVED_WORD | PRIMARY_WORD);
        assert_eq!(classify("limit"), CLAUSE_WORD);
        assert_eq!(classify("averyveryverylongidentifiername"), 0);
    }
}

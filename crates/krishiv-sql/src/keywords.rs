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
    ("END", N),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_sorted_for_binary_search() {
        assert!(KEYWORDS.windows(2).all(|w| w[0].0 < w[1].0));
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

// SPDX-License-Identifier: Apache-2.0
//! REPL history: `~/.krishiv/history`, at most [`MAX_ENTRIES`] lines.
//!
//! Lines that set a `sensitive` option never reach the file: whether an
//! option key is sensitive is the engine's typed-secret classification
//! ([`krishiv_common::secret::is_sensitive_key`]), reused here so the
//! history and the plan redaction agree on what a secret is.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

/// Most history entries kept; older ones are dropped on append.
pub(crate) const MAX_ENTRIES: usize = 10_000;

/// The history file (`~/.krishiv/history`); `None` when no home directory
/// is known, in which case history stays in memory only.
pub(crate) fn history_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;
    Some(home.join(".krishiv").join("history"))
}

/// Whether `line` may be persisted: false for statements that set a
/// sensitive option (`SET password=...`, `SET ... = '...'` with a
/// sensitive key, in any letter case). Everything else persists,
/// including `SET` of ordinary keys and failed statements.
pub(crate) fn should_persist(line: &str) -> bool {
    // Every `;`-separated statement must be persistable; one sensitive
    // `SET` vetoes the whole line, so a secret canary never lands in the
    // file through a multi-statement paste.
    split_top_level(line).iter().all(|stmt| {
        let Some(key) = set_key(stmt) else {
            return !assigns_secret(stmt);
        };
        !krishiv_common::secret::is_sensitive_key(&key)
    })
}

/// Whether a non-`SET` statement carries a credential: a sensitive option key assigned a value
/// (`OPTIONS (password 'x')`, `password = 'x'`) or credentials in a URL (`jdbc:...://user:pw@host`).
/// A column merely named `password` is not one, so ordinary queries still persist.
fn assigns_secret(stmt: &str) -> bool {
    let stmt = &blank_comments(stmt);
    let stmt = stmt.as_str();
    let mut i = 0;
    while let Some(off) = stmt.get(i..).and_then(|r| r.find("://")) {
        let rest = &stmt[i + off + 3..];
        let authority = rest
            .split(['/', ' ', '\'', '"', ')', ';'])
            .next()
            .unwrap_or("");
        if authority.contains('@') {
            return true;
        }
        i += off + 3;
    }
    let mut start = None;
    for (pos, c) in stmt
        .char_indices()
        .chain(std::iter::once((stmt.len(), ' ')))
    {
        let word = c.is_alphanumeric() || matches!(c, '.' | '_' | '-');
        match (start, word) {
            (None, true) => start = Some(pos),
            (Some(from), false) => {
                start = None;
                let next = stmt[pos..].trim_start().bytes().next();
                if matches!(next, Some(b'=' | b'\'' | b'"'))
                    && krishiv_common::secret::is_sensitive_key(&stmt[from..pos])
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// `s` without leading whitespace and comments (`-- ...` to the end of the
/// line, `/* ... */`), so a comment before `SET` does not hide it.
fn skip_blank_and_comments(mut s: &str) -> &str {
    loop {
        s = s.trim_start();
        if let Some(rest) = s.strip_prefix("--") {
            s = rest.find('\n').map_or("", |i| &rest[i + 1..]);
        } else if let Some(rest) = s.strip_prefix("/*") {
            s = rest.find("*/").map_or("", |i| &rest[i + 2..]);
        } else {
            return s;
        }
    }
}

/// The option key a `SET` statement assigns (`SET key=...`,
/// `SET key = value`, `` SET `key` = value ``; `SET -v` has none); `None`
/// for other statements. Backticks around key parts are removed.
fn set_key(stmt: &str) -> Option<String> {
    let trimmed = skip_blank_and_comments(stmt);
    let head = trimmed.get(..3)?;
    if !head.eq_ignore_ascii_case("SET") {
        return None;
    }
    let rest = skip_blank_and_comments(trimmed.get(3..).unwrap_or(""));
    if rest.is_empty() || rest.starts_with('-') {
        return None;
    }
    let mut key = String::new();
    let mut quoted = false;
    for c in rest.chars() {
        if c == '`' {
            quoted = !quoted;
        } else if quoted || c.is_alphanumeric() || c == '.' || c == '_' || c == '-' {
            key.push(c);
        } else {
            break;
        }
    }
    if key.is_empty() { None } else { Some(key) }
}

/// `s` with comments (outside quotes) replaced by a space, so words inside a comment are never read as code.
fn blank_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut quote: Option<char> = None;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) => {
                out.push(c);
                if c == '\\' && q != '`' {
                    out.extend(chars.next());
                } else if c == q {
                    quote = None;
                }
            }
            None if c == '-' && chars.peek() == Some(&'-') => {
                for n in chars.by_ref() {
                    if n == '\n' {
                        break;
                    }
                }
                out.push(' ');
            }
            None if c == '/' && chars.peek() == Some(&'*') => {
                chars.next();
                let mut prev = ' ';
                for n in chars.by_ref() {
                    if prev == '*' && n == '/' {
                        break;
                    }
                    prev = n;
                }
                out.push(' ');
            }
            None => {
                if matches!(c, '\'' | '"' | '`') {
                    quote = Some(c);
                }
                out.push(c);
            }
        }
    }
    out
}

/// `line` split at top-level `;` (outside quotes and comments), like the
/// statement splitter in `common.rs` but without line offsets.
fn split_top_level(line: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut quote: Option<char> = None;
    let mut start = 0;
    let mut chars = line.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match quote {
            Some(q) => {
                if c == '\\' && q != '`' {
                    chars.next();
                } else if c == q {
                    quote = None;
                }
            }
            None => match c {
                '\'' | '"' | '`' => quote = Some(c),
                '-' if chars.peek().is_some_and(|(_, n)| *n == '-') => {
                    // A line comment ends at the line, not at the buffer.
                    for (_, n) in chars.by_ref() {
                        if n == '\n' {
                            break;
                        }
                    }
                }
                '/' if chars.peek().is_some_and(|(_, n)| *n == '*') => {
                    // A block comment hides `;` too; an unterminated one runs to the end.
                    chars.next();
                    let mut prev = ' ';
                    for (_, n) in chars.by_ref() {
                        if prev == '*' && n == '/' {
                            break;
                        }
                        prev = n;
                    }
                }
                ';' => {
                    parts.push(&line[start..i]);
                    start = i + 1;
                }
                _ => {}
            },
        }
    }
    parts.push(&line[start..]);
    parts
}

/// Entries of the history file (oldest first); missing or unreadable
/// files read as empty, so a broken history never breaks the REPL.
pub(crate) fn load(path: &std::path::Path) -> Vec<String> {
    let file = std::fs::File::open(path);
    let Ok(file) = file else {
        return Vec::new();
    };
    // A broken line ends history here rather than breaking the REPL;
    // `map_while` (not `filter_map`) is what clippy requires of `io::Lines`.
    io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .collect()
}

/// Appends `line` to the history file unless it sets a sensitive option,
/// then drops the oldest entries past [`MAX_ENTRIES`]. Parent directories
/// are created. `line` is one whole statement as typed (the REPL calls this
/// once the statement is complete, so a secret on a continuation line is
/// seen together with its `SET`); embedded newlines become spaces, so the
/// file stays one entry per line. On Unix the directory is created `0700`
/// and the file `0600`.
///
/// # Errors
/// The I/O error that kept the entry from the file.
pub(crate) fn append(path: &std::path::Path, line: &str) -> io::Result<()> {
    if !should_persist(line) {
        return Ok(());
    }
    let flat: String = line.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        create_private_dir(parent)?;
    }
    let mut file = open_private(path, true)?;
    writeln!(file, "{flat}")?;
    file.flush()?;
    drop(file);
    // Rarely: over the cap, rewrite the newest entries (a unique temp name,
    // so two REPLs never share one).
    let mut entries = load(path);
    if entries.len() > MAX_ENTRIES {
        entries.drain(..entries.len() - MAX_ENTRIES);
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(format!(".{}.tmp", std::process::id()));
        let tmp = PathBuf::from(tmp);
        {
            let mut f = open_private(&tmp, false)?;
            for e in &entries {
                writeln!(f, "{e}")?;
            }
            f.flush()?;
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    }
    Ok(())
}

/// `dir` and its missing parents; `0700` on Unix for the ones created.
fn create_private_dir(dir: &std::path::Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

/// Opens `path` for writing (`append`, or truncating), created `0600` on Unix.
fn open_private(path: &std::path::Path, append: bool) -> io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true);
    if append {
        options.append(true);
    } else {
        options.write(true).truncate(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    options.open(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_lines_persist_but_sensitive_sets_do_not() {
        assert!(should_persist("select 1"));
        assert!(should_persist("SET spark.sql.ansi.enabled=false"));
        assert!(should_persist("SET"));
        assert!(should_persist("SET -v"));
        assert!(!should_persist("SET password=hunter2"));
        assert!(!should_persist("set spark.ssl.keyStorePassword = hunter2"));
        assert!(!should_persist("  SET fs.azure.account.key.x = hunter2"));
        assert!(!should_persist("SET token='abc'"));
        // One sensitive statement vetoes a multi-statement line.
        assert!(!should_persist("select 1; SET password=hunter2"));
        assert!(!should_persist("/* ; */ SET password=hunter2;"));
        assert!(!should_persist("/* a */ select 1; /* b; */ SET token='x'"));
        assert!(!should_persist(
            "CREATE TEMPORARY VIEW v USING jdbc OPTIONS (url 'jdbc:x', password 'hunter2')"
        ));
        assert!(!should_persist(
            "CREATE TEMPORARY VIEW v USING jdbc OPTIONS (password='hunter2')"
        ));
        assert!(!should_persist(
            "SELECT * FROM t USING 'jdbc:pg://user:pw@host/db'"
        ));
        assert!(should_persist("SELECT password FROM users WHERE id = 1"));
        // A secret-looking *value* under an ordinary key persists: the
        // rule keys on the option name, as the engine's classification does.
        assert!(should_persist("select 'hunter2' as s"));
    }

    #[test]
    fn secret_canary_is_never_written() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history");
        let canary = "krishiv-history-canary-9f3c1a";
        append(&path, "select 1").expect("append");
        append(&path, &format!("SET my.secret.token = '{canary}'")).expect("filtered");
        append(&path, "select 2").expect("append");
        let entries = load(&path);
        assert_eq!(entries, vec!["select 1".to_owned(), "select 2".to_owned()]);
        let raw = std::fs::read_to_string(&path).expect("read");
        assert!(!raw.contains(canary), "{raw:?}");
    }

    #[test]
    fn entries_are_capped_and_multiline_is_flat() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history");
        append(&path, "select\n1").expect("append");
        assert_eq!(load(&path), vec!["select 1".to_owned()]);
        append(&path, "   ").expect("blank");
        assert_eq!(load(&path).len(), 1);
        // Missing files read as empty.
        assert!(load(&dir.path().join("nope")).is_empty());
    }

    #[test]
    fn comments_backticks_and_line_comments_do_not_hide_a_secret() {
        assert!(!should_persist("/* c */ SET password=x"));
        assert!(!should_persist("-- note\nSET password=x"));
        assert!(!should_persist("SET `password` = x"));
        assert!(!should_persist("SET /* c */ token = 'a'"));
        // A line comment ends at its line: the statement after it counts.
        assert!(!should_persist("select 1 -- c\n; SET password = x"));
        assert!(should_persist("select 1 -- SET password = x"));
        assert!(should_persist("SET `spark.sql.shuffle.partitions` = 4"));
    }

    #[test]
    fn a_whole_multi_line_statement_is_judged_together() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history");
        append(&path, "SET my.token =\n 'abc';").expect("filtered");
        append(&path, "SET\npassword=x;").expect("filtered");
        assert!(load(&path).is_empty());
        append(&path, "select\n1;").expect("append");
        assert_eq!(load(&path), vec!["select 1;".to_owned()]);
    }

    #[cfg(unix)]
    #[test]
    fn the_history_file_and_directory_are_private() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".krishiv").join("history");
        append(&path, "select 1").expect("append");
        let file = std::fs::metadata(&path).expect("file").permissions().mode();
        let parent = std::fs::metadata(path.parent().expect("parent"))
            .expect("dir")
            .permissions()
            .mode();
        assert_eq!(file & 0o077, 0, "{file:o}");
        assert_eq!(parent & 0o077, 0, "{parent:o}");
    }

    #[test]
    fn the_oldest_entries_are_dropped_past_the_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history");
        let body: String = (0..MAX_ENTRIES + 5)
            .map(|i| format!("select {i}\n"))
            .collect();
        std::fs::write(&path, body).expect("seed");
        append(&path, "select last").expect("append");
        let entries = load(&path);
        assert_eq!(entries.len(), MAX_ENTRIES);
        assert_eq!(entries.last().map(String::as_str), Some("select last"));
        assert_eq!(entries[0], "select 6");
    }

    #[test]
    fn history_path_is_under_home() {
        let path = history_path().expect("home");
        assert!(path.ends_with(".krishiv/history"), "{path:?}");
    }
}

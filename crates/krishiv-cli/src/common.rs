// SPDX-License-Identifier: Apache-2.0
//! Shared input handling and session setup for the `sql` and
//! `debug-bundle` subcommands.

use std::io::Read;
use std::path::PathBuf;

use krishiv::Session;

/// Parses `KEY=VALUE` for `--conf` and `--table`.
pub(crate) fn key_value(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((k, v)) if !k.trim().is_empty() => Ok((k.trim().to_owned(), v.to_owned())),
        _ => Err(format!("expected KEY=VALUE, got `{s}`")),
    }
}

/// Reads SQL from `-e`, `-f` or stdin, as `krishiv sql` does.
pub(crate) fn read_input(
    execute: Option<&String>,
    file: Option<&PathBuf>,
) -> Result<String, String> {
    if let Some(sql) = execute {
        return Ok(sql.clone());
    }
    match file {
        Some(p) if p.as_os_str() != "-" => {
            std::fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display()))
        }
        _ => {
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| format!("cannot read stdin: {e}"))?;
            Ok(s)
        }
    }
}

/// Builds a session from `--conf` entries and registers `--table`
/// parquet paths as temp views.
pub(crate) fn build_session(
    conf: &[(String, String)],
    tables: &[(String, String)],
) -> Result<Session, String> {
    let mut builder = Session::builder();
    for (k, v) in conf {
        builder = builder.config(k, v);
    }
    let session = builder.build().map_err(|e| e.to_string())?;
    for (name, path) in tables {
        let df = session.read().parquet([path.as_str()]);
        futures::executor::block_on(df.create_or_replace_temp_view(name))
            .map_err(|e| format!("--table {name}={path}: {e}"))?;
    }
    Ok(session)
}

/// Splits `input` at `;` outside quotes and comments; blank statements are
/// dropped. Each statement comes with the number of input lines before it.
pub(crate) fn split_statements(input: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut done_lines = 0;
    let mut cur = String::new();
    let mut chars = input.chars().peekable();
    let mut quote: Option<char> = None;
    let mut code = false;
    while let Some(c) = chars.next() {
        cur.push(c);
        match quote {
            Some(q) => {
                if c == '\\' && q != '`' {
                    if let Some(n) = chars.next() {
                        cur.push(n);
                    }
                } else if c == q {
                    quote = None;
                }
            }
            None => match c {
                '\'' | '"' | '`' => {
                    code = true;
                    quote = Some(c);
                }
                '-' if chars.peek() == Some(&'-') => {
                    for n in chars.by_ref() {
                        cur.push(n);
                        if n == '\n' {
                            break;
                        }
                    }
                }
                '/' if chars.peek() == Some(&'*') => {
                    // Bracketed comments nest, as in Spark: the comment
                    // ends at the `*/` that closes the outermost one.
                    if let Some(open) = chars.next() {
                        cur.push(open);
                    }
                    let (mut depth, mut prev) = (1_usize, '\0');
                    for n in chars.by_ref() {
                        cur.push(n);
                        if prev == '*' && n == '/' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                            prev = '\0';
                        } else if prev == '/' && n == '*' {
                            depth += 1;
                            prev = '\0';
                        } else {
                            prev = n;
                        }
                    }
                }
                ';' => {
                    cur.pop();
                    push_stmt(&mut out, &mut done_lines, &mut cur, code);
                    code = false;
                }
                c if !c.is_whitespace() => code = true,
                _ => {}
            },
        }
    }
    push_stmt(&mut out, &mut done_lines, &mut cur, code);
    out
}

fn push_stmt(out: &mut Vec<(usize, String)>, done_lines: &mut usize, cur: &mut String, code: bool) {
    if code {
        let leading = cur.len() - cur.trim_start().len();
        let skipped = cur[..leading].matches('\n').count();
        out.push((*done_lines + skipped, cur.trim().to_owned()));
    }
    *done_lines += cur.matches('\n').count();
    cur.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_outside_quotes_and_comments() {
        let s = split_statements("select 'a;b'; -- x;y\nselect 2; /* ; */ select `c;`;;");
        assert_eq!(s.len(), 3, "{s:?}");
        assert_eq!(s[0].1, "select 'a;b'");
        assert_eq!(s[1].0, 0);
        let t = split_statements("select 1;\n\n  select 2");
        assert_eq!(t[1], (2, "select 2".to_owned()));
        let c = split_statements("-- only\n; /* x */ ; select 1 -- tail\n;\n-- end");
        assert_eq!(c, vec![(1, "select 1 -- tail".to_owned())]);
        let esc = split_statements("select 'it\\'s;' ; select 2");
        assert_eq!(esc.len(), 2, "{esc:?}");
        // Nested block comments hide the semicolons inside them.
        let nested = split_statements("select 1 /* a /* b; */ c; */ ; select 2");
        assert_eq!(
            nested,
            vec![
                (0, "select 1 /* a /* b; */ c; */".to_owned()),
                (0, "select 2".to_owned())
            ]
        );
        // `/*/` does not close the comment it opens.
        let tight = split_statements("select 1 /*/ ; */ ; select 2");
        assert_eq!(tight.len(), 2, "{tight:?}");
    }

    #[test]
    fn key_value_parsing() {
        assert_eq!(key_value("a=b=c"), Ok(("a".into(), "b=c".into())));
        assert_eq!(key_value(" k = v"), Ok(("k".into(), " v".into())));
        assert!(key_value("novalue").is_err());
        assert!(key_value("=v").is_err());
    }
}

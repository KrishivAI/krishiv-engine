// SPDX-License-Identifier: Apache-2.0
//! The REPL's line editor (P2-T132), on a terminal only: `reedline` gives
//! cursor editing, history search and a Tab menu fed by
//! [`crate::complete::complete`]. A statement that has no closing `;` yet
//! keeps the editor open on a new line, so a pasted multi-line query is
//! edited as one buffer. Piped input never comes here (the plain loop in
//! [`crate::repl`] reads it).

use std::borrow::Cow;
use std::sync::{Arc, Mutex, PoisonError};

use reedline::{
    ColumnarMenu, Completer, CompletionResult, Emacs, FileBackedHistory, HistoryItem, KeyCode,
    KeyModifiers, MenuBuilder, Prompt, PromptEditMode, PromptHistorySearch, Reedline,
    ReedlineEvent, ReedlineMenu, Signal, Span, Suggestion, ValidationResult, Validator,
    default_emacs_keybindings,
};

use crate::complete::{CompletionContext, complete, completion_start};

/// What one read of the editor gave.
pub(crate) enum Read {
    /// A line, or the several lines of a pasted statement.
    Text(String),
    /// Ctrl-C: drop what was typed.
    Interrupt,
    /// Ctrl-D, or the terminal went away.
    Eof,
    /// The terminal could not be driven (the editor failed to start or read): not an end of input.
    Error(String),
}

/// A terminal line editor with the REPL's completion, history and prompts.
pub(crate) struct Editor {
    line: Reedline,
    /// What Tab completes from; replaced as the session's catalog changes.
    ctx: Arc<Mutex<CompletionContext>>,
}

struct SqlPrompt;

impl Prompt for SqlPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("krishiv")
    }
    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }
    fn render_prompt_indicator(&self, _mode: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed("> ")
    }
    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("     ...> ")
    }
    fn render_prompt_history_search_indicator(&self, search: PromptHistorySearch) -> Cow<'_, str> {
        Cow::Owned(format!("(search: {}) ", search.term))
    }
}

struct SqlCompleter(Arc<Mutex<CompletionContext>>);

impl Completer for SqlCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> CompletionResult {
        let start = completion_start(line, pos);
        let end = line.floor_char_boundary(pos.min(line.len()));
        let ctx = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let suggestions: Vec<Suggestion> = complete(line, pos, &ctx)
            .into_iter()
            .map(|value| Suggestion {
                value,
                description: None,
                style: None,
                extra: None,
                span: Span { start, end },
                append_whitespace: true,
                match_indices: None,
                display_override: None,
            })
            .collect();
        CompletionResult::fresh(suggestions)
    }
}

/// Holds the editor open until the buffer is a whole statement (or a dot
/// command, or empty).
struct StatementValidator;

impl Validator for StatementValidator {
    fn validate(&self, line: &str) -> ValidationResult {
        // Pasted dot-commands are whole lines; what follows them must be whole statements.
        let (_, sql) = crate::repl::leading_dot_lines(line);
        if sql.trim().is_empty()
            || (crate::repl::statement_complete(sql) && !crate::repl::has_unfinished_tail(sql))
        {
            ValidationResult::Complete
        } else {
            ValidationResult::Incomplete
        }
    }
}

impl Editor {
    /// An editor seeded with `history` (oldest first). `None` when the
    /// editor cannot start, so the caller falls back to the plain loop.
    pub(crate) fn new(ctx: CompletionContext, history: &[String]) -> Option<Self> {
        let mut keys = default_emacs_keybindings();
        keys.add_binding(
            KeyModifiers::NONE,
            KeyCode::Tab,
            ReedlineEvent::UntilFound(vec![
                ReedlineEvent::Menu("completion_menu".to_owned()),
                ReedlineEvent::MenuNext,
            ]),
        );
        let mut store = FileBackedHistory::new(history.len().max(1000)).ok()?;
        for entry in history {
            reedline::History::save(&mut store, HistoryItem::from_command_line(entry.clone()))
                .ok()?;
        }
        let ctx = Arc::new(Mutex::new(ctx));
        let line = Reedline::create()
            .with_history(Box::new(store))
            .with_completer(Box::new(SqlCompleter(Arc::clone(&ctx))))
            .with_validator(Box::new(StatementValidator))
            .with_menu(ReedlineMenu::EngineCompleter(Box::new(
                ColumnarMenu::default().with_name("completion_menu"),
            )))
            .with_edit_mode(Box::new(Emacs::new(keys)));
        Some(Self { line, ctx })
    }

    /// Completes from `ctx` from now on (after statements changed the catalog).
    pub(crate) fn refresh(&mut self, ctx: CompletionContext) {
        *self.ctx.lock().unwrap_or_else(PoisonError::into_inner) = ctx;
    }

    /// Reads until Enter on a whole statement.
    pub(crate) fn read(&mut self) -> Read {
        match self.line.read_line(&SqlPrompt) {
            Ok(Signal::Success(text)) => Read::Text(text),
            Ok(Signal::CtrlD) => Read::Eof,
            Err(e) => Read::Error(e.to_string()),
            Ok(_) => Read::Interrupt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_validator_waits_for_the_semicolon() {
        let v = StatementValidator;
        assert!(matches!(v.validate(""), ValidationResult::Complete));
        assert!(matches!(v.validate("  "), ValidationResult::Complete));
        assert!(matches!(
            v.validate("select 1"),
            ValidationResult::Incomplete
        ));
        assert!(matches!(
            v.validate("select ';'"),
            ValidationResult::Incomplete
        ));
        assert!(matches!(
            v.validate("select 1;"),
            ValidationResult::Complete
        ));
        assert!(matches!(
            v.validate(".timer on"),
            ValidationResult::Complete
        ));
    }

    #[test]
    fn the_completer_replaces_the_word_under_the_cursor() {
        let ctx = CompletionContext {
            tables: vec!["lineitem".to_owned()],
            ..CompletionContext::default()
        };
        let mut c = SqlCompleter(Arc::new(Mutex::new(ctx)));
        let line = "select * from line";
        let got = c.complete(line, line.len());
        let found = got.suggestions();
        assert!(found.iter().any(|s| s.value == "lineitem"));
        assert!(
            found
                .iter()
                .all(|s| s.span.start == 14 && s.span.end == line.len())
        );
    }

    #[test]
    fn the_validator_waits_for_an_unfinished_trailing_statement() {
        let v = StatementValidator;
        assert!(matches!(
            v.validate("SELECT 1;\nINSERT INTO dst SELECT * FROM src"),
            ValidationResult::Incomplete
        ));
        assert!(matches!(
            v.validate("SELECT 1;\nSELECT 2;"),
            ValidationResult::Complete
        ));
    }

    #[test]
    fn a_refreshed_context_completes_new_names() {
        let mut ed = Editor::new(CompletionContext::default(), &[]).unwrap();
        ed.refresh(CompletionContext {
            tables: vec!["fresh_view".to_owned()],
            ..CompletionContext::default()
        });
        let mut c = SqlCompleter(Arc::clone(&ed.ctx));
        let line = "select * from fresh_";
        let found = c.complete(line, line.len());
        assert!(found.suggestions().iter().any(|s| s.value == "fresh_view"));
    }

    #[test]
    fn a_set_key_is_replaced_whole() {
        let ctx = CompletionContext::default();
        let mut c = SqlCompleter(Arc::new(Mutex::new(ctx)));
        let line = "set krishiv.sys";
        let got = c.complete(line, line.len());
        assert!(got.suggestions().iter().all(|s| s.span.start == 4));
        assert!(!got.suggestions().is_empty());
    }

    #[test]
    fn the_prompts_are_the_repls() {
        let p = SqlPrompt;
        assert_eq!(p.render_prompt_left(), "krishiv");
        assert_eq!(p.render_prompt_right(), "");
        assert_eq!(p.render_prompt_indicator(PromptEditMode::Default), "> ");
        assert_eq!(p.render_prompt_multiline_indicator(), "     ...> ");
        let search = PromptHistorySearch::new(
            reedline::PromptHistorySearchStatus::Passing,
            "sel".to_owned(),
        );
        assert_eq!(
            p.render_prompt_history_search_indicator(search),
            "(search: sel) "
        );
    }

    #[test]
    fn an_editor_starts_with_its_history_without_a_terminal() {
        let past = vec!["select 1;".to_owned(), "select 2;".to_owned()];
        assert!(Editor::new(CompletionContext::default(), &past).is_some());
        assert!(Editor::new(CompletionContext::default(), &[]).is_some());
    }
}

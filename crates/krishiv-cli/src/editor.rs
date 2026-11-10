// SPDX-License-Identifier: Apache-2.0
//! The REPL's line editor (P2-T132), on a terminal only: `reedline` gives
//! cursor editing, history search and a Tab menu fed by
//! [`crate::complete::complete`]. A statement that has no closing `;` yet
//! keeps the editor open on a new line, so a pasted multi-line query is
//! edited as one buffer. Piped input never comes here (the plain loop in
//! [`crate::repl`] reads it).

use std::borrow::Cow;

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
}

/// A terminal line editor with the REPL's completion, history and prompts.
pub(crate) struct Editor {
    line: Reedline,
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

struct SqlCompleter(CompletionContext);

impl Completer for SqlCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> CompletionResult {
        let start = completion_start(line, pos);
        let end = line.floor_char_boundary(pos.min(line.len()));
        let suggestions: Vec<Suggestion> = complete(line, pos, &self.0)
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
        if line.trim().is_empty() || crate::repl::statement_complete(line) {
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
        let line = Reedline::create()
            .with_history(Box::new(store))
            .with_completer(Box::new(SqlCompleter(ctx)))
            .with_validator(Box::new(StatementValidator))
            .with_menu(ReedlineMenu::EngineCompleter(Box::new(
                ColumnarMenu::default().with_name("completion_menu"),
            )))
            .with_edit_mode(Box::new(Emacs::new(keys)));
        Some(Self { line })
    }

    /// Reads until Enter on a whole statement.
    pub(crate) fn read(&mut self) -> Read {
        match self.line.read_line(&SqlPrompt) {
            Ok(Signal::Success(text)) => Read::Text(text),
            Ok(Signal::CtrlD) | Err(_) => Read::Eof,
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
        let mut c = SqlCompleter(ctx);
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
    fn a_set_key_is_replaced_whole() {
        let ctx = CompletionContext::default();
        let mut c = SqlCompleter(ctx);
        let line = "set krishiv.sys";
        let got = c.complete(line, line.len());
        assert!(got.suggestions().iter().all(|s| s.span.start == 4));
        assert!(!got.suggestions().is_empty());
    }
}

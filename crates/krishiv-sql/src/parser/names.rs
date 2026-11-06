// SPDX-License-Identifier: Apache-2.0
//! Identifier literals (`IDENTIFIER('name')`) and the multipart names that splice them (Spark 4.2
//! `strictIdentifier` `#identifierLiteral`, `DataTypeAstBuilder.visitMultipartIdentifier`).

use super::{PR, Parser, parse_multipart_identifier, quote_ident};
use crate::ast::{Ident, ObjectName, Span};
use crate::error::{self, ParseError};
use crate::lexer::{Sym, unescape_string};

impl Parser<'_> {
    /// The tokens of the identifier literal at token `n` (`IDENTIFIER`, `(`, a run of string
    /// literals, `)`), if one starts there. Not with only the legacy function syntax on.
    pub(super) fn identifier_literal_len(&self, n: usize) -> Option<usize> {
        if self.opts.legacy_identifier_clause
            || !self.kw_at(n, "IDENTIFIER")
            || !self.sym_at(n + 1, Sym::LParen)
        {
            return None;
        }
        // Spark's `stringLit` is a run of single strings, read as one string.
        let mut k = n + 2;
        while self.single_string_at(k) {
            k += 1;
        }
        (k > n + 2 && self.sym_at(k, Sym::RParen)).then_some(k + 1 - n)
    }

    /// Whether `IDENTIFIER('name')` starts at token `n`.
    pub(super) fn identifier_literal_at(&self, n: usize) -> bool {
        self.identifier_literal_len(n).is_some()
    }

    /// The identifier literal at token `n` as one quoted identifier: the run's strings joined,
    /// spanning `IDENTIFIER` to `)`.
    pub(super) fn identifier_literal(&self, n: usize) -> Option<Ident> {
        let len = self.identifier_literal_len(n)?;
        let (first, last) = (self.tok(n)?, self.tok(n + len - 1)?);
        let mut value = String::new();
        for k in n + 2..n + len - 1 {
            let piece = self.tok(k)?.span.text(self.src);
            value.push_str(&unescape_string(
                piece,
                !self.opts.escaped_string_literals,
                false,
            ));
        }
        Some(Ident {
            value,
            quoted: true,
            span: first.span.to(last.span),
        })
    }

    /// Spark parses the string of an identifier literal as a name when it builds the tree: it
    /// must be a name (not empty), of `parts` parts at most (`1` for a position taking one
    /// identifier). The first problem is deferred; the name is returned when there is none.
    pub(super) fn check_identifier_literal(
        &mut self,
        text: &str,
        span: Span,
        parts: usize,
    ) -> Option<ObjectName> {
        match parse_multipart_identifier(text, self.opts) {
            Err(e) => self.defer(ParseError { span, ..e }),
            Ok(name) if name.parts.len() > parts => {
                let identifier: Vec<String> =
                    name.parts.iter().map(|p| quote_ident(&p.value)).collect();
                self.defer(error::identifier_too_many_name_parts(
                    identifier.join("."),
                    parts,
                    span,
                ));
            }
            Ok(name) => return Some(name),
        }
        None
    }

    /// The parts one element of a multipart name contributes: an identifier, or every part of
    /// an identifier literal (`IDENTIFIER('cat.s').t` is `cat.s.t`, as Spark's
    /// `visitMultipartIdentifier` expands it). A literal that is not a name defers its error and
    /// stands as one part.
    pub(super) fn name_parts(&mut self, strong: bool) -> PR<Vec<Ident>> {
        let (Some(len), Some(id)) = (self.identifier_literal_len(0), self.identifier_literal(0))
        else {
            return Ok(vec![self.name_part(strong)?]);
        };
        self.pos += len;
        let Some(name) = self.check_identifier_literal(&id.value, id.span, usize::MAX) else {
            return Ok(vec![id]);
        };
        Ok(name
            .parts
            .into_iter()
            .map(|p| Ident {
                value: p.value,
                quoted: true,
                span: id.span,
            })
            .collect())
    }

    /// A session variable name in `DECLARE` or `DROP TEMPORARY VARIABLE`, which Spark 4.2's
    /// grammar reads as an `identifierReference` (`createVariable`, `dropVariable`):
    /// `IDENTIFIER('name')` names the variable even with only the legacy function syntax on,
    /// where the clause takes an expression (the session evaluates any other expression before
    /// parsing).
    pub(super) fn variable_reference(&mut self) -> PR<ObjectName> {
        let legacy = self.opts.legacy_identifier_clause;
        self.opts.legacy_identifier_clause = false;
        let name = self.variable_name();
        self.opts.legacy_identifier_clause = legacy;
        name
    }
}

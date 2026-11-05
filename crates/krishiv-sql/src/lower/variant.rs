// SPDX-License-Identifier: Apache-2.0
//! Lowering of the variant extraction `v:path` (Spark 4.2's
//! `AstBuilder.visitSemiStructuredExtract`).
//!
//! The path is the source text of the parts after the colon (backquotes
//! already gone, as Spark's lexer post-processor drops them): `$.` and that
//! text, or `$` and it when it starts with a bracket. It must parse as
//! `get_json_object`'s `JsonPathParser` (else `PARSE_SYNTAX_ERROR`); the
//! result is `variant_get(v, path)` (the analyzer's
//! `ExtractSemiStructuredFields`), aliased to the last named key when there
//! is one. `a:b.c` is `$.b.c`, so a backquoted key holding a dot is two
//! keys, as in Spark.
//!
//! Two deliberate divergences, where Spark 4.2.0 itself fails with
//! `INTERNAL_ERROR` (checked on the oracle): a path `JsonPathParser` refuses
//! (`v:['a''b']`) is `PARSE_SYNTAX_ERROR` naming the path (Spark raises the
//! class without its parameters), and `x:a` over a column that is not a
//! variant is `variant_get`'s `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`
//! (Spark names `COLUMN_IS_NOT_VARIANT_TYPE`, a class its error file lacks).

use std::sync::Arc;

use krishiv_plan::{Expr as IrExpr, ExprNode, Literal as IrLiteral};

use super::{Lowerer, R};
use crate::ast::{Expr, Span, VariantPath, VariantPathPartKind as K};
use crate::error::syntax_error;

/// The Spark path text of `path`.
fn path_text(path: &VariantPath) -> String {
    let mut out = String::from("$");
    for part in &path.parts {
        match &part.kind {
            K::Field(id) => {
                out.push('.');
                out.push_str(&id.value);
            }
            K::Key(k) => {
                out.push_str("['");
                out.push_str(k);
                out.push_str("']");
            }
            K::Index(n) => {
                out.push('[');
                out.push_str(n);
                out.push(']');
            }
            K::Name(id) => {
                out.push('[');
                out.push_str(&id.value);
                out.push(']');
            }
        }
    }
    out
}

/// `JsonPathParser.parse(path)`: `Some(last named key)` when `path`
/// parses, `None` when it does not.
fn json_path_alias(path: &str) -> Option<Option<String>> {
    let mut rest = path.strip_prefix('$')?;
    let mut last = None;
    while !rest.is_empty() {
        if rest.starts_with(".*") || rest.starts_with("['*']") {
            rest = &rest[if rest.starts_with(".*") { 2 } else { 5 }..];
        } else if let Some(r) = rest.strip_prefix('.') {
            let end = r.find(['.', '[']).unwrap_or(r.len());
            if end == 0 {
                return None;
            }
            last = Some(r[..end].to_owned());
            rest = &r[end..];
        } else if let Some(r) = rest.strip_prefix("['") {
            let end = r.find('\'')?;
            if end == 0 {
                return None;
            }
            last = Some(r[..end].to_owned());
            rest = r[end..].strip_prefix("']")?;
        } else {
            let r = rest.strip_prefix('[')?;
            let end = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
            let digits = &r[..end];
            if digits.is_empty() {
                rest = r.strip_prefix("*]")?;
            } else {
                digits.parse::<i64>().ok()?;
                rest = r[end..].strip_prefix(']')?;
            }
        }
    }
    Some(last)
}

impl Lowerer {
    /// `base:path`.
    pub(super) fn variant_get(&mut self, base: &Expr, path: &VariantPath, sp: Span) -> R<IrExpr> {
        let text = path_text(path);
        let Some(alias) = json_path_alias(&text) else {
            return Err(syntax_error(format!("'{text}'"), String::new(), path.span));
        };
        let base = self.arg(base)?;
        let path = Arc::new(self.ex(ExprNode::Literal(IrLiteral::String(text)), sp));
        let call = self.call("variant_get", vec![base, path], sp);
        Ok(match alias {
            Some(name) => self.ex(
                ExprNode::Alias {
                    expr: Arc::new(call),
                    name,
                },
                sp,
            ),
            None => call,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::json_path_alias;

    #[test]
    fn json_paths_parse_as_spark() {
        let alias = |p| json_path_alias(p).map(Option::unwrap_or_default);
        assert_eq!(alias("$.price").as_deref(), Some("price"));
        assert_eq!(alias("$.item[0].model").as_deref(), Some("model"));
        assert_eq!(alias("$['field.name']").as_deref(), Some("field.name"));
        assert_eq!(alias("$.field.name").as_deref(), Some("name"));
        assert_eq!(alias("$[0]").as_deref(), Some(""));
        assert_eq!(alias("$.scores[0]").as_deref(), Some("scores"));
        for bad in ["$['']", "$[x]", "$['a'b']", "$.", "$[\"a\"]", "$['a''b']"] {
            assert_eq!(json_path_alias(bad), None, "{bad}");
        }
    }
}

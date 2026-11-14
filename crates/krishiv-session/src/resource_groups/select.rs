// SPDX-License-Identifier: Apache-2.0
//! Selectors: the first one whose every set field matches routes the query; no match goes to
//! the default leaf.

use super::config::{Config, SelectorCfg};
use crate::admission::QueryRequest;

/// The dotted id of the leaf group `request` belongs to.
pub(super) fn select<'a>(config: &'a Config, request: &QueryRequest) -> &'a str {
    config
        .selectors
        .iter()
        .find(|s| matches(s, request))
        .map_or(config.default_group.as_str(), |s| s.group.as_str())
}

fn matches(s: &SelectorCfg, r: &QueryRequest) -> bool {
    s.user
        .as_ref()
        .is_none_or(|re| re.is_match(&r.principal.name))
        && s.user_group
            .as_ref()
            .is_none_or(|re| r.principal.groups.iter().any(|g| re.is_match(g)))
        && s.source
            .as_ref()
            .is_none_or(|re| r.source.as_deref().is_some_and(|x| re.is_match(x)))
        && s.client_tags.iter().all(|t| has_tag(&r.tags, t))
        && s.query_type.is_none_or(|c| r.statement_class == Some(c))
}

/// A selector tag `s` matches a pair `(k, v)` when `s == k` or `s == "k=v"`. A Connect request tag
/// `job=42` is the pair `("job=42", "")` (tags are opaque and may contain `=`), so it matches the
/// selector `job=42` through the first rule, and the session pair `("job", "42")` matches it
/// through the second: the over-match is accepted (design P3-T064 §4.3).
fn has_tag(tags: &[(String, String)], want: &str) -> bool {
    tags.iter().any(|(k, v)| {
        want == k
            || want
                .split_once('=')
                .is_some_and(|(wk, wv)| wk == k && wv == v)
    })
}

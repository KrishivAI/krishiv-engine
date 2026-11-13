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

/// A selector tag `k` matches any pair with key `k`; `k=v` matches the exact pair.
fn has_tag(tags: &[(String, String)], want: &str) -> bool {
    match want.split_once('=') {
        Some((k, v)) => tags.iter().any(|(tk, tv)| tk == k && tv == v),
        None => tags.iter().any(|(tk, _)| tk == want),
    }
}

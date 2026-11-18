// SPDX-License-Identifier: Apache-2.0
//! Registry keys named like a credential (design P3-T066 §5.5).
use super::super::*;

/// Design P3-T066 §5.5: a key that names a credential holds a secret reference (ADR 0038), never
/// the secret, so `GetConfig` can show its value. A key whose last segment ends in one of these
/// words is refused unless it is listed here with the reason it holds no secret.
#[test]
fn a_key_named_like_a_credential_is_on_the_allow_list() {
    const WORDS: [&str; 5] = ["password", "secret", "token", "credential", "key"];
    const ALLOWED: &[(&str, &str)] = &[];
    let mut found = Vec::new();
    for e in REGISTRY {
        let last = e
            .key
            .rsplit('.')
            .next()
            .unwrap_or(e.key)
            .to_ascii_lowercase();
        if WORDS.iter().any(|w| last.ends_with(w)) && !ALLOWED.iter().any(|(k, _)| *k == e.key) {
            found.push(e.key);
        }
    }
    assert!(
        found.is_empty(),
        "keys named like a credential that are not on the allow list: {found:?}"
    );
}

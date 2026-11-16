// SPDX-License-Identifier: Apache-2.0
//! Keyset pagination of the admin lists (design §6.2): a page token is a cursor `(sort key, id)`
//! bound to the request's filters, so rows inserted between pages are neither skipped nor
//! repeated, and a token cannot be reused with other filters.

use super::{AdminError, AdminResult};

const DEFAULT_SIZE: usize = 100;
const MAX_SIZE: usize = 1_000;

/// Where a page starts and how long it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    /// The most rows on the page.
    pub size: usize,
    /// Rows strictly after this `(sort key, id)` come next.
    pub after: Option<(i64, String)>,
}

/// FNV-1a 64 of the filter text: the token's binding to its request.
fn fingerprint(filter: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in filter.bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

impl Page {
    /// The page a request asks for. `size` 0 is the default (100), above 1,000 it is clamped,
    /// negative is refused; a `token` from another filter, or not one of ours, is refused.
    ///
    /// # Errors
    ///
    /// `KRISHIV_ADMIN_INVALID_REQUEST` for a negative size or a bad token.
    pub fn parse(size: i32, token: &str, filter: &str) -> AdminResult<Self> {
        let size = match usize::try_from(size) {
            Ok(0) => DEFAULT_SIZE,
            Ok(n) => n.min(MAX_SIZE),
            Err(_) => return Err(AdminError::invalid_argument("page_size is negative")),
        };
        if token.is_empty() {
            return Ok(Self { size, after: None });
        }
        let bad = || AdminError::invalid_argument("the page token is not valid for this request");
        let text = String::from_utf8(unhex(token).ok_or_else(bad)?).map_err(|_| bad())?;
        let mut parts = text.splitn(3, '|');
        let (Some(bound), Some(key), Some(id)) = (parts.next(), parts.next(), parts.next()) else {
            return Err(bad());
        };
        if bound != format!("{:016x}", fingerprint(filter)) {
            return Err(bad());
        }
        let key: i64 = key.parse().map_err(|_| bad())?;
        Ok(Self {
            size,
            after: Some((key, id.to_owned())),
        })
    }

    /// The token for the page after a page ending at `last`, bound to `filter`.
    #[must_use]
    pub fn token(last: (i64, &str), filter: &str) -> String {
        hex(format!("{:016x}|{}|{}", fingerprint(filter), last.0, last.1).as_bytes())
    }

    /// The page of `rows` (already sorted by `(key, id)`), and the token of the next one.
    pub fn apply<T>(
        &self,
        rows: Vec<T>,
        key: impl Fn(&T) -> (i64, String),
        filter: &str,
    ) -> (Vec<T>, String) {
        let mut rest: Vec<T> = rows
            .into_iter()
            .filter(|r| {
                self.after.as_ref().is_none_or(|(k, id)| {
                    let (rk, rid) = key(r);
                    (rk, rid.as_str()) > (*k, id.as_str())
                })
            })
            .collect();
        let more = rest.len() > self.size;
        rest.truncate(self.size);
        let next = if more {
            rest.last()
                .map(|r| {
                    let (k, id) = key(r);
                    Self::token((k, &id), filter)
                })
                .unwrap_or_default()
        } else {
            String::new()
        };
        (rest, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: i64) -> Vec<(i64, String)> {
        (1..=n).map(|i| (i * 10, format!("id{i}"))).collect()
    }

    #[test]
    fn pages_walk_the_rows_once_and_inserts_do_not_shift_them() {
        let page = Page::parse(2, "", "f").unwrap_or_else(|e| panic!("{}", e.message));
        let (first, token) = page.apply(rows(5), Clone::clone, "f");
        assert_eq!(first.len(), 2);
        assert!(!token.is_empty());
        // A row inserted before the cursor between the pages changes nothing for the next one.
        let mut now = rows(5);
        now.insert(0, (5, "early".to_owned()));
        let next = Page::parse(2, &token, "f").unwrap_or_else(|e| panic!("{}", e.message));
        let (second, token) = next.apply(now.clone(), Clone::clone, "f");
        assert_eq!(
            second.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(),
            ["id3", "id4"]
        );
        let last = Page::parse(2, &token, "f").unwrap_or_else(|e| panic!("{}", e.message));
        let (third, token) = last.apply(now, Clone::clone, "f");
        assert_eq!(third.len(), 1);
        assert!(token.is_empty(), "the last page has no token");
    }

    #[test]
    fn the_size_is_defaulted_clamped_and_checked() {
        assert_eq!(Page::parse(0, "", "f").map(|p| p.size), Ok(100));
        assert_eq!(Page::parse(5_000, "", "f").map(|p| p.size), Ok(1_000));
        assert!(Page::parse(-1, "", "f").is_err());
    }

    #[test]
    fn a_token_is_bound_to_its_filters_and_garbage_is_refused() {
        let token = Page::token((10, "a"), "state=running");
        assert!(Page::parse(10, &token, "state=running").is_ok());
        assert!(Page::parse(10, &token, "state=failed").is_err());
        for bad in ["zz", "abc", "0102"] {
            assert!(Page::parse(10, bad, "f").is_err(), "{bad}");
        }
    }
}

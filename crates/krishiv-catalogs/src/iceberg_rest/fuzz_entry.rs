// SPDX-License-Identifier: Apache-2.0
//! The Iceberg REST response parsers behind one function, for the `iceberg_rest_response` fuzz target. A server
//! is input, not a trusted party (design `P3-lakehouse.md` §5.6): every parser must return a value or an error
//! for any bytes, never panic, recurse past the depth limit or allocate without bound. Not a public surface.

use super::types::{self, CatalogConfig, LoadedTable};

/// Runs one fuzz input: `which`, then a depth byte (`1 + b % 128`, so every limit the client allows is fuzzed;
/// seeds use 0x3f for the default 64), then the body. `None` when the input is too short.
pub fn parse_input(data: &[u8]) -> Option<bool> {
    let [which, depth, body @ ..] = data else {
        return None;
    };
    Some(parse_response(*which, body, 1 + usize::from(*depth) % 128))
}

/// Runs the parser picked by `which` (modulo the parser count) on `body`, nested at most `max_depth` deep.
/// Returns whether the body parsed. Parsed values are walked so a lazily wrong shape cannot hide.
pub fn parse_response(which: u8, body: &[u8], max_depth: usize) -> bool {
    match which % 8 {
        0 => types::parse_limited(body, max_depth).is_ok(),
        1 => types::parse_limited(body, max_depth)
            .and_then(CatalogConfig::from_json)
            .map(|c| {
                let _ = c.get("warehouse");
                let _ = c.supports("GET /v1/{prefix}/namespaces");
            })
            .is_ok(),
        2 => types::parse_limited(body, max_depth)
            .and_then(types::namespaces_page)
            .is_ok(),
        3 => types::parse_limited(body, max_depth)
            .and_then(types::tables_page)
            .is_ok(),
        4 => types::parse_limited(body, max_depth)
            .and_then(LoadedTable::from_json)
            .map(|t| {
                let _ = (
                    t.location.len(),
                    t.format_version,
                    t.storage_credentials.len(),
                );
            })
            .is_ok(),
        5 => types::parse_limited(body, max_depth)
            .and_then(types::credentials)
            .is_ok(),
        6 => {
            let model = types::error_model(body).0.is_some();
            types::oauth_error(body).is_some() || model
        }
        _ => types::parse_limited(body, max_depth)
            .and_then(types::token_response)
            .is_ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_input, parse_response};

    /// The fuzz seeds are well-formed responses: each parses with its own parser, and the 70-deep one is refused
    /// by the depth limit instead of recursing.
    #[test]
    fn the_seed_bodies_parse_and_the_deep_one_is_refused() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fuzz/seeds/iceberg_rest_response");
        let mut seen = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let data = std::fs::read(&path).unwrap();
            let ok = parse_input(&data).unwrap();
            let deep = path.file_name().unwrap().to_string_lossy().contains("deep");
            assert_eq!(ok, !deep, "{}", path.display());
            seen += 1;
        }
        assert_eq!(seen, 8);
    }

    /// Every allowed depth limit is reachable from the input's depth byte, and a body nested past it is refused.
    #[test]
    fn the_depth_byte_picks_the_limit() {
        let nested = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
        let input = |depth: u8, n: usize| {
            let mut v = vec![0, depth];
            v.extend_from_slice(nested(n).as_bytes());
            v
        };
        // Depth byte 127 allows 128 levels; byte 0 allows 1.
        assert_eq!(parse_input(&input(127, 120)), Some(true));
        assert_eq!(parse_input(&input(0, 5)), Some(false));
        assert_eq!(parse_input(&input(0, 1)), Some(true));
        assert_eq!(parse_input(&[0]), None);
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        for which in 0..16u8 {
            for body in [
                &b""[..],
                b"{",
                b"null",
                b"[1,2",
                b"{\"error\":1}",
                &[0xff, 0xfe, 0x00],
            ] {
                let _ = parse_response(which, body, 64);
            }
        }
    }
}

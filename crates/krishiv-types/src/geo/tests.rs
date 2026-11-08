// SPDX-License-Identifier: Apache-2.0
use serde_json::Value;

use super::{ByteOrder, SpatialKind, rewrite};

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.2.0/wkb-goldens.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}"))
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap_or_else(|e| panic!("{s}: {e}")))
        .collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02X}")).collect()
}

/// What `rewrite` gives, in the fixture's shape.
fn ours(wkb: &[u8], kind: SpatialKind) -> Value {
    let ndr = rewrite(wkb, kind, ByteOrder::Ndr);
    let xdr = rewrite(wkb, kind, ByteOrder::Xdr);
    match (ndr, xdr) {
        (Ok(n), Ok(x)) => serde_json::json!({"ndr": hex(&n), "xdr": hex(&x)}),
        (Err(e), Err(e2)) => {
            assert_eq!(e, e2, "the byte order of the output never changes an error");
            let params: serde_json::Map<String, Value> = e
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), Value::from(v.as_str())))
                .collect();
            serde_json::json!({
                "error_class": e.error_class().name(),
                "message_parameters": params,
            })
        }
        (n, x) => panic!("{n:?} / {x:?}"),
    }
}

#[test]
fn reads_and_writes_wkb_as_spark_does() {
    let doc = fixture();
    assert_eq!(doc["spark_version"], "4.2.0");
    let cases = doc["cases"].as_array().map_or(&[][..], Vec::as_slice);
    assert!(cases.len() > 100, "{} cases", cases.len());
    let (mut ok, mut failed) = (0, 0);
    for case in cases {
        let wkb = unhex(case["wkb"].as_str().unwrap_or_default());
        for (key, kind) in [
            ("geometry", SpatialKind::Geometry),
            ("geography", SpatialKind::Geography),
        ] {
            let spark = &case[key];
            assert_eq!(&ours(&wkb, kind), spark, "{} as {key}", case["name"]);
            if spark.get("ndr").is_some() {
                ok += 1;
            } else {
                failed += 1;
            }
        }
    }
    assert!(ok > 60 && failed > 100, "{ok} valid, {failed} refused");
}

#[test]
fn rewriting_round_trips_between_byte_orders() {
    let doc = fixture();
    for case in doc["cases"].as_array().map_or(&[][..], Vec::as_slice) {
        let wkb = unhex(case["wkb"].as_str().unwrap_or_default());
        let Ok(ndr) = rewrite(&wkb, SpatialKind::Geometry, ByteOrder::Ndr) else {
            continue;
        };
        let xdr = rewrite(&ndr, SpatialKind::Geometry, ByteOrder::Xdr);
        let xdr = xdr.unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        assert_eq!(xdr.len(), ndr.len());
        assert_eq!(
            rewrite(&xdr, SpatialKind::Geometry, ByteOrder::Ndr).as_ref(),
            Ok(&ndr),
            "{}",
            case["name"]
        );
        // The storage form is a fixed point.
        assert_eq!(
            rewrite(&ndr, SpatialKind::Geometry, ByteOrder::Ndr).as_ref(),
            Ok(&ndr)
        );
        assert!(ndr.len() <= wkb.len());
    }
}

#[test]
fn deep_nesting_does_not_recurse() {
    // A collection nested 200,000 deep: Spark's reader fails on such input
    // without an error class; this one reads it with an explicit stack.
    let mut wkb = Vec::new();
    for _ in 0..200_000 {
        wkb.extend_from_slice(&[1, 7, 0, 0, 0, 1, 0, 0, 0]);
    }
    wkb.extend_from_slice(&[1, 4, 0, 0, 0, 0, 0, 0, 0]);
    let out = rewrite(&wkb, SpatialKind::Geometry, ByteOrder::Xdr);
    assert_eq!(out.map(|o| o.len()), Ok(wkb.len()));
    // A count that claims more children than follow fails at the end.
    let last = wkb.len() - 9;
    wkb.truncate(last);
    assert!(rewrite(&wkb, SpatialKind::Geometry, ByteOrder::Ndr).is_err());
}

#[test]
fn arbitrary_bytes_never_panic() {
    // Deterministic xorshift mutations of every fixture input.
    let doc = fixture();
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for case in doc["cases"].as_array().map_or(&[][..], Vec::as_slice) {
        let base = unhex(case["wkb"].as_str().unwrap_or_default());
        for _ in 0..200 {
            let mut wkb = base.clone();
            if !wkb.is_empty() {
                for _ in 0..1 + next() % 3 {
                    let i = usize::try_from(next() % wkb.len() as u64).unwrap_or(0);
                    wkb[i] = u8::try_from(next() & 0xFF).unwrap_or(0);
                }
                let keep = usize::try_from(next() % (wkb.len() as u64 + 1)).unwrap_or(0);
                if next() % 4 == 0 {
                    wkb.truncate(keep);
                }
            }
            for kind in [SpatialKind::Geometry, SpatialKind::Geography] {
                if let Ok(out) = rewrite(&wkb, kind, ByteOrder::Ndr) {
                    assert!(out.len() <= wkb.len());
                    assert_eq!(rewrite(&out, kind, ByteOrder::Ndr).as_ref(), Ok(&out));
                }
            }
        }
    }
}

#[test]
fn points_print_as_extended_well_known_text() {
    use super::to_ewkt;
    let point = unhex("0101000000000000000000F03F0000000000000040");
    assert_eq!(to_ewkt(0, &point).unwrap(), "POINT(1 2)");
    assert_eq!(to_ewkt(3857, &point).unwrap(), "SRID=3857;POINT(1 2)");
    // Big-endian input prints the same.
    let xdr = unhex("00000000013FF00000000000004000000000000000");
    assert_eq!(to_ewkt(0, &xdr).unwrap(), "POINT(1 2)");
    assert!(to_ewkt(0, &[1]).is_err());
}

/// Every case of a real Spark 4.2.0 run (`tests/scripts` note
/// `research/2026-11-06-spark-geo-wkt-and-geography-cast.md`): a geometry
/// prints without an SRID, a geography with `SRID=4326;`. Only the accepted
/// cases are checked here; refusals are covered by the malformed-value test.
#[test]
fn every_spark_text_case_prints_the_same() {
    use super::to_ewkt;
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.2.0/geo-wkt-goldens.json"
    ))
    .unwrap_or_else(|e| panic!("{e}"));
    let doc: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}"));
    let cases = doc["cases"].as_array().expect("cases");
    assert!(cases.len() >= 30);
    for c in cases {
        let name = c["name"].as_str().expect("name");
        let wkb = unhex(c["wkb_hex"].as_str().expect("wkb"));
        if let Some(want) = c["geometry"].as_str() {
            assert_eq!(to_ewkt(0, &wkb).as_deref().ok(), Some(want), "{name}");
        }
        if let Some(want) = c["geography"].as_str() {
            assert_eq!(to_ewkt(4326, &wkb).as_deref().ok(), Some(want), "{name}");
        }
    }
}

#[test]
fn text_of_a_malformed_value_is_an_error_not_a_panic() {
    use super::to_ewkt;
    for bad in [
        &[][..],
        &[1, 1, 0, 0, 0][..],
        &[
            2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ][..],
        &[1, 99, 0, 0, 0][..],
        // A point followed by a stray byte.
        &[
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xF0, 0x3F, 0, 0, 0, 0, 0, 0, 0x40, 0, 7,
        ][..],
        // A line string that claims 2^32 - 1 points.
        &[1, 2, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF][..],
    ] {
        assert!(to_ewkt(0, bad).is_err(), "{bad:?}");
    }
    // A collection nesting past the bound is refused, not recursed into.
    let mut deep = Vec::new();
    for _ in 0..200 {
        deep.extend_from_slice(&[1, 7, 0, 0, 0, 1, 0, 0, 0]);
    }
    deep.extend_from_slice(&[1, 7, 0, 0, 0, 0, 0, 0, 0]);
    assert!(to_ewkt(0, &deep).is_err());
    // A multi-geometry holds only its own member type, and never another multi-geometry.
    let mut mixed = vec![1, 4, 0, 0, 0, 1, 0, 0, 0];
    mixed.extend_from_slice(&[1, 2, 0, 0, 0, 0, 0, 0, 0]);
    assert!(to_ewkt(0, &mixed).is_err());
    let mut nested = Vec::new();
    for _ in 0..200 {
        nested.extend_from_slice(&[1, 4, 0, 0, 0, 1, 0, 0, 0]);
    }
    assert!(to_ewkt(0, &nested).is_err());
}

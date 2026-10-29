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

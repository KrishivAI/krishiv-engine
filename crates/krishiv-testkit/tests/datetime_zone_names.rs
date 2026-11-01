// SPDX-License-Identifier: Apache-2.0
//! `z` and `zzzz` for every region zone, against the names a real Spark prints
//! (`tests/fixtures/spark-4.2.0/zone-names.json`, made by `tests/scripts/gen_zone_names.py`).

use krishiv_kernels::cast::TimeZoneRules;
use krishiv_kernels::datetime::resolve_zone;
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose, Zone};
use serde_json::Value;

type R = Result<(), Box<dyn std::error::Error>>;

#[test]
fn every_region_zone_is_named_as_in_spark() -> R {
    let path = krishiv_testkit::tests_dir().join("fixtures/spark-4.2.0/zone-names.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let short = DateTimePattern::new("z", Purpose::Format)?;
    let full = DateTimePattern::new("zzzz", Purpose::Format)?;
    let zones = fixture["zones"].as_object().ok_or("zones")?;
    let mut wrong = Vec::new();
    for (id, entry) in zones.iter().filter(|(id, _)| !id.starts_with("SystemV/")) {
        let rules = resolve_zone(id)?;
        let names: Vec<&str> = entry["names"]
            .as_array()
            .ok_or("names")?
            .iter()
            .filter_map(Value::as_str)
            .collect();
        for (kind, at, (s, f)) in [
            ("std", "std_at", (names[0], names[1])),
            ("dst", "dst_at", (names[2], names[3])),
        ] {
            let secs = entry[at].as_i64().ok_or("instant")?;
            let zone = Zone { id, rules: &rules };
            let micros = secs * 1_000_000;
            let got = (
                short.format_timestamp(micros, zone)?,
                full.format_timestamp(micros, zone)?,
            );
            if got != (s.to_owned(), f.to_owned()) {
                wrong.push(format!(
                    "{id} {kind}: spark ({s}, {f}), got {got:?} (offset {})",
                    rules.offset_at_utc(secs)
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} mismatches:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    Ok(())
}

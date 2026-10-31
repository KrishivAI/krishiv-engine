// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.session.timeZone` is checked against the time zone database
//! when it is set, as Spark's `INVALID_CONF_VALUE.TIME_ZONE` check does
//! (`timezone.sql`): a well-formed region that does not exist fails at
//! `SET`, not later in a call that reads the zone.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

const KEY: &str = "spark.sql.session.timeZone";

#[test]
fn an_unknown_zone_is_refused_when_it_is_set() {
    for bad in ["Mars/Base", "Europe/Pariss", "invalid/zone", "+25:00", ""] {
        let s = Session::new().unwrap();
        let before = s.conf().get(KEY);
        let e = s.conf().set(KEY, bad).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InvalidConfValueTimeZone, "{bad}");
        let text = e.to_string();
        assert!(text.contains(KEY) && text.contains(bad), "{text}");
        assert_eq!(s.conf().get(KEY), before, "a refused SET changes nothing: {bad}");
    }
}

#[test]
fn the_sql_forms_are_refused_the_same_way() {
    let s = Session::new().unwrap();
    for sql in [
        "SET spark.sql.session.timeZone = Mars/Base",
        "SET TIME ZONE 'Europe/Pariss'",
    ] {
        let e = block_on(s.sql(sql)).and_then(|df| block_on(df.collect())).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InvalidConfValueTimeZone, "{sql}");
    }
}

#[test]
fn known_zones_and_offsets_are_accepted() {
    let s = Session::new().unwrap();
    for ok in [
        "UTC",
        "Z",
        "America/Los_Angeles",
        "Europe/Paris",
        "+08:00",
        "-05:30",
        "GMT+02:00",
        "UTC+8",
        "PST",
    ] {
        s.conf().set(KEY, ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
        assert_eq!(s.conf().get(KEY).as_deref(), Some(ok));
    }
}

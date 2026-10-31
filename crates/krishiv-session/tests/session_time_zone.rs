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

const CANONICAL: &str = "krishiv.sql.session.timeZone";

#[test]
fn an_unknown_zone_is_refused_when_it_is_set() {
    // Under the Spark key and the canonical one; a zone with whitespace
    // around it is not a zone (Spark's `ZoneId.of` does not trim).
    for key in [KEY, CANONICAL] {
        for bad in [
            "Mars/Base",
            "Europe/Pariss",
            "invalid/zone",
            "+25:00",
            "",
            " UTC ",
            "UTC ",
        ] {
            let s = Session::new().unwrap();
            let before = s.conf().get(key);
            let e = s.conf().set(key, bad).unwrap_err();
            assert_eq!(
                e.error_class(),
                ErrorClass::InvalidConfValueTimeZone,
                "{key} {bad:?}"
            );
            assert_eq!(
                e.message_parameters(),
                [("confValue", bad.to_owned()), ("confName", key.to_owned())],
                "{key} {bad:?}"
            );
            assert_eq!(
                s.conf().get(key),
                before,
                "a refused SET changes nothing: {key} {bad:?}"
            );
        }
    }
}

#[test]
fn the_sql_forms_are_refused_the_same_way() {
    let s = Session::new().unwrap();
    for sql in [
        "SET spark.sql.session.timeZone = Mars/Base",
        "SET TIME ZONE 'Europe/Pariss'",
    ] {
        let e = block_on(s.sql(sql))
            .and_then(|df| block_on(df.collect()))
            .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::InvalidConfValueTimeZone,
            "{sql}"
        );
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
        s.conf()
            .set(KEY, ok)
            .unwrap_or_else(|e| panic!("{ok}: {e}"));
        assert_eq!(s.conf().get(KEY).as_deref(), Some(ok));
    }
}

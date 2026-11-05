// SPDX-License-Identifier: Apache-2.0
//! Week of the year and the week year of a date in the hybrid calendar
//! (`Locale.US`: weeks start on Sunday, week 1 holds January 1st), as
//! `GregorianCalendar.getWeekYear` computes them.

use super::{days_from_civil, hybrid_year_len, real_day};

/// `Calendar.WEEK_OF_YEAR` for `Locale.US` (weeks start on Sunday, week 1
/// is the one with January 1st), for the real day `real` whose hybrid
/// calendar year and day of year are `year` and `doy`.
pub(super) fn week_of_year(real: i64, year: i64, doy: i64) -> i64 {
    // Day of week of January 1st of the year (Sunday = 0).
    let jan1 = real - (doy - 1);
    let jan1_sun0 = (jan1 + 4).rem_euclid(7);
    let week = (doy - 1 + jan1_sun0) / 7 + 1;
    // The last days of December in the week that holds January 1st of the
    // next year belong to its week 1.
    let next_jan1 = jan1 + hybrid_year_len(year);
    let next_sun0 = (next_jan1 + 4).rem_euclid(7);
    let dec31 = next_jan1 - 1;
    if real > dec31 - next_sun0 && next_sun0 != 0 {
        1
    } else {
        week
    }
}

/// The day that decides the week year of `days`: itself, unless the week
/// (Sunday first, week 1 holds January 1st) is week 1 of the next year.
///
/// `days` is the written date's label and `real` its real epoch day (they
/// differ before the cutover); `doy` is the hybrid ordinal.
pub(super) fn week_year_anchor(days: i64, real: i64, year: i64, doy: i64) -> i64 {
    if year <= 0 {
        // A BC date takes `GregorianCalendar.getWeekYear`'s slow path, which
        // sets the *astronomical* year as an era year: the "next year" it
        // inspects is the mirrored year `-year`.
        return days_from_civil(bc_week_year(real, year, doy), 1, 1).unwrap_or(days);
    }
    if week_of_year(real, year, doy) == 1 && doy > 300 {
        // Late December in next year's week 1.
        days + 31
    } else {
        days
    }
}

/// The week year of a BC date (`year <= 0`, Julian), as the JDK computes it.
fn bc_week_year(real: i64, year: i64, doy: i64) -> i64 {
    let max_doy = hybrid_year_len(year);
    if doy > 1 && doy < max_doy - 6 {
        return year;
    }
    // Days from the year's first day to its first Sunday.
    let to_sunday = |jan1: i64| (7 - (jan1 + 4).rem_euclid(7)) % 7;
    let min_doy = 1 + to_sunday(real - (doy - 1));
    if doy < min_doy {
        return year;
    }
    let mirror = -year;
    let first = days_from_civil(mirror, 1, 1).map_or(0, |d| real_day(d, mirror, 1, 1));
    let del = to_sunday(first);
    let min_next = if del == 0 { 7 } else { del };
    if max_doy - doy < 7 - min_next {
        year + 1
    } else {
        year
    }
}

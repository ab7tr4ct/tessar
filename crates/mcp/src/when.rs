//! Dates: parsing the `when` and `as_of` arguments, formatting timestamps.
//!
//! Dates follow a calendar: the civil days of a time zone, the daemon's
//! local zone in service ([`Calendar::Local`]). `Timestamp` stays
//! milliseconds since the epoch in UTC, so the journal does not depend on
//! the zone; only how a moment is shown as a date, where a day begins and
//! how `when` / `as_of` days are read depend on it.
//!
//! The functions are generic over a [`chrono::TimeZone`], so tests can use a
//! fixed offset instead of the process zone.

use chrono::{FixedOffset, Local, LocalResult, NaiveDate, Offset, TimeZone, Utc};
use tessar_core::model::{Interval, TimeSpan, Timestamp};

/// Milliseconds in a day.
pub const DAY_MS: i64 = 86_400_000;

/// Days since 1970-01-01 of a civil date (proleptic Gregorian).
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date `(year, month, day)` of a day count since 1970-01-01.
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// A time zone to read and show dates in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Calendar {
    /// The local zone of this process.
    Local,
    /// A fixed offset from UTC.
    Fixed(FixedOffset),
}

impl Calendar {
    /// UTC days.
    pub fn utc() -> Self {
        Self::Fixed(Utc.fix())
    }

    /// `YYYY-MM-DD` of a timestamp.
    pub fn ymd(&self, t: Timestamp) -> String {
        match self {
            Self::Local => ymd(&Local, t),
            Self::Fixed(tz) => ymd(tz, t),
        }
    }

    /// The start of the day containing `t`.
    pub fn day_start(&self, t: Timestamp) -> Timestamp {
        match self {
            Self::Local => day_start(&Local, t),
            Self::Fixed(tz) => day_start(tz, t),
        }
    }

    /// See [`parse_when`].
    pub fn parse_when(&self, s: &str) -> Result<TimeSpan, String> {
        match self {
            Self::Local => parse_when(&Local, s),
            Self::Fixed(tz) => parse_when(tz, s),
        }
    }

    /// See [`parse_as_of`].
    pub fn parse_as_of(&self, s: &str) -> Result<Timestamp, String> {
        match self {
            Self::Local => parse_as_of(&Local, s),
            Self::Fixed(tz) => parse_as_of(tz, s),
        }
    }

    /// See [`span_text`].
    pub fn span_text(&self, span: TimeSpan) -> String {
        match self {
            Self::Local => span_text(&Local, span),
            Self::Fixed(tz) => span_text(tz, span),
        }
    }

    /// See [`tense`].
    pub fn tense(&self, span: TimeSpan, now: Timestamp) -> Option<Tense> {
        match self {
            Self::Local => tense(&Local, span, now),
            Self::Fixed(tz) => tense(tz, span, now),
        }
    }
}

/// The offset of `tz` from UTC, in milliseconds, at the instant `ms`.
fn offset_ms<Tz: TimeZone>(tz: &Tz, ms: i64) -> i64 {
    let at = chrono::DateTime::from_timestamp_millis(ms)
        .unwrap_or_default()
        .naive_utc();
    i64::from(tz.offset_from_utc_datetime(&at).fix().local_minus_utc()) * 1000
}

/// The number of the civil day in `tz` that contains `t` (days since
/// 1970-01-01 of its local date).
fn day_number<Tz: TimeZone>(tz: &Tz, t: Timestamp) -> i64 {
    (t.as_millis() + offset_ms(tz, t.as_millis())).div_euclid(DAY_MS)
}

/// The instant, in milliseconds, at which civil day `day` begins in `tz`.
fn day_start_ms<Tz: TimeZone>(tz: &Tz, day: i64) -> i64 {
    let (y, m, d) = civil_from_days(day);
    let midnight =
        NaiveDate::from_ymd_opt(y as i32, m as u32, d as u32).and_then(|d| d.and_hms_opt(0, 0, 0));
    let Some(midnight) = midnight else {
        return day * DAY_MS;
    };
    match tz.from_local_datetime(&midnight) {
        LocalResult::Single(t) | LocalResult::Ambiguous(t, _) => t.timestamp_millis(),
        // A zone that skips midnight: read it with the offset in force then.
        LocalResult::None => {
            let as_utc = midnight.and_utc().timestamp_millis();
            as_utc - offset_ms(tz, as_utc)
        }
    }
}

/// `YYYY-MM-DD` of a timestamp in `tz`.
pub fn ymd<Tz: TimeZone>(tz: &Tz, t: Timestamp) -> String {
    let (y, m, d) = civil_from_days(day_number(tz, t));
    format!("{y:04}-{m:02}-{d:02}")
}

/// The start of the day containing `t` in `tz`.
pub fn day_start<Tz: TimeZone>(tz: &Tz, t: Timestamp) -> Timestamp {
    Timestamp::from_millis(day_start_ms(tz, day_number(tz, t)))
}

fn days_in_month(y: i64, m: i64) -> i64 {
    days_from_civil(
        if m == 12 { y + 1 } else { y },
        if m == 12 { 1 } else { m + 1 },
        1,
    ) - days_from_civil(y, m, 1)
}

/// A date part as a half-open range of day numbers `[start, end)`.
fn parse_part(s: &str) -> Result<(i64, i64), String> {
    let bad = || format!("`{s}` is not a date: use YYYY, YYYY-MM or YYYY-MM-DD");
    let num = |p: &str, len: usize| -> Result<i64, String> {
        if p.len() == len && p.bytes().all(|b| b.is_ascii_digit()) {
            p.parse().map_err(|_| bad())
        } else {
            Err(bad())
        }
    };
    let parts: Vec<&str> = s.trim().split('-').collect();
    match parts[..] {
        [y] => {
            let y = num(y, 4)?;
            Ok((days_from_civil(y, 1, 1), days_from_civil(y + 1, 1, 1)))
        }
        [y, m] => {
            let (y, m) = (num(y, 4)?, num(m, 2)?);
            if !(1..=12).contains(&m) {
                return Err(format!("`{s}`: month must be 01 to 12"));
            }
            let start = days_from_civil(y, m, 1);
            Ok((start, start + days_in_month(y, m)))
        }
        [y, m, d] => {
            let (y, m, d) = (num(y, 4)?, num(m, 2)?, num(d, 2)?);
            if !(1..=12).contains(&m) {
                return Err(format!("`{s}`: month must be 01 to 12"));
            }
            if !(1..=days_in_month(y, m)).contains(&d) {
                return Err(format!("`{s}`: no such day in that month"));
            }
            let start = days_from_civil(y, m, d);
            Ok((start, start + 1))
        }
        _ => Err(bad()),
    }
}

/// Parses `when`: `YYYY`, `YYYY-MM`, `YYYY-MM-DD`, or `a/b` of those.
///
/// Every form is a range of whole days of `tz`: a day is that day, a month
/// is that month, `a/b` runs from the start of `a` to the end of `b`.
pub fn parse_when<Tz: TimeZone>(tz: &Tz, s: &str) -> Result<TimeSpan, String> {
    let ms = |d: i64| Timestamp::from_millis(day_start_ms(tz, d));
    let (start, end) = match s.split_once('/') {
        Some((a, _)) if a.trim().is_empty() => {
            return Err(format!("`{s}`: missing date before `/`"));
        }
        Some((_, b)) if b.trim().is_empty() => {
            return Err(format!("`{s}`: missing date after `/`"));
        }
        Some((a, b)) => {
            let (start, _) = parse_part(a)?;
            let (_, end) = parse_part(b)?;
            (start, end)
        }
        None => parse_part(s)?,
    };
    Interval::new(ms(start), ms(end))
        .map(TimeSpan::Range)
        .map_err(|_| format!("`{s}`: the end is before the start"))
}

/// The first instant of a span, if it has one.
pub fn span_start(span: TimeSpan) -> Option<Timestamp> {
    match span {
        TimeSpan::Point(t) | TimeSpan::Since(t) => Some(t),
        TimeSpan::Range(r) => Some(r.start()),
        TimeSpan::Until(_) | TimeSpan::Duration(_) => None,
    }
}

/// Parses `as_of` (`YYYY-MM-DD`, a day of `tz`) into the last millisecond of
/// that day.
pub fn parse_as_of<Tz: TimeZone>(tz: &Tz, s: &str) -> Result<Timestamp, String> {
    if s.trim().matches('-').count() != 2 {
        return Err(format!("`{s}` is not a date: use YYYY-MM-DD"));
    }
    let (_, end) = parse_part(s)?;
    Ok(Timestamp::from_millis(day_start_ms(tz, end) - 1))
}

/// A span for people: `2026-07-01`, `2026-07-01 to 2026-07-14`, `since …`.
pub fn span_text<Tz: TimeZone>(tz: &Tz, span: TimeSpan) -> String {
    match span {
        TimeSpan::Point(t) => ymd(tz, t),
        TimeSpan::Range(r) => {
            let (a, b) = (
                ymd(tz, r.start()),
                ymd(tz, Timestamp::from_millis(r.end().as_millis() - 1)),
            );
            if a == b { a } else { format!("{a} to {b}") }
        }
        TimeSpan::Since(t) => format!("since {}", ymd(tz, t)),
        TimeSpan::Until(t) => format!("until {}", ymd(tz, t)),
        TimeSpan::Duration(d) => format!("{} days", d.as_millis() / DAY_MS),
    }
}

/// Where a span lies relative to the day containing `now`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tense {
    /// Entirely before today.
    Past,
    /// Includes today.
    Now,
    /// Entirely after today.
    Future,
}

impl Tense {
    /// The word shown to the agent.
    pub fn word(self) -> &'static str {
        match self {
            Self::Past => "past",
            Self::Now => "now",
            Self::Future => "future",
        }
    }
}

/// Compares a span with today in `tz`. `None` for a bare duration.
pub fn tense<Tz: TimeZone>(tz: &Tz, span: TimeSpan, now: Timestamp) -> Option<Tense> {
    let day = day_number(tz, now);
    let today = day_start_ms(tz, day);
    let tomorrow = day_start_ms(tz, day + 1);
    let (start, end) = match span {
        TimeSpan::Point(t) => (t.as_millis(), t.as_millis() + 1),
        TimeSpan::Range(r) => (r.start().as_millis(), r.end().as_millis()),
        TimeSpan::Since(t) => (t.as_millis(), i64::MAX),
        TimeSpan::Until(t) => (i64::MIN, t.as_millis()),
        TimeSpan::Duration(_) => return None,
    };
    Some(if end <= today {
        Tense::Past
    } else if start >= tomorrow {
        Tense::Future
    } else {
        Tense::Now
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const UTC: Utc = Utc;

    fn range(s: &str) -> (String, String) {
        match parse_when(&UTC, s).unwrap() {
            TimeSpan::Range(r) => (
                ymd(&UTC, r.start()),
                ymd(&UTC, Timestamp::from_millis(r.end().as_millis() - 1)),
            ),
            other => panic!("not a range: {other:?}"),
        }
    }

    #[test]
    fn parses_day_month_year() {
        assert_eq!(
            range("2026-07-04"),
            ("2026-07-04".into(), "2026-07-04".into())
        );
        assert_eq!(range("2026-02"), ("2026-02-01".into(), "2026-02-28".into()));
        assert_eq!(range("2024-02"), ("2024-02-01".into(), "2024-02-29".into()));
        assert_eq!(range("2026-12"), ("2026-12-01".into(), "2026-12-31".into()));
        assert_eq!(range("2026"), ("2026-01-01".into(), "2026-12-31".into()));
    }

    #[test]
    fn parses_ranges() {
        assert_eq!(
            range("2026-07-01/2026-07-14"),
            ("2026-07-01".into(), "2026-07-14".into())
        );
        assert_eq!(
            range("2026-06/2026-08"),
            ("2026-06-01".into(), "2026-08-31".into())
        );
    }

    #[test]
    fn rejects_bad_dates() {
        for bad in [
            "",
            "soon",
            "2026-13",
            "2026-02-30",
            "2026-7-4",
            "26",
            "2026-07-01/2026-06-01",
            "2026-07-01/",
            "2026-01-01-01",
        ] {
            assert!(parse_when(&UTC, bad).is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn slash_without_a_date_is_named() {
        assert!(parse_when(&UTC, "2026/").unwrap_err().contains("after `/`"));
        assert!(
            parse_when(&UTC, "/2026")
                .unwrap_err()
                .contains("before `/`")
        );
    }

    #[test]
    fn as_of_is_end_of_day() {
        let t = parse_as_of(&UTC, "2026-07-04").unwrap();
        assert_eq!(ymd(&UTC, t), "2026-07-04");
        assert_eq!(
            ymd(&UTC, Timestamp::from_millis(t.as_millis() + 1)),
            "2026-07-05"
        );
        assert!(parse_as_of(&UTC, "2026-07").is_err());
    }

    #[test]
    fn civil_round_trip() {
        for d in [-1000, 0, 1, 59, 60, 365, 20_000, 30_000] {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
        assert_eq!(ymd(&UTC, Timestamp::from_millis(0)), "1970-01-01");
    }

    #[test]
    fn tense_against_today() {
        let now = Timestamp::from_millis(days_from_civil(2026, 11, 8) * DAY_MS + 5_000);
        let t = |s: &str| tense(&UTC, parse_when(&UTC, s).unwrap(), now);
        assert_eq!(t("2026-11-07"), Some(Tense::Past));
        assert_eq!(t("2026-11-08"), Some(Tense::Now));
        assert_eq!(t("2026-11"), Some(Tense::Now));
        assert_eq!(t("2026-11-09"), Some(Tense::Future));
        assert_eq!(t("2025"), Some(Tense::Past));
        assert_eq!(tense(&UTC, TimeSpan::Since(now), now), Some(Tense::Now));
    }

    fn plus(hours: i32) -> FixedOffset {
        FixedOffset::east_opt(hours * 3600).unwrap()
    }

    /// 2026-11-08T22:00:00Z.
    fn late_evening_utc() -> Timestamp {
        Timestamp::from_millis(days_from_civil(2026, 11, 8) * DAY_MS + 22 * 3_600_000)
    }

    #[test]
    fn a_recorded_moment_shows_its_local_date() {
        let t = late_evening_utc();
        assert_eq!(ymd(&plus(3), t), "2026-11-09");
        assert_eq!(ymd(&plus(0), t), "2026-11-08");
        assert_eq!(ymd(&plus(-5), t), "2026-11-08");
        assert_eq!(
            day_start(&plus(3), t).as_millis(),
            days_from_civil(2026, 11, 9) * DAY_MS - 3 * 3_600_000
        );
    }

    #[test]
    fn today_is_the_local_day() {
        let now = late_evening_utc();
        let at = |tz: &FixedOffset, s: &str| tense(tz, parse_when(tz, s).unwrap(), now);
        // At UTC+3 it is already the 9th: an event on the 9th is happening now.
        assert_eq!(at(&plus(3), "2026-11-09"), Some(Tense::Now));
        assert_eq!(at(&plus(3), "2026-11-08"), Some(Tense::Past));
        assert_eq!(at(&plus(3), "2026-11-10"), Some(Tense::Future));
        // At UTC it is still the 8th, and the 9th is ahead.
        assert_eq!(at(&plus(0), "2026-11-09"), Some(Tense::Future));
        assert_eq!(at(&plus(0), "2026-11-08"), Some(Tense::Now));
    }

    #[test]
    fn when_and_as_of_days_are_local_days() {
        let tz = plus(3);
        match parse_when(&tz, "2026-11-09").unwrap() {
            TimeSpan::Range(r) => {
                assert_eq!(
                    r.start().as_millis(),
                    days_from_civil(2026, 11, 9) * DAY_MS - 3 * 3_600_000
                );
                assert_eq!(r.end().as_millis() - r.start().as_millis(), DAY_MS);
            }
            other => panic!("not a range: {other:?}"),
        }
        let end = parse_as_of(&tz, "2026-11-09").unwrap();
        assert_eq!(ymd(&tz, end), "2026-11-09");
        assert_eq!(
            ymd(&tz, Timestamp::from_millis(end.as_millis() + 1)),
            "2026-11-10"
        );
        assert_eq!(
            span_text(&tz, parse_when(&tz, "2026-11-09").unwrap()),
            "2026-11-09"
        );
    }

    #[test]
    fn calendar_dispatches_to_its_zone() {
        let t = late_evening_utc();
        assert_eq!(Calendar::utc().ymd(t), "2026-11-08");
        assert_eq!(Calendar::Fixed(plus(3)).ymd(t), "2026-11-09");
    }
}

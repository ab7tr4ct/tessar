//! Dates: parsing the `when` and `as_of` arguments, formatting timestamps.
//!
//! All dates are UTC days. `Timestamp` is milliseconds since the epoch.

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

/// `YYYY-MM-DD` of a timestamp.
pub fn ymd(t: Timestamp) -> String {
    let (y, m, d) = civil_from_days(t.as_millis().div_euclid(DAY_MS));
    format!("{y:04}-{m:02}-{d:02}")
}

/// The start of the UTC day containing `t`.
pub fn day_start(t: Timestamp) -> Timestamp {
    Timestamp::from_millis(t.as_millis().div_euclid(DAY_MS) * DAY_MS)
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
/// Every form is a range of whole UTC days: a day is that day, a month is
/// that month, `a/b` runs from the start of `a` to the end of `b`.
pub fn parse_when(s: &str) -> Result<TimeSpan, String> {
    let ms = |d: i64| Timestamp::from_millis(d * DAY_MS);
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

/// Parses `as_of` (`YYYY-MM-DD`) into the last millisecond of that day.
pub fn parse_as_of(s: &str) -> Result<Timestamp, String> {
    if s.trim().matches('-').count() != 2 {
        return Err(format!("`{s}` is not a date: use YYYY-MM-DD"));
    }
    let (_, end) = parse_part(s)?;
    Ok(Timestamp::from_millis(end * DAY_MS - 1))
}

/// A span for people: `2026-07-01`, `2026-07-01 to 2026-07-14`, `since …`.
pub fn span_text(span: TimeSpan) -> String {
    match span {
        TimeSpan::Point(t) => ymd(t),
        TimeSpan::Range(r) => {
            let (a, b) = (
                ymd(r.start()),
                ymd(Timestamp::from_millis(r.end().as_millis() - 1)),
            );
            if a == b { a } else { format!("{a} to {b}") }
        }
        TimeSpan::Since(t) => format!("since {}", ymd(t)),
        TimeSpan::Until(t) => format!("until {}", ymd(t)),
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

/// Compares a span with today (UTC). `None` for a bare duration.
pub fn tense(span: TimeSpan, now: Timestamp) -> Option<Tense> {
    let today = day_start(now).as_millis();
    let tomorrow = today + DAY_MS;
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

    fn range(s: &str) -> (String, String) {
        match parse_when(s).unwrap() {
            TimeSpan::Range(r) => (
                ymd(r.start()),
                ymd(Timestamp::from_millis(r.end().as_millis() - 1)),
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
            assert!(parse_when(bad).is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn slash_without_a_date_is_named() {
        assert!(parse_when("2026/").unwrap_err().contains("after `/`"));
        assert!(parse_when("/2026").unwrap_err().contains("before `/`"));
    }

    #[test]
    fn as_of_is_end_of_day() {
        let t = parse_as_of("2026-07-04").unwrap();
        assert_eq!(ymd(t), "2026-07-04");
        assert_eq!(ymd(Timestamp::from_millis(t.as_millis() + 1)), "2026-07-05");
        assert!(parse_as_of("2026-07").is_err());
    }

    #[test]
    fn civil_round_trip() {
        for d in [-1000, 0, 1, 59, 60, 365, 20_000, 30_000] {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
        assert_eq!(ymd(Timestamp::from_millis(0)), "1970-01-01");
    }

    #[test]
    fn tense_against_today() {
        let now = Timestamp::from_millis(days_from_civil(2026, 11, 8) * DAY_MS + 5_000);
        let t = |s: &str| tense(parse_when(s).unwrap(), now);
        assert_eq!(t("2026-11-07"), Some(Tense::Past));
        assert_eq!(t("2026-11-08"), Some(Tense::Now));
        assert_eq!(t("2026-11"), Some(Tense::Now));
        assert_eq!(t("2026-11-09"), Some(Tense::Future));
        assert_eq!(t("2025"), Some(Tense::Past));
        assert_eq!(tense(TimeSpan::Since(now), now), Some(Tense::Now));
    }
}

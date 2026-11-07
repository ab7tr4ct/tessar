//! Time: instants, durations, intervals.
//!
//! Every instant is milliseconds since the Unix epoch, UTC. Phrases like
//! "last Friday" are not parsed here: the core only takes numbers.

use serde::{Deserialize, Serialize};

use super::ModelError;

/// An instant: milliseconds since the Unix epoch, UTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(i64);

impl Timestamp {
    /// The instant `ms` milliseconds after the Unix epoch.
    pub const fn from_millis(ms: i64) -> Self {
        Self(ms)
    }

    /// Milliseconds since the Unix epoch.
    pub const fn as_millis(self) -> i64 {
        self.0
    }

    /// The instant `delta` after this one, or `None` on overflow.
    pub const fn checked_add(self, delta: TimeDelta) -> Option<Self> {
        match self.0.checked_add(delta.0) {
            Some(ms) => Some(Self(ms)),
            None => None,
        }
    }
}

/// A duration in milliseconds; always positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "i64", into = "i64")]
pub struct TimeDelta(i64);

impl TimeDelta {
    /// A duration of `ms` milliseconds; fails if `ms <= 0`.
    pub const fn from_millis(ms: i64) -> Result<Self, ModelError> {
        if ms > 0 {
            Ok(Self(ms))
        } else {
            Err(ModelError::NonPositiveDuration(ms))
        }
    }

    /// The duration in milliseconds.
    pub const fn as_millis(self) -> i64 {
        self.0
    }
}

impl TryFrom<i64> for TimeDelta {
    type Error = ModelError;

    fn try_from(ms: i64) -> Result<Self, ModelError> {
        Self::from_millis(ms)
    }
}

impl From<TimeDelta> for i64 {
    fn from(delta: TimeDelta) -> i64 {
        delta.0
    }
}

/// A half-open interval `[start, end)` with start strictly before end.
///
/// An imprecise date is an interval too: "in March" is
/// `[March 1, April 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawInterval", into = "RawInterval")]
pub struct Interval {
    start: Timestamp,
    end: Timestamp,
}

/// An interval as stored in the journal, before its invariant is checked.
#[derive(Serialize, Deserialize)]
struct RawInterval {
    start: Timestamp,
    end: Timestamp,
}

impl Interval {
    /// The interval `[start, end)`; fails if it is empty.
    pub const fn new(start: Timestamp, end: Timestamp) -> Result<Self, ModelError> {
        if start.0 < end.0 {
            Ok(Self { start, end })
        } else {
            Err(ModelError::EmptyInterval {
                start: start.0,
                end: end.0,
            })
        }
    }

    /// The start, inclusive.
    pub const fn start(self) -> Timestamp {
        self.start
    }

    /// The end, exclusive.
    pub const fn end(self) -> Timestamp {
        self.end
    }

    /// Whether `t` lies inside.
    pub fn contains(self, t: Timestamp) -> bool {
        self.start <= t && t < self.end
    }

    /// Whether the two intervals share any instant.
    pub fn overlaps(self, other: Interval) -> bool {
        self.start < other.end && other.start < self.end
    }
}

impl TryFrom<RawInterval> for Interval {
    type Error = ModelError;

    fn try_from(raw: RawInterval) -> Result<Self, ModelError> {
        Self::new(raw.start, raw.end)
    }
}

impl From<Interval> for RawInterval {
    fn from(interval: Interval) -> Self {
        Self {
            start: interval.start,
            end: interval.end,
        }
    }
}

/// When the described thing happens in the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeSpan {
    /// An exact instant.
    Point(Timestamp),
    /// An interval, including an imprecise date.
    Range(Interval),
    /// From an instant on, end unknown: "since March".
    Since(Timestamp),
    /// Up to an instant, start unknown: "until summer".
    Until(Timestamp),
    /// Only the length is known: "two weeks", dates unknown.
    Duration(TimeDelta),
}

impl TimeSpan {
    /// Whether this span overlaps `window`.
    ///
    /// `None` means unknown: a [`TimeSpan::Duration`] has no position in
    /// time.
    pub fn overlaps(self, window: Interval) -> Option<bool> {
        match self {
            Self::Point(t) => Some(window.contains(t)),
            Self::Range(range) => Some(range.overlaps(window)),
            Self::Since(start) => Some(start < window.end()),
            Self::Until(end) => Some(window.start() < end),
            Self::Duration(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(ms: i64) -> Timestamp {
        Timestamp::from_millis(ms)
    }

    fn iv(start: i64, end: i64) -> Interval {
        Interval::new(ts(start), ts(end)).unwrap()
    }

    #[test]
    fn interval_rejects_empty() {
        assert_eq!(
            Interval::new(ts(5), ts(5)),
            Err(ModelError::EmptyInterval { start: 5, end: 5 })
        );
        assert!(Interval::new(ts(6), ts(5)).is_err());
    }

    #[test]
    fn interval_is_half_open() {
        let i = iv(10, 20);
        assert!(!i.contains(ts(9)));
        assert!(i.contains(ts(10)));
        assert!(i.contains(ts(19)));
        assert!(!i.contains(ts(20)));
    }

    #[test]
    fn touching_intervals_do_not_overlap() {
        assert!(!iv(0, 10).overlaps(iv(10, 20)));
        assert!(iv(0, 11).overlaps(iv(10, 20)));
        assert!(iv(12, 15).overlaps(iv(10, 20)));
        assert!(iv(10, 20).overlaps(iv(12, 15)));
    }

    #[test]
    fn delta_must_be_positive() {
        assert_eq!(
            TimeDelta::from_millis(0),
            Err(ModelError::NonPositiveDuration(0))
        );
        assert!(TimeDelta::from_millis(-1).is_err());
        assert_eq!(TimeDelta::from_millis(1).unwrap().as_millis(), 1);
    }

    #[test]
    fn checked_add_detects_overflow() {
        let day = TimeDelta::from_millis(86_400_000).unwrap();
        assert_eq!(ts(0).checked_add(day), Some(ts(86_400_000)));
        assert_eq!(ts(i64::MAX).checked_add(day), None);
    }

    #[test]
    fn span_overlaps_window() {
        let window = iv(100, 200);
        assert_eq!(TimeSpan::Point(ts(150)).overlaps(window), Some(true));
        assert_eq!(TimeSpan::Point(ts(200)).overlaps(window), Some(false));
        assert_eq!(TimeSpan::Range(iv(190, 300)).overlaps(window), Some(true));
        assert_eq!(TimeSpan::Since(ts(199)).overlaps(window), Some(true));
        assert_eq!(TimeSpan::Since(ts(200)).overlaps(window), Some(false));
        assert_eq!(TimeSpan::Until(ts(101)).overlaps(window), Some(true));
        assert_eq!(TimeSpan::Until(ts(100)).overlaps(window), Some(false));
        let two_weeks = TimeDelta::from_millis(14 * 86_400_000).unwrap();
        assert_eq!(TimeSpan::Duration(two_weeks).overlaps(window), None);
    }

    #[test]
    fn invalid_values_cannot_be_read_from_journal() {
        assert!(serde_json::from_str::<Interval>(r#"{"start":5,"end":5}"#).is_err());
        assert!(serde_json::from_str::<TimeDelta>("0").is_err());
        assert!(serde_json::from_str::<TimeSpan>(r#"{"duration":-3}"#).is_err());
    }

    #[test]
    fn span_json_is_readable() {
        let span = TimeSpan::Range(iv(1, 2));
        let json = serde_json::to_string(&span).unwrap();
        assert_eq!(json, r#"{"range":{"start":1,"end":2}}"#);
        assert_eq!(serde_json::from_str::<TimeSpan>(&json).unwrap(), span);
    }
}

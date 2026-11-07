//! Heat: how useful a fact has been recently.
//!
//! Heat decides retention only — which facts may cool down and leave hot
//! memory. It never boosts ranking: otherwise facts that are recalled often
//! would be recalled even more often, and a few clusters would crowd out
//! everything else.
//!
//! Heat is kept per [`Lineage`](super::Lineage) in a side table built from
//! the access log, not in the fact record: a fact's content never changes,
//! and a read must not turn into a write to the main journal.

use serde::{Deserialize, Serialize};

use super::ModelError;
use super::time::{TimeDelta, Timestamp};

/// Heat of one fact lineage: a non-negative value that halves every
/// half-life since `updated_at`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawHeat", into = "RawHeat")]
pub struct Heat {
    value: f32,
    updated_at: Timestamp,
}

/// Heat as stored in the access log, before its invariant is checked.
#[derive(Serialize, Deserialize)]
struct RawHeat {
    value: f32,
    updated_at: Timestamp,
}

impl Heat {
    /// Heat `value` as of `at`; fails if the value is negative, NaN or
    /// infinite.
    pub fn new(value: f32, at: Timestamp) -> Result<Self, ModelError> {
        if value.is_finite() && value >= 0.0 {
            Ok(Self {
                value,
                updated_at: at,
            })
        } else {
            Err(ModelError::InvalidHeat(value))
        }
    }

    /// No heat yet, as of `at`.
    pub const fn cold(at: Timestamp) -> Self {
        Self {
            value: 0.0,
            updated_at: at,
        }
    }

    /// The value as of the last update, before decay.
    pub const fn value(self) -> f32 {
        self.value
    }

    /// When the value was last updated.
    pub const fn updated_at(self) -> Timestamp {
        self.updated_at
    }

    /// The value at `now` after decay. A `now` before the last update gives
    /// the undecayed value.
    pub fn at(self, now: Timestamp, half_life: TimeDelta) -> f32 {
        let elapsed = now.as_millis().saturating_sub(self.updated_at.as_millis());
        if elapsed <= 0 {
            return self.value;
        }
        let half_lives = elapsed as f64 / half_life.as_millis() as f64;
        (f64::from(self.value) * (-half_lives).exp2()) as f32
    }

    /// Decays to `now` and then adds `gain`.
    pub fn warmed(
        self,
        gain: f32,
        now: Timestamp,
        half_life: TimeDelta,
    ) -> Result<Self, ModelError> {
        let at = now.max(self.updated_at);
        Self::new(self.at(at, half_life) + gain, at)
    }
}

impl TryFrom<RawHeat> for Heat {
    type Error = ModelError;

    fn try_from(raw: RawHeat) -> Result<Self, ModelError> {
        Self::new(raw.value, raw.updated_at)
    }
}

impl From<Heat> for RawHeat {
    fn from(heat: Heat) -> Self {
        Self {
            value: heat.value,
            updated_at: heat.updated_at,
        }
    }
}

/// How useful a recalled fact turned out to be for one query.
///
/// Stored as a number in `-1..=2`, the scale an agent answers with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "i8", into = "i8")]
pub enum Usefulness {
    /// Completely off for this query.
    Miss,
    /// Shown, but not used.
    Neutral,
    /// Used.
    Useful,
    /// Exactly what was needed.
    Bingo,
}

impl Usefulness {
    /// The score on the `-1..=2` scale.
    pub const fn score(self) -> i8 {
        match self {
            Self::Miss => -1,
            Self::Neutral => 0,
            Self::Useful => 1,
            Self::Bingo => 2,
        }
    }

    /// How much heat the fact gains.
    ///
    /// Only usefulness warms a fact. A miss says that this query and this
    /// fact do not match — a signal for tuning ranking, not a reason to
    /// cool a fact that may matter elsewhere.
    pub const fn heat_gain(self) -> f32 {
        match self {
            Self::Miss | Self::Neutral => 0.0,
            Self::Useful => 1.0,
            Self::Bingo => 2.0,
        }
    }
}

impl TryFrom<i8> for Usefulness {
    type Error = ModelError;

    fn try_from(score: i8) -> Result<Self, ModelError> {
        match score {
            -1 => Ok(Self::Miss),
            0 => Ok(Self::Neutral),
            1 => Ok(Self::Useful),
            2 => Ok(Self::Bingo),
            _ => Err(ModelError::InvalidUsefulness(score)),
        }
    }
}

impl From<Usefulness> for i8 {
    fn from(u: Usefulness) -> i8 {
        u.score()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400_000;

    fn ts(ms: i64) -> Timestamp {
        Timestamp::from_millis(ms)
    }

    fn week() -> TimeDelta {
        TimeDelta::from_millis(7 * DAY).unwrap()
    }

    #[test]
    fn heat_must_be_finite_and_non_negative() {
        assert!(Heat::new(0.0, ts(0)).is_ok());
        assert_eq!(Heat::new(-1.0, ts(0)), Err(ModelError::InvalidHeat(-1.0)));
        assert!(Heat::new(f32::NAN, ts(0)).is_err());
        assert!(Heat::new(f32::INFINITY, ts(0)).is_err());
    }

    #[test]
    fn heat_halves_every_half_life() {
        let heat = Heat::new(8.0, ts(0)).unwrap();
        assert_eq!(heat.at(ts(0), week()), 8.0);
        assert_eq!(heat.at(ts(7 * DAY), week()), 4.0);
        assert_eq!(heat.at(ts(21 * DAY), week()), 1.0);
        // A clock behind the last update does not heat up.
        assert_eq!(heat.at(ts(-DAY), week()), 8.0);
    }

    #[test]
    fn warming_decays_first_then_adds() {
        let heat = Heat::new(4.0, ts(0)).unwrap();
        let warmed = heat.warmed(2.0, ts(7 * DAY), week()).unwrap();
        assert_eq!(warmed.value(), 4.0);
        assert_eq!(warmed.updated_at(), ts(7 * DAY));
        assert_eq!(
            Heat::cold(ts(0))
                .warmed(1.0, ts(0), week())
                .unwrap()
                .value(),
            1.0
        );
    }

    #[test]
    fn only_usefulness_warms() {
        assert_eq!(Usefulness::Miss.heat_gain(), 0.0);
        assert_eq!(Usefulness::Neutral.heat_gain(), 0.0);
        assert!(Usefulness::Useful.heat_gain() > 0.0);
        assert!(Usefulness::Bingo.heat_gain() > Usefulness::Useful.heat_gain());
    }

    #[test]
    fn usefulness_is_stored_as_a_score() {
        let scores: Vec<Usefulness> = serde_json::from_str("[-1, 1, 2, 1, 0]").unwrap();
        assert_eq!(
            scores,
            [
                Usefulness::Miss,
                Usefulness::Useful,
                Usefulness::Bingo,
                Usefulness::Useful,
                Usefulness::Neutral
            ]
        );
        assert_eq!(serde_json::to_string(&Usefulness::Bingo).unwrap(), "2");
        assert!(serde_json::from_str::<Usefulness>("3").is_err());
        assert!(serde_json::from_str::<Usefulness>("-2").is_err());
    }

    #[test]
    fn invalid_heat_cannot_be_read_from_log() {
        assert!(serde_json::from_str::<Heat>(r#"{"value":-0.5,"updated_at":0}"#).is_err());
        let heat = Heat::new(1.5, ts(3)).unwrap();
        let json = serde_json::to_string(&heat).unwrap();
        assert_eq!(serde_json::from_str::<Heat>(&json).unwrap(), heat);
    }
}

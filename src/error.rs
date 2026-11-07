//! Library errors.

use thiserror::Error;

/// Library errors.
///
/// Model errors are returned when a value would break an invariant: an
/// empty interval, a NaN, a confidence outside `[0, 1]`. Such a value
/// cannot be created at all — neither by a constructor nor by reading the
/// journal.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    /// The interval `[start, end)` is empty: start is not before end.
    #[error("empty interval: start {start} ms is not before end {end} ms")]
    EmptyInterval {
        /// Start, in milliseconds.
        start: i64,
        /// End, in milliseconds.
        end: i64,
    },

    /// A duration must be positive.
    #[error("duration must be positive, got {0} ms")]
    NonPositiveDuration(i64),

    /// A float is NaN or infinite.
    #[error("number must be finite, got {0}")]
    NonFiniteFloat(f64),

    /// A confidence is outside `[0, 1]`.
    #[error("confidence must be in [0, 1], got {0}")]
    ConfidenceOutOfRange(f32),

    /// A key is empty or whitespace only.
    #[error("key must not be empty")]
    EmptyKey,

    /// A key is too long.
    #[error("key is longer than {max} characters")]
    KeyTooLong {
        /// Maximum length, in characters.
        max: usize,
    },

    /// A kind or predicate name breaks the naming rules.
    #[error("invalid name {0:?}: use 1 to 64 lowercase letters, digits or `_`")]
    InvalidName(String),
}

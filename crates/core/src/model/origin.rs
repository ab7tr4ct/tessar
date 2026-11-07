//! Where facts come from.

use serde::{Deserialize, Serialize};

use super::ModelError;
use super::text::Key;

/// Confidence in an inferred fact, from 0 to 1 inclusive.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Confidence(f32);

impl Confidence {
    /// The confidence `x`; fails if it is outside `[0, 1]` or NaN.
    pub fn new(x: f32) -> Result<Self, ModelError> {
        if (0.0..=1.0).contains(&x) {
            Ok(Self(x))
        } else {
            Err(ModelError::ConfidenceOutOfRange(x))
        }
    }

    /// The value.
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for Confidence {
    type Error = ModelError;

    fn try_from(x: f32) -> Result<Self, ModelError> {
        Self::new(x)
    }
}

impl From<Confidence> for f32 {
    fn from(c: Confidence) -> f32 {
        c.0
    }
}

/// Where a fact comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "origin", rename_all = "snake_case")]
pub enum Origin {
    /// Written by a person, an agent or a hook.
    Asserted {
        /// Who wrote it: `user`, `claude-code`, `hook:post_tool_use`.
        by: Key,
    },
    /// Inferred by a model or an algorithm. A hypothesis until confirmed.
    Inferred {
        /// Who inferred it.
        by: Key,
        /// How sure it is.
        confidence: Confidence,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_bounds() {
        assert!(Confidence::new(0.0).is_ok());
        assert!(Confidence::new(1.0).is_ok());
        assert!(Confidence::new(-0.01).is_err());
        assert!(Confidence::new(1.01).is_err());
        assert!(Confidence::new(f32::NAN).is_err());
    }

    #[test]
    fn origin_json_is_tagged() {
        let origin = Origin::Inferred {
            by: Key::new("extractor").unwrap(),
            confidence: Confidence::new(0.5).unwrap(),
        };
        let json = serde_json::to_string(&origin).unwrap();
        assert_eq!(
            json,
            r#"{"origin":"inferred","by":"extractor","confidence":0.5}"#
        );
        assert_eq!(serde_json::from_str::<Origin>(&json).unwrap(), origin);
        assert!(
            serde_json::from_str::<Origin>(r#"{"origin":"inferred","by":"x","confidence":2.0}"#)
                .is_err()
        );
    }
}

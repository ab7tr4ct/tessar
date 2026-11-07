//! Validated strings: entity keys and schema names.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::ModelError;

/// A key: `tessar`, `src/main.rs`, `Anna`.
///
/// Not empty, no surrounding whitespace, at most [`Key::MAX_CHARS`]
/// characters. Equality is by exact text; for lookup, keys with the same
/// [`Key::normalized`] form are the same key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Key(String);

impl Key {
    /// Maximum key length, in characters.
    pub const MAX_CHARS: usize = 256;

    /// A key from text; surrounding whitespace is trimmed.
    pub fn new(raw: &str) -> Result<Self, ModelError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(ModelError::EmptyKey);
        }
        if trimmed.chars().count() > Self::MAX_CHARS {
            return Err(ModelError::KeyTooLong {
                max: Self::MAX_CHARS,
            });
        }
        Ok(Self(trimmed.to_owned()))
    }

    /// The key text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The form used for lookup: lowercase, inner whitespace collapsed to a
    /// single space.
    pub fn normalized(&self) -> String {
        self.0
            .split_whitespace()
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl TryFrom<String> for Key {
    type Error = ModelError;

    fn try_from(raw: String) -> Result<Self, ModelError> {
        Self::new(&raw)
    }
}

impl From<Key> for String {
    fn from(key: Key) -> String {
        key.0
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The name of an entity kind or a predicate: `person`, `lives_in`.
///
/// 1 to [`Name::MAX_CHARS`] characters: lowercase letters of any script,
/// digits and `_`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Name(String);

impl Name {
    /// Maximum name length, in characters.
    pub const MAX_CHARS: usize = 64;

    /// A name from text; fails if the text breaks the naming rules.
    pub fn new(raw: &str) -> Result<Self, ModelError> {
        let allowed =
            |c: char| c == '_' || c.is_numeric() || (c.is_alphabetic() && !c.is_uppercase());
        let len = raw.chars().count();
        if (1..=Self::MAX_CHARS).contains(&len) && raw.chars().all(allowed) {
            Ok(Self(raw.to_owned()))
        } else {
            Err(ModelError::InvalidName(raw.to_owned()))
        }
    }

    /// The name text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Name {
    type Error = ModelError;

    fn try_from(raw: String) -> Result<Self, ModelError> {
        Self::new(&raw)
    }
}

impl From<Name> for String {
    fn from(name: Name) -> String {
        name.0
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_trimmed_and_not_empty() {
        assert_eq!(Key::new("  tessar ").unwrap().as_str(), "tessar");
        assert_eq!(Key::new("   "), Err(ModelError::EmptyKey));
        assert_eq!(Key::new(""), Err(ModelError::EmptyKey));
    }

    #[test]
    fn key_length_is_counted_in_chars_not_bytes() {
        let max = "я".repeat(Key::MAX_CHARS);
        assert!(Key::new(&max).is_ok());
        let over = "я".repeat(Key::MAX_CHARS + 1);
        assert_eq!(
            Key::new(&over),
            Err(ModelError::KeyTooLong {
                max: Key::MAX_CHARS
            })
        );
    }

    #[test]
    fn normalized_ignores_case_and_inner_spaces_in_any_script() {
        let a = Key::new("My   Sister").unwrap();
        let b = Key::new("my sister").unwrap();
        assert_ne!(a, b);
        assert_eq!(a.normalized(), b.normalized());
        assert_eq!(Key::new("Моя  Сестра").unwrap().normalized(), "моя сестра");
    }

    #[test]
    fn name_rules() {
        for ok in ["person", "lives_in", "цель", "v2", "_"] {
            assert!(Name::new(ok).is_ok(), "{ok}");
        }
        for bad in ["", "Person", "lives in", "lives-in", "Цель"] {
            assert!(Name::new(bad).is_err(), "{bad}");
        }
        assert!(Name::new(&"a".repeat(Name::MAX_CHARS + 1)).is_err());
    }

    #[test]
    fn invalid_strings_cannot_be_read_from_journal() {
        assert!(serde_json::from_str::<Key>(r#""  ""#).is_err());
        assert!(serde_json::from_str::<Name>(r#""Bad Name""#).is_err());
        assert_eq!(
            serde_json::from_str::<Key>(r#"" tessar ""#)
                .unwrap()
                .as_str(),
            "tessar"
        );
    }
}

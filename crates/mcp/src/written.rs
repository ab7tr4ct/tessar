//! What a write tool (`remember`, `lesson`, `update`) did, as data. The tool
//! attaches it to its result as structured content; `TessarServer::call_tool`
//! takes it out again for the decision log, so the client only ever sees the
//! text report. Handles, outcomes and spaces only; never a fact text.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What happened to one fact. The one place the outcome words are defined:
/// the text reports and the structured content both take them from here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WriteOutcome {
    /// A new fact.
    Created,
    /// An inferred value recorded beside an asserted current one.
    Conflicting,
    /// Already known; nothing was written.
    Unchanged,
    /// Already known, and the contradicting current values were closed.
    Kept,
    /// The fact stopped being true.
    Closed,
    /// A new version replaced the previous one.
    Superseded,
    /// The fact was never true: closed, or replaced by a new fact.
    Corrected,
    /// The content was destroyed.
    Redacted,
}

impl WriteOutcome {
    /// Every outcome, in the order of the type.
    pub const ALL: [Self; 8] = [
        Self::Created,
        Self::Conflicting,
        Self::Unchanged,
        Self::Kept,
        Self::Closed,
        Self::Superseded,
        Self::Corrected,
        Self::Redacted,
    ];

    /// The word the text reports start with and the structured content uses.
    pub const fn word(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Conflicting => "conflicting",
            Self::Unchanged => "unchanged",
            Self::Kept => "kept",
            Self::Closed => "closed",
            Self::Superseded => "superseded",
            Self::Corrected => "corrected",
            Self::Redacted => "redacted",
        }
    }
}

/// One fact of a write call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WrittenFact {
    /// The fact the outcome is about: the new version when one was written,
    /// otherwise the fact that was addressed. Every outcome the tools
    /// produce today names one; it is optional so that a result without a
    /// fact can be told.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub outcome: WriteOutcome,
    /// `superseded`, or `corrected` with a replacement: the version replaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<String>,
    /// `conflicting`: the asserted current value it sits beside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conflicts_with: Option<String>,
    /// `kept`: a contradicting current value that was closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed: Option<String>,
    /// `kept`: how many more were closed besides `closed`; absent when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed_others: Option<u32>,
}

impl WrittenFact {
    pub fn new(id: Option<String>, outcome: WriteOutcome) -> Self {
        Self {
            id,
            outcome,
            previous: None,
            conflicts_with: None,
            closed: None,
            closed_others: None,
        }
    }
}

/// The result of a write call: `{"space": ..., "facts": [...]}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Written {
    /// The space the call wrote to.
    pub space: String,
    /// One entry per fact, in call order.
    pub facts: Vec<WrittenFact>,
}

impl Written {
    pub fn to_value(&self) -> Value {
        // A plain struct of strings, numbers and an enum: cannot fail.
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_agree_with_the_serialised_form() {
        for outcome in WriteOutcome::ALL {
            assert_eq!(
                serde_json::to_value(outcome).unwrap(),
                Value::String(outcome.word().to_owned())
            );
        }
    }

    #[test]
    fn absent_fields_are_not_serialised() {
        let written = Written {
            space: "user".into(),
            facts: vec![WrittenFact::new(None, WriteOutcome::Created)],
        };
        assert_eq!(
            written.to_value(),
            serde_json::json!({"space": "user", "facts": [{"outcome": "created"}]})
        );
    }
}

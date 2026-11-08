//! Memory records: spaces, entities, episodes, facts.

use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};

use super::ids::{EntityId, EntityKindId, EpisodeId, FactId, Lineage, PredicateId, SpaceId};
use super::origin::Origin;
use super::text::Key;
use super::time::{TimeSpan, Timestamp};
use super::value::Value;

/// The kind of a space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceKind {
    /// Long-term memory: a project, a user, an agent.
    LongTerm,
    /// Working memory of an agent team for the duration of one workflow.
    Working,
}

/// The state of a space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceState {
    /// Writable.
    Open,
    /// Read-only; nothing is deleted.
    Archived,
}

/// A memory space: the boundary of data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Space {
    /// The space id.
    pub id: SpaceId,
    /// The name: `tessar`, `personal`.
    pub name: Key,
    /// The parent; for working memory, the project space.
    pub parent: Option<SpaceId>,
    /// Long-term or working.
    pub kind: SpaceKind,
    /// Open or archived.
    pub state: SpaceState,
    /// When it was created.
    pub created_at: Timestamp,
}

/// An entity is identity only. Everything known about it is a fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    /// The entity id.
    pub id: EntityId,
    /// The space of the entity.
    pub space: SpaceId,
    /// The kind from the space schema.
    pub kind: EntityKindId,
    /// The key; unique within the space and kind (by [`Key::normalized`]).
    pub key: Key,
    /// Other names: "my sister" for "Anna".
    pub aliases: Vec<Key>,
    /// When it was created.
    pub created_at: Timestamp,
}

/// An episode: the source text facts were taken from — a message, a
/// session, a document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Episode {
    /// The episode id.
    pub id: EpisodeId,
    /// The space of the episode.
    pub space: SpaceId,
    /// The source text.
    pub text: String,
    /// When the episode happened.
    pub at: Timestamp,
    /// Where it came from: `claude-code`, `voice-call`, `import`.
    pub source: Key,
    /// When memory recorded it.
    pub recorded_at: Timestamp,
}

/// The content of a fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FactBody {
    /// A free-form statement: "decided not to use MongoDB".
    Statement {
        /// The statement text.
        text: String,
        /// When the described thing happens in the world, if known.
        when: Option<TimeSpan>,
    },
    /// A triple: (tessar, goal) = "memory engine"; (vacation, span) =
    /// July 1–14.
    ///
    /// A link between two entities is a triple whose value is a
    /// [`Value::Entity`].
    Triple {
        /// Who or what the fact is about.
        subject: EntityId,
        /// A predicate from a space schema.
        predicate: PredicateId,
        /// The value.
        object: Value,
        /// The fact in one sentence, as the writer put it: "User leads a
        /// team of 5 engineers". Carries the context the triple drops and
        /// is what a question resembles. `None` if written without one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    /// An event. It cannot be created without `when`.
    Event {
        /// What happened.
        text: String,
        /// Who or what took part.
        participants: Vec<EntityId>,
        /// When.
        when: TimeSpan,
    },
}

impl FactBody {
    /// Entities referenced by the content, in order of appearance.
    pub fn entities(&self) -> Vec<EntityId> {
        match self {
            Self::Statement { .. } => Vec::new(),
            Self::Triple {
                subject,
                object: Value::Entity(object),
                ..
            } => vec![*subject, *object],
            Self::Triple { subject, .. } => vec![*subject],
            Self::Event { participants, .. } => participants.clone(),
        }
    }

    /// When the described thing happens in the world, if known.
    ///
    /// For a triple, only if the value itself is a span ([`Value::Span`]).
    pub fn when(&self) -> Option<TimeSpan> {
        match self {
            Self::Statement { when, .. } => *when,
            Self::Event { when, .. } => Some(*when),
            Self::Triple {
                object: Value::Span(span),
                ..
            } => Some(*span),
            Self::Triple { .. } => None,
        }
    }
}

/// How a fact version ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "ending", rename_all = "snake_case")]
pub enum Ending {
    /// The world changed: the version was true until `valid_to`.
    Changed {
        /// Until when the version was true in the world.
        valid_to: Timestamp,
    },
    /// The version was a mistake: it was never true.
    Corrected,
}

/// The closing of a fact version. Set once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactEnd {
    /// A change in the world or a correction of a mistake.
    #[serde(flatten)]
    pub ending: Ending,
    /// When memory learned that the version is closed.
    pub recorded_at: Timestamp,
    /// The next version; `None` if the fact simply stopped being true.
    pub next: Option<FactId>,
}

/// One version of a fact.
///
/// The content never changes. `end` is set once, when the version is
/// closed; the next version is a new record with its own id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    /// The id of this version.
    pub id: FactId,
    /// The space of the fact.
    pub space: SpaceId,
    /// The version chain this version belongs to.
    pub lineage: Lineage,
    /// The version number within the chain, starting at 1.
    pub version: NonZeroU32,
    /// The previous version.
    pub prev: Option<FactId>,
    /// The content.
    pub body: FactBody,
    /// Entities mentioned beyond those in the content, for example in the
    /// text of a statement.
    pub mentions: Vec<EntityId>,
    /// The episode the fact was taken from.
    pub episode: Option<EpisodeId>,
    /// Who wrote or inferred it.
    pub origin: Origin,
    /// When memory learned the fact.
    pub recorded_at: Timestamp,
    /// From when the fact is true in the world.
    pub valid_from: Timestamp,
    /// The closing of this version; `None` means it is current.
    pub end: Option<FactEnd>,
}

impl Fact {
    /// Whether this is the current version.
    pub fn is_current(&self) -> bool {
        self.end.is_none()
    }

    /// The next version, if any.
    pub fn next(&self) -> Option<FactId> {
        self.end.and_then(|end| end.next)
    }

    /// Whether the fact was true in the world at `t`, according to what
    /// memory knows now. A corrected version was never true.
    pub fn was_true_at(&self, t: Timestamp) -> bool {
        if t < self.valid_from {
            return false;
        }
        match self.end.map(|end| end.ending) {
            None => true,
            Some(Ending::Changed { valid_to }) => t < valid_to,
            Some(Ending::Corrected) => false,
        }
    }

    /// Whether memory held this version as current at `t`.
    pub fn was_known_at(&self, t: Timestamp) -> bool {
        self.recorded_at <= t && self.end.is_none_or(|end| t < end.recorded_at)
    }

    /// All entities of the fact — from the content and the mentions —
    /// sorted by id, without duplicates.
    pub fn entities(&self) -> Vec<EntityId> {
        let mut all = self.body.entities();
        all.extend_from_slice(&self.mentions);
        all.sort_unstable();
        all.dedup();
        all
    }
}

#[cfg(test)]
mod tests {
    use super::super::time::Interval;
    use super::*;

    fn ts(ms: i64) -> Timestamp {
        Timestamp::from_millis(ms)
    }

    fn e(raw: u32) -> EntityId {
        EntityId::new(raw)
    }

    /// The version "the user lives in Moscow": recorded at 100, true from 50.
    fn moscow() -> Fact {
        Fact {
            id: FactId::new(0),
            space: SpaceId::new(0),
            lineage: Lineage::starting_at(FactId::new(0)),
            version: NonZeroU32::MIN,
            prev: None,
            body: FactBody::Triple {
                subject: e(0),
                predicate: PredicateId::new(0),
                object: Value::Entity(e(1)),
                text: None,
            },
            mentions: vec![e(2), e(0)],
            episode: None,
            origin: Origin::Asserted {
                by: Key::new("user").unwrap(),
            },
            recorded_at: ts(100),
            valid_from: ts(50),
            end: None,
        }
    }

    #[test]
    fn current_version_is_true_from_valid_from_on() {
        let f = moscow();
        assert!(f.is_current());
        assert!(!f.was_true_at(ts(49)));
        assert!(f.was_true_at(ts(50)));
        assert!(f.was_true_at(ts(1_000_000)));
    }

    #[test]
    fn known_only_after_recording() {
        let f = moscow();
        // True from 50, but memory learned it only at 100.
        assert!(f.was_true_at(ts(60)));
        assert!(!f.was_known_at(ts(60)));
        assert!(f.was_known_at(ts(100)));
    }

    #[test]
    fn backdated_change_splits_two_time_axes() {
        // At 300 memory learned that the move happened at 200.
        let mut f = moscow();
        f.end = Some(FactEnd {
            ending: Ending::Changed { valid_to: ts(200) },
            recorded_at: ts(300),
            next: Some(FactId::new(1)),
        });
        assert!(!f.is_current());
        assert_eq!(f.next(), Some(FactId::new(1)));
        assert!(f.was_true_at(ts(199)));
        assert!(!f.was_true_at(ts(200)));
        // At 250 memory still held this version as current.
        assert!(f.was_known_at(ts(250)));
        assert!(!f.was_known_at(ts(300)));
    }

    #[test]
    fn corrected_version_was_never_true_but_was_known() {
        let mut f = moscow();
        f.end = Some(FactEnd {
            ending: Ending::Corrected,
            recorded_at: ts(300),
            next: Some(FactId::new(1)),
        });
        assert!(!f.was_true_at(ts(60)));
        assert!(!f.was_true_at(ts(250)));
        assert!(f.was_known_at(ts(250)));
        assert!(!f.was_known_at(ts(300)));
    }

    #[test]
    fn entities_merge_body_and_mentions() {
        assert_eq!(moscow().entities(), vec![e(0), e(1), e(2)]);
        let text_only = FactBody::Triple {
            subject: e(5),
            predicate: PredicateId::new(0),
            object: Value::Text("memory engine".into()),
            text: None,
        };
        assert_eq!(text_only.entities(), vec![e(5)]);
    }

    #[test]
    fn when_comes_from_statement_event_or_span_value() {
        let july = TimeSpan::Range(Interval::new(ts(10), ts(20)).unwrap());
        let statement = FactBody::Statement {
            text: "x".into(),
            when: None,
        };
        let event = FactBody::Event {
            text: "vacation".into(),
            participants: vec![e(0)],
            when: july,
        };
        let span = FactBody::Triple {
            subject: e(3),
            predicate: PredicateId::new(1),
            object: Value::Span(july),
            text: None,
        };
        let deadline = FactBody::Triple {
            subject: e(3),
            predicate: PredicateId::new(2),
            object: Value::Time(ts(15)),
            text: None,
        };
        assert_eq!(statement.when(), None);
        assert_eq!(event.when(), Some(july));
        assert_eq!(span.when(), Some(july));
        assert_eq!(deadline.when(), None);
    }

    #[test]
    fn fact_round_trips_through_json() {
        let mut f = moscow();
        f.end = Some(FactEnd {
            ending: Ending::Changed { valid_to: ts(200) },
            recorded_at: ts(300),
            next: Some(FactId::new(1)),
        });
        let json = serde_json::to_string(&f).unwrap();
        assert!(json.contains(r#""kind":"triple""#), "{json}");
        assert!(
            json.contains(
                r#""end":{"ending":"changed","valid_to":200,"recorded_at":300,"next":1}"#
            ),
            "{json}"
        );
        assert_eq!(serde_json::from_str::<Fact>(&json).unwrap(), f);
    }

    #[test]
    fn version_zero_cannot_be_read_from_journal() {
        let json = serde_json::to_string(&moscow())
            .unwrap()
            .replace(r#""version":1"#, r#""version":0"#);
        assert!(serde_json::from_str::<Fact>(&json).is_err());
    }
}

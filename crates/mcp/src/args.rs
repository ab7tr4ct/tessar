//! Tool inputs. The doc comments become the descriptions the agent sees in
//! the JSON schema, so they are written for the agent.

use schemars::JsonSchema;
use serde::Deserialize;

/// A space to write to or to read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Space {
    /// The current repository: its decisions, conventions, architecture.
    Project,
    /// The person: preferences, plans, habits, anything true of them in every project.
    User,
}

/// Where `recall` looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum RecallSpace {
    /// Only the current repository.
    Project,
    /// Only facts about the person.
    User,
    /// Both (default).
    #[default]
    All,
}

/// An entity: a person, place, project, thing the facts are about.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct Entity {
    /// The entity's name as it is usually written: "Anna", "tessar", "Berlin". The person the agent works for is `{"key": "user", "kind": "person"}`.
    pub key: String,
    /// One of: person, place, organization, item, project, event, topic, other.
    pub kind: String,
    /// Where the entity lives. Defaults to the fact's space (`about` looks in project, then user); the entity `user` always lives in the user space.
    #[serde(default)]
    pub space: Option<Space>,
}

/// The kind of a fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum FactKind {
    /// A free-form sentence: a decision, a convention, a note. Needs `text`.
    Statement,
    /// A (subject, predicate, value) property that can change over time. Needs `subject`, `predicate`, `value`; add `text` too.
    Attribute,
    /// Something that happened or will happen on a date. Needs `text` and `when`.
    Event,
}

/// One fact to remember.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct Fact {
    /// statement, attribute or event.
    pub kind: FactKind,
    /// The fact in one self-contained sentence ("User plans a trip to Japan in May 2027"). Required for statement and event, strongly recommended for attribute: it is what later searches match.
    #[serde(default)]
    pub text: Option<String>,
    /// When it happens: "YYYY-MM-DD", "YYYY-MM", "YYYY" or "YYYY-MM-DD/YYYY-MM-DD". Required for event, optional for statement.
    #[serde(default)]
    pub when: Option<String>,
    /// Attribute only: who or what the property belongs to.
    #[serde(default)]
    pub subject: Option<Entity>,
    /// Attribute only: the property, snake_case, at most 4 words, universal (`trip_destination`, not `plans_trip_to_japan`). Reuse a name from the vocabulary when one fits.
    #[serde(default)]
    pub predicate: Option<String>,
    /// Attribute only: the value as text ("Japan", "Berlin").
    #[serde(default)]
    pub value: Option<String>,
    /// Attribute, new predicate only: true if a new value always replaces the old one (`lives_in`, `job_title`). Default false: values accumulate.
    #[serde(default)]
    pub single: Option<bool>,
    /// Attribute, new predicate only: one line saying what the predicate means and what its value is. Required when the predicate is new.
    #[serde(default)]
    pub predicate_description: Option<String>,
    /// Attribute: declare `predicate` a synonym of this existing predicate, so both mean the same from now on.
    #[serde(default)]
    pub synonym_of: Option<String>,
    /// Attribute: define `predicate` as new even though memory found a similar one. Use only after looking at the similar predicate.
    #[serde(default)]
    pub force_new_predicate: Option<bool>,
    /// Other entities the fact is about besides the subject.
    #[serde(default)]
    pub mentions: Vec<Entity>,
}

/// Input of `remember`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RememberArgs {
    /// Where the facts belong. Default `project`. Facts about the person go to `user`.
    #[serde(default)]
    pub space: Option<Space>,
    /// The text the facts were taken from (a message, a note), kept next to them as their source.
    #[serde(default)]
    pub source: Option<String>,
    /// The facts. All are written, or none.
    pub facts: Vec<Fact>,
}

/// Input of `recall`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RecallArgs {
    /// What to look for, in natural language.
    pub query: String,
    /// How many facts to return. Default 10.
    #[serde(default)]
    pub limit: Option<usize>,
    /// project, user or all (default).
    #[serde(default)]
    pub space: Option<RecallSpace>,
    /// "YYYY-MM-DD": show what was true on that day instead of now.
    #[serde(default)]
    pub as_of: Option<String>,
}

/// Input of `about`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AboutArgs {
    /// The entity to list facts about.
    pub entity: Entity,
    /// Also list the replaced values of changed facts. Default false.
    #[serde(default)]
    pub history: Option<bool>,
}

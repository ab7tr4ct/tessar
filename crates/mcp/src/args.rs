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
    /// All spaces (default): this repository, the user, and other repositories. Each hit shows its space.
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
    /// When it happens: "YYYY-MM-DD", "YYYY-MM", "YYYY" or "YYYY-MM-DD/YYYY-MM-DD". Required for event, optional for statement. On an attribute it is when the described thing happens, e.g. the date of a planned trip; the value holds from now.
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
    /// Attribute: define `predicate` as new even though memory found a similar one. Skips the similarity check and the universality check (no numbers or entity names in the name), so use it only after looking at the similar predicate.
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
    /// How many facts to return, 1 to 50. Default 10.
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

/// Why a fact is updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum UpdateReason {
    /// It stopped being true, or has a new value now. The old version stays in history as true until `since`.
    Changed,
    /// It was never true (a mistake). The old version is hidden from "what was true" but stays in the record of what memory knew.
    Wrong,
    /// The content must be destroyed (a secret, personal data the person wants gone). Irreversible. Takes only `fact`.
    Redact,
}

/// Input of `update`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct UpdateArgs {
    /// The handle of the fact, the leading `fNN` of a line from `recall` or `about`, for example "f12". It must be a current version.
    pub fact: String,
    /// "changed": it stopped being true or has a new value. "wrong": it was never true. "redact": destroy the content for good (only `fact` is allowed).
    pub reason: UpdateReason,
    /// reason "changed" only: the day it stopped being true, "YYYY-MM-DD" (it was not true on that day). Default now.
    #[serde(default)]
    pub since: Option<String>,
    /// The replacement sentence. Statement and event: the new text. Attribute: the new sentence ("User lives in Berlin"). Leave out `text`, `value` and `when` to only close the fact.
    #[serde(default)]
    pub text: Option<String>,
    /// Attribute only: the new value as text ("Berlin"). The subject and predicate stay.
    #[serde(default)]
    pub value: Option<String>,
    /// The replacement's `when`, same formats as in `remember`: "YYYY-MM-DD", "YYYY-MM", "YYYY" or "YYYY-MM-DD/YYYY-MM-DD". Kept from the old version when left out.
    #[serde(default)]
    pub when: Option<String>,
}

//! The mapping layer: tool requests become core calls, results become text.
//!
//! [`Tessar`] holds a [`Memory`], the two spaces every memory has (`user`
//! and one per repository) and their entity kinds. It knows nothing about
//! MCP: methods take the argument types of [`crate::args`] and return the
//! text the agent reads. `Ok` is a result, `Err` is a tool error the agent
//! should act on (nothing was written).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use tessar_core::clock::Clock;
use tessar_core::model::{
    Cardinality, EntityId, EntityKindId, Fact, FactBody, Key, Name, Origin, PredicateDef,
    PredicateId, SpaceId, SpaceKind, Timestamp, Value, ValueType,
};
use tessar_core::store::{
    BodyDraft, EntityRef, EpisodeDraft, FactDraft, Memory, ObjectDraft, Outcome, PredicateDraft,
    Recall, Remember, StoreError, View, Why,
};
use tessar_core::vocab;

use crate::args::{
    AboutArgs, Entity, Fact as FactArgs, FactKind, RecallArgs, RecallSpace, RememberArgs, Space,
};
use crate::when::{self, Tense};

/// The entity kinds defined in both spaces.
pub const KINDS: [&str; 8] = [
    "person",
    "place",
    "organization",
    "item",
    "project",
    "event",
    "topic",
    "other",
];

/// Who the facts are attributed to.
const AGENT: &str = "claude-code";

/// The most facts `recall` returns.
const MAX_LIMIT: usize = 50;

/// The memory with its two spaces, ready to serve tools.
pub struct Tessar {
    mem: Memory,
    clock: Arc<dyn Clock>,
    user: SpaceId,
    project: SpaceId,
    user_kinds: HashMap<String, EntityKindId>,
    project_kinds: HashMap<String, EntityKindId>,
}

impl Tessar {
    /// Prepares a memory for serving, idempotently: the `user` space, the
    /// project space, the entity kinds in both, the entity `user` and the
    /// starter vocabulary. Reports non-fatal trouble through `warn`.
    pub fn bootstrap(
        mem: Memory,
        clock: Arc<dyn Clock>,
        project_name: &str,
        warn: &dyn Fn(&str),
    ) -> Result<Self, StoreError> {
        let key = |s: &str| Key::new(s).map_err(StoreError::from);
        let user = mem.create_space(key("user")?, SpaceKind::LongTerm, None)?;
        // A repository called "user" must not become the user space.
        let project_key = match project_name.trim().to_lowercase().as_str() {
            "" | "user" => "user-project".to_owned(),
            other => other.to_owned(),
        };
        let project = mem.create_space(key(&project_key)?, SpaceKind::LongTerm, None)?;

        let define_kinds = |space: SpaceId| -> Result<HashMap<String, EntityKindId>, StoreError> {
            KINDS
                .iter()
                .map(|k| {
                    let id = mem.define_kind(space, Name::new(k)?)?;
                    Ok(((*k).to_owned(), id))
                })
                .collect()
        };
        let user_kinds = define_kinds(user)?;
        let project_kinds = define_kinds(project)?;

        mem.create_entity(user_kinds["person"], key("user")?)?;
        for draft in vocab::starter() {
            let name = draft.name.as_str().to_owned();
            match mem.define_predicate_unchecked(user, draft) {
                Ok(_) => {}
                Err(StoreError::PredicateConflict { .. }) => warn(&format!(
                    "starter predicate `{name}` was redefined by the user; keeping their definition"
                )),
                Err(e) => return Err(e),
            }
        }
        Ok(Self {
            mem,
            clock,
            user,
            project,
            user_kinds,
            project_kinds,
        })
    }

    fn space_id(&self, space: Space) -> SpaceId {
        match space {
            Space::User => self.user,
            Space::Project => self.project,
        }
    }

    fn kinds(&self, space: Space) -> &HashMap<String, EntityKindId> {
        match space {
            Space::User => &self.user_kinds,
            Space::Project => &self.project_kinds,
        }
    }

    fn space_name(&self, id: SpaceId) -> String {
        self.mem
            .space(id)
            .map_or_else(|| id.to_string(), |s| s.name.as_str().to_owned())
    }

    // ---- vocabulary ------------------------------------------------

    /// A predicate by name, in the user space first, then the project's.
    fn find_predicate(&self, name: &str) -> Option<PredicateDef> {
        self.mem
            .predicate_named(self.user, name)
            .or_else(|| self.mem.predicate_named(self.project, name))
    }

    /// `name (single|many): description`, one line per canonical predicate.
    pub fn vocabulary_summary(&self) -> String {
        self.mem
            .predicates()
            .iter()
            .map(|p| {
                format!(
                    "{} ({}): {}",
                    p.name,
                    cardinality_word(p.cardinality),
                    p.description
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The `vocabulary` tool: every predicate with its synonyms.
    pub fn vocabulary(&self) -> String {
        let mut out = String::new();
        for p in self.mem.predicates() {
            let _ = write!(
                out,
                "{} ({}): {}",
                p.name,
                cardinality_word(p.cardinality),
                p.description
            );
            let synonyms: Vec<String> = self
                .mem
                .synonyms_of(p.id)
                .iter()
                .map(|s| s.name.to_string())
                .collect();
            if !synonyms.is_empty() {
                let _ = write!(out, " | synonyms: {}", synonyms.join(", "));
            }
            out.push('\n');
        }
        out.trim_end().to_owned()
    }

    // ---- remember --------------------------------------------------

    /// The `remember` tool.
    pub fn remember(&self, args: RememberArgs) -> Result<String, String> {
        let space = args.space.unwrap_or(Space::Project);
        if args.facts.is_empty() {
            return Err("`facts` is empty: nothing to remember".to_owned());
        }
        let mut drafts = Vec::new();
        let mut problems = Vec::new();
        for (i, fact) in args.facts.iter().enumerate() {
            match self.draft(fact, space) {
                Ok(d) => drafts.push(d),
                Err(found) => {
                    problems.extend(found.into_iter().map(|p| format!("fact {}: {p}", i + 1)));
                }
            }
        }
        if !problems.is_empty() {
            return Err(format!(
                "Nothing was written. Fix these and call `remember` again:\n- {}",
                problems.join("\n- ")
            ));
        }
        let episode = args
            .source
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|text| EpisodeDraft {
                text: text.to_owned(),
                at: None,
                source: agent_key(),
            });
        let request = Remember {
            space: self.space_id(space),
            origin: Origin::Asserted { by: agent_key() },
            episode,
            facts: drafts,
        };
        let done = self
            .mem
            .remember(request)
            .map_err(|e| format!("Nothing was written. {}", self.store_message(&e)))?;

        let now = self.clock.now();
        let mut out = format!(
            "Remembered {} fact(s) in space `{}`:",
            done.facts.len(),
            self.space_name(self.space_id(space))
        );
        for (i, recorded) in done.facts.iter().enumerate() {
            let line = self
                .mem
                .fact(recorded.fact)
                .map_or_else(String::new, |f| self.line(&f, now, true));
            let verb = match recorded.outcome {
                Outcome::Created => "created".to_owned(),
                Outcome::Unchanged => "unchanged (already known)".to_owned(),
                Outcome::Closed => "closed".to_owned(),
                Outcome::Superseded { previous } => {
                    let was = self.mem.fact(previous).map_or_else(String::new, |p| {
                        format!(
                            " (was: {}, recorded {})",
                            self.value_of(&p.body),
                            when::ymd(p.recorded_at)
                        )
                    });
                    format!("superseded{was}")
                }
            };
            let _ = write!(out, "\n{}. {verb}: {line}", i + 1);
        }
        Ok(out)
    }

    /// Turns one fact of the request into a draft, or lists what is wrong.
    /// Defines a new predicate if the fact proposes one.
    fn draft(&self, f: &FactArgs, space: Space) -> Result<FactDraft, Vec<String>> {
        let mut problems = Vec::new();
        let text = f
            .text
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_owned);
        let when = match f.when.as_deref().map(str::trim).filter(|w| !w.is_empty()) {
            Some(w) => ok(
                &mut problems,
                when::parse_when(w).map_err(|e| format!("`when`: {e}")),
            ),
            None => None,
        };
        let mut mentions: Vec<EntityRef> = f
            .mentions
            .iter()
            .filter_map(|e| ok(&mut problems, self.resolve(e, space)))
            .collect();

        let body = match f.kind {
            FactKind::Statement => {
                if let Some(subject) = &f.subject {
                    mentions.extend(ok(&mut problems, self.resolve(subject, space)));
                }
                match &text {
                    Some(text) => Some(BodyDraft::Statement {
                        text: text.clone(),
                        when,
                    }),
                    None => ok(&mut problems, Err("a statement needs `text`".to_owned())),
                }
            }
            FactKind::Event => {
                if let Some(subject) = &f.subject
                    && let Some(r) = ok(&mut problems, self.resolve(subject, space))
                {
                    mentions.insert(0, r);
                }
                match (&text, when) {
                    (Some(text), Some(when)) => Some(BodyDraft::Event {
                        text: text.clone(),
                        participants: std::mem::take(&mut mentions),
                        when,
                    }),
                    (None, _) => ok(&mut problems, Err("an event needs `text`".to_owned())),
                    (_, None) => ok(
                        &mut problems,
                        Err("an event needs `when` (YYYY-MM-DD, YYYY-MM, YYYY or a/b)".to_owned()),
                    ),
                }
            }
            FactKind::Attribute => {
                let subject = match &f.subject {
                    Some(s) => ok(&mut problems, self.resolve(s, space)),
                    None => ok(
                        &mut problems,
                        Err("an attribute needs `subject`".to_owned()),
                    ),
                };
                let value = f
                    .value
                    .as_deref()
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(str::to_owned);
                if value.is_none() {
                    ok::<()>(&mut problems, Err("an attribute needs `value`".to_owned()));
                }
                let predicate = match f.predicate.as_deref().map(str::trim) {
                    Some(p) if !p.is_empty() => ok(&mut problems, self.resolve_predicate(p, f)),
                    _ => ok(
                        &mut problems,
                        Err("an attribute needs `predicate`".to_owned()),
                    ),
                };
                match (subject, predicate, value) {
                    (Some(subject), Some(predicate), Some(value)) => Some(BodyDraft::Triple {
                        subject,
                        predicate,
                        object: ObjectDraft::Value(Value::Text(value)),
                        text: text.clone(),
                    }),
                    _ => None,
                }
            }
        };
        match (body, problems.is_empty()) {
            (Some(body), true) => Ok(FactDraft {
                body,
                mentions,
                valid_from: None,
                expected_version: None,
            }),
            _ => Err(problems),
        }
    }

    /// The predicate for an attribute: an existing one, a synonym declared
    /// now or a new one. Explains what to do if none can be had.
    fn resolve_predicate(&self, name: &str, f: &FactArgs) -> Result<PredicateId, String> {
        let parsed = Name::new(name).map_err(|_| {
            format!(
                "predicate `{name}` is not snake_case: use lowercase letters, digits and `_`, \
                 for example `trip_destination`"
            )
        })?;
        if let Some(existing) = self.find_predicate(name) {
            return Ok(existing.id);
        }
        let description = f
            .predicate_description
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .to_owned();

        if let Some(target) = f
            .synonym_of
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let Some(into) = self.find_predicate(target) else {
                return Err(format!(
                    "synonym_of `{target}` is not an existing predicate; call `vocabulary` for the list"
                ));
            };
            return self
                .mem
                .alias_predicate(self.user, parsed, description, into.id)
                .map_err(|e| {
                    format!(
                        "cannot make `{name}` a synonym of `{target}`: {}",
                        self.predicate_message(name, &e)
                    )
                });
        }

        if description.is_empty() {
            return Err(format!(
                "predicate `{name}` is new: give `predicate_description` (one line: what it means \
                 and what its value is), and `single: true` if a new value always replaces the old one. \
                 If an existing predicate fits, use it: call `vocabulary`"
            ));
        }
        let draft = PredicateDraft {
            name: parsed,
            description,
            object: ValueType::Text,
            cardinality: if f.single.unwrap_or(false) {
                Cardinality::Single
            } else {
                Cardinality::Many
            },
        };
        let defined = if f.force_new_predicate.unwrap_or(false) {
            self.mem.define_predicate_unchecked(self.user, draft)
        } else {
            self.mem.define_predicate(self.user, draft)
        };
        defined.map_err(|e| self.predicate_message(name, &e))
    }

    /// What a failed predicate definition means for the agent.
    fn predicate_message(&self, name: &str, e: &StoreError) -> String {
        match e {
            StoreError::PredicateSimilar { existing, .. } => {
                let (similar, description) = self
                    .mem
                    .predicate(*existing)
                    .map(|p| (p.name.to_string(), p.description))
                    .unwrap_or_else(|| (existing.to_string(), String::new()));
                format!(
                    "predicate `{name}` looks like `{similar}` ({description}); reuse `{similar}`, \
                     or set synonym_of: \"{similar}\", or force_new_predicate: true"
                )
            }
            StoreError::PredicateNotUniversal { word, reason, .. } => format!(
                "predicate `{name}` is not universal: {word:?} {reason}. Name the relation, not the thing \
                 (`trip_destination` with value \"Japan\", not `plans_trip_to_japan`)"
            ),
            StoreError::PredicateNameTooLong { max_words, .. } => format!(
                "predicate `{name}` has more than {max_words} words; shorten it to a general relation"
            ),
            StoreError::PredicateConflict { .. } => format!(
                "predicate `{name}` already exists with another type or cardinality; \
                 use it as it is (see `vocabulary`) or pick another name"
            ),
            other => format!("predicate `{name}`: {}", self.store_message(other)),
        }
    }

    /// Explains a store error to the agent.
    fn store_message(&self, e: &StoreError) -> String {
        match e {
            StoreError::DuplicatePair { subject, predicate } => format!(
                "the call sets two values of the single-valued predicate `{}` for `{}`; send only the last one",
                self.mem
                    .predicate(*predicate)
                    .map_or_else(|| predicate.to_string(), |p| p.name.to_string()),
                self.entity_key(*subject)
            ),
            StoreError::Conflict { .. } => {
                "the fact was changed by someone else meanwhile; recall it again and decide again"
                    .to_owned()
            }
            StoreError::EmptyText => "a text field is empty".to_owned(),
            StoreError::Embed(e) => format!("the embedder failed ({e}); try again"),
            StoreError::SpaceArchived(_) => "the space is archived and read-only".to_owned(),
            other => other.to_string(),
        }
    }

    // ---- entities --------------------------------------------------

    /// The space an entity lives in: `user` (person) always in the user
    /// space, else the one named, else the fact's.
    fn entity_space(e: &Entity, default: Space) -> Space {
        if is_user(e) {
            Space::User
        } else {
            e.space.unwrap_or(default)
        }
    }

    fn resolve(&self, e: &Entity, default: Space) -> Result<EntityRef, String> {
        let space = Self::entity_space(e, default);
        let kind = self.kind_id(space, &e.kind)?;
        let key =
            Key::new(&e.key).map_err(|_| format!("entity key {:?} is empty or too long", e.key))?;
        Ok(EntityRef::Key { kind, key })
    }

    fn kind_id(&self, space: Space, kind: &str) -> Result<EntityKindId, String> {
        self.kinds(space).get(kind.trim()).copied().ok_or_else(|| {
            format!(
                "unknown entity kind {:?}; use one of: {}",
                kind,
                KINDS.join(", ")
            )
        })
    }

    fn entity_key(&self, id: EntityId) -> String {
        self.mem
            .entity(id)
            .map_or_else(|| id.to_string(), |e| e.key.as_str().to_owned())
    }

    // ---- rendering -------------------------------------------------

    /// A value as text.
    fn value_of(&self, body: &FactBody) -> String {
        match body {
            FactBody::Triple { object, .. } => self.value_text(object),
            FactBody::Statement { text, .. } | FactBody::Event { text, .. } => text.clone(),
        }
    }

    fn value_text(&self, v: &Value) -> String {
        match v {
            Value::Text(t) => t.clone(),
            Value::Entity(id) => self.entity_key(*id),
            Value::Int(n) => n.to_string(),
            Value::Float(x) => x.get().to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Time(t) => when::ymd(*t),
            Value::Span(s) => when::span_text(*s),
        }
    }

    /// The sentence of a fact: its text, or `subject predicate: value`.
    fn sentence(&self, fact: &Fact) -> String {
        match &fact.body {
            FactBody::Statement { text, .. } | FactBody::Event { text, .. } => text.clone(),
            FactBody::Triple {
                subject,
                predicate,
                object,
                text,
            } => match text.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
                Some(t) => t.to_owned(),
                None => {
                    let name = self.mem.predicate(*predicate).map_or_else(
                        || predicate.to_string(),
                        |p| p.name.as_str().replace('_', " "),
                    );
                    format!(
                        "{} {name}: {}",
                        self.entity_key(*subject),
                        self.value_text(object)
                    )
                }
            },
        }
    }

    /// `[recorded] [space] sentence [current; earlier: v (date)] (tense)`.
    fn line(&self, fact: &Fact, now: Timestamp, with_earlier: bool) -> String {
        let mut line = format!(
            "[{}] [{}] {}",
            when::ymd(fact.recorded_at),
            self.space_name(fact.space),
            self.sentence(fact)
        );
        if with_earlier && matches!(fact.body, FactBody::Triple { .. }) {
            let earlier: Vec<String> = self
                .mem
                .history(fact.lineage)
                .iter()
                .filter(|h| h.version < fact.version)
                .rev()
                .map(|h| format!("{} ({})", self.value_of(&h.body), when::ymd(h.recorded_at)))
                .collect();
            if !earlier.is_empty() || !fact.is_current() {
                let state = if fact.is_current() {
                    "current"
                } else {
                    "no longer current"
                };
                if earlier.is_empty() {
                    let _ = write!(line, " [{state}]");
                } else {
                    let _ = write!(line, " [{state}; earlier: {}]", earlier.join(", "));
                }
            }
        }
        if let Some(span) = fact.body.when()
            && let Some(tense) = when::tense(span, now)
        {
            let _ = write!(line, " ({}: {})", Tense::word(tense), when::span_text(span));
        }
        line
    }

    fn why(&self, why: &Why) -> String {
        let mut parts = Vec::new();
        if let Some(rank) = why.entity_rank {
            let names: Vec<String> = why.entities.iter().map(|&e| self.entity_key(e)).collect();
            parts.push(format!("entity #{rank} ({})", names.join(", ")));
        }
        if let Some(rank) = why.keyword_rank {
            parts.push(format!("words #{rank} ({})", why.words.join(" ")));
        }
        if let Some(rank) = why.meaning_rank {
            match why.similarity {
                Some(s) => parts.push(format!("meaning #{rank} (similarity {s:.2})")),
                None => parts.push(format!("meaning #{rank}")),
            }
        }
        if parts.is_empty() {
            "other".to_owned()
        } else {
            parts.join("; ")
        }
    }

    // ---- recall and about ------------------------------------------

    /// The `recall` tool.
    pub fn recall(&self, args: RecallArgs) -> Result<String, String> {
        let query = args.query.trim();
        if query.is_empty() {
            return Err("`query` is empty".to_owned());
        }
        let view = match args
            .as_of
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            Some(day) => View::TrueAt(when::parse_as_of(day).map_err(|e| format!("`as_of`: {e}"))?),
            None => View::Current,
        };
        let spaces = match args.space.unwrap_or_default() {
            RecallSpace::Project => vec![self.project],
            RecallSpace::User => vec![self.user],
            RecallSpace::All => Vec::new(),
        };
        let limit = args.limit.unwrap_or(10).clamp(1, MAX_LIMIT);
        let recalled = self.mem.recall(&Recall {
            spaces,
            near: vec![self.project, self.user],
            view,
            limit,
            ..Recall::new(query)
        });
        let now = self.clock.now();
        let mut out = String::new();
        if recalled.hits.is_empty() {
            out.push_str("Nothing found.");
        }
        for (i, hit) in recalled.hits.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            let _ = write!(
                out,
                "{}. {} | found by: {}",
                i + 1,
                self.line(&hit.fact, now, true),
                self.why(&hit.why)
            );
        }
        if let Some(degraded) = &recalled.degraded {
            let _ = write!(
                out,
                "\nWarning: {degraded}. Results come from the other channels only."
            );
        }
        Ok(out)
    }

    /// The `about` tool.
    pub fn about(&self, args: AboutArgs) -> Result<String, String> {
        let e = &args.entity;
        let spaces: Vec<Space> = if is_user(e) {
            vec![Space::User]
        } else if let Some(space) = e.space {
            vec![space]
        } else {
            vec![Space::Project, Space::User]
        };
        let mut found = None;
        for space in spaces {
            let kind = self.kind_id(space, &e.kind)?;
            if let Some(entity) = self.mem.find_entity(self.space_id(space), kind, &e.key) {
                found = Some(entity);
                break;
            }
        }
        let Some(entity) = found else {
            return Ok(format!(
                "No {} named {:?} is known.",
                e.kind.trim(),
                e.key.trim()
            ));
        };
        let now = self.clock.now();
        let mut facts = self.mem.facts_about(entity.id, View::Current);
        let history = args.history.unwrap_or(false);
        if history {
            let mut lineages = Vec::new();
            for f in &facts {
                if !lineages.contains(&f.lineage) {
                    lineages.push(f.lineage);
                }
            }
            facts = lineages
                .into_iter()
                .flat_map(|l| self.mem.history(l))
                .collect();
        }
        facts.sort_by_key(|f| (f.recorded_at, f.id));

        let mut out = format!(
            "{} ({} in space `{}`): {} fact(s)",
            entity.key,
            e.kind.trim(),
            self.space_name(entity.space),
            facts.len()
        );
        for f in &facts {
            let mut line = self.line(f, now, !history);
            if history && matches!(f.body, FactBody::Triple { .. }) {
                let _ = match f.end.map(|end| end.ending) {
                    None => write!(line, " [version {}, current]", f.version),
                    Some(tessar_core::model::Ending::Changed { valid_to }) => {
                        write!(
                            line,
                            " [version {}, until {}]",
                            f.version,
                            when::ymd(valid_to)
                        )
                    }
                    Some(tessar_core::model::Ending::Corrected) => {
                        write!(line, " [version {}, corrected]", f.version)
                    }
                };
            }
            let _ = write!(out, "\n- {line}");
        }
        Ok(out)
    }
}

/// Whether the entity is `user`, the person: it always lives in the user
/// space.
fn is_user(e: &Entity) -> bool {
    e.key.trim().eq_ignore_ascii_case("user") && e.kind.trim() == "person"
}

/// Keeps the value of `r`, or records its problem.
fn ok<T>(problems: &mut Vec<String>, r: Result<T, String>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(p) => {
            problems.push(p);
            None
        }
    }
}

fn cardinality_word(c: Cardinality) -> &'static str {
    match c {
        Cardinality::Single => "single",
        Cardinality::Many => "many",
    }
}

fn agent_key() -> Key {
    Key::new(AGENT).expect("the agent name is a valid key")
}

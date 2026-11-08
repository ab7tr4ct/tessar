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
    Cardinality, Ending, EntityId, EntityKindId, Fact, FactBody, FactId, Key, Name, Origin,
    PredicateDef, PredicateId, SpaceId, SpaceKind, TimeSpan, Timestamp, Value, ValueType,
};
use tessar_core::store::{
    BodyDraft, EntityRef, EpisodeDraft, FactDraft, Memory, ObjectDraft, Outcome, PredicateDraft,
    Reason, Recall, Remember, Replacement, StoreError, Update, View, Why,
};
use tessar_core::vocab;

use crate::args::{
    AboutArgs, Entity, Fact as FactArgs, FactKind, RecallArgs, RecallSpace, RememberArgs, Space,
    UpdateArgs, UpdateReason,
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
    /// The key of the project entity: the project space's name.
    project_key: Key,
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
        let project_key = key(&project_key)?;
        mem.create_entity(project_kinds["project"], project_key.clone())?;
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
            project_key,
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
        out.pop();
        out
    }

    // ---- remember --------------------------------------------------

    /// The `remember` tool.
    pub fn remember(&self, args: RememberArgs) -> Result<String, String> {
        let space = args.space.unwrap_or(Space::Project);
        if args.facts.is_empty() {
            return Err("`facts` is empty: nothing to remember".to_owned());
        }
        // Phase 1: check everything that can be checked without writing.
        let mut plan = Plan::default();
        let mut pending = Vec::new();
        let mut problems = Vec::new();
        for (i, fact) in args.facts.iter().enumerate() {
            match self.draft(fact, space, i, &mut plan) {
                Ok(p) => pending.push(p),
                Err(found) => {
                    problems.extend(found.into_iter().map(|p| format!("fact {}: {p}", i + 1)));
                }
            }
        }
        if !problems.is_empty() {
            return Err(format!(
                "Nothing was written, no predicates either. Fix these and call `remember` again:\n- {}",
                problems.join("\n- ")
            ));
        }

        // Phase 2: define the new predicates. Memory checks a new predicate
        // for near-duplicates only while defining it, so a problem found
        // here may follow predicates that are already defined; they stay.
        let mut ids = Vec::new();
        let mut kept: Vec<String> = Vec::new();
        for new in &plan.news {
            match self.define_new(new) {
                Ok(id) => {
                    ids.push(id);
                    kept.push(format!("`{}`", new.name));
                }
                Err(message) => {
                    let which: Vec<String> =
                        new.facts.iter().map(|i| (i + 1).to_string()).collect();
                    problems.push(format!("fact {}: {message}", which.join(", ")));
                }
            }
        }
        let kept_note = if kept.is_empty() {
            String::new()
        } else {
            format!(
                "\nPredicates defined and kept: {}. They are part of the vocabulary now.",
                kept.join(", ")
            )
        };
        if !problems.is_empty() {
            return Err(format!(
                "No facts were written. Fix these and call `remember` again:\n- {}{kept_note}",
                problems.join("\n- ")
            ));
        }

        // Phase 3: one transaction for the facts.
        let drafts: Vec<FactDraft> = pending.into_iter().map(|p| p.into_draft(&ids)).collect();
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
        let done = self.mem.remember(request).map_err(|e| {
            format!(
                "No facts were written. {}{kept_note}",
                self.store_message(&e)
            )
        })?;

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
                .map_or_else(String::new, |f| self.line(&f, now, true, None));
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

    /// Checks one fact of the request and prepares it, or lists what is
    /// wrong. Writes nothing: a new predicate is added to `plan`.
    fn draft(
        &self,
        f: &FactArgs,
        space: Space,
        index: usize,
        plan: &mut Plan,
    ) -> Result<Pending, Vec<String>> {
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

        let subject_ref = f
            .subject
            .as_ref()
            .and_then(|s| ok(&mut problems, self.resolve(s, space)));
        // The fact is about the person (user space) or the project (project
        // space) even when the agent does not say so.
        let own = self.own_entity(space);
        if subject_ref.as_ref() != Some(&own) && !mentions.contains(&own) {
            mentions.push(own);
        }

        let body = match f.kind {
            FactKind::Statement => {
                if let Some(r) = subject_ref
                    && !mentions.contains(&r)
                {
                    mentions.push(r);
                }
                match &text {
                    Some(text) => Some(PendingBody::Ready(BodyDraft::Statement {
                        text: text.clone(),
                        when,
                    })),
                    None => ok(&mut problems, Err("a statement needs `text`".to_owned())),
                }
            }
            FactKind::Event => {
                if let Some(r) = subject_ref {
                    mentions.retain(|m| m != &r);
                    mentions.insert(0, r);
                }
                match (&text, when) {
                    (Some(text), Some(when)) => Some(PendingBody::Ready(BodyDraft::Event {
                        text: text.clone(),
                        participants: std::mem::take(&mut mentions),
                        when,
                    })),
                    (None, _) => ok(&mut problems, Err("an event needs `text`".to_owned())),
                    (_, None) => ok(
                        &mut problems,
                        Err("an event needs `when` (YYYY-MM-DD, YYYY-MM, YYYY or a/b)".to_owned()),
                    ),
                }
            }
            FactKind::Attribute => {
                let subject = match &f.subject {
                    Some(_) => subject_ref,
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
                    problems.push("an attribute needs `value`".to_owned());
                }
                let predicate = match f.predicate.as_deref().map(str::trim) {
                    Some(p) if !p.is_empty() => {
                        ok(&mut problems, self.resolve_predicate(p, f, index, plan))
                    }
                    _ => ok(
                        &mut problems,
                        Err("an attribute needs `predicate`".to_owned()),
                    ),
                };
                match (subject, predicate, value) {
                    (Some(subject), Some(pred), Some(value)) => Some(PendingBody::Triple {
                        subject,
                        pred,
                        value,
                        text: text.clone(),
                        when,
                    }),
                    _ => None,
                }
            }
        };
        match (body, problems.is_empty()) {
            (Some(body), true) => Ok(Pending { body, mentions }),
            _ => Err(problems),
        }
    }

    /// Plans the predicate for an attribute: an existing one, or a new one
    /// (defined or declared a synonym) to create in phase 2. Explains what
    /// to do if none can be had. Writes nothing.
    fn resolve_predicate(
        &self,
        name: &str,
        f: &FactArgs,
        index: usize,
        plan: &mut Plan,
    ) -> Result<PredRef, String> {
        let parsed = Name::new(name).map_err(|_| {
            format!(
                "predicate `{name}` is not snake_case: use lowercase letters, digits and `_`, \
                 for example `trip_destination`"
            )
        })?;
        if let Some(existing) = self.find_predicate(name) {
            return Ok(PredRef::Existing(existing.id));
        }
        // The same new predicate in two facts is defined once.
        if let Some(i) = plan.news.iter().position(|n| n.name == name) {
            plan.news[i].facts.push(index);
            return Ok(PredRef::New(i));
        }
        let description = f
            .predicate_description
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .to_owned();

        let how = if let Some(target) = f
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
            NewHow::Alias { into: into.id }
        } else {
            if description.is_empty() {
                return Err(format!(
                    "predicate `{name}` is new: give `predicate_description` (one line: what it means \
                     and what its value is), and `single: true` if a new value always replaces the old one. \
                     If an existing predicate fits, use it: call `vocabulary`"
                ));
            }
            NewHow::Define {
                single: f.single.unwrap_or(false),
                force: f.force_new_predicate.unwrap_or(false),
            }
        };
        plan.news.push(NewPredicate {
            name: name.to_owned(),
            parsed,
            description,
            how,
            facts: vec![index],
        });
        Ok(PredRef::New(plan.news.len() - 1))
    }

    /// Defines or aliases a planned predicate in the user space.
    fn define_new(&self, new: &NewPredicate) -> Result<PredicateId, String> {
        let name = &new.name;
        match new.how {
            NewHow::Alias { into } => self
                .mem
                .alias_predicate(self.user, new.parsed.clone(), new.description.clone(), into)
                .map_err(|e| {
                    format!(
                        "cannot make `{name}` a synonym: {}",
                        self.predicate_message(name, &e)
                    )
                }),
            NewHow::Define { single, force } => {
                let draft = PredicateDraft {
                    name: new.parsed.clone(),
                    description: new.description.clone(),
                    object: ValueType::Text,
                    cardinality: if single {
                        Cardinality::Single
                    } else {
                        Cardinality::Many
                    },
                };
                let defined = if force {
                    self.mem.define_predicate_unchecked(self.user, draft)
                } else {
                    self.mem.define_predicate(self.user, draft)
                };
                defined.map_err(|e| self.predicate_message(name, &e))
            }
        }
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
            StoreError::NotCurrent(_) => {
                "the fact is already closed; `recall` or `about` shows its current version"
                    .to_owned()
            }
            StoreError::ShapeMismatch(_) => "a replacement must keep the kind of the fact, and \
                 for an attribute its subject and predicate. To change those, close this fact \
                 with `update` (no `text`, `value` or `when`) and `remember` a new one"
                .to_owned(),
            StoreError::ChangeBeforeValid { at, valid_from } => format!(
                "`since` {} is before the fact became true ({}); use a later date or leave `since` out",
                when::ymd(*at),
                when::ymd(*valid_from)
            ),
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

    /// The entity to write: `user` in the user space; else in the space
    /// named; else where it already exists (project first, then user);
    /// else a new one in the fact's space.
    fn resolve(&self, e: &Entity, default: Space) -> Result<EntityRef, String> {
        let key =
            Key::new(&e.key).map_err(|_| format!("entity key {:?} is empty or too long", e.key))?;
        let space = if is_user(e) || e.space.is_some() {
            Self::entity_space(e, default)
        } else {
            [Space::Project, Space::User]
                .into_iter()
                .find(|&s| self.exists(s, e))
                .unwrap_or(default)
        };
        let kind = self.kind_id(space, &e.kind)?;
        Ok(EntityRef::Key { kind, key })
    }

    /// The entity facts of `space` are about by default: `user` in the user
    /// space, the project in the project space.
    fn own_entity(&self, space: Space) -> EntityRef {
        match space {
            Space::User => EntityRef::Key {
                kind: self.user_kinds["person"],
                key: Key::new("user").expect("`user` is a valid key"),
            },
            Space::Project => EntityRef::Key {
                kind: self.project_kinds["project"],
                key: self.project_key.clone(),
            },
        }
    }

    /// Whether the entity already exists in `space`.
    fn exists(&self, space: Space, e: &Entity) -> bool {
        self.kinds(space).get(e.kind.trim()).is_some_and(|&kind| {
            self.mem
                .find_entity(self.space_id(space), kind, &e.key)
                .is_some()
        })
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
                ..
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

    /// `fNN [recorded] [space] sentence [current; earlier: v (date)] (tense)`.
    /// `fNN` is the handle `update` takes.
    fn line(&self, fact: &Fact, now: Timestamp, with_earlier: bool, as_of: Option<&str>) -> String {
        let mut line = format!(
            "{} [{}] [{}] {}",
            fact.id,
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
            let state = if fact.is_current() {
                "current".to_owned()
            } else if let (Some(day), Some(now_fact)) = (as_of, self.mem.current(fact.lineage)) {
                format!("true on {day}; now: {}", self.value_of(&now_fact.body))
            } else {
                "no longer current".to_owned()
            };
            if !earlier.is_empty() || !fact.is_current() {
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

    // ---- update ----------------------------------------------------

    /// The `update` tool.
    pub fn update(&self, args: UpdateArgs) -> Result<String, String> {
        let handle = args.fact.trim();
        let id = parse_handle(handle)
            .ok_or_else(|| format!("`fact` {handle:?} is not a handle: use the leading `fNN` of a line from `recall` or `about`, for example \"f12\""))?;
        let fact = self.mem.fact(id).ok_or_else(|| format!("no fact {id}"))?;
        if fact.end.is_some() {
            return Err(match self.mem.current(fact.lineage) {
                Some(current) => format!(
                    "{id} is already closed; the current version is {}. Update that one",
                    current.id
                ),
                None => format!(
                    "{id} is already closed and has no current version; nothing to update. \
                     `remember` the fact again if it holds"
                ),
            });
        }

        let clean = |s: &Option<String>| {
            s.as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        };
        let (text, value, when_arg) = (clean(&args.text), clean(&args.value), clean(&args.when));
        let since = clean(&args.since);
        let reason = match (args.reason, since) {
            (UpdateReason::Changed, None) => Reason::Change { at: None },
            (UpdateReason::Changed, Some(day)) => {
                let end = when::parse_as_of(&day).map_err(|e| format!("`since`: {e}"))?;
                Reason::Change {
                    at: Some(when::day_start(end)),
                }
            }
            (UpdateReason::Wrong, None) => Reason::Correction,
            (UpdateReason::Wrong, Some(_)) => {
                return Err(
                    "`since` goes with reason \"changed\"; a \"wrong\" fact was never true, \
                     so it has no end date"
                        .to_owned(),
                );
            }
        };
        let when_new = when_arg
            .as_deref()
            .map(|w| when::parse_when(w).map_err(|e| format!("`when`: {e}")))
            .transpose()?;

        let replacement = if text.is_none() && value.is_none() && when_new.is_none() {
            None
        } else {
            Some(Replacement {
                body: self.replacement_body(&fact, text, value, when_new)?,
                mentions: self.replacement_mentions(&fact),
            })
        };
        let had_replacement = replacement.is_some();
        let recorded = self
            .mem
            .update(Update {
                lineage: fact.lineage,
                expected_version: fact.version,
                reason,
                replacement,
                origin: Origin::Asserted { by: agent_key() },
            })
            .map_err(|e| format!("Nothing was changed. {}", self.store_message(&e)))?;

        let now = self.clock.now();
        let new_line = |id: FactId| {
            self.mem
                .fact(id)
                .map_or_else(|| id.to_string(), |f| self.line(&f, now, true, None))
        };
        let wrong = args.reason == UpdateReason::Wrong;
        Ok(match recorded.outcome {
            Outcome::Superseded { previous } if wrong => format!(
                "corrected: {previous} was never true. Replaced by: {}",
                new_line(recorded.fact)
            ),
            Outcome::Superseded { previous } => {
                let until = self
                    .valid_to(previous)
                    .map_or_else(String::new, |d| format!(" ({previous} was true until {d})"));
                format!("superseded: {}{until}", new_line(recorded.fact))
            }
            Outcome::Closed if wrong => format!("corrected: {id} (was never true)"),
            Outcome::Closed => format!(
                "closed: {id} (true until {})",
                self.valid_to(id).unwrap_or_else(|| when::ymd(now))
            ),
            Outcome::Unchanged => {
                let hint = if matches!(fact.body, FactBody::Triple { .. }) && had_replacement {
                    " An attribute changes with a new `value` (or `when`); a new `text` alone is not a new version."
                } else {
                    ""
                };
                format!("unchanged: {id} already says that; nothing was written.{hint}")
            }
            Outcome::Created => format!("created: {}", new_line(recorded.fact)),
        })
    }

    /// The mentions of a replacement: the old ones, plus the entity the
    /// space is about (the person or the project) unless the fact already
    /// names it. Facts of other repositories keep their mentions as they are.
    fn replacement_mentions(&self, fact: &Fact) -> Vec<EntityRef> {
        let mut mentions: Vec<EntityRef> =
            fact.mentions.iter().map(|&e| EntityRef::Id(e)).collect();
        let space = if fact.space == self.user {
            Space::User
        } else if fact.space == self.project {
            Space::Project
        } else {
            return mentions;
        };
        let (kind, key) = match space {
            Space::User => (self.user_kinds["person"], "user"),
            Space::Project => (self.project_kinds["project"], self.project_key.as_str()),
        };
        if let Some(own) = self.mem.find_entity(fact.space, kind, key)
            && !fact.entities().contains(&own.id)
        {
            mentions.push(EntityRef::Id(own.id));
        }
        mentions
    }

    /// The day a version stopped being true, if it was closed as changed.
    fn valid_to(&self, id: FactId) -> Option<String> {
        match self.mem.fact(id)?.end?.ending {
            Ending::Changed { valid_to } => Some(when::ymd(valid_to)),
            Ending::Corrected => None,
        }
    }

    /// The body of an update's replacement: the old one with the given
    /// fields swapped.
    fn replacement_body(
        &self,
        fact: &Fact,
        text: Option<String>,
        value: Option<String>,
        when_new: Option<TimeSpan>,
    ) -> Result<BodyDraft, String> {
        let handle = fact.id;
        if value.is_some() && !matches!(fact.body, FactBody::Triple { .. }) {
            return Err(format!(
                "`value` is for attributes, and {handle} is not one; put the new sentence in `text`"
            ));
        }
        Ok(match &fact.body {
            FactBody::Statement { text: old, when } => BodyDraft::Statement {
                text: text.unwrap_or_else(|| old.clone()),
                when: when_new.or(*when),
            },
            FactBody::Event {
                text: old,
                participants,
                when,
            } => BodyDraft::Event {
                text: text.unwrap_or_else(|| old.clone()),
                participants: participants.iter().map(|&e| EntityRef::Id(e)).collect(),
                when: when_new.unwrap_or(*when),
            },
            FactBody::Triple {
                subject,
                predicate,
                object,
                text: old,
                when,
            } => BodyDraft::Triple {
                subject: EntityRef::Id(*subject),
                predicate: *predicate,
                object: ObjectDraft::Value(value.map_or_else(|| object.clone(), Value::Text)),
                text: text.or_else(|| old.clone()),
                when: when_new.or(*when),
            },
        })
    }

    // ---- recall and about ------------------------------------------

    /// The `recall` tool.
    pub fn recall(&self, args: RecallArgs) -> Result<String, String> {
        let query = args.query.trim();
        if query.is_empty() {
            return Err("`query` is empty".to_owned());
        }
        let day = args
            .as_of
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let view = match day {
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
                self.line(&hit.fact, now, true, day),
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

    /// Every version of every fact about an entity, closed and corrected
    /// ones included. The store has no such read, so this walks the dense
    /// fact ids.
    fn all_facts_about(&self, entity: EntityId) -> Vec<Fact> {
        (0..)
            .map_while(|n| parse_handle(&format!("f{n}")).and_then(|id| self.mem.fact(id)))
            .filter(|f| f.entities().contains(&entity))
            .collect()
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
        let history = args.history.unwrap_or(false);
        let mut facts = if history {
            self.all_facts_about(entity.id)
        } else {
            self.mem.facts_about(entity.id, View::Current)
        };
        facts.sort_by_key(|f| (f.recorded_at, f.id));

        let mut out = format!(
            "{} ({} in space `{}`): {} fact(s)",
            entity.key,
            e.kind.trim(),
            self.space_name(entity.space),
            facts.len()
        );
        for f in &facts {
            let mut line = self.line(f, now, !history, None);
            if history {
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

/// Predicates a `remember` call will have to create, found in phase 1.
#[derive(Default)]
struct Plan {
    news: Vec<NewPredicate>,
}

struct NewPredicate {
    name: String,
    parsed: Name,
    description: String,
    how: NewHow,
    /// The facts (0-based) that use it.
    facts: Vec<usize>,
}

enum NewHow {
    Define { single: bool, force: bool },
    Alias { into: PredicateId },
}

/// A predicate in a prepared fact: existing, or the n-th of the plan.
#[derive(Clone, Copy)]
enum PredRef {
    Existing(PredicateId),
    New(usize),
}

enum PendingBody {
    Ready(BodyDraft),
    Triple {
        subject: EntityRef,
        pred: PredRef,
        value: String,
        text: Option<String>,
        when: Option<TimeSpan>,
    },
}

/// A checked fact that waits for its predicate to exist.
struct Pending {
    body: PendingBody,
    mentions: Vec<EntityRef>,
}

impl Pending {
    /// The draft, given the ids of the plan's new predicates.
    fn into_draft(self, ids: &[PredicateId]) -> FactDraft {
        let body = match self.body {
            PendingBody::Ready(body) => body,
            PendingBody::Triple {
                subject,
                pred,
                value,
                text,
                when,
            } => BodyDraft::Triple {
                subject,
                predicate: match pred {
                    PredRef::Existing(id) => id,
                    PredRef::New(i) => ids[i],
                },
                object: ObjectDraft::Value(Value::Text(value)),
                text,
                when,
            },
        };
        FactDraft {
            body,
            mentions: self.mentions,
            valid_from: None,
            expected_version: None,
        }
    }
}

/// The id in a handle like `f12`.
fn parse_handle(s: &str) -> Option<FactId> {
    use serde::Deserialize as _;
    use serde::de::value::{Error, U32Deserializer};
    let digits = s.strip_prefix(['f', 'F'])?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // Ids are plain numbers on the wire; the store checks the fact exists.
    FactId::deserialize(U32Deserializer::<Error>::new(digits.parse().ok()?)).ok()
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

//! Planning a write: check a request against the state and turn it into
//! operations, without changing anything.
//!
//! A [`Planner`] runs under the writer lock, so nothing else changes the
//! state between planning and applying. It mints ids as the vector lengths
//! the new records will get, and keeps a small overlay of what the current
//! transaction has already created (new entities), so that two facts about
//! the same new entity resolve to one entity.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use super::draft::{
    BodyDraft, EntityRef, EpisodeDraft, FactDraft, ObjectDraft, Outcome, Reason, Recorded,
    Remember, Remembered, Update,
};
use super::error::StoreError;
use super::op::Op;
use super::state::{Lens, State};
use crate::model::{
    Cardinality, Ending, Entity, EntityId, EntityKindDef, EntityKindId, Episode, EpisodeId, Fact,
    FactBody, FactEnd, FactId, Key, Lineage, Name, Origin, PredicateId, Space, SpaceId, SpaceKind,
    SpaceState, Timestamp, Value,
};

/// Planning of predicate definitions and merges lives in
/// [`super::predicate`]; it shares these fields.
pub(crate) struct Planner<'s> {
    pub(super) state: &'s State,
    pub(super) at: Timestamp,
    pub(super) ops: Vec<Op>,
    pub(super) next: Lens,
    /// Entities created by this transaction: (space, kind, name) → id.
    new_entities: HashMap<(SpaceId, EntityKindId, String), EntityId>,
    new_entity_kinds: HashMap<EntityId, EntityKindId>,
    /// Single-valued (subject, predicate) pairs set by this transaction.
    single_pairs: HashSet<(EntityId, PredicateId)>,
}

impl<'s> Planner<'s> {
    pub(crate) fn new(state: &'s State, at: Timestamp) -> Self {
        Self {
            state,
            at,
            ops: Vec::new(),
            next: state.len(),
            new_entities: HashMap::new(),
            new_entity_kinds: HashMap::new(),
            single_pairs: HashSet::new(),
        }
    }

    /// The planned operations.
    pub(crate) fn into_ops(self) -> Vec<Op> {
        self.ops
    }

    pub(crate) fn create_space(
        &mut self,
        name: Key,
        kind: SpaceKind,
        parent: Option<SpaceId>,
    ) -> Result<SpaceId, StoreError> {
        let state = self.state;
        if let Some(id) = state.space_named(&name.normalized()) {
            let existing = state.space(id).ok_or(StoreError::UnknownSpace(id))?;
            return if existing.kind == kind && existing.parent == parent {
                Ok(id)
            } else {
                Err(StoreError::NameTaken(format!("space {name}")))
            };
        }
        if let Some(parent) = parent {
            state
                .space(parent)
                .ok_or(StoreError::UnknownSpace(parent))?;
        }
        let id = mint(&mut self.next.spaces, SpaceId::new)?;
        self.ops.push(Op::CreateSpace(Space {
            id,
            name,
            parent,
            kind,
            state: SpaceState::Open,
            created_at: self.at,
        }));
        Ok(id)
    }

    pub(crate) fn define_kind(
        &mut self,
        space: SpaceId,
        name: Name,
    ) -> Result<EntityKindId, StoreError> {
        self.open_space(space)?;
        if let Some(id) = self.state.kind_named(space, &name) {
            return Ok(id);
        }
        let id = mint(&mut self.next.kinds, EntityKindId::new)?;
        self.ops
            .push(Op::DefineKind(EntityKindDef { id, space, name }));
        Ok(id)
    }

    pub(crate) fn add_alias(&mut self, entity: EntityId, alias: Key) -> Result<(), StoreError> {
        let state = self.state;
        let e = state
            .entity(entity)
            .ok_or(StoreError::UnknownEntity(entity))?;
        self.open_space(e.space)?;
        match state.entity_named(e.space, e.kind, &alias.normalized()) {
            Some(id) if id == entity => Ok(()),
            Some(_) => Err(StoreError::NameTaken(format!("name {alias}"))),
            None => {
                self.ops.push(Op::AddAlias { entity, alias });
                Ok(())
            }
        }
    }

    pub(crate) fn remember(&mut self, request: Remember) -> Result<Remembered, StoreError> {
        let Remember {
            space,
            origin,
            episode,
            facts,
        } = request;
        self.open_space(space)?;
        let episode = match episode {
            Some(draft) => Some(self.add_episode(space, draft)?),
            None => None,
        };
        let mut recorded = Vec::with_capacity(facts.len());
        for draft in facts {
            recorded.push(self.remember_fact(space, &origin, episode, draft)?);
        }
        Ok(Remembered {
            episode,
            facts: recorded,
        })
    }

    pub(crate) fn update(&mut self, update: Update) -> Result<Recorded, StoreError> {
        let state = self.state;
        let head = state
            .head(update.lineage)
            .ok_or(StoreError::UnknownLineage(update.lineage))?;
        self.open_space(head.space)?;
        if !head.is_current() {
            return Err(StoreError::NotCurrent(update.lineage));
        }
        check_expected(
            Some(update.expected_version),
            Some(update.lineage),
            Some(head.version),
        )?;
        let (ending, valid_from) = match update.reason {
            Reason::Change { at } => {
                let at = at.unwrap_or(self.at);
                if at < head.valid_from {
                    return Err(StoreError::ChangeBeforeValid {
                        at,
                        valid_from: head.valid_from,
                    });
                }
                (Ending::Changed { valid_to: at }, at)
            }
            Reason::Correction => (Ending::Corrected, head.valid_from),
        };

        let Some(replacement) = update.replacement else {
            self.close(head.id, ending, None);
            return Ok(Recorded {
                fact: head.id,
                outcome: Outcome::Closed,
            });
        };
        let body = self.body(replacement.body)?;
        if !self.same_shape(&head.body, &body) {
            return Err(StoreError::ShapeMismatch(update.lineage));
        }
        if self.same_content(&body, &head.body) {
            return Ok(Recorded {
                fact: head.id,
                outcome: Outcome::Unchanged,
            });
        }
        let mentions = self.resolve_all(&replacement.mentions)?;
        let id = self.push_version(
            head,
            head.space,
            body,
            mentions,
            None,
            &update.origin,
            valid_from,
        )?;
        self.close(head.id, ending, Some(id));
        Ok(Recorded {
            fact: id,
            outcome: Outcome::Superseded { previous: head.id },
        })
    }

    /// Plans one fact of a write, lying in `space`.
    ///
    /// For a single-valued pair, the replacement rule works across spaces:
    /// the pair index is global, so a new value written to `space`
    /// supersedes the current version written to another one, and the new
    /// version lies in `space`.
    ///
    /// A write with the same value and another `text` is
    /// [`Outcome::Unchanged`] and does not store the new text (see
    /// [`FactBody::same_content`]).
    fn remember_fact(
        &mut self,
        space: SpaceId,
        origin: &Origin,
        episode: Option<EpisodeId>,
        draft: FactDraft,
    ) -> Result<Recorded, StoreError> {
        let state = self.state;
        let body = self.body(draft.body)?;
        let mentions = self.resolve_all(&draft.mentions)?;
        let valid_from = draft.valid_from.unwrap_or(self.at);

        if let FactBody::Triple {
            subject, predicate, ..
        } = &body
        {
            let (subject, predicate) = (*subject, *predicate);
            let def = state
                .predicate(predicate)
                .ok_or(StoreError::UnknownPredicate(predicate))?;
            let lineages = state.lineages_of_pair(subject, predicate);
            match def.cardinality {
                Cardinality::Single => {
                    if !self.single_pairs.insert((subject, predicate)) {
                        return Err(StoreError::DuplicatePair { subject, predicate });
                    }
                    // A single-valued pair has at most one lineage: new
                    // values become its new versions.
                    if let Some(&lineage) = lineages.first() {
                        let head = state.head(lineage).ok_or_else(|| {
                            StoreError::Internal(format!("{lineage} has no head"))
                        })?;
                        if !head.is_current() {
                            check_expected(draft.expected_version, Some(lineage), None)?;
                            let id = self.push_version(
                                head, space, body, mentions, episode, origin, valid_from,
                            )?;
                            return Ok(Recorded {
                                fact: id,
                                outcome: Outcome::Created,
                            });
                        }
                        check_expected(draft.expected_version, Some(lineage), Some(head.version))?;
                        if self.same_content(&head.body, &body) {
                            return Ok(Recorded {
                                fact: head.id,
                                outcome: Outcome::Unchanged,
                            });
                        }
                        if valid_from < head.valid_from {
                            return Err(StoreError::ChangeBeforeValid {
                                at: valid_from,
                                valid_from: head.valid_from,
                            });
                        }
                        let id = self.push_version(
                            head, space, body, mentions, episode, origin, valid_from,
                        )?;
                        self.close(
                            head.id,
                            Ending::Changed {
                                valid_to: valid_from,
                            },
                            Some(id),
                        );
                        return Ok(Recorded {
                            fact: id,
                            outcome: Outcome::Superseded { previous: head.id },
                        });
                    }
                    check_expected(draft.expected_version, None, None)?;
                }
                Cardinality::Many => {
                    let same = lineages
                        .iter()
                        .filter_map(|&l| state.head(l))
                        .find(|h| h.is_current() && self.same_content(&h.body, &body));
                    if let Some(existing) = same {
                        return Ok(Recorded {
                            fact: existing.id,
                            outcome: Outcome::Unchanged,
                        });
                    }
                }
            }
        }

        let id = mint(&mut self.next.facts, FactId::new)?;
        self.ops.push(Op::AddFact(Fact {
            id,
            space,
            lineage: Lineage::starting_at(id),
            version: NonZeroU32::MIN,
            prev: None,
            body,
            mentions,
            episode,
            origin: origin.clone(),
            recorded_at: self.at,
            valid_from,
            end: None,
        }));
        Ok(Recorded {
            fact: id,
            outcome: Outcome::Created,
        })
    }

    fn add_episode(
        &mut self,
        space: SpaceId,
        draft: EpisodeDraft,
    ) -> Result<EpisodeId, StoreError> {
        let text = non_empty(draft.text)?;
        let id = mint(&mut self.next.episodes, EpisodeId::new)?;
        self.ops.push(Op::AddEpisode(Episode {
            id,
            space,
            text,
            at: draft.at.unwrap_or(self.at),
            source: draft.source,
            recorded_at: self.at,
        }));
        Ok(id)
    }

    /// Plans the next version of `head`'s lineage, lying in `space`.
    #[allow(clippy::too_many_arguments)]
    fn push_version(
        &mut self,
        head: &Fact,
        space: SpaceId,
        body: FactBody,
        mentions: Vec<EntityId>,
        episode: Option<EpisodeId>,
        origin: &Origin,
        valid_from: Timestamp,
    ) -> Result<FactId, StoreError> {
        let version = head
            .version
            .checked_add(1)
            .ok_or(StoreError::TooManyRecords)?;
        let id = mint(&mut self.next.facts, FactId::new)?;
        self.ops.push(Op::AddFact(Fact {
            id,
            space,
            lineage: head.lineage,
            version,
            prev: Some(head.id),
            body,
            mentions,
            episode,
            origin: origin.clone(),
            recorded_at: self.at,
            valid_from,
            end: None,
        }));
        Ok(id)
    }

    fn close(&mut self, fact: FactId, ending: Ending, next: Option<FactId>) {
        self.ops.push(Op::CloseFact {
            fact,
            end: FactEnd {
                ending,
                recorded_at: self.at,
                next,
            },
        });
    }

    /// Checks a fact draft's content and resolves its entity references.
    fn body(&mut self, draft: BodyDraft) -> Result<FactBody, StoreError> {
        let state = self.state;
        match draft {
            BodyDraft::Statement { text, when } => Ok(FactBody::Statement {
                text: non_empty(text)?,
                when,
            }),
            BodyDraft::Event {
                text,
                participants,
                when,
            } => Ok(FactBody::Event {
                text: non_empty(text)?,
                participants: self.resolve_all(&participants)?,
                when,
            }),
            BodyDraft::Triple {
                subject,
                predicate,
                object,
                text,
            } => {
                let subject = self.resolve(&subject)?;
                // A synonym is written under its canonical predicate; the
                // predicate may belong to any space.
                let predicate = state.canonical(predicate);
                let def = state
                    .predicate(predicate)
                    .ok_or(StoreError::UnknownPredicate(predicate))?;
                let object = match object {
                    ObjectDraft::Value(Value::Entity(id)) => {
                        Value::Entity(self.resolve(&EntityRef::Id(id))?)
                    }
                    ObjectDraft::Value(value) => value,
                    ObjectDraft::Entity(entity) => Value::Entity(self.resolve(&entity)?),
                };
                if !def.object.accepts(&object, |e| self.kind_of(e)) {
                    return Err(StoreError::TypeMismatch {
                        predicate,
                        expected: def.object,
                    });
                }
                let text = match text {
                    Some(text) => Some(non_empty(text)?),
                    None => None,
                };
                Ok(FactBody::Triple {
                    subject,
                    predicate,
                    object,
                    text,
                })
            }
        }
    }

    fn resolve_all(&mut self, refs: &[EntityRef]) -> Result<Vec<EntityId>, StoreError> {
        refs.iter().map(|r| self.resolve(r)).collect()
    }

    /// Finds the entity, or plans creating it. Within one transaction the
    /// same key always resolves to the same entity.
    ///
    /// An entity may lie in any space: the space of a fact says where the
    /// fact lies, not what it may refer to. A new entity is created in the
    /// space of its kind.
    fn resolve(&mut self, entity: &EntityRef) -> Result<EntityId, StoreError> {
        let state = self.state;
        match entity {
            EntityRef::Id(id) => {
                state.entity(*id).ok_or(StoreError::UnknownEntity(*id))?;
                Ok(*id)
            }
            EntityRef::Key { kind, key } => {
                let space = state
                    .kind(*kind)
                    .ok_or(StoreError::UnknownKind(*kind))?
                    .space;
                let name = key.normalized();
                if let Some(id) = state.entity_named(space, *kind, &name) {
                    return Ok(id);
                }
                let slot = (space, *kind, name);
                if let Some(&id) = self.new_entities.get(&slot) {
                    return Ok(id);
                }
                self.open_space(space)?;
                let id = mint(&mut self.next.entities, EntityId::new)?;
                self.ops.push(Op::CreateEntity(Entity {
                    id,
                    space,
                    kind: *kind,
                    key: key.clone(),
                    aliases: Vec::new(),
                    created_at: self.at,
                }));
                self.new_entities.insert(slot, id);
                self.new_entity_kinds.insert(id, *kind);
                Ok(id)
            }
        }
    }

    /// Finds or creates the entity of `kind` with `key` (or that alias).
    pub(crate) fn create_entity(
        &mut self,
        kind: EntityKindId,
        key: Key,
    ) -> Result<EntityId, StoreError> {
        self.resolve(&EntityRef::Key { kind, key })
    }

    /// [`FactBody::same_content`] with the predicates of triples compared
    /// by their canonical predicate, so a synonym written before a merge
    /// equals its canonical.
    fn same_content(&self, old: &FactBody, new: &FactBody) -> bool {
        match (old, new) {
            (
                FactBody::Triple {
                    predicate: p1,
                    subject: s1,
                    object: o1,
                    ..
                },
                FactBody::Triple {
                    predicate: p2,
                    subject: s2,
                    object: o2,
                    ..
                },
            ) => s1 == s2 && o1 == o2 && self.state.canonical(*p1) == self.state.canonical(*p2),
            _ => old.same_content(new),
        }
    }

    /// Whether a replacement keeps the kind, subject and (canonical)
    /// predicate.
    fn same_shape(&self, old: &FactBody, new: &FactBody) -> bool {
        match (old, new) {
            (FactBody::Statement { .. }, FactBody::Statement { .. })
            | (FactBody::Event { .. }, FactBody::Event { .. }) => true,
            (
                FactBody::Triple {
                    subject: s1,
                    predicate: p1,
                    ..
                },
                FactBody::Triple {
                    subject: s2,
                    predicate: p2,
                    ..
                },
            ) => s1 == s2 && self.state.canonical(*p1) == self.state.canonical(*p2),
            _ => false,
        }
    }

    fn kind_of(&self, entity: EntityId) -> Option<EntityKindId> {
        self.state
            .entity(entity)
            .map(|e| e.kind)
            .or_else(|| self.new_entity_kinds.get(&entity).copied())
    }

    pub(super) fn open_space(&self, space: SpaceId) -> Result<(), StoreError> {
        let s = self
            .state
            .space(space)
            .ok_or(StoreError::UnknownSpace(space))?;
        match s.state {
            SpaceState::Open => Ok(()),
            SpaceState::Archived => Err(StoreError::SpaceArchived(space)),
        }
    }
}

/// The id of the next record in a vector of length `*next`.
pub(super) fn mint<I>(next: &mut usize, make: fn(u32) -> I) -> Result<I, StoreError> {
    let raw = u32::try_from(*next).map_err(|_| StoreError::TooManyRecords)?;
    *next += 1;
    Ok(make(raw))
}

/// Optimistic concurrency: the writer must have seen the current version.
fn check_expected(
    expected: Option<NonZeroU32>,
    lineage: Option<Lineage>,
    current: Option<NonZeroU32>,
) -> Result<(), StoreError> {
    match expected {
        Some(expected) if Some(expected) != current => Err(StoreError::Conflict {
            lineage,
            expected,
            current,
        }),
        _ => Ok(()),
    }
}

fn non_empty(text: String) -> Result<String, StoreError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        Err(StoreError::EmptyText)
    } else if trimmed.len() == text.len() {
        Ok(text)
    } else {
        Ok(trimmed.to_owned())
    }
}

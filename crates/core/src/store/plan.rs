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

/// The parts of a write of a single-valued fact that do not depend on the
/// current state.
#[derive(Clone, Copy)]
struct SingleWrite<'a> {
    space: SpaceId,
    origin: &'a Origin,
    episode: Option<EpisodeId>,
    expected_version: Option<NonZeroU32>,
    valid_from: Timestamp,
    /// The `valid_from` the draft gave, if it gave one.
    dated: Option<Timestamp>,
    /// Not resolved yet: a write that changes nothing creates no entity.
    mentions: &'a [EntityRef],
}

/// What a write of a single-valued fact came to.
enum Resolved {
    /// Planned and finished.
    Done(Recorded),
    /// Starts a lineage of its own, beside the asserted current head.
    Beside { asserted: FactId },
    /// Starts a lineage of its own; nothing was current.
    Fresh,
}

fn is_asserted(origin: &Origin) -> bool {
    match origin {
        Origin::Asserted { .. } => true,
        Origin::Inferred { .. } => false,
    }
}

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

    /// Plans the permanent removal of a fact version's content. A current
    /// version is closed first as a correction without a successor, so a
    /// single-valued pair is free for a new value and a closed version is
    /// never left looking current.
    ///
    /// Works in an archived space too: removing something that must not
    /// be kept is not a write of new memory.
    pub(crate) fn redact(&mut self, fact: FactId) -> Result<(), StoreError> {
        let f = self.state.fact(fact).ok_or(StoreError::UnknownFact(fact))?;
        if f.is_redacted() {
            return Err(StoreError::AlreadyRedacted(fact));
        }
        if f.is_current() {
            self.close(fact, Ending::Corrected, None);
        }
        self.ops.push(Op::Redact { fact });
        Ok(())
    }

    /// Plans blanking the text of an episode.
    pub(crate) fn redact_episode(&mut self, episode: EpisodeId) -> Result<(), StoreError> {
        let e = self
            .state
            .episode(episode)
            .ok_or(StoreError::UnknownEpisode(episode))?;
        if e.text.is_empty() {
            return Err(StoreError::EpisodeAlreadyRedacted(episode));
        }
        self.ops.push(Op::RedactEpisode { episode });
        Ok(())
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
    /// After a merge a single-valued pair may have several current values
    /// (one per lineage). The next write resolves the contradiction: the
    /// head recorded last (ties: the highest id) continues with the new
    /// version, and the other current heads are closed as changed at the
    /// same moment, with no successor. A pair with no current head
    /// continues its latest lineage; the new version may not start before
    /// the earlier values of the pair stopped being true
    /// ([`StoreError::ChangeBeforeValid`]). Corrected and redacted
    /// versions do not count: they are true in no view.
    ///
    /// An [`Origin::Inferred`] write never replaces an [`Origin::Asserted`]
    /// current value. If it differs from one, it continues a current
    /// inferred head if there is one (closing only that), and otherwise
    /// starts a lineage of its own beside the asserted value
    /// ([`Outcome::Conflicting`]). An asserted write closes all current
    /// heads as above, and continues the latest asserted one if there is
    /// one.
    ///
    /// A value that is already current in a head the write may keep (any
    /// equal head for an inferred write, an asserted one for an asserted
    /// write; an asserted head first, then the one recorded last) gets no
    /// new version. The write closes the other heads it may close, with
    /// no successor: [`Outcome::Resolved`], or [`Outcome::Unchanged`] if
    /// there are none. `expected_version` is then checked against the
    /// kept head, and mentions are not resolved.
    ///
    /// An asserted write of a value that is current only as inferred
    /// confirms it with a new asserted version
    /// ([`Outcome::Superseded`]). Beside an asserted head of another
    /// value this is an ordinary change of the asserted lineage. Alone,
    /// the new version follows the inferred head and, unless the draft
    /// is dated, starts when the inferred one did; the inferred head is
    /// closed as changed at the later of the two starts, so it is true
    /// at no time and a `TrueAt` read shows the value once, with no
    /// change in it, while `KnownAt` and the history still show that it
    /// was inferred first.
    ///
    /// A new version is dated against the version it follows only
    /// ([`StoreError::ChangeBeforeValid`]); a head closed with no
    /// successor stops at the new start or at its own, whichever is
    /// later.
    ///
    /// A write with the same value and another `text` is
    /// [`Outcome::Unchanged`] and does not store the new text (see
    /// [`FactBody::same_content`]). An unchanged write plans nothing, not
    /// even the entities its mentions name, and its mentions are not
    /// checked at all: an unknown entity id or kind in them goes unnoticed.
    fn remember_fact(
        &mut self,
        space: SpaceId,
        origin: &Origin,
        episode: Option<EpisodeId>,
        draft: FactDraft,
    ) -> Result<Recorded, StoreError> {
        let state = self.state;
        let body = self.body(draft.body)?;
        let valid_from = draft.valid_from.unwrap_or(self.at);
        let mut outcome = Outcome::Created;

        if let FactBody::Triple {
            subject, predicate, ..
        } = &body
        {
            let (subject, predicate) = (*subject, *predicate);
            let def = state
                .predicate(predicate)
                .ok_or(StoreError::UnknownPredicate(predicate))?;
            match def.cardinality {
                Cardinality::Single => {
                    if !self.single_pairs.insert((subject, predicate)) {
                        return Err(StoreError::DuplicatePair { subject, predicate });
                    }
                    let write = SingleWrite {
                        space,
                        origin,
                        episode,
                        expected_version: draft.expected_version,
                        valid_from,
                        dated: draft.valid_from,
                        mentions: &draft.mentions,
                    };
                    match self.resolve_single(subject, predicate, &write, &body)? {
                        Resolved::Done(recorded) => return Ok(recorded),
                        Resolved::Beside { asserted } => {
                            outcome = Outcome::Conflicting { current: asserted };
                        }
                        Resolved::Fresh => {}
                    }
                }
                Cardinality::Many => {
                    let same = state
                        .lineages_of_pair(subject, predicate)
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

        // Only now, past every `Unchanged` return: resolving a mention may
        // plan a new entity.
        let mentions = self.resolve_all(&draft.mentions)?;
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
        Ok(Recorded { fact: id, outcome })
    }

    /// Resolves a write of a value to a single-valued pair against its
    /// current heads (see [`Self::remember_fact`] for the rules).
    fn resolve_single(
        &mut self,
        subject: EntityId,
        predicate: PredicateId,
        write: &SingleWrite<'_>,
        body: &FactBody,
    ) -> Result<Resolved, StoreError> {
        let state = self.state;
        let SingleWrite {
            space,
            origin,
            episode,
            expected_version,
            valid_from,
            dated,
            mentions,
        } = *write;
        let heads: Vec<&'s Fact> = state
            .lineages_of_pair(subject, predicate)
            .iter()
            .map(|&l| {
                state
                    .head(l)
                    .ok_or_else(|| StoreError::Internal(format!("{l} has no head")))
            })
            .collect::<Result<_, _>>()?;
        let latest =
            |facts: &[&'s Fact]| facts.iter().copied().max_by_key(|h| (h.recorded_at, h.id));
        let current: Vec<&'s Fact> = heads.iter().copied().filter(|h| h.is_current()).collect();

        if current.is_empty() {
            // No current value: the latest lineage continues, if any.
            let Some(head) = latest(&heads) else {
                check_expected(expected_version, None, None)?;
                return Ok(Resolved::Fresh);
            };
            check_expected(expected_version, Some(head.lineage), None)?;
            check_lower_bound(state, &heads, valid_from)?;
            // `head` was closed before this version existed and an end is
            // set once, so it keeps `end.next == None`: across this gap
            // the chain is linked by `prev` only.
            let mentions = self.resolve_all(mentions)?;
            let id = self.push_version(
                head,
                space,
                body.clone(),
                mentions,
                episode,
                origin,
                valid_from,
            )?;
            return Ok(Resolved::Done(Recorded {
                fact: id,
                outcome: Outcome::Created,
            }));
        }

        let (asserted, inferred): (Vec<&'s Fact>, Vec<&'s Fact>) = current
            .iter()
            .copied()
            .partition(|h| is_asserted(&h.origin));
        // An inferred write never closes an asserted value: it may close
        // only the inferred heads, and if there are none it starts a
        // lineage of its own. Any other write may close all current heads
        // and continues the latest asserted one (else the latest of all).
        let beside_asserted = !is_asserted(origin) && !asserted.is_empty();
        let closing = if beside_asserted {
            inferred.clone()
        } else {
            current.clone()
        };
        // A head closed with no successor stops being true when the new
        // value starts, or at its own start if that is later: then it is
        // true at no time, and no end ever precedes its start.
        let stopped = |h: &Fact, at: Timestamp| Ending::Changed {
            valid_to: at.max(h.valid_from),
        };
        let equal: Vec<&'s Fact> = current
            .iter()
            .copied()
            .filter(|h| self.same_content(&h.body, body))
            .collect();

        // The value is already current in a head the write may keep (any
        // equal head for an inferred write, an asserted one for an asserted
        // write): no new version. The write only closes the other heads it
        // may close.
        let keep = equal
            .iter()
            .copied()
            .filter(|h| !is_asserted(origin) || is_asserted(&h.origin))
            .max_by_key(|h| (is_asserted(&h.origin), h.recorded_at, h.id));
        if let Some(keep) = keep {
            check_expected(expected_version, Some(keep.lineage), Some(keep.version))?;
            let others: Vec<&'s Fact> = closing
                .iter()
                .copied()
                .filter(|h| h.id != keep.id)
                .collect();
            for other in &others {
                self.close(other.id, stopped(other, valid_from), None);
            }
            let outcome = match latest(&others) {
                None => Outcome::Unchanged,
                Some(last) => Outcome::Resolved {
                    closed: last.id,
                    others: u32::try_from(others.len() - 1).unwrap_or(u32::MAX),
                },
            };
            return Ok(Resolved::Done(Recorded {
                fact: keep.id,
                outcome,
            }));
        }

        // Left in `equal`: inferred heads under an asserted write, which
        // confirms the value with a new asserted version.
        let confirmed = latest(&equal);
        let target = if beside_asserted {
            latest(&inferred)
        } else {
            latest(&asserted).or(confirmed).or_else(|| latest(&current))
        };
        let Some(head) = target else {
            // Only reachable for an inferred write beside asserted heads.
            check_expected(expected_version, None, None)?;
            let beside = latest(&asserted)
                .ok_or_else(|| StoreError::Internal("conflict without an asserted head".into()))?;
            return Ok(Resolved::Beside {
                asserted: beside.id,
            });
        };
        check_expected(expected_version, Some(head.lineage), Some(head.version))?;
        let starts = if confirmed.is_some_and(|c| c.id == head.id) {
            // The value did not change, only who vouches for it: unless
            // the write is dated, the asserted version takes over the
            // whole period of the inferred one, which then is true at no
            // time (`stopped`), so no read shows the value twice or a
            // change that did not happen. A dated confirmation is taken
            // at its word, earlier or later.
            // Reaching back before the inferred value's own start may not
            // cross an earlier value of the pair. From that start on, any
            // overlap is with another lineage: the conflict that was
            // already there, and this write is what resolves it.
            let starts = dated.unwrap_or(head.valid_from);
            if starts < head.valid_from {
                check_lower_bound(state, &heads, starts)?;
            }
            starts
        } else if valid_from < head.valid_from {
            return Err(StoreError::ChangeBeforeValid {
                at: valid_from,
                valid_from: head.valid_from,
            });
        } else {
            valid_from
        };
        let mentions = self.resolve_all(mentions)?;
        let id = self.push_version(head, space, body.clone(), mentions, episode, origin, starts)?;
        for other in closing.iter().filter(|h| h.id != head.id) {
            self.close(other.id, stopped(other, starts), None);
        }
        self.close(head.id, stopped(head, starts), Some(id));
        Ok(Resolved::Done(Recorded {
            fact: id,
            outcome: Outcome::Superseded { previous: head.id },
        }))
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
                when,
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
                    // Checked like every other text of a fact: trimmed, so
                    // "Moscow " is the value "Moscow", and never empty.
                    ObjectDraft::Value(Value::Text(text)) => Value::Text(non_empty(text)?),
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
                    when,
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
                    when: w1,
                    ..
                },
                FactBody::Triple {
                    predicate: p2,
                    subject: s2,
                    object: o2,
                    when: w2,
                    ..
                },
            ) => {
                s1 == s2
                    && o1 == o2
                    && w1 == w2
                    && self.state.canonical(*p1) == self.state.canonical(*p2)
            }
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

/// A new version of a pair may not start before an earlier value of the
/// pair ended, or a `TrueAt` read would show two values. Only versions
/// closed as changed were ever true: a corrected one was not, and a
/// redacted one is in no view, so neither bounds anything. Every version
/// of every lineage of the pair counts, not the heads alone: under a
/// corrected head lies the version it replaced. A close dated in the
/// future bounds too; an end is set once and cannot be pulled back, so
/// until then the pair accepts only later values.
fn check_lower_bound(state: &State, heads: &[&Fact], starts: Timestamp) -> Result<(), StoreError> {
    let floor = heads
        .iter()
        .flat_map(|h| std::iter::successors(Some(*h), |f| f.prev.and_then(|p| state.fact(p))))
        .filter(|f| !f.is_redacted())
        .filter_map(|f| match f.end?.ending {
            // An end at or before the start: true at no time.
            Ending::Changed { valid_to } => (valid_to > f.valid_from).then_some(valid_to),
            Ending::Corrected => None,
        })
        .max();
    match floor {
        Some(floor) if starts < floor => Err(StoreError::ChangeBeforeValid {
            at: starts,
            valid_from: floor,
        }),
        _ => Ok(()),
    }
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

//! The in-memory state: records in vectors plus lookup indexes.
//!
//! The state changes only through [`State::apply`], the same code for live
//! writes and for replaying the log, so both always build the same state.

use std::collections::HashMap;
use std::num::NonZeroU32;

use super::op::Op;
use crate::model::{
    Entity, EntityId, EntityKindDef, EntityKindId, Episode, Fact, FactBody, FactId, Lineage, Name,
    PredicateDef, PredicateId, Space, SpaceId, ValueType,
};

#[derive(Debug, Default)]
pub(crate) struct State {
    spaces: Vec<Space>,
    kinds: Vec<EntityKindDef>,
    predicates: Vec<PredicateDef>,
    entities: Vec<Entity>,
    episodes: Vec<Episode>,
    facts: Vec<Fact>,

    /// Normalized space name → space.
    space_by_name: HashMap<String, SpaceId>,
    kind_by_name: HashMap<(SpaceId, Name), EntityKindId>,
    predicate_by_name: HashMap<(SpaceId, Name), PredicateId>,
    /// (space, kind, normalized key or alias) → entity.
    entity_by_name: HashMap<(SpaceId, EntityKindId, String), EntityId>,
    /// Entity → every fact version that refers to it, oldest first.
    facts_by_entity: Vec<Vec<FactId>>,
    /// Lineage → its latest version.
    heads: HashMap<Lineage, FactId>,
    /// (subject, predicate) → lineages of triples with that pair.
    pairs: HashMap<(EntityId, PredicateId), Vec<Lineage>>,
}

impl State {
    pub(crate) fn space(&self, id: SpaceId) -> Option<&Space> {
        self.spaces.get(id.index())
    }

    pub(crate) fn kind(&self, id: EntityKindId) -> Option<&EntityKindDef> {
        self.kinds.get(id.index())
    }

    pub(crate) fn predicate(&self, id: PredicateId) -> Option<&PredicateDef> {
        self.predicates.get(id.index())
    }

    pub(crate) fn entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(id.index())
    }

    pub(crate) fn fact(&self, id: FactId) -> Option<&Fact> {
        self.facts.get(id.index())
    }

    pub(crate) fn len(&self) -> Lens {
        Lens {
            spaces: self.spaces.len(),
            kinds: self.kinds.len(),
            predicates: self.predicates.len(),
            entities: self.entities.len(),
            episodes: self.episodes.len(),
            facts: self.facts.len(),
        }
    }

    pub(crate) fn space_named(&self, normalized: &str) -> Option<SpaceId> {
        self.space_by_name.get(normalized).copied()
    }

    pub(crate) fn kind_named(&self, space: SpaceId, name: &Name) -> Option<EntityKindId> {
        self.kind_by_name.get(&(space, name.clone())).copied()
    }

    pub(crate) fn predicate_named(&self, space: SpaceId, name: &Name) -> Option<PredicateId> {
        self.predicate_by_name.get(&(space, name.clone())).copied()
    }

    pub(crate) fn entity_named(
        &self,
        space: SpaceId,
        kind: EntityKindId,
        normalized: &str,
    ) -> Option<EntityId> {
        self.entity_by_name
            .get(&(space, kind, normalized.to_owned()))
            .copied()
    }

    /// The latest version of a lineage, current or not.
    pub(crate) fn head(&self, lineage: Lineage) -> Option<&Fact> {
        self.heads.get(&lineage).and_then(|&id| self.fact(id))
    }

    pub(crate) fn lineages_of_pair(&self, subject: EntityId, predicate: PredicateId) -> &[Lineage] {
        self.pairs
            .get(&(subject, predicate))
            .map_or(&[], Vec::as_slice)
    }

    pub(crate) fn facts_of(&self, entity: EntityId) -> &[FactId] {
        self.facts_by_entity
            .get(entity.index())
            .map_or(&[], Vec::as_slice)
    }

    /// Applies one operation, checking that it fits the current state.
    pub(crate) fn apply(&mut self, op: Op) -> Result<(), String> {
        match op {
            Op::CreateSpace(space) => {
                expect_next("space", space.id.index(), self.spaces.len())?;
                if let Some(parent) = space.parent {
                    self.require_space(parent)?;
                }
                let name = space.name.normalized();
                if self.space_by_name.contains_key(&name) {
                    return Err(format!("space name {name:?} is taken"));
                }
                self.space_by_name.insert(name, space.id);
                self.spaces.push(space);
            }
            Op::DefineKind(kind) => {
                expect_next("kind", kind.id.index(), self.kinds.len())?;
                self.require_space(kind.space)?;
                let key = (kind.space, kind.name.clone());
                if self.kind_by_name.contains_key(&key) {
                    return Err(format!("kind {} is already defined", kind.name));
                }
                self.kind_by_name.insert(key, kind.id);
                self.kinds.push(kind);
            }
            Op::DefinePredicate(predicate) => {
                expect_next("predicate", predicate.id.index(), self.predicates.len())?;
                self.require_space(predicate.space)?;
                if let ValueType::Entity(Some(kind)) = predicate.object {
                    self.require_kind_in(kind, predicate.space)?;
                }
                let key = (predicate.space, predicate.name.clone());
                if self.predicate_by_name.contains_key(&key) {
                    return Err(format!("predicate {} is already defined", predicate.name));
                }
                self.predicate_by_name.insert(key, predicate.id);
                self.predicates.push(predicate);
            }
            Op::CreateEntity(entity) => {
                expect_next("entity", entity.id.index(), self.entities.len())?;
                self.require_space(entity.space)?;
                self.require_kind_in(entity.kind, entity.space)?;
                let names: Vec<String> = std::iter::once(&entity.key)
                    .chain(&entity.aliases)
                    .map(|name| name.normalized())
                    .collect();
                for (i, name) in names.iter().enumerate() {
                    let slot = (entity.space, entity.kind, name.clone());
                    if self.entity_by_name.contains_key(&slot) || names[..i].contains(name) {
                        return Err(format!("entity name {name:?} is taken"));
                    }
                }
                for name in names {
                    self.entity_by_name
                        .insert((entity.space, entity.kind, name), entity.id);
                }
                self.facts_by_entity.push(Vec::new());
                self.entities.push(entity);
            }
            Op::AddAlias { entity, alias } => {
                let e = self
                    .entities
                    .get_mut(entity.index())
                    .ok_or_else(|| format!("unknown entity {entity}"))?;
                let slot = (e.space, e.kind, alias.normalized());
                if self.entity_by_name.contains_key(&slot) {
                    return Err(format!("entity name {:?} is taken", slot.2));
                }
                self.entity_by_name.insert(slot, entity);
                e.aliases.push(alias);
            }
            Op::AddEpisode(episode) => {
                expect_next("episode", episode.id.index(), self.episodes.len())?;
                self.require_space(episode.space)?;
                self.episodes.push(episode);
            }
            Op::AddFact(fact) => self.add_fact(fact)?,
            Op::CloseFact { fact, end } => {
                if let Some(next) = end.next {
                    let n = self
                        .fact(next)
                        .ok_or_else(|| format!("unknown next version {next}"))?;
                    if n.prev != Some(fact) {
                        return Err(format!("{next} does not follow {fact}"));
                    }
                }
                let f = self
                    .facts
                    .get_mut(fact.index())
                    .ok_or_else(|| format!("unknown fact {fact}"))?;
                if f.end.is_some() {
                    return Err(format!("{fact} is already closed"));
                }
                f.end = Some(end);
            }
        }
        Ok(())
    }

    fn add_fact(&mut self, fact: Fact) -> Result<(), String> {
        expect_next("fact", fact.id.index(), self.facts.len())?;
        self.require_space(fact.space)?;
        let entities = fact.entities();
        for &e in &entities {
            let entity = self
                .entity(e)
                .ok_or_else(|| format!("unknown entity {e}"))?;
            if entity.space != fact.space {
                return Err(format!("{e} is in another space"));
            }
        }
        if let FactBody::Triple { predicate, .. } = &fact.body {
            let p = self
                .predicate(*predicate)
                .ok_or_else(|| format!("unknown predicate {predicate}"))?;
            if p.space != fact.space {
                return Err(format!("{predicate} is in another space"));
            }
        }
        if let Some(episode) = fact.episode
            && self.episodes.get(episode.index()).is_none()
        {
            return Err(format!("unknown episode {episode}"));
        }
        if fact.version == NonZeroU32::MIN {
            if fact.lineage.first() != fact.id || fact.prev.is_some() {
                return Err(format!("{} starts a lineage it does not own", fact.id));
            }
        } else {
            let prev = fact
                .prev
                .ok_or_else(|| format!("{} has no previous version", fact.id))?;
            let p = self
                .fact(prev)
                .ok_or_else(|| format!("unknown previous version {prev}"))?;
            let follows = p.lineage == fact.lineage
                && p.version.checked_add(1) == Some(fact.version)
                && self.heads.get(&fact.lineage) == Some(&prev);
            if !follows {
                return Err(format!("{} does not follow {prev}", fact.id));
            }
        }

        for e in entities {
            self.facts_by_entity[e.index()].push(fact.id);
        }
        self.heads.insert(fact.lineage, fact.id);
        if fact.version == NonZeroU32::MIN
            && let FactBody::Triple {
                subject, predicate, ..
            } = &fact.body
        {
            self.pairs
                .entry((*subject, *predicate))
                .or_default()
                .push(fact.lineage);
        }
        self.facts.push(fact);
        Ok(())
    }

    fn require_space(&self, id: SpaceId) -> Result<&Space, String> {
        self.space(id).ok_or_else(|| format!("unknown space {id}"))
    }

    fn require_kind_in(&self, kind: EntityKindId, space: SpaceId) -> Result<(), String> {
        match self.kind(kind) {
            Some(k) if k.space == space => Ok(()),
            Some(_) => Err(format!("{kind} is in another space")),
            None => Err(format!("unknown kind {kind}")),
        }
    }
}

/// How many records of each kind the state holds: the next ids.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lens {
    pub(crate) spaces: usize,
    pub(crate) kinds: usize,
    pub(crate) predicates: usize,
    pub(crate) entities: usize,
    pub(crate) episodes: usize,
    pub(crate) facts: usize,
}

fn expect_next(what: &str, index: usize, len: usize) -> Result<(), String> {
    if index == len {
        Ok(())
    } else {
        Err(format!("{what} {index} is out of order: expected {len}"))
    }
}

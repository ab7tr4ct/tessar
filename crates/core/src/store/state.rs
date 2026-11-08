//! The in-memory state: records in vectors plus lookup indexes.
//!
//! The state changes only through [`State::apply`], the same code for live
//! writes and for replaying the log, so both always build the same state.

use std::collections::HashMap;
use std::num::NonZeroU32;

use super::op::Op;
use super::text::{TextIndex, fact_text};
use super::vector::{PredicateVectors, Quantized, VectorIndex};
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
    /// (subject, canonical predicate) → lineages of triples with that
    /// pair. Keyed by the canonical predicate, so synonyms share pairs.
    pairs: HashMap<(EntityId, PredicateId), Vec<Lineage>>,
    /// Predicate → its canonical predicate; itself unless merged.
    canonical: Vec<PredicateId>,
    /// Normalized key or alias → entities with that name, in any space or
    /// kind. For finding entities named in a query.
    names: HashMap<String, Vec<EntityId>>,
    /// Keyword index over every fact version.
    text: TextIndex,
    /// Vectors of every fact version, if the meaning channel is on. Not
    /// built by [`State::apply`]: vectors are not in the log, the writer
    /// adds them right after applying a commit.
    vectors: Option<VectorIndex>,
    /// Vectors of predicates, if the meaning channel is on. Like
    /// `vectors`, not built by [`State::apply`].
    predicate_vectors: Option<PredicateVectors>,
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

    /// The canonical predicate of the predicate named `name` in `space`.
    pub(crate) fn predicate_named(&self, space: SpaceId, name: &Name) -> Option<PredicateId> {
        self.predicate_by_name
            .get(&(space, name.clone()))
            .map(|&id| self.canonical(id))
    }

    /// The canonical predicate of `id`: `id` itself unless it was merged
    /// into another. An unknown id is returned unchanged.
    pub(crate) fn canonical(&self, id: PredicateId) -> PredicateId {
        self.canonical.get(id.index()).copied().unwrap_or(id)
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

    /// Entities whose normalized key or alias is `normalized`.
    pub(crate) fn entities_named(&self, normalized: &str) -> &[EntityId] {
        self.names.get(normalized).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn text(&self) -> &TextIndex {
        &self.text
    }

    /// Every fact version, by index.
    pub(crate) fn facts(&self) -> &[Fact] {
        &self.facts
    }

    pub(crate) fn vectors(&self) -> Option<&VectorIndex> {
        self.vectors.as_ref()
    }

    /// Turns the meaning channel on. Vectors of the predicates and of the
    /// facts already here must be pushed next, in order.
    pub(crate) fn enable_vectors(&mut self, dims: usize) {
        self.vectors = Some(VectorIndex::new(dims));
        self.predicate_vectors = Some(PredicateVectors::new(dims));
    }

    /// Adds the vector of a predicate (canonical or synonym).
    pub(crate) fn push_predicate_vector(
        &mut self,
        id: PredicateId,
        vector: Quantized,
    ) -> Result<(), String> {
        let index = self
            .predicate_vectors
            .as_mut()
            .ok_or("the meaning channel is off")?;
        if self.predicates.get(id.index()).is_none() {
            return Err(format!("vector for unknown predicate {id}"));
        }
        index.push(id, vector)
    }

    /// The predicate whose vector is closest to `query`, with the cosine.
    /// `None` if the meaning channel is off or no predicate has a vector.
    pub(crate) fn most_similar_predicate(&self, query: &Quantized) -> Option<(PredicateId, f32)> {
        self.predicate_vectors.as_ref()?.most_similar(query)
    }

    /// Every canonical predicate (synonyms left out), in id order.
    pub(crate) fn canonical_predicates(&self) -> impl Iterator<Item = &PredicateDef> {
        self.predicates
            .iter()
            .filter(|p| self.canonical(p.id) == p.id)
    }

    /// The synonyms of the canonical predicate of `id` (it excluded), in
    /// id order.
    pub(crate) fn synonyms_of(&self, id: PredicateId) -> impl Iterator<Item = &PredicateDef> {
        let canonical = self.canonical(id);
        self.predicates
            .iter()
            .filter(move |p| p.id != canonical && self.canonical(p.id) == canonical)
    }

    /// Every predicate, synonyms included, in id order.
    pub(crate) fn all_predicates(&self) -> &[PredicateDef] {
        &self.predicates
    }

    /// Adds the vector of the next fact that has none.
    pub(crate) fn push_vector(&mut self, fact: FactId, vector: Quantized) -> Result<(), String> {
        let index = self.vectors.as_mut().ok_or("the meaning channel is off")?;
        if fact.index() >= self.facts.len() {
            return Err(format!("vector for unknown fact {fact}"));
        }
        index.push(fact, vector)
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
            Op::DefinePredicate(predicate) => self.add_predicate(predicate)?,
            Op::MergePredicate { from, into } => self.merge_predicate(from, into)?,
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
                    self.names.entry(name.clone()).or_default().push(entity.id);
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
                self.names.entry(slot.2.clone()).or_default().push(entity);
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

    /// Adds a predicate: its space and entity kind exist, the name is free
    /// in the space.
    fn add_predicate(&mut self, predicate: PredicateDef) -> Result<(), String> {
        expect_next("predicate", predicate.id.index(), self.predicates.len())?;
        self.require_space(predicate.space)?;
        if let ValueType::Entity(Some(kind)) = predicate.object
            && self.kind(kind).is_none()
        {
            return Err(format!("unknown kind {kind}"));
        }
        let key = (predicate.space, predicate.name.clone());
        if self.predicate_by_name.contains_key(&key) {
            return Err(format!("predicate {} is already defined", predicate.name));
        }
        self.predicate_by_name.insert(key, predicate.id);
        self.canonical.push(predicate.id);
        self.predicates.push(predicate);
        Ok(())
    }

    /// Makes `from` a synonym of `into`: both exist, `into` is canonical,
    /// `from` is canonical (not merged yet) and not `into`, and value type
    /// and cardinality match. The pairs of `from` move under `into`, their
    /// lineages ordered by first version so the order never depends on how
    /// the state was built.
    fn merge_predicate(&mut self, from: PredicateId, into: PredicateId) -> Result<(), String> {
        let f = self
            .predicate(from)
            .ok_or_else(|| format!("unknown predicate {from}"))?;
        let t = self
            .predicate(into)
            .ok_or_else(|| format!("unknown predicate {into}"))?;
        if from == into {
            return Err(format!("cannot merge {from} into itself"));
        }
        if self.canonical(into) != into {
            return Err(format!("{into} is itself a synonym"));
        }
        if self.canonical(from) != from {
            return Err(format!("{from} is already merged"));
        }
        if f.object != t.object || f.cardinality != t.cardinality {
            return Err(format!(
                "{from} and {into} differ in value type or cardinality"
            ));
        }
        self.canonical[from.index()] = into;
        // Flatten chains: synonyms of `from` now point at `into`. Their
        // pairs already sit under `from` and move with it below.
        for c in &mut self.canonical {
            if *c == from {
                *c = into;
            }
        }
        let moved: Vec<(EntityId, PredicateId)> = self
            .pairs
            .keys()
            .filter(|(_, p)| *p == from)
            .copied()
            .collect();
        for key in moved {
            let Some(lineages) = self.pairs.remove(&key) else {
                continue;
            };
            let target = self.pairs.entry((key.0, into)).or_default();
            target.extend(lineages);
            target.sort_by_key(|l| l.first());
        }
        Ok(())
    }

    /// Adds a fact. Its entities and predicate may live in any space: the
    /// fact's space says where the fact lies, not what it may refer to.
    fn add_fact(&mut self, fact: Fact) -> Result<(), String> {
        expect_next("fact", fact.id.index(), self.facts.len())?;
        self.require_space(fact.space)?;
        let entities = fact.entities();
        for &e in &entities {
            self.entity(e)
                .ok_or_else(|| format!("unknown entity {e}"))?;
        }
        if let FactBody::Triple { predicate, .. } = &fact.body {
            self.predicate(*predicate)
                .ok_or_else(|| format!("unknown predicate {predicate}"))?;
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

        let text = fact_text(self, &fact);
        self.text.add(fact.id, &text)?;
        for e in entities {
            self.facts_by_entity[e.index()].push(fact.id);
        }
        self.heads.insert(fact.lineage, fact.id);
        if fact.version == NonZeroU32::MIN
            && let FactBody::Triple {
                subject, predicate, ..
            } = &fact.body
        {
            let predicate = self.canonical(*predicate);
            self.pairs
                .entry((*subject, predicate))
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

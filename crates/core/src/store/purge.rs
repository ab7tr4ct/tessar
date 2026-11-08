//! Purge: rewriting a log so that redacted content is gone from it.
//!
//! A redaction ([`Op::Redact`], [`Op::RedactEpisode`]) removes content from
//! the state, but the log is append-only: the text is still in the
//! `AddFact` and `AddEpisode` ops written earlier. [`purge`] rewrites
//! those ops. It is a pure function over commits; writing the result back
//! is the log adapter's job (for the file journal,
//! `FileJournal::rewrite`).
//!
//! Replaying the purged commits gives the same state as replaying the
//! originals, with two documented exceptions:
//!
//! - Entity keys and aliases are not redacted. An entity created for a
//!   secret keeps its name in `CreateEntity`.
//! - A lineage whose every version is redacted loses its (subject,
//!   predicate) pair, because only the content of the first version told
//!   it. The next write of that pair starts a new lineage instead of
//!   continuing the old one. A lineage with a surviving later version
//!   keeps its pair.
//!
//! The process holding the log must be closed while it is rewritten, and
//! reopened afterwards.

use std::collections::HashSet;

use super::op::Op;
use crate::log::Commit;
use crate::model::{EpisodeId, FactBody, FactId};

/// How many records [`purge_with_report`] rewrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PurgeReport {
    /// `AddFact` ops whose content was removed.
    pub facts: usize,
    /// `AddEpisode` ops whose text was removed.
    pub episodes: usize,
}

/// Rewrites earlier `AddFact` and `AddEpisode` ops of redacted records so
/// that they carry no content, and returns the commits otherwise
/// unchanged: same numbers, same order, same ids. See the module docs.
pub fn purge(commits: Vec<Commit<Op>>) -> Vec<Commit<Op>> {
    purge_with_report(commits).0
}

/// [`purge`] that also counts the records it rewrote. A record that is
/// already clean is not counted, so purging a purged log reports nothing.
pub fn purge_with_report(mut commits: Vec<Commit<Op>>) -> (Vec<Commit<Op>>, PurgeReport) {
    let mut facts: HashSet<FactId> = HashSet::new();
    let mut episodes: HashSet<EpisodeId> = HashSet::new();
    for op in commits.iter().flat_map(|c| &c.ops) {
        match op {
            Op::Redact { fact } => {
                facts.insert(*fact);
            }
            Op::RedactEpisode { episode } => {
                episodes.insert(*episode);
            }
            _ => {}
        }
    }

    let mut report = PurgeReport::default();
    for op in commits.iter_mut().flat_map(|c| &mut c.ops) {
        match op {
            Op::AddFact(fact)
                if facts.contains(&fact.id)
                    && (!fact.is_redacted()
                        || !fact.mentions.is_empty()
                        || fact.episode.is_some()) =>
            {
                fact.body = FactBody::Redacted;
                fact.mentions.clear();
                fact.episode = None;
                report.facts += 1;
            }
            Op::AddEpisode(episode)
                if episodes.contains(&episode.id) && !episode.text.is_empty() =>
            {
                episode.text.clear();
                report.episodes += 1;
            }
            _ => {}
        }
    }
    (commits, report)
}

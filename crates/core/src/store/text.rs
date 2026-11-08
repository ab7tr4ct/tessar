//! Words: splitting text into words and the BM25 keyword index.
//!
//! No stemming yet: "решили" and "решение" are different words (known
//! trade-off; a stemmer or `tantivy` comes when measurements ask for it).

use std::collections::HashMap;

use super::state::State;
use crate::model::{Entity, Fact, FactBody, FactId, Value};

/// BM25 term-frequency saturation.
const K1: f32 = 1.2;
/// BM25 length normalization.
const B: f32 = 0.75;

/// Splits text into lowercase words of letters and digits, in any script.
/// Single letters are dropped; single digits are kept.
pub(crate) fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            let mut chars = w.chars();
            match (chars.next(), chars.next()) {
                (None, _) => false,
                (Some(c), None) => c.is_numeric(),
                _ => true,
            }
        })
        .map(str::to_lowercase)
}

/// Splits a query into candidate entity names: runs of up to
/// `max_words` whitespace-separated tokens, with sentence punctuation
/// trimmed from the edges and normalized like [`Key::normalized`].
///
/// [`Key::normalized`]: crate::model::Key::normalized
pub(crate) fn name_candidates(text: &str, max_words: usize) -> Vec<String> {
    let tokens: Vec<String> = text
        .split_whitespace()
        .map(|t| t.trim_matches(is_sentence_punctuation).to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    let mut names = Vec::new();
    for start in 0..tokens.len() {
        for end in start + 1..=(start + max_words).min(tokens.len()) {
            names.push(tokens[start..end].join(" "));
        }
    }
    names
}

fn is_sentence_punctuation(c: char) -> bool {
    matches!(
        c,
        '.' | ','
            | ';'
            | ':'
            | '!'
            | '?'
            | '"'
            | '\''
            | '«'
            | '»'
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '…'
            | '—'
    )
}

/// The text of a fact as the keyword index sees it: the text of a
/// statement or event, or the subject, predicate and value of a triple,
/// plus the keys of the entities in the content.
pub(crate) fn fact_text(state: &State, fact: &Fact) -> String {
    fact_text_with(state, fact, &[])
}

/// [`fact_text`] for a fact that is not applied yet: its entities may be
/// among `new`, created in the same commit.
pub(crate) fn fact_text_with(state: &State, fact: &Fact, new: &[&Entity]) -> String {
    let key = |id| {
        state
            .entity(id)
            .or_else(|| new.iter().copied().find(|e| e.id == id))
            .map_or("", |e| e.key.as_str())
    };
    let mut parts: Vec<String> = Vec::new();
    match &fact.body {
        FactBody::Statement { text, .. } => parts.push(text.clone()),
        FactBody::Event {
            text, participants, ..
        } => {
            parts.push(text.clone());
            parts.extend(participants.iter().map(|&p| key(p).to_owned()));
        }
        FactBody::Triple {
            subject,
            predicate,
            object,
            text,
        } => {
            // TODO(core-A): put the sentence first, then the structured
            // form, so both the phrase and the exact key are found.
            if let Some(text) = text {
                parts.push(text.clone());
            }
            parts.push(key(*subject).to_owned());
            if let Some(p) = state.predicate(*predicate) {
                parts.push(p.name.as_str().replace('_', " "));
            }
            match object {
                Value::Text(t) => parts.push(t.clone()),
                Value::Int(n) => parts.push(n.to_string()),
                Value::Float(x) => parts.push(x.get().to_string()),
                Value::Bool(b) => parts.push(b.to_string()),
                Value::Entity(e) => parts.push(key(*e).to_owned()),
                Value::Time(_) | Value::Span(_) => {}
            }
        }
    }
    parts.join(" ")
}

/// An inverted index of words for BM25, over every fact version.
#[derive(Debug, Default)]
pub(crate) struct TextIndex {
    /// Word → (fact, how many times the word occurs in it).
    postings: HashMap<String, Vec<(FactId, u32)>>,
    /// Number of words per fact, by fact index.
    lengths: Vec<u32>,
    total_length: u64,
}

impl TextIndex {
    /// Indexes the next fact. Facts must come in id order.
    pub(crate) fn add(&mut self, fact: FactId, text: &str) -> Result<(), String> {
        if fact.index() != self.lengths.len() {
            return Err(format!(
                "text index expected fact {}, got {fact}",
                self.lengths.len()
            ));
        }
        let mut counts: HashMap<String, u32> = HashMap::new();
        let mut length = 0u32;
        for word in words(text) {
            *counts.entry(word).or_default() += 1;
            length += 1;
        }
        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_unstable();
        for (word, count) in counts {
            self.postings.entry(word).or_default().push((fact, count));
        }
        self.lengths.push(length);
        self.total_length += u64::from(length);
        Ok(())
    }

    /// BM25 score of every fact containing at least one of `terms`.
    /// Terms are summed in the given order, so scores are deterministic.
    pub(crate) fn search(&self, terms: &[String]) -> HashMap<FactId, f32> {
        let docs = self.lengths.len() as f32;
        let mut scores: HashMap<FactId, f32> = HashMap::new();
        if docs == 0.0 {
            return scores;
        }
        let average = self.total_length as f32 / docs;
        for term in terms {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let df = postings.len() as f32;
            let idf = (1.0 + (docs - df + 0.5) / (df + 0.5)).ln();
            for &(fact, tf) in postings {
                let tf = tf as f32;
                let length = self.lengths[fact.index()] as f32;
                let norm = K1 * (1.0 - B + B * length / average.max(1.0));
                *scores.entry(fact).or_default() += idf * tf * (K1 + 1.0) / (tf + norm);
            }
        }
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(text: &str) -> Vec<String> {
        words(text).collect()
    }

    #[test]
    fn words_split_on_anything_but_letters_and_digits() {
        assert_eq!(
            all("Решили НЕ брать MongoDB, v2.0 — src/main.rs"),
            [
                "решили",
                "не",
                "брать",
                "mongodb",
                "v2",
                "0",
                "src",
                "main",
                "rs"
            ]
        );
        assert_eq!(all("a и 7 ok"), ["7", "ok"]);
    }

    #[test]
    fn name_candidates_keep_inner_punctuation() {
        let names = name_candidates("Что с cortex-db и «HGVL»?", 2);
        assert!(names.contains(&"cortex-db".to_owned()));
        assert!(names.contains(&"hgvl".to_owned()));
        assert!(names.contains(&"с cortex-db".to_owned()));
        assert!(
            !names.contains(&"что с cortex-db".to_owned()),
            "max 2 words"
        );
    }

    #[test]
    fn rare_words_weigh_more() {
        let mut index = TextIndex::default();
        index.add(FactId::new(0), "memory engine in rust").unwrap();
        index.add(FactId::new(1), "rust rust rust").unwrap();
        index.add(FactId::new(2), "graph engine").unwrap();
        let scores = index.search(&["memory".to_owned(), "rust".to_owned()]);
        assert!(scores[&FactId::new(0)] > scores[&FactId::new(1)]);
        assert!(!scores.contains_key(&FactId::new(2)));
    }

    #[test]
    fn facts_must_be_indexed_in_order() {
        let mut index = TextIndex::default();
        assert!(index.add(FactId::new(1), "x").is_err());
    }
}

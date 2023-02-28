//! Space Saving Sets
//!
//! The initial algorithm presented in the paper.
//!
//! The `SetCounter` can use any cardinality sketch, but we only provide HyperLogLog

mod cached;
mod config;
mod counter;

use std::{collections::HashMap, error, fmt, hash::Hash};

use sketch_traits::{CardinalitySketch, HeavyDistinctHitterSketch, New};

use crate::{cached::Cached, counter::Counter};
pub use crate::{
    config::{Config, ConfigError},
    counter::ResetStrategy,
};

#[derive(Clone, Debug)]
pub struct SpaceSavingSets<L, S>
where
    S: New,
{
    config: Config<S::Config>,
    counters: HashMap<L, Counter<Cached<S>>>,
}

impl<L, S> New for SpaceSavingSets<L, S>
where
    S: New,
    S::Config: Clone,
{
    type Config = Config<S::Config>;

    fn new(config: &Self::Config) -> Self {
        Self {
            config: config.clone(),
            counters: HashMap::new(),
        }
    }
}

impl<L, S> HeavyDistinctHitterSketch for SpaceSavingSets<L, S>
where
    L: Eq + Hash + Clone,
    S: CardinalitySketch + New,
    S::Config: Eq,
{
    type Label = L;
    type Item = S::Item;
    type MergeError = MergeError;

    fn insert(&mut self, label: L, item: &S::Item) {
        let full = self.full();
        let key_exists = self.counters.contains_key(&label);
        let counter = if !key_exists {
            if full {
                let min_label = self.get_min_label();
                let mut counter = self.counters.remove(&min_label).unwrap();
                counter.reset(&self.config.reset_strategy);
                self.counters.entry(label).or_insert(counter)
            } else {
                self.counters.entry(label).or_insert_with(|| {
                    Counter::new(Cached::new(&self.config.cardinality_sketch_config))
                })
            }
        } else {
            self.counters.get_mut(&label).unwrap()
        };
        counter.sketch.insert(item);
    }

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        if self.config != other.config {
            return Err(MergeError::ConfigMismatch);
        }

        for (l, c) in other.counters.iter() {
            self.counters
                .entry(l.clone())
                .or_insert_with(|| {
                    Counter::new(Cached::new(&self.config.cardinality_sketch_config))
                })
                .sketch
                .merge(&c.sketch)
                .unwrap();
        }
        let mut entries = self
            .counters
            .iter()
            .map(|(label, counter)| (label, counter.offset_cardinality()))
            .collect::<Vec<_>>();
        entries.sort_by_key(|&(_, cardinality)| cardinality);
        entries
            .into_iter()
            .rev()
            .skip(self.config.max_num_counters)
            .map(|(label, _)| label)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|label| {
                self.counters.remove(&label);
            });
        Ok(())
    }

    fn clear(&mut self) {
        todo!()
    }

    fn cardinality(&self, label: &L) -> u64 {
        self.counters
            .get(label)
            .map(|c| c.offset_cardinality())
            .unwrap_or_else(|| {
                self.counters
                    .values()
                    .map(|c| c.offset_cardinality())
                    .min()
                    .unwrap_or(0)
            })
    }

    fn top(&self, k: usize) -> Vec<(&L, u64)> {
        let mut entries = self
            .counters
            .iter()
            .map(|(label, counter)| (label, counter.offset_cardinality()))
            .collect::<Vec<_>>();
        entries.sort_by_key(|&(_, cardinality)| cardinality);
        entries.into_iter().rev().take(k).collect::<Vec<_>>()
    }
}

impl<L, S> SpaceSavingSets<L, S>
where
    L: Clone,
    S: CardinalitySketch + New,
{
    fn full(&self) -> bool {
        debug_assert!(self.counters.len() <= self.config.max_num_counters);
        self.counters.len() == self.config.max_num_counters
    }

    // TODO: see if using a min-heap makes things faster. Since a SetCounter
    // only ever increases, we only need to push the node down the tree on
    // insert if it gets larger than its children.
    fn get_min_label(&self) -> L {
        self.counters
            .iter()
            .min_by_key(|(_, counter)| counter.offset_cardinality())
            .map(|(label, _)| (*label).clone())
            .unwrap()
    }
}

impl<L, S> SpaceSavingSets<L, S>
where
    S: New,
{
    pub fn config(&self) -> &Config<S::Config> {
        &self.config
    }

    pub fn num_counters(&self) -> usize {
        self.counters.len()
    }
}

#[derive(Clone, Debug)]
pub enum MergeError {
    ConfigMismatch,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MergeError::ConfigMismatch => write!(f, "sketch configs do not match"),
        }
    }
}

impl error::Error for MergeError {}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use hll::HyperLogLog;
    use sketch_traits::HeavyDistinctHitterSketch;

    use super::*;
    use crate::ResetStrategy;

    const SIZE: usize = 10;
    const COUNTER_SIZE: usize = 512;
    const HLL_SEEDS: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

    fn config(reset_strategy: ResetStrategy) -> Config<hll::Config> {
        Config::new(
            SIZE,
            reset_strategy,
            hll::Config::new(COUNTER_SIZE, Some(HLL_SEEDS)).unwrap(),
        )
        .unwrap()
    }

    fn relative_error(a: u64, b: u64) -> f64 {
        let fa = a as f64;
        let fb = b as f64;
        (fa - fb).abs() / fa
    }

    #[test]
    fn offset_sss_counts_set_cardinality() {
        let mut sketch: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Offset));
        let mut exact: HashMap<char, HashSet<u64>> = HashMap::new();

        // fill up the sketch
        for label in 'a'..='j' {
            for i in 0..100 {
                sketch.insert(label, &i);
                exact.entry(label).or_default().insert(i);
            }
        }
        assert!(sketch.full());
        let label = 'a';
        assert!(
            relative_error(
                sketch.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) < 0.1
        );

        // adding a new label will start the sketch off with an offset of ~100, so
        // after adding the same 100 the sketch should be off by a factor of 2
        let label = 'k';
        for i in 0..100 {
            sketch.insert(label, &i);
            exact.entry(label).or_default().insert(i);
        }
        assert!(
            (relative_error(
                sketch.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) - 0.5)
                .abs()
                < 0.1
        );

        // adding back the removed label will start the sketch off with an offset of ~100, so
        // after adding 100 different items the sketch should be accurate
        let sketch_labels: HashSet<char> = sketch.counters.keys().cloned().collect();
        let original_labels = ('a'..='j').collect::<HashSet<_>>();
        let label = *(&original_labels - &sketch_labels).iter().next().unwrap();
        for i in 101..200 {
            sketch.insert(label, &i);
            exact.entry(label).or_default().insert(i);
        }
        assert!(
            relative_error(
                sketch.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) < 0.1
        );
    }

    #[test]
    fn recycling_sss_counts_set_cardinality() {
        let mut sketch: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Recycle));
        let mut exact: HashMap<char, HashSet<u64>> = HashMap::new();

        // fill up the sketch
        for label in 'a'..='j' {
            for i in 0..100 {
                sketch.insert(label, &i);
                exact.entry(label).or_default().insert(i);
            }
        }
        assert!(sketch.full());
        let label = 'a';
        assert!(
            relative_error(
                sketch.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) < 0.1
        );

        // adding a new label will start the sketch off with a used cardinality
        // sketch but we're inserting the same items so the cardinality should
        // stay the same
        let label = 'k';
        for i in 0..100 {
            sketch.insert(label, &i);
            exact.entry(label).or_default().insert(i);
        }
        assert!(
            relative_error(
                sketch.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) < 0.1
        );

        // adding a new label will start the sketch off with a used cardinality
        // sketch but we're inserting different items so the cardinality should
        // be off by a factor of 2
        let label = 'l';
        for i in 100..200 {
            sketch.insert(label, &i);
            exact.entry(label).or_default().insert(i);
        }
        assert!(
            (relative_error(
                sketch.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) - 0.5)
                .abs()
                < 0.1
        );
    }

    #[test]
    fn merged_recycling_sss_counts_set_cardinality() {
        let mut sketch1: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Recycle));
        let mut sketch2: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Recycle));
        let mut sketch3: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Recycle));
        let mut exact: HashMap<char, HashSet<u64>> = HashMap::new();

        // all 3 sketches have the same information, so merging should have no effect
        for label in 'a'..='j' {
            for i in 0..100 {
                sketch1.insert(label, &i);
                sketch2.insert(label, &i);
                sketch3.insert(label, &i);
                exact.entry(label).or_default().insert(i);
            }
        }
        let label = 'a';
        assert!(sketch1.merge(&sketch2).is_ok());
        assert!(sketch1.merge(&sketch3).is_ok());
        assert!(
            relative_error(
                sketch1.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) < 0.1
        );
        assert!(relative_error(sketch1.cardinality(&label), sketch2.cardinality(&label)) < 0.1);

        // this sketch has disjoint labels and larger sets (a..j should get pushed out)
        let mut sketch4: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Recycle));
        for label in 'k'..='t' {
            for i in 1..200 {
                sketch4.insert(label, &i);
                exact.entry(label).or_default().insert(i);
            }
        }
        let label = 'k';
        assert!(sketch1.merge(&sketch4).is_ok());
        assert!(
            relative_error(
                sketch1.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) < 0.1
        );

        // this sketch overlaps on half the labels and half-disjoint items
        let mut sketch5: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Recycle));
        for label in 'p'..='y' {
            for i in 100..300 {
                sketch5.insert(label, &i);
                exact.entry(label).or_default().insert(i);
            }
        }
        assert!(sketch1.merge(&sketch5).is_ok());

        let label = 'p';
        assert!(
            relative_error(
                sketch1.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) < 0.1
        );
    }

    #[test]
    fn merged_offset_sss_counts_set_cardinality() {
        let mut sketch1: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Offset));
        let mut exact: HashMap<char, HashSet<u64>> = HashMap::new();

        // this sketch has disjoint labels and larger sets (a..j should get pushed out)
        let mut sketch2: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Offset));
        for label in 'k'..='t' {
            for i in 1..200 {
                sketch2.insert(label, &i);
                exact.entry(label).or_default().insert(i);
            }
        }
        assert!(sketch1.merge(&sketch2).is_ok());

        // this sketch overlaps on half the labels and half-disjoint items
        let mut sketch3: SpaceSavingSets<char, HyperLogLog<u64>> =
            SpaceSavingSets::new(&config(ResetStrategy::Offset));
        for label in 'p'..='y' {
            for i in 100..300 {
                sketch3.insert(label, &i);
                exact.entry(label).or_default().insert(i);
            }
        }
        assert!(sketch1.merge(&sketch3).is_ok());

        let label = 'p';
        assert!(
            relative_error(
                sketch1.cardinality(&label),
                exact.get(&label).unwrap().len() as u64
            ) < 0.1
        );
    }
}

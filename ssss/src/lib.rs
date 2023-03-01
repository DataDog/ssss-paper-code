//! Sampling Space Saving Sets
//!
//! A high-performance fixed-size mergeable "distinct heavy-hitter sketch" that
//! takes streaming (label, item) pairs. The sketch holds `size` labels and
//! corresponding cardinality sketches (HLL).
//!
//! If the label exists in the sketch, the item is added to the cardinality
//! sketch associated with the label. If the label does not exist in the sketch,
//! it samples in two stages. The cardinality of the label's set is crudely
//! estimated by taking the the trailing zeros of the item's hash value raised
//! to the power two. This estimate is then compare to the `threshold`
//! value kept by the sketch. If the estimate is greater, it is then compared to
//! the minimum cardinality in the sketch. Only if the input is greater than the
//! minimum cardinality do we drop the minimum label and replace with the new
//! label. (The threshold is then replaced by the min cardinality.)
//!
//! This has nice effects. 1) it prevents small cardinality labels from always
//! replacing the bottom of the sketch and lessens the churn that happens in the
//! Space-Saving algorithm's entries; and 2) because we're not even considering
//! any input that doesn't pass the initial threshold, we don't have calculate
//! the minimum cardinality on every input which improves the speed
//! considerably.

mod cached;
mod config;
use std::{collections::HashMap, error, fmt, fmt::Debug, hash::Hash};

use hll::HyperLogLog;
use sketch_traits::{CardinalitySketch, HeavyDistinctHitterSketch, New};

use crate::cached::Cached;
pub use crate::config::{Config, ConfigError};

#[derive(Clone, Debug)]
pub struct SamplingSpaceSavingSets<L, S>
where
    S: New,
{
    config: Config<S::Config>,
    counters: HashMap<L, Cached<S>>,
    /// the initial bar for an item to pass before being considered
    threshold: u64,
}

pub type HllSamplingSpaceSavingSets<L, I> = SamplingSpaceSavingSets<L, HyperLogLog<I>>;

impl<L, S> New for SamplingSpaceSavingSets<L, S>
where
    S: New,
    S::Config: Clone,
{
    type Config = Config<S::Config>;

    fn new(config: &Self::Config) -> Self {
        Self {
            config: config.clone(),
            counters: HashMap::with_capacity(config.max_num_counters),
            threshold: 0,
        }
    }
}

impl<L, S> HeavyDistinctHitterSketch for SamplingSpaceSavingSets<L, S>
where
    L: Eq + Hash + Clone,
    S: CardinalitySketch + New,
    S::Item: Hash,
    S::Config: Eq,
{
    type Label = L;
    type Item = S::Item;
    type MergeError = MergeError;

    #[inline]
    fn insert(&mut self, label: Self::Label, item: &Self::Item) {
        if let Some(counter) = if let Some(counter) = self.counters.get_mut(&label) {
            // The counter for the label exists; use it.
            Some(counter)
        } else if self.counters.len() < self.config.max_num_counters {
            // We have space; create a new counter.
            Some(
                self.counters
                    .entry(label)
                    .or_insert(Cached::new(&self.config.cardinality_sketch_config)),
            )
        } else {
            let cardinality_estimate = self.cardinality_estimate(&label, item);
            if cardinality_estimate > self.threshold {
                let (min_label, min_cardinality) = self
                    .counters
                    .iter()
                    .map(|(label, counter)| (label, counter.cardinality()))
                    .min_by_key(|(_, cardinality)| *cardinality)
                    .unwrap(); // set threshold to min cardinality
                self.threshold = min_cardinality;
                if cardinality_estimate > min_cardinality {
                    // The sampling threshold is reached, remap the existing counter with the minimum cardinality to the label.
                    // Remove the counter with the minimum cardinality.
                    let min_counter = self.counters.remove(&min_label.clone()).unwrap();
                    // Set threshold to the minimum cardinality.
                    self.threshold = min_counter.cardinality();
                    // Map the counter to the new label.
                    Some(self.counters.entry(label).or_insert(min_counter))
                } else {
                    None
                }
            } else {
                None
            }
        } {
            counter.insert(item)
        }
    }

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        if self.config != other.config {
            return Err(MergeError::ConfigMismatch);
        }

        self.threshold = other.threshold.min(self.threshold);

        // merge the two sets of counters
        for (l, c) in other.counters.iter() {
            self.counters
                .entry(l.clone())
                .or_insert_with(|| Cached::new(&self.config.cardinality_sketch_config))
                .merge(c)
                .unwrap_or_else(
                    // By construction, parameters cannot mismatch.
                    |_| unreachable!(),
                );
        }

        // only keep the top self.size counters
        let mut entries = self
            .counters
            .iter()
            .map(|(label, counter)| (label, counter.cardinality()))
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
            .map(|c| c.cardinality())
            .unwrap_or_else(|| {
                self.counters
                    .values()
                    .map(|c| c.cardinality())
                    .min()
                    .unwrap_or(0)
            })
    }

    fn top(&self, k: usize) -> Vec<(&L, u64)> {
        let mut entries = self
            .counters
            .iter()
            .map(|(label, counter)| (label, counter.cardinality()))
            .collect::<Vec<_>>();
        entries.sort_by_key(|&(_, cardinality)| cardinality);
        entries.into_iter().rev().take(k).collect::<Vec<_>>()
    }
}

impl<L, S> SamplingSpaceSavingSets<L, S>
where
    L: Hash,
    S: CardinalitySketch + New,
    S::Item: Hash,
{
    #[inline]
    fn cardinality_estimate(&self, _label: &L, item: &S::Item) -> u64 {
        (u64::MAX as f64 / self.config.hash_builder.hash_one(item) as f64) as u64
    }
}

impl<L, S> SamplingSpaceSavingSets<L, S>
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

    use super::*;

    const SIZE: usize = 10;
    const SEEDS: [u64; 4] = [0, 1, 2, 3];
    const COUNTER_SIZE: usize = 512;
    const HLL_SEEDS: [u64; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

    fn config() -> Config<hll::Config> {
        Config::new(
            SIZE,
            hll::Config::new(COUNTER_SIZE, Some(HLL_SEEDS)).unwrap(),
            Some(SEEDS),
        )
        .unwrap()
    }

    fn relative_error(a: u64, b: u64) -> f64 {
        let fa: f64 = a as f64;
        let fb: f64 = b as f64;
        (fa - fb).abs() / fa
    }

    #[test]
    fn ssss_counts_set_cardinality() {
        let mut sketch: HllSamplingSpaceSavingSets<char, u64> =
            SamplingSpaceSavingSets::new(&config());
        let mut exact: HashMap<char, HashSet<u64>> = HashMap::new();

        // fill up the sketch
        for label in 'a'..='j' {
            for i in 0..100 {
                sketch.insert(label, &i);
                exact.entry(label).or_default().insert(i);
            }
        }
        assert!(sketch.num_counters() == sketch.config().max_num_counters());
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
    }

    #[test]
    fn merged_ssss_counts_set_cardinality() {
        let mut sketch1: HllSamplingSpaceSavingSets<char, u64> =
            SamplingSpaceSavingSets::new(&config());
        let mut exact: HashMap<char, HashSet<u64>> = HashMap::new();

        // this sketch has disjoint labels and larger sets (a..j should get pushed out)
        let mut sketch2: HllSamplingSpaceSavingSets<char, u64> =
            SamplingSpaceSavingSets::new(&config());
        for label in 'k'..='t' {
            for i in 1..200 {
                sketch2.insert(label, &i);
                exact.entry(label).or_default().insert(i);
            }
        }
        assert!(sketch1.merge(&sketch2).is_ok());

        // this sketch overlaps on half the labels and half-disjoint items
        let mut sketch3: HllSamplingSpaceSavingSets<char, u64> =
            SamplingSpaceSavingSets::new(&config());
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

    #[test]
    fn should_merge_iff_same_config() {
        let hll_config = hll::Config::new(COUNTER_SIZE, Some(HLL_SEEDS)).unwrap();
        let config1 = Config::new(SIZE, hll_config.clone(), None).unwrap();
        let config2 = Config::new(SIZE, hll_config, None).unwrap();

        assert!(HllSamplingSpaceSavingSets::<usize, usize>::new(&config1)
            .merge(&HllSamplingSpaceSavingSets::<usize, usize>::new(&config1))
            .is_ok());
        assert!(HllSamplingSpaceSavingSets::<usize, usize>::new(&config1)
            .merge(&HllSamplingSpaceSavingSets::<usize, usize>::new(&config2))
            .is_err());
    }
}

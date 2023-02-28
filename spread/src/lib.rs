//! SpreadSketch
//!
//! A "distinct heavy-hitter sketch" that takes streaming (label, item)
//! pairs. Rust implementation of:
//!
//! Lu Tang, Qun Huang, and Patrick P. C. Lee. Spreadsketch: Toward invertible
//! and network-wide detection of superspread- ers. In IEEE INFOCOM 2020 - IEEE
//! Conference on Computer Communications, pages 1608–1617, 2020.

mod config;

use std::{
    error,
    fmt::{self, Debug},
    hash::Hash,
    iter::repeat_with,
};

use itertools::Itertools;
use sketch_traits::{CardinalitySketch, HeavyDistinctHitterSketch, New};

pub use crate::config::{Config, ConfigError};

#[derive(Clone, Debug)]
struct Bucket<L, S> {
    label: Option<L>,
    sketch: S,
    level: u8,
}

impl<L, S> Bucket<L, S>
where
    S: New,
{
    fn new(config: &S::Config) -> Self {
        Self {
            label: None,
            sketch: S::new(config),
            level: 0,
        }
    }
}

impl<L, S> Bucket<L, S>
where
    S: CardinalitySketch,
    L: Clone,
{
    fn update(&mut self, label: L, item: &S::Item, l: u8) {
        // TODO: insert the (label, item) pair instead of just item
        self.sketch.insert(item);
        if self.level <= l {
            self.label = Some(label);
            self.level = l;
        }
    }

    fn count(&self) -> u64 {
        self.sketch.cardinality()
    }

    fn merge(&mut self, other: &Self) -> Result<(), S::MergeError> {
        if other.level > self.level {
            self.level = other.level;
            self.label = other.label.clone();
        }
        self.sketch.merge(&other.sketch)
    }
}

#[derive(Clone, Debug)]
pub struct SpreadSketch<L, S>
where
    S: New,
{
    config: Config<S::Config>,
    buckets: Vec<Bucket<L, S>>,
}

impl<L, S> New for SpreadSketch<L, S>
where
    S: New,
    S::Config: Clone,
{
    type Config = Config<S::Config>;

    fn new(config: &Self::Config) -> Self {
        Self {
            config: config.clone(),
            buckets: repeat_with(|| Bucket::new(&config.cardinality_sketch_config))
                .take(config.num_rows * config.num_cols)
                .collect(),
        }
    }
}
impl<L, S> HeavyDistinctHitterSketch for SpreadSketch<L, S>
where
    L: Eq + Hash + Clone,
    S: CardinalitySketch + New,
    S::Item: Hash,
    S::Config: Eq,
{
    type Label = L;
    type Item = S::Item;
    type MergeError = MergeError;

    fn insert(&mut self, label: L, item: &S::Item) {
        let l = u8::try_from(self.global_hash(&label, item).leading_zeros()).unwrap();
        for r in 0..self.config.num_rows {
            let c = self.row_hash(r, &label);
            let bucket_index = self.bucket_index(r, c);
            self.buckets[bucket_index].update(label.clone(), item, l);
        }
    }

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        if self.config != other.config {
            return Err(MergeError::ConfigMismatch);
        }
        self.buckets
            .iter_mut()
            .zip(other.buckets.iter())
            .for_each(|(s, o)| s.merge(o).unwrap());
        Ok(())
    }

    fn clear(&mut self) {
        todo!()
    }

    fn cardinality(&self, label: &L) -> u64 {
        (0..self.config.num_rows)
            .map(|r| {
                let c = self.row_hash(r, label);
                self.bucket_index(r, c)
            })
            .map(|bucket_index| &self.buckets[bucket_index])
            .map(|bucket| bucket.count())
            .min()
            .unwrap()
    }

    fn top(&self, k: usize) -> Vec<(&L, u64)> {
        self.buckets
            .iter()
            .filter_map(|b| b.label.as_ref())
            .into_iter()
            .unique()
            .map(|l| (l, self.cardinality(l)))
            .sorted_by_key(|&(_, cardinality)| -(cardinality as i32))
            .take(k)
            .collect()
    }
}

impl<L, S> SpreadSketch<L, S>
where
    S: New,
{
    pub fn config(&self) -> &Config<S::Config> {
        &self.config
    }
}

impl<L, S> SpreadSketch<L, S>
where
    S: CardinalitySketch + New,
    L: Hash,
    S::Item: Hash,
{
    #[inline]
    fn global_hash(&self, label: &L, item: &S::Item) -> u64 {
        self.config.hash_builders[0].hash_one((item, label))
    }

    // hash function for each row;
    #[inline]
    fn row_hash(&self, row: usize, label: &L) -> usize {
        self.config.hash_builders[1].hash_one((row, label)) as usize % self.config.num_cols
    }

    #[inline]
    fn bucket_index(&self, row_index: usize, col_index: usize) -> usize {
        row_index * self.config.num_cols + col_index
    }
}

#[derive(Clone, Debug)]
pub enum MergeError {
    ConfigMismatch,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

    const SEEDS: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    const COUNTER_SIZE: usize = 512;
    const HLL_SEEDS: [u64; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

    fn seeded_hll_config(num_registers: usize) -> hll::Config {
        hll::Config::new(num_registers, Some(HLL_SEEDS)).unwrap()
    }

    fn seeded_config(num_rows: usize, num_cols: usize) -> Config<hll::Config> {
        Config::new(
            num_rows,
            num_cols,
            seeded_hll_config(COUNTER_SIZE),
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
    fn bucket_counts() {
        let mut bucket: Bucket<String, HyperLogLog<u32>> = Bucket::new(&seeded_hll_config(32));
        let label = String::from("label");
        for i in 0..100 {
            bucket.update(label.clone(), &i, 1);
        }
        assert!(relative_error(bucket.count(), 100) < 0.5);
    }

    #[test]
    fn new_sketch_works() {
        let mut sketch: SpreadSketch<String, HyperLogLog<u32>> =
            SpreadSketch::new(&Config::new(4, 10, seeded_hll_config(COUNTER_SIZE), None).unwrap());
        let set_mult = 10;
        for l in 1..100 {
            let label = l.to_string();
            for i in 0..set_mult * l {
                sketch.insert(label.clone(), &i);
            }
        }
        for l in 90..100 {
            let label = l.to_string();
            assert!(relative_error(sketch.cardinality(&label), (set_mult * l) as u64) < 0.2);
        }
    }

    #[test]
    fn seeded_sketch_works() {
        let mut sketch = SpreadSketch::<_, HyperLogLog<_>>::new(&seeded_config(4, 100));
        for l in 1..10 {
            let label = l.to_string();
            for i in 0..10 * l {
                sketch.insert(label.clone(), &i);
            }
            assert!(relative_error(sketch.cardinality(&label), (10 * l) as u64) < 0.2);
        }
    }

    #[test]
    fn merge_works() {
        let mut sketch1 = SpreadSketch::<_, HyperLogLog<_>>::new(&seeded_config(4, 100));
        let mut exact: HashMap<String, HashSet<u32>> = HashMap::new();
        for l in 1..10 {
            let label = l.to_string();
            for i in 0..10 * l {
                exact.entry(label.clone()).or_default().insert(i);
                sketch1.insert(label.clone(), &i);
            }
        }

        let mut sketch2 = SpreadSketch::<_, HyperLogLog<_>>::new(&seeded_config(4, 100));
        for l in 5..15 {
            let label = l.to_string();
            for i in 0..10 * l {
                exact.entry(label.clone()).or_default().insert(i);
                sketch2.insert(label.clone(), &i);
            }
        }

        assert!(sketch1.merge(&sketch2).is_ok());
        for l in 1..15 {
            let label = l.to_string();
            assert!(
                relative_error(
                    sketch1.cardinality(&label),
                    exact.get(&label).unwrap().len() as u64
                ) < 0.1
            );
        }
    }
}

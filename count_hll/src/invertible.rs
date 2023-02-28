use std::{collections::HashSet, hash::Hash, iter, marker::PhantomData};

use itertools::Itertools;
use sketch_traits::{HeavyDistinctHitterSketch, New};

use crate::{Config, MergeError, PointwiseSketch};

#[derive(Clone, Debug)]
pub struct LabelSetCountHLL<L, I> {
    sketch: PointwiseSketch,
    labels: HashSet<L>,
    item_type: PhantomData<I>,
}

impl<L, I> New for LabelSetCountHLL<L, I> {
    type Config = Config;

    fn new(config: &Self::Config) -> Self {
        Self {
            sketch: PointwiseSketch::new(config),
            labels: HashSet::new(),
            item_type: PhantomData,
        }
    }
}

impl<L, I> HeavyDistinctHitterSketch for LabelSetCountHLL<L, I>
where
    L: Eq + Hash + Clone,
    I: Hash,
{
    type Label = L;
    type Item = I;
    type MergeError = MergeError;

    fn insert(&mut self, label: Self::Label, item: &Self::Item) {
        self.sketch.insert(&label, &item);
        self.labels.insert(label);
    }

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        self.sketch.merge(&other.sketch)?;
        self.labels.extend(other.labels.iter().cloned());
        Ok(())
    }

    fn clear(&mut self) {
        self.sketch.clear();
        self.labels.clear();
    }

    fn cardinality(&self, label: &Self::Label) -> u64 {
        self.sketch.cardinality(label)
    }

    fn top(&self, k: usize) -> Vec<(&Self::Label, u64)> {
        self.labels
            .iter()
            .map(|label| (label, self.cardinality(label)))
            .sorted_by_key(|&(_, cardinality)| cardinality)
            .rev()
            .take(k)
            .collect::<Vec<_>>()
    }
}

impl<L, I> LabelSetCountHLL<L, I> {
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    pub fn num_registers(&self) -> usize {
        self.sketch.num_registers()
    }
}

#[derive(Clone, Debug)]
pub struct LabelArrayCountHLL<L, I> {
    sketch: PointwiseSketch,
    labels: Vec<(Option<L>, u8)>, // and their respective levels
    item_type: PhantomData<I>,
}

impl<L, I> New for LabelArrayCountHLL<L, I> {
    type Config = Config;

    fn new(config: &Self::Config) -> Self {
        Self {
            sketch: PointwiseSketch::new(config),
            labels: iter::repeat_with(|| (None, 0))
                .take(config.depth * config.width)
                .collect(),
            item_type: PhantomData,
        }
    }
}

impl<L, I> HeavyDistinctHitterSketch for LabelArrayCountHLL<L, I>
where
    L: Eq + Hash + Clone,
    I: Hash,
{
    type Label = L;
    type Item = I;
    type MergeError = MergeError;

    fn insert(&mut self, label: Self::Label, item: &Self::Item) {
        self.sketch.insert(&label, item);
        let index = self.sketch.get_index(&label, item);
        let z = self.sketch.get_z(&label, item);
        let (label_at_index, level_at_index) = &mut self.labels[index];
        if z > *level_at_index {
            (*label_at_index, *level_at_index) = (Some(label), z);
        }
    }

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        self.sketch.merge(&other.sketch)?;
        self.labels.iter_mut().zip(other.labels.iter()).for_each(
            |((s_label, s_level), (o_label, o_level))| {
                if o_level > s_level {
                    (*s_label, *s_level) = (o_label.clone(), *o_level);
                }
            },
        );
        Ok(())
    }

    fn clear(&mut self) {
        self.sketch.clear();
        self.labels.iter_mut().for_each(|l| *l = (None, 0));
    }

    fn cardinality(&self, label: &Self::Label) -> u64 {
        self.sketch.cardinality(label)
    }

    fn top(&self, k: usize) -> Vec<(&Self::Label, u64)> {
        self.labels
            .iter()
            .flat_map(|(label, _)| label)
            .unique()
            .map(|label| (label, self.cardinality(label)))
            .sorted_by_key(|&(_, cardinality)| cardinality)
            .rev()
            .take(k)
            .collect::<Vec<_>>()
    }
}

impl<L, I> LabelArrayCountHLL<L, I> {
    pub fn num_labels(&self) -> usize {
        self.sketch.config.depth * self.sketch.config.width
    }

    pub fn num_registers(&self) -> usize {
        self.sketch.num_registers()
    }
}

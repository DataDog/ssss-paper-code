use std::error;

pub trait CardinalitySketch {
    type Item;
    type MergeError: error::Error;

    fn insert(&mut self, item: &Self::Item);

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError>;

    fn clear(&mut self);

    fn cardinality(&self) -> u64;
}

pub trait HeavyDistinctHitterSketch {
    type Label;
    type Item;
    type MergeError: error::Error;

    fn insert(&mut self, label: Self::Label, item: &Self::Item);

    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError>;

    fn clear(&mut self);

    fn cardinality(&self, label: &Self::Label) -> u64;

    fn top(&self, k: usize) -> Vec<(&Self::Label, u64)>;
}

pub trait New {
    type Config;

    fn new(config: &Self::Config) -> Self;
}

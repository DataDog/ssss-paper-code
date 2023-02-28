use sketch_traits::{CardinalitySketch, New};

#[derive(Clone, Debug)]
pub(crate) struct Cached<S> {
    sketch: S,
    cardinality: u64,
}

impl<S> New for Cached<S>
where
    S: New,
{
    type Config = S::Config;

    #[inline]
    fn new(config: &Self::Config) -> Self {
        Self {
            sketch: S::new(config),
            cardinality: 0,
        }
    }
}

impl<S> CardinalitySketch for Cached<S>
where
    S: CardinalitySketch,
{
    type Item = S::Item;
    type MergeError = S::MergeError;

    #[inline]
    fn insert(&mut self, item: &Self::Item) {
        self.sketch.insert(item);
        self.cardinality = self.sketch.cardinality();
    }

    #[inline]
    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        self.sketch.merge(&other.sketch)?;
        self.cardinality = self.sketch.cardinality();
        Ok(())
    }

    #[inline]
    fn clear(&mut self) {
        self.sketch.clear();
        self.cardinality = 0;
    }

    #[inline]
    fn cardinality(&self) -> u64 {
        self.cardinality
    }
}

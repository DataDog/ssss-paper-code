use sketch_traits::CardinalitySketch;

/// What to do with a sketch before mapping it to a different label.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ResetStrategy {
    /// Sketches are reused as they are across labels.
    Recycle,
    /// Sketches are cleared and their cardinalities are offset by their cardinality before clearing.
    Offset,
}

#[derive(Clone, Debug)]
pub(crate) struct Counter<S> {
    pub(crate) sketch: S,
    pub(crate) offset: u64,
}

impl<S> Counter<S> {
    pub(crate) fn new(sketch: S) -> Self {
        Self { sketch, offset: 0 }
    }
}

impl<S> Counter<S>
where
    S: CardinalitySketch,
{
    #[inline]
    pub(crate) fn reset(&mut self, reset_strategy: &ResetStrategy) {
        match reset_strategy {
            ResetStrategy::Recycle => {
                // do nothing; we're going to keep using the sketch as is
            }
            ResetStrategy::Offset => {
                self.offset += self.sketch.cardinality();
                self.sketch.clear();
            }
        }
    }

    pub(crate) fn offset_cardinality(&self) -> u64 {
        self.sketch.cardinality() + self.offset
    }
}

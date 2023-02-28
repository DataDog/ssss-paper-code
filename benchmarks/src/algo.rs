use std::fmt;

use hll::HyperLogLog;
use sketch_traits::{HeavyDistinctHitterSketch, New};

pub trait Algorithm: fmt::Display {
    type Sketch<L, I>;

    fn new_sketch<L, I>(&self, sketch_size: usize, counter_size: usize) -> Self::Sketch<L, I>
    where
        Self::Sketch<L, I>: HeavyDistinctHitterSketch<Label = L, Item = I>;

    fn optimal_counter_size(&self) -> usize;
}

#[derive(Clone, Debug)]
pub struct Achll;
impl Algorithm for Achll {
    type Sketch<L, I> = count_hll::LabelArrayCountHLL<L, I>;

    fn new_sketch<L, I>(&self, sketch_size: usize, counter_size: usize) -> Self::Sketch<L, I>
    where
        Self::Sketch<L, I>: HeavyDistinctHitterSketch<Label = L, Item = I>,
    {
        count_hll::LabelArrayCountHLL::new(
            &count_hll::Config::new(
                counter_size,
                sketch_size,
                Some([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            )
            .unwrap(),
        )
    }

    fn optimal_counter_size(&self) -> usize {
        512
    }
}
impl fmt::Display for Achll {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Count-HLL")
    }
}

#[derive(Clone, Debug)]
pub struct Schll;
impl Algorithm for Schll {
    type Sketch<L, I> = count_hll::LabelSetCountHLL<L, I>;

    fn new_sketch<L, I>(&self, sketch_size: usize, counter_size: usize) -> Self::Sketch<L, I>
    where
        Self::Sketch<L, I>: HeavyDistinctHitterSketch<Label = L, Item = I>,
    {
        count_hll::LabelSetCountHLL::new(
            &count_hll::Config::new(
                counter_size,
                sketch_size,
                Some([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            )
            .unwrap(),
        )
    }

    fn optimal_counter_size(&self) -> usize {
        512
    }
}
impl fmt::Display for Schll {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SCHLL")
    }
}

#[derive(Clone, Debug)]
pub struct Osss;
impl Algorithm for Osss {
    type Sketch<L, I> = sss::SpaceSavingSets<L, HyperLogLog<I>>;

    fn new_sketch<L, I>(&self, sketch_size: usize, counter_size: usize) -> Self::Sketch<L, I>
    where
        Self::Sketch<L, I>: HeavyDistinctHitterSketch<Label = L, Item = I>,
    {
        sss::SpaceSavingSets::new(
            &sss::Config::new(
                sketch_size,
                sss::ResetStrategy::Offset,
                hll::Config::new(counter_size, Some([0, 1, 2, 3, 4, 5, 6, 7])).unwrap(),
            )
            .unwrap(),
        )
    }

    fn optimal_counter_size(&self) -> usize {
        1024
    }
}
impl fmt::Display for Osss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OSSS")
    }
}

#[derive(Clone, Debug)]
pub struct Rsss;
impl Algorithm for Rsss {
    type Sketch<L, I> = sss::SpaceSavingSets<L, HyperLogLog<I>>;

    fn new_sketch<L, I>(&self, sketch_size: usize, counter_size: usize) -> Self::Sketch<L, I>
    where
        Self::Sketch<L, I>: HeavyDistinctHitterSketch<Label = L, Item = I>,
    {
        sss::SpaceSavingSets::new(
            &sss::Config::new(
                sketch_size,
                sss::ResetStrategy::Recycle,
                hll::Config::new(counter_size, Some([0, 1, 2, 3, 4, 5, 6, 7])).unwrap(),
            )
            .unwrap(),
        )
    }

    fn optimal_counter_size(&self) -> usize {
        1024
    }
}
impl fmt::Display for Rsss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RSSS")
    }
}

#[derive(Clone, Debug)]
pub struct Spread;
impl Spread {
    pub const DEPTH: usize = 4;
}

impl Algorithm for Spread {
    type Sketch<L, I> = spread::SpreadSketch<L, HyperLogLog<I>>;

    fn new_sketch<L, I>(&self, sketch_size: usize, counter_size: usize) -> Self::Sketch<L, I>
    where
        Self::Sketch<L, I>: HeavyDistinctHitterSketch<Label = L, Item = I>,
    {
        spread::SpreadSketch::new(
            &spread::Config::new(
                Self::DEPTH,
                sketch_size,
                hll::Config::new(counter_size, Some([0, 1, 2, 3, 4, 5, 6, 7])).unwrap(),
                Some([0, 1, 2, 3, 4, 5, 6, 7]),
            )
            .unwrap(),
        )
    }

    fn optimal_counter_size(&self) -> usize {
        64
    }
}
impl fmt::Display for Spread {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Spread")
    }
}

#[derive(Clone, Debug)]
pub struct Ssss;
impl Algorithm for Ssss {
    type Sketch<L, I> = ssss::HllSamplingSpaceSavingSets<L, I>;

    fn new_sketch<L, I>(&self, sketch_size: usize, counter_size: usize) -> Self::Sketch<L, I>
    where
        Self::Sketch<L, I>: HeavyDistinctHitterSketch<Label = L, Item = I>,
    {
        ssss::SamplingSpaceSavingSets::new(
            &ssss::Config::new(
                sketch_size,
                hll::Config::new(counter_size, Some([0, 1, 2, 3, 4, 5, 6, 7])).unwrap(),
                Some([0, 1, 2, 3]),
            )
            .unwrap(),
        )
    }

    fn optimal_counter_size(&self) -> usize {
        1024
    }
}
impl fmt::Display for Ssss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SSSS")
    }
}

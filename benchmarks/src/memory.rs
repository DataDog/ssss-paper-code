use std::mem::size_of;

use ahash::RandomState;
use hll::HyperLogLog;
use sketch_traits::New;

use crate::algo;

const LABEL_SIZE: usize = size_of::<u32>(); // using u32 as a proxy for the label size
const MEGABYTE: usize = 1_048_576;

pub trait MemorySize {
    /// Estimates the memory footprint of the object.
    // TODO: comment about it being a lower bound for HashMap.
    fn mem_size(&self) -> usize;
}

fn hll_mem_size(size: usize) -> usize {
    size_of::<usize>() * 2
        + size_of::<u8>() * size
        + size_of::<RandomState>() * 2
        + size_of::<f64>() * 2
}

fn recycling_mem_size(counter_size: usize) -> usize {
    size_of::<usize>() + hll_mem_size(counter_size)
}

impl<L, I> MemorySize for count_hll::LabelArrayCountHLL<L, I> {
    fn mem_size(&self) -> usize {
        let _constants =
            size_of::<usize>() * 3 + size_of::<RandomState>() * 3 + size_of::<f64>() * 64;
        let pointwise_mem_size = size_of::<u8>() * self.num_registers();
        let size_of_labels = LABEL_SIZE * self.num_labels();
        let size_of_levels = size_of::<u8>() * self.num_registers();
        pointwise_mem_size + size_of_labels + size_of_levels
    }
}

impl<L, I> MemorySize for count_hll::LabelSetCountHLL<L, I> {
    fn mem_size(&self) -> usize {
        let _constants =
            size_of::<usize>() * 3 + size_of::<RandomState>() * 3 + size_of::<f64>() * 64;
        let pointwise_mem_size = size_of::<u8>() * self.num_registers();
        let size_of_labels = size_of::<L>() * self.num_labels();
        pointwise_mem_size + size_of_labels
    }
}

impl<L, S> MemorySize for spread::SpreadSketch<L, S>
where
    S: New<Config = hll::Config>,
{
    fn mem_size(&self) -> usize {
        let _constants = size_of::<usize>() * 2 + size_of::<RandomState>() * 2;
        let size_of_bucket = size_of::<u8>()
            + LABEL_SIZE
            + hll_mem_size(self.config().cardinality_sketch_config().num_registers());
        size_of_bucket * self.config().num_rows() * self.config().num_cols()
    }
}

fn sss_counter_size(
    reset_strategy: &sss::ResetStrategy,
    cardinality_sketch_config: &hll::Config,
) -> usize {
    match reset_strategy {
        sss::ResetStrategy::Recycle => {
            size_of::<usize>() * 2 + hll_mem_size(cardinality_sketch_config.num_registers())
        }
        sss::ResetStrategy::Offset => {
            size_of::<usize>() + hll_mem_size(cardinality_sketch_config.num_registers())
        }
    }
}

impl<L, I> MemorySize for sss::SpaceSavingSets<L, HyperLogLog<I>>
where
    HyperLogLog<I>: New<Config = hll::Config>,
{
    fn mem_size(&self) -> usize {
        let _constants = size_of::<usize>();
        (sss_counter_size(
            self.config().reset_strategy(),
            self.config().cardinality_sketch_config(),
        ) + LABEL_SIZE)
            * self.num_counters()
    }
}

impl<L, I> MemorySize for ssss::SamplingSpaceSavingSets<L, HyperLogLog<I>>
where
    HyperLogLog<I>: New<Config = hll::Config>,
{
    fn mem_size(&self) -> usize {
        let _config_mem_size = size_of::<usize>() * 2 + size_of::<u64>();
        let _constants = _config_mem_size + size_of::<usize>() + size_of::<RandomState>() * 2;

        (recycling_mem_size(self.config().cardinality_sketch_config().num_registers()) + LABEL_SIZE)
            * self.num_counters()
    }
}

pub trait MaxCapacity {
    /// Returns the maximum number of entries for a sketch to be less than `memory` MBs
    fn entries_for_mbs(&self, memory: f32, counter_size: usize) -> usize;
}

impl MaxCapacity for algo::Achll {
    fn entries_for_mbs(&self, memory: f32, counter_size: usize) -> usize {
        let column_size = (LABEL_SIZE + size_of::<u8>() + size_of::<u8>()) * counter_size;
        (memory * MEGABYTE as f32 / column_size as f32) as usize
    }
}

impl MaxCapacity for algo::Schll {
    fn entries_for_mbs(&self, _memory: f32, _counter_size: usize) -> usize {
        unimplemented!("who knows how big the label set will be")
    }
}

impl MaxCapacity for algo::Spread {
    fn entries_for_mbs(&self, memory: f32, counter_size: usize) -> usize {
        let size_of_bucket = size_of::<u8>() + LABEL_SIZE + hll_mem_size(counter_size);
        let size_of_row = size_of_bucket * algo::Spread::DEPTH;
        (memory * MEGABYTE as f32 / size_of_row as f32) as usize
    }
}

impl MaxCapacity for algo::Osss {
    fn entries_for_mbs(&self, memory: f32, counter_size: usize) -> usize {
        (memory * MEGABYTE as f32
            / (sss_counter_size(
                &sss::ResetStrategy::Offset,
                &hll::Config::new(counter_size, None).unwrap(),
            ) + LABEL_SIZE) as f32) as usize
    }
}

impl MaxCapacity for algo::Rsss {
    fn entries_for_mbs(&self, memory: f32, counter_size: usize) -> usize {
        (memory * MEGABYTE as f32
            / (sss_counter_size(
                &sss::ResetStrategy::Recycle,
                &hll::Config::new(counter_size, None).unwrap(),
            ) + LABEL_SIZE) as f32) as usize
    }
}

impl MaxCapacity for algo::Ssss {
    fn entries_for_mbs(&self, memory: f32, counter_size: usize) -> usize {
        (memory * MEGABYTE as f32 / (recycling_mem_size(counter_size) + LABEL_SIZE) as f32) as usize
    }
}

#[cfg(test)]
mod tests {
    use sketch_traits::HeavyDistinctHitterSketch;

    use super::{MaxCapacity, MemorySize, MEGABYTE};
    use crate::algo;

    #[test]
    fn figures_out_entry_sizes_for_memory() {
        for memory in [1.0, 2.0, 3.0, 4.0, 5.0, 10.0] {
            for counter_size in [256, 512, 1024] {
                print_sizes(&algo::Achll, memory, counter_size);
                print_sizes(&algo::Osss, memory, counter_size);
                print_sizes(&algo::Rsss, memory, counter_size);
                print_sizes(&algo::Spread, memory, counter_size);
                print_sizes(&algo::Ssss, memory, counter_size);
            }
        }

        fn print_sizes<A>(algo: &A, memory: f32, counter_size: usize)
        where
            A: algo::Algorithm + MaxCapacity,
            A::Sketch<u32, u32>: HeavyDistinctHitterSketch<Label = u32, Item = u32> + MemorySize,
        {
            let entries = algo.entries_for_mbs(memory, counter_size);
            let sketch = algo.new_sketch::<u32, u32>(entries, counter_size);
            assert!(sketch.mem_size() as f32 / MEGABYTE as f32 <= memory);
            println!(
                "{:.1} MB limit; {} Entries; {} Counter Size; {:.1} kB; {}",
                memory,
                entries,
                counter_size,
                sketch.mem_size() as f32 / 1024.0,
                algo,
            );
        }
    }
}

use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    fmt::Debug,
    hash::Hash,
    mem::size_of_val,
};

use itertools::Itertools;
use sketch_traits::HeavyDistinctHitterSketch;

use crate::memory::MemorySize;

#[derive(Clone, Debug, Default)]
pub struct GroundTruth<L, I> {
    sets: HashMap<L, HashSet<I>>,
}

impl<L, I> HeavyDistinctHitterSketch for GroundTruth<L, I>
where
    L: Eq + Hash + Clone + Debug,
    I: Eq + Hash + Clone + Debug,
{
    type Label = L;
    type Item = I;
    type MergeError = Infallible;

    #[inline]
    fn insert(&mut self, label: Self::Label, item: &Self::Item) {
        self.sets.entry(label).or_default().insert(item.clone());
    }

    #[inline]
    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        for (k, v) in other.sets.iter() {
            self.sets.entry(k.clone()).or_default().extend(v.clone());
        }
        Ok(())
    }

    #[inline]
    fn clear(&mut self) {
        todo!()
    }

    #[inline]
    fn cardinality(&self, label: &Self::Label) -> u64 {
        self.sets.get(label).map(HashSet::len).unwrap_or(0) as u64
    }

    #[inline]
    fn top(&self, k: usize) -> Vec<(&Self::Label, u64)> {
        self.top_cardinalities().take(k).collect::<Vec<_>>()
    }
}

impl<L, I> MemorySize for GroundTruth<L, I> {
    fn mem_size(&self) -> usize {
        self.sets
            .iter()
            .map(|(l, i)| size_of_val(l) + size_of_val(i))
            .sum::<usize>()
    }
}

impl<L, I> GroundTruth<L, I>
where
    L: Eq + Hash + Clone + Debug,
    I: Debug,
{
    pub fn new() -> Self {
        Self {
            sets: HashMap::new(),
        }
    }

    pub fn num_labels(&self) -> usize {
        self.sets.len()
    }

    pub fn top_cardinalities(&self) -> impl Iterator<Item = (&L, u64)> {
        self.sets
            .iter()
            .map(|(label, items)| (label, items.len() as u64))
            .sorted_by_key(|&(_, cardinality)| cardinality)
            .rev()
    }

    pub fn print_top(
        &self,
        sketch: &impl HeavyDistinctHitterSketch<Label = L, Item = I>,
        k: usize,
    ) {
        println!("Top {} (actual), real count, sketch count", k);
        let top_k: Vec<_> = self.top_cardinalities().take(k).collect();
        let mut counter = 0;
        for (label, cardinality) in top_k {
            counter += 1;
            println!(
                "{}: {:?} {:?} {:?}",
                counter,
                label,
                cardinality,
                sketch.cardinality(label)
            );
        }
    }

    #[allow(dead_code)]
    pub fn print_sketch_top(&self, sketch_top: Vec<(&L, usize)>, k: usize) {
        println!("Top {} (sketch), real count, sketch count", k);
        let mut counter = 0;
        for (label, cardinality) in sketch_top {
            counter += 1;
            if counter > k {
                break;
            }
            println!(
                "{}: {:?} {:?} {:?}",
                counter,
                label,
                self.sets.get(label).unwrap().len(),
                cardinality
            );
        }
    }

    pub fn l1norm(&self) -> f64 {
        self.sets.values().map(|items| (items.len())).sum::<usize>() as f64
    }

    pub fn l2norm2(&self) -> f64 {
        self.sets
            .values()
            .map(|items| usize::pow(items.len(), 2))
            .sum::<usize>() as f64
    }

    pub fn mean(&self) -> f64 {
        self.l1norm() / self.sets.len() as f64
    }

    pub fn percentile(&self, p: f64) -> usize {
        let mut sizes: Vec<usize> = self.sets.values().map(|items| (items.len())).collect();
        sizes.sort();
        // TODO: deal with boundary conditions
        let location = (p * sizes.len() as f64) as usize;
        sizes[location]
    }

    pub fn max(&self) -> usize {
        self.sets.values().map(|items| (items.len())).max().unwrap()
    }

    /// Return an iterator (in true cardinality order) over the relative errors
    /// of each cardinality sketch
    pub fn rel_errors<'a>(
        &'a self,
        sketch: &'a impl HeavyDistinctHitterSketch<Label = L, Item = I>,
    ) -> impl Iterator<Item = f64> + 'a {
        self.top_cardinalities().map(|(label, cardinality)| {
            let sketch_cardinality = sketch.cardinality(label);
            (sketch_cardinality as f64 - cardinality as f64).abs() / cardinality as f64
        })
    }

    /// Relative Mean Absolute Error over Actual Top
    pub fn actual_rmae<'a>(
        &'a self,
        sketch: &'a impl HeavyDistinctHitterSketch<Label = L, Item = I>,
        k: usize,
    ) -> f64 {
        rel_l1(&mut self.rel_errors(sketch), k)
    }

    /// Relative Root Mean Square Error over Actual Top
    pub fn actual_rrmse<'a>(
        &'a self,
        sketch: &'a impl HeavyDistinctHitterSketch<Label = L, Item = I>,
        k: usize,
    ) -> f64 {
        rel_l2(&mut self.rel_errors(sketch), k)
    }

    /// Return an iterator (in true cardinality order) over the absolute errors
    /// of each cardinality sketch
    pub fn abs_errors<'a>(
        &'a self,
        sketch: &'a impl HeavyDistinctHitterSketch<Label = L, Item = I>,
    ) -> impl Iterator<Item = f64> + 'a {
        self.top_cardinalities().map(|(label, cardinality)| {
            let sketch_cardinality = sketch.cardinality(label);
            (sketch_cardinality as f64 - cardinality as f64).abs()
        })
    }

    /// Calculate the normalized absolute error over everything (n things)
    pub fn total_nae(&self, sketch: &impl HeavyDistinctHitterSketch<Label = L, Item = I>) -> f64 {
        self.abs_errors(sketch)
            .sum::<f64>()
            .mul_add(self.l1norm().recip(), 0.0) // divide by l1 norm
    }

    /// Calculate the normalized absolute error over the top k
    pub fn top_nae(
        &self,
        sketch: &impl HeavyDistinctHitterSketch<Label = L, Item = I>,
        k: usize,
    ) -> f64 {
        let l1 = self
            .top_cardinalities()
            .take(k)
            .map(|(_, c)| c)
            .sum::<u64>() as f64;

        self.abs_errors(sketch).take(k).map(|e| e / l1).sum::<f64>()
    }

    /// Calculate the normalized square error over everything (n things)
    pub fn total_nrse(&self, sketch: &impl HeavyDistinctHitterSketch<Label = L, Item = I>) -> f64 {
        let l2 = self.l2norm2();
        self.abs_errors(sketch)
            .map(|e| e.powf(2.0) / l2)
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate the normalized square error over the top k
    pub fn top_nrse(
        &self,
        sketch: &impl HeavyDistinctHitterSketch<Label = L, Item = I>,
        k: usize,
    ) -> f64 {
        let l2 = self
            .top_cardinalities()
            .take(k)
            .map(|(_, c)| c.pow(2))
            .sum::<u64>() as f64;

        self.abs_errors(sketch)
            .take(k)
            .map(|e| e.powf(2.0) / l2)
            .sum::<f64>()
            .sqrt()
    }

    pub fn worst_err(
        &self,
        sketch: &impl HeavyDistinctHitterSketch<Label = L, Item = I>,
        k: usize,
    ) {
        let top_k: Vec<_> = self.top_cardinalities().take(k).collect();
        let mut worst_error_entry: ErrorEntry<L> = ErrorEntry::new(top_k[0].0.clone());

        top_k
            .into_iter()
            .enumerate()
            .for_each(|(counter, (label, cardinality))| {
                let error_entry: ErrorEntry<L> = ErrorEntry {
                    label: label.clone(),
                    true_rank: counter,
                    true_size: cardinality,
                    approx_size: sketch.cardinality(label),
                };
                if error_entry.rel_l1() > worst_error_entry.rel_l1() {
                    worst_error_entry = error_entry;
                }
            });
        println!("Worst Error Entry:");
        worst_error_entry.print();
    }

    /// Return an iterator (in sketch cardinality order) over the relative errors of
    /// each cardinality sketch
    pub fn rel_sketch_errors<'a>(
        &'a self,
        sketch_top: impl Iterator<Item = &'a (&'a L, u64)> + 'a,
    ) -> impl Iterator<Item = f64> + 'a {
        sketch_top.map(|(label, cardinality)| {
            let true_cardinality = self.sets.get(label).unwrap().len();
            (true_cardinality as f64 - *cardinality as f64).abs() / true_cardinality as f64
        })
    }

    /// Relative Mean Absolute Error over Sketch Top
    pub fn sketch_rmae(&self, sketch_top: &Vec<(&L, u64)>) -> f64 {
        let n = sketch_top.len();

        rel_l1(&mut self.rel_sketch_errors(sketch_top.iter()), n)
    }

    /// Relative Root Mean Square Error over Sketch Top
    pub fn sketch_rrmse(&self, sketch_top: &Vec<(&L, u64)>) -> f64 {
        let n = sketch_top.len();
        rel_l2(&mut self.rel_sketch_errors(sketch_top.iter()), n)
    }

    /// Return an iterator (in sketch cardinality order) over the absolute errors of
    /// each cardinality sketch
    pub fn abs_sketch_errors<'a>(
        &'a self,
        sketch_top: impl Iterator<Item = &'a (&'a L, u64)> + 'a,
    ) -> impl Iterator<Item = f64> + 'a {
        sketch_top.map(|(label, cardinality)| {
            let true_cardinality = self.sets.get(label).unwrap().len();
            (true_cardinality as f64 - *cardinality as f64).abs()
        })
    }

    /// Normalized Absolute Error over Sketch Top
    pub fn sketch_nae(&self, sketch_top: &[(&L, u64)]) -> f64 {
        let l1 = sketch_top
            .iter()
            .map(|(l, _)| self.sets.get(l).unwrap().len())
            .sum::<usize>() as f64;

        self.abs_sketch_errors(sketch_top.iter())
            .sum::<f64>()
            .mul_add(l1.recip(), 0.0) // divide by l1
    }

    /// Normalized Root Squared Error over Sketch Top
    pub fn sketch_nrse(&self, sketch_top: &[(&L, u64)]) -> f64 {
        let l2 = sketch_top
            .iter()
            .map(|(l, _)| usize::pow(self.sets.get(l).unwrap().len(), 2))
            .sum::<usize>() as f64;

        self.abs_sketch_errors(sketch_top.iter())
            .map(|e| e.powf(2.0) / l2)
            .sum::<f64>()
            .sqrt()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ErrorEntry<L> {
    label: L,
    true_rank: usize,
    true_size: u64,
    approx_size: u64,
}

impl<L> ErrorEntry<L>
where
    L: Eq + Hash + Debug,
{
    fn new(label: L) -> Self {
        Self {
            label,
            true_rank: 0,
            true_size: 0,
            approx_size: 0,
        }
    }

    fn rel_l1(&self) -> f64 {
        if self.true_size == 0 {
            0.0
        } else {
            (self.true_size as f64 - self.approx_size as f64).abs() / self.true_size as f64
        }
    }

    fn rel_l2(&self) -> f64 {
        if self.true_size == 0 {
            0.0
        } else {
            ((self.true_size as f64 - self.approx_size as f64).abs() / self.true_size as f64)
                .powf(2.0)
        }
    }

    fn print(&self) {
        println!(
            "{}: {:?} {} {} L1: {:.2}% L2: {:.2}%",
            self.true_rank,
            self.label,
            self.true_size,
            self.approx_size,
            self.rel_l1() * 100.0,
            self.rel_l2() * 100.0,
        )
    }
}

/// Calculate the L1 relative error
pub fn rel_l1(rel_errs: &mut dyn Iterator<Item = f64>, k: usize) -> f64 {
    rel_errs
        .take(k)
        .sum::<f64>()
        .mul_add((k as f64).recip(), 0.0) // divide by k
}

/// Calculate the L2 relative error over the top k
pub fn rel_l2(rel_errs: &mut dyn Iterator<Item = f64>, k: usize) -> f64 {
    rel_errs
        .take(k)
        .map(|error| error.powf(2.0) / k as f64)
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use sketch_traits::HeavyDistinctHitterSketch;

    use super::GroundTruth;
    use crate::memory::MemorySize;

    #[test]
    fn ground_truth_is_exact() {
        let mut ground_truth: GroundTruth<u32, u32> = GroundTruth::new();
        let mut great_sketch: GroundTruth<u32, u32> = GroundTruth::new();
        let mut bad_sketch: GroundTruth<u32, u32> = GroundTruth::new();
        for i in 1..101 {
            for j in 0..i {
                ground_truth.insert(i, &j);
                great_sketch.insert(i, &j);
                bad_sketch.insert(101 - i, &j);
            }
        }
        assert!(ground_truth.num_labels() == 100);
        assert!(ground_truth.mean() == 50.5);
        assert!(ground_truth.percentile(0.5) == 51); // TODO: verify that this is what we want
        assert!(ground_truth.max() == 100);
        assert!(ground_truth.actual_rmae(&great_sketch, 10) == 0.0);
        assert!(ground_truth.actual_rrmse(&great_sketch, 10) == 0.0);
        assert!(ground_truth.top_nae(&great_sketch, 10) == 0.0);
        assert!(ground_truth.total_nae(&great_sketch) == 0.0);
        assert!(ground_truth.top_nrse(&great_sketch, 10) == 0.0);
        assert!(ground_truth.total_nrse(&great_sketch) == 0.0);
        assert!(ground_truth.sketch_rmae(&great_sketch.top(10)) == 0.0);
        assert!(ground_truth.sketch_rrmse(&great_sketch.top(10)) == 0.0);
        assert!(ground_truth.sketch_nae(&great_sketch.top(10)) == 0.0);
        assert!(ground_truth.sketch_nrse(&great_sketch.top(10)) == 0.0);

        let rel_errs = [99.0_f64 / 1., 97. / 2., 95. / 3., 93. / 4., 91. / 5.];
        assert!(
            (ground_truth.sketch_rmae(&bad_sketch.top(5)) - rel_errs.iter().sum::<f64>() / 5.)
                .abs()
                < 0.0001
        );
        assert!(
            (ground_truth.sketch_rrmse(&bad_sketch.top(5))
                - rel_errs
                    .iter()
                    .map(|r| r.powf(2.0) / 5.0)
                    .sum::<f64>()
                    .sqrt()
                < 0.0001)
        );

        // test merging
        great_sketch.insert(1, &1);
        great_sketch.insert(101, &1);
        assert!(ground_truth.merge(&great_sketch).is_ok());
        assert!(ground_truth.num_labels() == 101);
        assert!(ground_truth.sets.get(&1).unwrap().len() == 2);

        ground_truth.worst_err(&great_sketch, 10);
        ground_truth.print_top(&great_sketch, 10);
        ground_truth.mem_size();
    }
}

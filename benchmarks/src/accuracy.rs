use std::{collections::HashSet, fmt::Debug, hash::Hash, iter};

use sketch_traits::HeavyDistinctHitterSketch;

use crate::{
    algo::{self, Algorithm},
    data::Dataset,
    exact::{rel_l1, rel_l2, GroundTruth},
};

#[allow(dead_code)]
fn run_generative_case<A, D>(num_entries: usize, k: u32, algorithm: &A, dataset: &D, verbose: bool)
where
    A: Algorithm,
    D: Dataset,
    A::Sketch<D::Label, D::Item>: HeavyDistinctHitterSketch<Label = D::Label, Item = D::Item>,
    D::Label: Eq + Hash + Clone + Debug,
    D::Item: Eq + Hash + Clone + Debug,
{
    let mut entries = HashSet::new();
    let mut items = HashSet::new();
    let mut ground_truth = GroundTruth::new();
    let counter_size = 1024;
    let mut sketch = algorithm.new_sketch(k.try_into().unwrap(), counter_size);

    iter::repeat_with(|| dataset.iter())
        .flatten()
        .take(num_entries)
        .for_each(|(label, item)| {
            entries.insert((label.clone(), item.clone()));
            items.insert(item.clone());
            ground_truth.insert(label.clone(), &item);
            sketch.insert(label, &item);
        });

    if verbose {
        println!("{}", dataset);
        println!("Num Entries: {}", num_entries);
        println!("Num Labels: {}", ground_truth.num_labels());
        println!("Unique Entries: {}", entries.len());
        println!("Unique Items: {}", items.len());
        println!("k: {}", k);
        let heavy_sets = ground_truth
            .top_cardinalities()
            .take(k as usize)
            .collect::<Vec<_>>();
        println!("Ground Truth: {:?}", heavy_sets);
        println!();

        println!("{}:", algorithm);
        println!(
            "Relative L1 Error over Top {}: {:.2}%",
            k,
            rel_l1(&mut ground_truth.rel_errors(&sketch), k as usize) * 100.0,
        );
        println!(
            "Relative L2 Error over Top {}: {:.2}%",
            k,
            rel_l2(&mut ground_truth.rel_errors(&sketch), k as usize) * 100.0,
        );
        println!("{:?}", sketch.top(k as usize));
        println!()
    }
}

#[allow(dead_code)]
fn run_generative_case_for_all_algo<D>(num_entries: usize, k: u32, dataset: &D, verbose: bool)
where
    D: Dataset,
    D::Label: Eq + Hash + Clone + Debug,
    D::Item: Eq + Hash + Clone + Debug,
{
    run_generative_case(num_entries, k, &algo::Achll, dataset, verbose);
    run_generative_case(num_entries, k, &algo::Schll, dataset, verbose);
    run_generative_case(num_entries, k, &algo::Osss, dataset, verbose);
    run_generative_case(num_entries, k, &algo::Rsss, dataset, verbose);
    run_generative_case(num_entries, k, &algo::Ssss, dataset, verbose);
}

#[test]
fn run_generative_cases() {
    let k = 10;
    let verbose = true;

    use crate::data;
    run_generative_case_for_all_algo(1000, k, &data::synth::Uniform::new(k), verbose);
    run_generative_case_for_all_algo(1000, k, &data::synth::Poisson::new(k), verbose);
    run_generative_case_for_all_algo(1000, k, &data::synth::Repeats::new(k), verbose);
    run_generative_case_for_all_algo(1000, k, &data::synth::CycleSingleItem::new(k), verbose);
    run_generative_case_for_all_algo(1000, k, &data::synth::CycleUniqueItems::new(k), verbose);
    run_generative_case_for_all_algo(100_000, 1000, &data::synth::OneLabel, verbose);
    run_generative_case_for_all_algo(100, 1000, &data::synth::OneLabel, verbose);
    run_generative_case_for_all_algo(100, 100, &data::synth::OneLabel, verbose);
}

use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    fs::read_dir,
    hash::Hash,
    mem::size_of,
    path::PathBuf,
    time::Instant,
};

use num_format::{Locale, ToFormattedString};
use sketch_traits::HeavyDistinctHitterSketch;

use crate::{
    algo::Algorithm,
    data::{Dataset, FileDataset, FolderDataset},
    data::synth::{Overlap, Zipf},
    exact::GroundTruth,
    memory::{MaxCapacity, MemorySize},
    specialized_dispatch, SketchType,
};

pub fn dataset_ground_truth<L, I>(
    dataset: &impl Dataset<Label = L, Item = I>,
    verbose: bool,
) -> GroundTruth<L, I>
where
    L: Eq + Hash + Clone + Debug,
    I: Eq + Hash + Clone + Debug,
{
    let mut ground_truth = GroundTruth::new();
    let mut entries = HashSet::new();
    let mut items = HashSet::new();
    let mut label_count = HashMap::new();
    let mut num_entries = 0;
    let start = Instant::now();

    // get ground truth
    for (label, item) in dataset.iter() {
        entries.insert((label.clone(), item.clone()));
        items.insert(item.clone());
        *label_count.entry(label.clone()).or_insert(0) += 1;
        ground_truth.insert(label.clone(), &item);
        num_entries += 1;
    }

    if verbose {
        let top100: HashSet<L> = ground_truth
            .top_cardinalities()
            .take(100)
            .map(|(l, _)| l.clone())
            .collect::<HashSet<_>>();

        let top100_entries: usize = label_count
            .iter()
            .filter(|(l, _)| top100.contains(l))
            .map(|(_, c)| c)
            .sum();

        let top1000: HashSet<L> = ground_truth
            .top_cardinalities()
            .take(1000)
            .map(|(l, _)| l.clone())
            .collect::<HashSet<_>>();

        let top1000_entries: usize = label_count
            .iter()
            .filter(|(l, _)| top1000.contains(l))
            .map(|(_, c)| c)
            .sum();

        println!(
            "Num Entries: {}",
            num_entries.to_formatted_string(&Locale::en)
        );
        println!(
            "Unique Entries: {} ({:.0}%)",
            entries.len(),
            100.0 * (entries.len() as f64 / num_entries as f64)
        );
        println!(
            "Top 100 Label Entries: {} ({:.0}%)",
            top100_entries,
            100.0 * (top100_entries as f64 / num_entries as f64)
        );
        println!(
            "Top 1000 Label Entries: {} ({:.0}%)",
            top1000_entries,
            100.0 * (top1000_entries as f64 / num_entries as f64)
        );
        println!("Unique Items: {}", items.len());
        println!(
            "Num Labels: {} ({:.1} MB)",
            ground_truth.num_labels(),
            (size_of::<u64>() * ground_truth.num_labels()) as f64 / 1_048_576.0
        );
        println!("Mean Label Set Sizes: {:.1?}", ground_truth.mean());
        println!(
            "p25/p50/p75 Set Sizes: {:.0?} {:.0?} {:.0?}",
            ground_truth.percentile(0.25),
            ground_truth.percentile(0.5),
            ground_truth.percentile(0.75),
        );
        println!(
            "p90/p95/p99 Set Sizes: {:.0?} {:.0?} {:.0?}",
            ground_truth.percentile(0.90),
            ground_truth.percentile(0.95),
            ground_truth.percentile(0.99),
        );
        println!(
            "p999/p9999/max Set Sizes: {:.0?} {:.0?} {:.0?}",
            ground_truth.percentile(0.999),
            ground_truth.percentile(0.9999),
            ground_truth.max(),
        );
        println!(
            "Ground Truth Memory: {:.1} MB ({:.0} kB)",
            ground_truth.mem_size() as f64 / 1_048_576.0,
            ground_truth.mem_size() as f64 / 1024.0,
        );

        let top_k = 10;
        let heavy_sets = ground_truth
            .top_cardinalities()
            .take(top_k)
            .collect::<Vec<_>>();
        println!("Ground Truth Top {}: {:?}", top_k, heavy_sets);
        println!("Ground Truth Time: {:.2?}", start.elapsed());
        println!();
    }
    ground_truth
}

pub fn run_sketch<A>(
    folder_path: &PathBuf,
    max_per_file: usize,
    sketch_type: &A,
    entries: usize,
    counter_size: usize,
    verbose: bool,
) where
    A: Algorithm,
    A::Sketch<String, String>:
        HeavyDistinctHitterSketch<Label = String, Item = String> + MemorySize,
{
    let dataset = FolderDataset::new(folder_path, max_per_file);
    let ground_truth = Box::new(dataset_ground_truth(&dataset, verbose));
    sketch_dataset(entries, counter_size, sketch_type, &ground_truth, &dataset);
}

pub enum ComboType {
    SingleSketch,
    MergeSketches,
}

pub fn run_combos(
    combo_type: ComboType,
    folder_path: &PathBuf,
    max_per_file: usize,
    sketch_types: &[SketchType],
    memories: &[f32],
    counter_sizes: &[usize],
    verbose: bool,
) {
    let dataset = FolderDataset::new(folder_path, max_per_file);
    let ground_truth = Box::new(dataset_ground_truth(&dataset, verbose));

    for sketch_type in sketch_types {
        specialized_dispatch! {
            sketch_type,
            |algorithm| {
                match combo_type {
                    ComboType::SingleSketch => println!("Algo: {}", algorithm),
                    ComboType::MergeSketches => println!("Algo: {} (Merged)", algorithm),
                }
                for memory in memories {
                    for counter_size in counter_sizes {
                        let entries = MaxCapacity::entries_for_mbs(&algorithm, *memory, *counter_size);
                        match combo_type {
                            ComboType::SingleSketch => {
                                sketch_dataset(entries, *counter_size, &algorithm, &ground_truth, &dataset)
                            }
                            ComboType::MergeSketches => merge_on_data(
                                read_dir(folder_path).unwrap().map(|path| FileDataset::new(path.unwrap().path(), max_per_file)),
                                entries,
                                *counter_size,
                                &algorithm,
                                &ground_truth,
                            ),
                        }
                    }
                }
            }

        }
    }
}


pub fn run_zipf(
    num_labels: usize,
    exponent: f64,
    num_samples: usize,
    sketch_types: &[SketchType],
    memories: &[f32],
    counter_sizes: &[usize],
    verbose: bool,
) {
    let dataset = Zipf::new(num_labels, exponent, num_samples, true);
    let ground_truth = Box::new(dataset_ground_truth(&dataset, verbose));

    for sketch_type in sketch_types {
        specialized_dispatch! {
            sketch_type,
            |algorithm| {
                println!("Algo: {}", algorithm);
                for memory in memories {
                    for counter_size in counter_sizes {
                        let entries = MaxCapacity::entries_for_mbs(&algorithm, *memory, *counter_size);
                        sketch_dataset(entries, *counter_size, &algorithm, &ground_truth, &dataset)
                    }
                }
            }

        }
    }
}

pub fn run_overlap(
    k_small: u32,
    n_big: usize,
    sketch_types: &[SketchType],
    entries: usize,
    counter_size: usize,
    verbose: bool,
) {
    let dataset = Overlap::new(k_small, n_big, true);
    let ground_truth = Box::new(dataset_ground_truth(&dataset, verbose));

    for sketch_type in sketch_types {
        specialized_dispatch!(sketch_type, |algorithm| {
            println!("Algo: {}", algorithm);
            sketch_dataset(entries, counter_size, &algorithm, &ground_truth, &dataset);
        })
    }
}

pub fn sketch_dataset<L, I, A>(
    entries: usize,
    counter_size: usize,
    algorithm: &A,
    ground_truth: &GroundTruth<L, I>,
    dataset: &impl Dataset<Label = L, Item = I>,
) where
    L: Eq + Hash + Clone + Debug,
    I: Eq + Hash + Clone + Debug,
    A: Algorithm,
    A::Sketch<L, I>: HeavyDistinctHitterSketch<Label = L, Item = I> + MemorySize,
{
    let mut sketch = algorithm.new_sketch(entries, counter_size);
    let start = Instant::now();
    for (label, item) in dataset.iter() {
        sketch.insert(label.clone(), &item);
    }
    println!("Insertion Time: {:.2?}", start.elapsed());
    println!(
        "Memory: {:.1} MB ({:.0} kB); Counter Size: {}; Entries: {}",
        sketch.mem_size() as f64 / 1_048_576.0,
        sketch.mem_size() as f64 / 1024.0,
        counter_size,
        entries,
    );
    print_stats(ground_truth, &sketch);
    println!();
}

fn merge_on_data<L, I, A>(
    sketch_datasets: impl Iterator<Item = impl Dataset<Label = L, Item = I>>,
    entries: usize,
    counter_size: usize,
    algorithm: &A,
    ground_truth: &GroundTruth<L, I>,
) where
    L: Eq + Hash + Clone + Debug,
    I: Eq + Hash + Clone + Debug,
    A: Algorithm,
    A::Sketch<L, I>: HeavyDistinctHitterSketch<Label = L, Item = I> + MemorySize,
{
    let mut sketch = algorithm.new_sketch(entries, counter_size);
    println!("Running {}:", algorithm);
    let mut file_count = 0;
    let start = Instant::now();
    for dataset in sketch_datasets {
        let mut file_sketch = algorithm.new_sketch(entries, counter_size);
        for (label, item) in dataset.iter() {
            file_sketch.insert(label.clone(), &item);
        }
        sketch
            .merge(&file_sketch)
            .unwrap_or_else(|e| panic!("{:?}", e));
        file_count += 1;
    }
    println!("Merged {} sketches.", file_count);
    println!("Insertion Time: {:.2?}:", start.elapsed());
    println!(
        "Memory: {:.1} MB ({:.0} kB); Entries: {}",
        sketch.mem_size() as f64 / 1_048_576.0,
        sketch.mem_size() as f64 / 1024.0,
        entries,
    );
    print_stats(ground_truth, &sketch);
    println!();
}

fn print_stats<L, I>(
    ground_truth: &GroundTruth<L, I>,
    sketch: &impl HeavyDistinctHitterSketch<Label = L, Item = I>,
) where
    L: Eq + Hash + Clone + Debug,
    I: Eq + Hash + Clone + Debug,
{
    let start = Instant::now();
    let top_k = 1000;
    let sketch_top_k = sketch.top(top_k);
    println!(
        "Query Time to retrieve Sketch Top {}: {:.2?}",
        top_k,
        start.elapsed()
    );

    // print header
    print!("Top\tNAE(T)\tNAE(S)\tNAE(M)\tNAE(Q)\t");
    print!("NRSE(T)\tNRSE(S)\tNRSE(Q)\t");
    print!("RMAE(T)\tRMAE(S)\tRMAE(Q)\t");
    print!("RMSE(T)\tRMSE(S)\tRMSE(Q)\t");
    println!("RMAX(T)\tRMAX(S)\tRMAX(Q)");

    for p in 1..4 {
        let k = usize::pow(10, p);
        let mut sketch_k = k;
        if k > sketch_top_k.len() {
            sketch_k = sketch_top_k.len();
        }

        let true_nae = ground_truth.top_nae(sketch, k);
        let sketch_nae = ground_truth.sketch_nae(&sketch_top_k[..sketch_k]);
        print!(
            "{:05}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t",
            k,
            true_nae,
            sketch_nae,
            mean(true_nae, sketch_nae),
            quadratic_mean(true_nae, sketch_nae,)
        );
        let true_nrse = ground_truth.top_nrse(sketch, k);
        let sketch_nrse = ground_truth.sketch_nrse(&sketch_top_k[..sketch_k]);
        print!(
            "{:.3}\t{:.3}\t{:.3}\t",
            true_nrse,
            sketch_nrse,
            quadratic_mean(true_nrse, sketch_nrse,)
        );
        let true_rmae = ground_truth.actual_rmae(sketch, k);
        let sketch_rmae = ground_truth.sketch_rmae(&sketch_top_k[..sketch_k].to_vec());
        print!(
            "{:.3}\t{:.3}\t{:.3}\t",
            true_rmae,
            sketch_rmae,
            quadratic_mean(true_rmae, sketch_rmae,)
        );
        let true_rrmse = ground_truth.actual_rrmse(sketch, k);
        let sketch_rrmse = ground_truth.sketch_rrmse(&sketch_top_k[..sketch_k].to_vec());
        print!(
            "{:.3}\t{:.3}\t{:.3}\t",
            true_rrmse,
            sketch_rrmse,
            quadratic_mean(true_rrmse, sketch_rrmse,)
        );
        let true_rel_max = ground_truth.actual_rel_max(sketch, k);
        let sketch_rel_max = ground_truth.sketch_rel_max(&sketch_top_k[..sketch_k].to_vec());
        println!(
            "{:.3}\t{:.3}\t{:.3}",
            true_rel_max,
            sketch_rel_max,
            quadratic_mean(true_rel_max, sketch_rel_max,)
        );
    }
}

fn mean(a: f64, b: f64) -> f64 {
    (a + b) / 2.0
}

fn quadratic_mean(a: f64, b: f64) -> f64 {
    ((a * a + b * b) / 2.0).sqrt()
}

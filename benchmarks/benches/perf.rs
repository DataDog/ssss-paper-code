use std::hint::black_box;

use benchmarks::{
    algo::{self, Algorithm},
    data::{self, Dataset},
    memory::MaxCapacity,
};
use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion,
};
use pprof::criterion::{Output, PProfProfiler};
use sketch_traits::HeavyDistinctHitterSketch;

const MEMORY_SIZE: f32 = 1.0; // 1MB
const MAX_NUM_ENTRIES: usize = usize::MAX; // cached entries

macro_rules! for_all_datasets {
    ($fn: expr) => {
        // $fn(data::FolderDataset::new("data/example", usize::MAX));
        $fn(data::FolderDataset::new("data/Witty", usize::MAX));
        $fn(data::FolderDataset::new("data/PubMed", usize::MAX));
        $fn(data::FolderDataset::new("data/KASANDR", usize::MAX));

        // $fn(data::synth::Random::<u64, u64>::new());
        // $fn(data::synth::Random::<String, String>::new());
    };
}

macro_rules! for_all_algorithms {
    ($fn: expr) => {
        $fn(algo::Ssss);
        $fn(algo::Achll);
        $fn(algo::Spread);
        // $fn(algo::Schll);
        // $fn(algo::Osss);
        // $fn(algo::Rsss);
    };
}

fn bench_insertion(c: &mut Criterion) {
    let mut benchmark_group = c.benchmark_group("Insertion");

    for_all_datasets!(|dataset| {
        // Load all data before benchmarking.
        let entries = Dataset::iter(&dataset)
            .take(MAX_NUM_ENTRIES)
            .collect::<Vec<_>>();
        println!("Dataset {} has {} entries", dataset, entries.len());
        for_all_algorithms!(|algorithm| bench_insertion_with(
            &mut benchmark_group,
            &algorithm,
            &dataset,
            &entries
        ));
    });

    benchmark_group.finish();
}

fn bench_insertion_with<'a, A, D>(
    benchmark_group: &mut BenchmarkGroup<impl Measurement>,
    algorithm: &A,
    dataset: &D,
    entries: &'a [(D::Label, D::Item)],
) where
    A: Algorithm + MaxCapacity,
    D: Dataset,
    A::Sketch<&'a D::Label, D::Item>:
        HeavyDistinctHitterSketch<Label = &'a D::Label, Item = D::Item>,
{
    benchmark_group.bench_function(
        BenchmarkId::new(format!("{}", algorithm), format!("{}", dataset)),
        |b| {
            let mut sketch = algorithm.new_sketch(
                algorithm.entries_for_mbs(MEMORY_SIZE, algorithm.optimal_counter_size()),
                algorithm.optimal_counter_size(),
            );
            let mut item_index = 0;
            b.iter(|| {
                let (label, item) = &entries[item_index];
                item_index += 1;
                if item_index == entries.len() {
                    item_index = 0;
                }
                sketch.insert(black_box(label), black_box(item));
            })
        },
    );
}

fn bench_merge(c: &mut Criterion) {
    let mut benchmark_group = c.benchmark_group("Merge");

    for_all_datasets!(|dataset| {
        for_all_algorithms!(|algorithm| bench_merge_with(
            &mut benchmark_group,
            &algorithm,
            &dataset,
        ));
    });

    benchmark_group.finish()
}

fn bench_merge_with<A, D>(
    benchmark_group: &mut BenchmarkGroup<impl Measurement>,
    algorithm: &A,
    dataset: &D,
) where
    A: Algorithm + MaxCapacity,
    D: Dataset,
    A::Sketch<D::Label, D::Item>:
        HeavyDistinctHitterSketch<Label = D::Label, Item = D::Item> + Clone,
{
    const NUM_ENTRIES: usize = 1_000_000;

    let sketch = {
        let mut sketch = algorithm.new_sketch(
            algorithm.entries_for_mbs(MEMORY_SIZE, algorithm.optimal_counter_size()),
            algorithm.optimal_counter_size(),
        );
        dataset
            .iter()
            .take(NUM_ENTRIES)
            .for_each(|(label, item)| sketch.insert(label, &item));
        sketch
    };

    benchmark_group.bench_with_input(
        BenchmarkId::new(format!("{}", algorithm), dataset),
        &NUM_ENTRIES,
        |b, _| {
            b.iter_batched_ref(
                || sketch.clone(),
                |s| black_box(s).merge(black_box(&sketch)),
                BatchSize::SmallInput,
            )
        },
    );
}

fn bench_top(c: &mut Criterion) {
    let mut benchmark_group = c.benchmark_group("Top");
    benchmark_group.sample_size(10); // Querying Count-HLL takes a lot of time.

    for_all_datasets!(|dataset| {
        for_all_algorithms!(
            |algorithm| bench_top_with(&mut benchmark_group, &algorithm, &dataset,)
        );
    });

    benchmark_group.finish()
}

fn bench_top_with<A, D>(
    benchmark_group: &mut BenchmarkGroup<impl Measurement>,
    algorithm: &A,
    dataset: &D,
) where
    A: Algorithm + MaxCapacity,
    D: Dataset,
    A::Sketch<D::Label, D::Item>: HeavyDistinctHitterSketch<Label = D::Label, Item = D::Item>,
{
    const K: usize = 1000;

    let sketch = {
        let mut sketch = algorithm.new_sketch(
            algorithm.entries_for_mbs(MEMORY_SIZE, algorithm.optimal_counter_size()),
            algorithm.optimal_counter_size(),
        );
        dataset
            .iter()
            .take(MAX_NUM_ENTRIES)
            .for_each(|(label, item)| sketch.insert(label, &item));
        sketch
    };

    benchmark_group.bench_with_input(
        BenchmarkId::new(format!("{}", algorithm), dataset),
        &MAX_NUM_ENTRIES,
        |b, _| {
            b.iter_with_large_drop(|| {
                black_box(&sketch).top(black_box(K));
            })
        },
    );
}

criterion_group!(
    name = hs_benchmarks;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_insertion, bench_merge, bench_top,
);
criterion_main!(hs_benchmarks);

/// Utilities to load from files or generate data to be used as an input to the sketches.
use std::{
    fmt,
    fs::{read_dir, File},
    io::{BufRead, BufReader},
    iter::Iterator,
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;

pub trait Dataset: fmt::Display {
    type Label;
    type Item;

    fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>>;
}

#[derive(Clone, Debug)]
pub struct FolderDataset {
    path: PathBuf,
    max_per_file: usize,
}

impl FolderDataset {
    pub fn new(path: impl AsRef<Path>, max_per_file: usize) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            max_per_file,
        }
    }
}

impl fmt::Display for FolderDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path.file_name().unwrap().to_str().unwrap())
    }
}

impl Dataset for FolderDataset {
    type Label = String;
    type Item = String;

    fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>> {
        let max_per_file = self.max_per_file;
        Box::new(
            read_dir(&self.path)
                .unwrap()
                .map(|path| path.unwrap())
                .map(|dir_entry| dir_entry.path())
                .flat_map(move |file_path| {
                    BufReader::new(GzDecoder::new(File::open(file_path).unwrap()))
                        .lines()
                        .take(max_per_file)
                        .map(|line| {
                            let line_str = line.unwrap();
                            let fields = line_str.split(',').take(2).collect::<Vec<&str>>();
                            (fields[1].to_string(), fields[0].to_string())
                        })
                }),
        )
    }
}

#[derive(Clone, Debug)]
pub struct FileDataset {
    path: PathBuf,
    max_per_file: usize,
}

impl FileDataset {
    pub fn new(path: impl AsRef<Path>, max_per_file: usize) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            max_per_file,
        }
    }
}

impl fmt::Display for FileDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path.file_name().unwrap().to_str().unwrap())
    }
}

impl Dataset for FileDataset {
    type Label = String;
    type Item = String;

    fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>> {
        Box::new(
            BufReader::new(GzDecoder::new(File::open(&self.path).unwrap()))
                .lines()
                .take(self.max_per_file)
                .map(|line| {
                    let line_str = line.unwrap();
                    let fields = line_str.split(',').take(2).collect::<Vec<&str>>();
                    (fields[1].to_string(), fields[0].to_string())
                }),
        )
    }
}

pub mod synth {
    use std::{any::type_name, fmt, iter, marker::PhantomData};

    use rand::prelude::*;
    use uuid::Uuid;

    use super::Dataset;

    fn make_label(u: u32) -> String {
        char::from_u32(u + 0x41).unwrap().to_string()
    }

    macro_rules! impl_dataset {
        ($dataset: ty, $name: expr, $label: ty, $item: ty, $gen: expr) => {
            impl std::fmt::Display for $dataset {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, $name)
                }
            }

            impl Dataset for $dataset {
                type Label = $label;
                type Item = $item;

                fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>> {
                    Box::new(std::iter::from_fn($gen(&self)))
                }
            }
        };
    }

    #[derive(Clone, Debug)]
    pub struct Uniform {
        k: u32,
    }

    impl Uniform {
        pub fn new(k: u32) -> Self {
            Self { k }
        }
    }

    impl_dataset!(Uniform, "Uniform", String, u64, |dataset: &Uniform| {
        let dist = rand_distr::Uniform::new(0, dataset.k * 2);
        let mut rng = thread_rng();
        move || Some((make_label(rng.sample(dist)), rng.gen()))
    });

    #[derive(Clone, Debug)]
    pub struct Poisson {
        k: f64,
    }

    impl Poisson {
        pub fn new(k: u32) -> Self {
            Self { k: k as f64 }
        }
    }

    impl_dataset!(Poisson, "Poisson", String, u64, |dataset: &Poisson| {
        let dist = rand_distr::Poisson::new(dataset.k).unwrap();
        let mut rng = thread_rng();
        move || Some((make_label(rng.sample(dist) as u32), rng.gen()))
    });

    #[derive(Clone, Debug)]
    pub struct Repeats {
        k: u32,
    }

    impl Repeats {
        const LABEL: &str = "Z";
        const ITEM: <Self as Dataset>::Item = 10;

        pub fn new(k: u32) -> Self {
            Self { k }
        }
    }

    impl_dataset!(Repeats, "Repeats", String, u64, |dataset: &Repeats| {
        let mut poisson_iter = Poisson::new(dataset.k).iter();
        let mut rng = thread_rng();
        move || match rng.gen() {
            false => poisson_iter.next(),
            true => Some((Self::LABEL.to_string(), Self::ITEM)),
        }
    });

    #[derive(Clone, Debug)]
    pub struct CycleSingleItem {
        k: u32,
    }

    impl CycleSingleItem {
        const ITEM: <Self as Dataset>::Item = 0;

        pub fn new(k: u32) -> Self {
            Self { k }
        }
    }

    impl_dataset!(
        CycleSingleItem,
        "Single x item, alternating labels",
        String,
        u64,
        |dataset: &CycleSingleItem| {
            let mut cycle = (0..=dataset.k).cycle();
            move || Some((make_label(cycle.next().unwrap()), Self::ITEM))
        }
    );

    #[derive(Clone, Debug)]
    pub struct CycleUniqueItems {
        k: u32,
    }

    impl CycleUniqueItems {
        pub fn new(k: u32) -> Self {
            Self { k }
        }
    }

    impl_dataset!(
        CycleUniqueItems,
        "Alternating labels, each with unique x item",
        String,
        u64,
        |dataset: &CycleUniqueItems| {
            let mut cycle = (0..=dataset.k).cycle();
            move || {
                let next = cycle.next().unwrap();
                Some((make_label(next), next as u64))
            }
        }
    );

    #[derive(Clone, Debug)]
    pub struct OneLabel;

    impl OneLabel {
        const LABEL: &str = "Z";
    }

    impl_dataset!(
        OneLabel,
        "One label, uniformly random items",
        String,
        u64,
        |_| {
            let mut rng = thread_rng();
            move || Some((Self::LABEL.to_string(), rng.gen()))
        }
    );

    #[derive(Clone, Debug)]
    pub struct SingleEntry<L, I> {
        label: L,
        item: I,
    }

    impl<L, I> SingleEntry<L, I> {
        pub fn new(label: L, item: I) -> Self {
            Self { label, item }
        }
    }

    impl<L, I> fmt::Display for SingleEntry<L, I>
    where
        L: fmt::Display,
        I: fmt::Display,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "SingleEntry<{}, {}>({}, {})",
                type_name::<L>(),
                type_name::<I>(),
                self.label,
                self.item
            )
        }
    }

    impl<L, I> Dataset for SingleEntry<L, I>
    where
        Self: fmt::Display,
        L: Clone + 'static,
        I: Clone + 'static,
    {
        type Label = L;
        type Item = I;

        fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>> {
            Box::new(iter::repeat((self.label.clone(), self.item.clone())))
        }
    }

    #[derive(Default, Clone, Debug)]
    pub struct Random<L, I> {
        label_type: PhantomData<L>,
        item_type: PhantomData<I>,
    }

    impl<L, I> Random<L, I> {
        pub fn new() -> Self {
            Self {
                label_type: PhantomData,
                item_type: PhantomData,
            }
        }
    }

    impl<L, I> fmt::Display for Random<L, I> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Random<{}, {}>", type_name::<L>(), type_name::<I>())
        }
    }

    impl Dataset for Random<u64, u64> {
        type Label = u64;
        type Item = u64;

        fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>> {
            let mut rng = thread_rng();
            Box::new(iter::from_fn(move || Some((rng.gen(), rng.gen()))))
        }
    }

    impl Dataset for Random<String, String> {
        type Label = String;
        type Item = String;

        fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>> {
            const STRING_LEN: usize = 16;
            Box::new(iter::from_fn(|| {
                let label = thread_rng()
                    .sample_iter(&rand::distributions::Alphanumeric)
                    .take(STRING_LEN)
                    .map(char::from)
                    .collect();
                let item = thread_rng()
                    .sample_iter(&rand::distributions::Alphanumeric)
                    .take(STRING_LEN)
                    .map(char::from)
                    .collect();
                Some((label, item))
            }))
        }
    }

    pub struct Overlap {
        k_small: u32,
        n_big: usize,
        verbose: bool,
    }

    impl Overlap {
        const UNIVERSE_SIZE: u32 = 1_000_000;
        const COMMON_SIZE: u32 = 100_000;
        const N_SMALL: usize = 1000; // size of small sets from common
        pub const K_BIG: usize = 1000; // # of big sets from full universe

        // const N_BIG: [u32; 5] = [20_000, 50_000, 100_000, 200_000, 500_000];  // size of big sets from full universe
        // const K_SMALL: [u32; 2] = [100_000, 1_000_000];    // # of small sets from common

        pub fn new(k_small: u32, n_big: usize, verbose: bool) -> Self {
            Self {
                k_small,
                n_big,
                verbose,
            }
        }
    }

    impl fmt::Display for Overlap {
        fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
            todo!()
        }
    }

    impl Dataset for Overlap {
        type Label = Uuid;
        type Item = u32;

        fn iter(&self) -> Box<dyn Iterator<Item = (Self::Label, Self::Item)>> {
            let mut rng = rand::thread_rng();
            let mut data = Vec::new();

            let common_items: Vec<Self::Item> = (0..Self::COMMON_SIZE).collect();
            for _ in 0..self.k_small {
                let label = Uuid::new_v4();
                let mut small_set: Vec<_> = common_items
                    .choose_multiple(&mut rng, Self::N_SMALL)
                    .into_iter()
                    .map(|&i| (label, i))
                    .collect();
                data.append(&mut small_set);
            }

            let full_universe: Vec<Self::Item> = (0..Self::UNIVERSE_SIZE).collect();
            for _ in 0..Self::K_BIG {
                let label = Uuid::new_v4();
                let mut large_set: Vec<_> = full_universe
                    .choose_multiple(&mut rng, self.n_big)
                    .into_iter()
                    .map(|&i| (label, i))
                    .collect();
                data.append(&mut large_set);
            }

            data.shuffle(&mut rng);

            if self.verbose {
                println!();
                println!(
                    "{} Small Sets of size: {} (sets chosen randomly from [0, {}))",
                    self.k_small,
                    Self::N_SMALL,
                    Self::COMMON_SIZE,
                );
                println!(
                    "{} Large Sets of size: {} (sets chosen randomly from [0, {}))",
                    Self::K_BIG,
                    self.n_big,
                    Self::UNIVERSE_SIZE,
                );
            }
            Box::new(data.into_iter())
        }
    }
}

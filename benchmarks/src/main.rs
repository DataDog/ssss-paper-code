extern crate clap;
use std::{fmt, path::PathBuf};

use clap::{ArgAction, Parser, Subcommand};

use crate::dataset::{run_combos, run_overlap, run_sketch, ComboType};

pub mod accuracy;
pub mod algo;
pub mod data;
pub mod dataset;
pub mod exact;
pub mod memory;

const DEFAULT_COUNTER_SIZE: usize = 1024;
const DEFAULT_COUNTER_SIZES: [usize; 7] = [32, 64, 128, 256, 512, 1024, 2048];
const DEFAULT_MAX_PER_FILE: usize = 100_000_000;
const DEFAULT_MEMORY: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
const DEFAULT_NUM_SKETCH_ENTRIES: usize = 100;
const DEFAULT_SKETCH_TYPES: [SketchType; 3] =
    [SketchType::Ssss, SketchType::Spread, SketchType::Achll];

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run sketches against a real dataset
    Sketch {
        /// Path to dataset
        input: PathBuf,

        /// Number of lines to take per file
        #[clap(short, long, value_parser, default_value_t=DEFAULT_MAX_PER_FILE)]
        max_per_file: usize,

        /// Sketch type
        #[clap(short, long, value_parser)]
        sketch_type: SketchType,

        /// Number of entries kept by sketches
        #[clap(short, long, value_parser, default_value_t=DEFAULT_NUM_SKETCH_ENTRIES)]
        entries: usize,

        /// The size of the cardinality counters
        #[clap(short, long, value_parser, default_value_t=DEFAULT_COUNTER_SIZE)]
        counter_size: usize,

        /// Control the amount of output
        #[clap(short, long, action = ArgAction::SetTrue)]
        verbose: bool,
    },

    /// Run memory-constrained sketches against a real dataset
    Combo {
        /// Path to dataset
        input: PathBuf,

        /// Number of lines to take per file
        #[clap(long, value_parser, default_value_t=DEFAULT_MAX_PER_FILE)]
        max_per_file: usize,

        /// Sketch type
        #[clap(short, long, value_parser, default_values_t=DEFAULT_SKETCH_TYPES)]
        sketch_type: Vec<SketchType>,

        /// Max amount of memory used by sketch (in MB)
        #[clap(short, long, value_parser, default_values_t=DEFAULT_MEMORY)]
        memory: Vec<f32>,

        /// The size of the cardinality counters
        #[clap(short, long, value_parser, default_values_t=DEFAULT_COUNTER_SIZES)]
        counter_size: Vec<usize>,

        /// Control the amount of output
        #[clap(short, long, action = ArgAction::SetTrue)]
        verbose: bool,
    },

    /// Run sketches against a real dataset a file at a time then merge them
    Merge {
        /// Path to dataset
        input: PathBuf,

        /// Number of lines to take per file
        #[clap(long, value_parser, default_value_t=DEFAULT_MAX_PER_FILE)]
        max_per_file: usize,

        /// Sketch type
        #[clap(short, long, value_parser, default_values_t=DEFAULT_SKETCH_TYPES)]
        sketch_type: Vec<SketchType>,

        /// Max amount of memory used by sketch (in MB)
        #[clap(short, long, value_parser, default_values_t=DEFAULT_MEMORY)]
        memory: Vec<f32>,

        /// The size of the cardinality counters
        #[clap(short, long, value_parser, default_values_t=DEFAULT_COUNTER_SIZES)]
        counter_size: Vec<usize>,

        /// Control the amount of output
        #[clap(short, long, action = ArgAction::SetTrue)]
        verbose: bool,
    },

    Overlap {
        /// k_small, # of small sets from common
        #[clap(short, long, value_parser)]
        k_small: u32,

        /// n_big, size of big sets from full universe
        #[clap(short, long, value_parser)]
        n_big: usize,

        /// Sketch type
        #[clap(short, long, value_parser, default_values_t=DEFAULT_SKETCH_TYPES)]
        sketch_type: Vec<SketchType>,

        /// Number of entries kept by sketches
        #[clap(short, long, value_parser, default_value_t=data::synth::Overlap::K_BIG)]
        entries: usize,

        /// The size of the cardinality counters
        #[clap(short, long, value_parser, default_value_t=DEFAULT_COUNTER_SIZE)]
        counter_size: usize,

        /// Control the amount of output
        #[clap(short, long, action = ArgAction::SetTrue)]
        verbose: bool,
    },
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum SketchType {
    Achll,
    Schll,
    Osss,
    Rsss,
    Spread,
    Ssss,
}

impl fmt::Display for SketchType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_lowercase())
    }
}

macro_rules! specialized_dispatch {
    ($sketch_type:ident, $fn:expr) => {
        match $sketch_type {
            SketchType::Achll => $fn(crate::algo::Achll),
            SketchType::Schll => $fn(crate::algo::Schll),
            SketchType::Osss => $fn(crate::algo::Osss),
            SketchType::Rsss => $fn(crate::algo::Rsss),
            SketchType::Spread => $fn(crate::algo::Spread),
            SketchType::Ssss => $fn(crate::algo::Ssss),
        }
    };
}
pub(crate) use specialized_dispatch;

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Command::Sketch {
            input,
            max_per_file,
            sketch_type,
            entries,
            counter_size,
            verbose,
        } => {
            specialized_dispatch!(sketch_type, |algorithm| run_sketch(
                input,
                *max_per_file,
                &algorithm,
                *entries,
                *counter_size,
                *verbose,
            ))
        }
        Command::Combo {
            input,
            max_per_file,
            sketch_type,
            memory,
            counter_size,
            verbose,
        } => {
            run_combos(
                ComboType::SingleSketch,
                input,
                *max_per_file,
                sketch_type,
                memory,
                counter_size,
                *verbose,
            );
        }
        Command::Merge {
            input,
            max_per_file,
            sketch_type,
            memory,
            counter_size,
            verbose,
        } => {
            run_combos(
                ComboType::MergeSketches,
                input,
                *max_per_file,
                sketch_type,
                memory,
                counter_size,
                *verbose,
            );
        }
        Command::Overlap {
            k_small,
            n_big,
            sketch_type,
            entries,
            counter_size,
            verbose,
        } => {
            run_overlap(
                *k_small,
                *n_big,
                sketch_type,
                *entries,
                *counter_size,
                *verbose,
            );
        }
    }
}

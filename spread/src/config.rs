use std::{error, fmt};

use ahash::RandomState;
use rand::random;

#[derive(Clone, Debug)]
pub struct Config<C> {
    /// The level of redundancy of the underlying Count-Min Sketch, a.k.a the depth.
    pub(crate) num_rows: usize,
    /// The "width" of the sketch; corresponds to the number of labels we should
    /// be able to accuractely estimate.
    pub(crate) num_cols: usize,
    seeds: [u64; 8],
    pub(crate) hash_builders: [RandomState; 2],
    pub(crate) cardinality_sketch_config: C,
}

impl<C> Config<C> {
    pub fn new(
        num_rows: usize,
        num_cols: usize,
        cardinality_sketch_config: C,
        seeds: Option<[u64; 8]>,
    ) -> Result<Self, ConfigError> {
        if num_rows == 0 {
            return Err(ConfigError::ZeroNumRows);
        } else if num_cols == 0 {
            return Err(ConfigError::ZeroNumCols);
        }
        let seeds_or_random = seeds.unwrap_or_else(random);
        Ok(Self {
            num_rows,
            num_cols,
            seeds: seeds_or_random,
            hash_builders: [
                RandomState::with_seeds(
                    seeds_or_random[0],
                    seeds_or_random[1],
                    seeds_or_random[2],
                    seeds_or_random[3],
                ),
                RandomState::with_seeds(
                    seeds_or_random[4],
                    seeds_or_random[5],
                    seeds_or_random[6],
                    seeds_or_random[7],
                ),
            ],
            cardinality_sketch_config,
        })
    }

    /// The amount of redundancy (usually a small constant)
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The number of entries kept by the sketch.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    pub fn cardinality_sketch_config(&self) -> &C {
        &self.cardinality_sketch_config
    }
}

impl<C> PartialEq for Config<C>
where
    C: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.num_rows == other.num_rows
            && self.num_cols == other.num_cols
            && self.seeds == other.seeds
            && self.cardinality_sketch_config == other.cardinality_sketch_config
    }
}

impl<C> Eq for Config<C> where C: Eq {}

#[derive(Clone, Debug)]
pub enum ConfigError {
    ZeroNumRows,
    ZeroNumCols,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConfigError::ZeroNumRows => {
                write!(f, "the number of rows should not be zero")
            }
            ConfigError::ZeroNumCols => {
                write!(f, "the number of columns should not be zero")
            }
        }
    }
}

impl error::Error for ConfigError {}

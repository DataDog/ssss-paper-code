use std::{error, fmt};

use ahash::RandomState;
use rand::random;

#[derive(Clone, Debug)]
pub struct Config<C> {
    /// The maximum number of counters to keep.
    pub(crate) max_num_counters: usize,
    seeds: [u64; 4],
    pub(crate) hash_builder: RandomState,
    pub(crate) cardinality_sketch_config: C,
}

impl<C> Config<C> {
    pub fn new(
        max_num_counters: usize,
        cardinality_sketch_config: C,
        seeds: Option<[u64; 4]>,
    ) -> Result<Self, ConfigError> {
        if max_num_counters == 0 {
            return Err(ConfigError::ZeroMaxNumCounters);
        }
        let seeds_or_random = seeds.unwrap_or_else(random);
        Ok(Self {
            max_num_counters,
            seeds: seeds_or_random,
            hash_builder: RandomState::with_seeds(
                seeds_or_random[0],
                seeds_or_random[1],
                seeds_or_random[2],
                seeds_or_random[3],
            ),
            cardinality_sketch_config,
        })
    }

    pub fn max_num_counters(&self) -> usize {
        self.max_num_counters
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
        self.max_num_counters == other.max_num_counters
            && self.seeds == other.seeds
            && self.cardinality_sketch_config == other.cardinality_sketch_config
    }
}

impl<C> Eq for Config<C> where C: Eq {}

#[derive(Clone, Debug)]
pub enum ConfigError {
    ZeroMaxNumCounters,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConfigError::ZeroMaxNumCounters => {
                write!(f, "the size should not be zero")
            }
        }
    }
}

impl error::Error for ConfigError {}

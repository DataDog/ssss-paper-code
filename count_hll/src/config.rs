use std::{error, fmt};

use ahash::RandomState;
use rand::random;

use crate::dist::{geometric, Distribution};

#[derive(Clone, Debug)]
pub struct Config {
    pub(crate) depth: usize,
    pub(crate) depth_log2: usize,
    pub(crate) width: usize,
    seeds: [u64; 12],
    pub(crate) hash_builders: [RandomState; 3],
    pub(crate) cardinality_estimation_method: CardinalityEstimationMethod,
    // FIXME: Use the same across one across sketch instances.
    pub(crate) geometric: Distribution,
}

impl Config {
    pub fn new(d: usize, w: usize, seeds: Option<[u64; 12]>) -> Result<Self, ConfigError> {
        if d & (d - 1) != 0 {
            return Err(ConfigError::NonPowerOfTwoDepth);
        } else if w == 0 {
            return Err(ConfigError::ZeroWidth);
        }
        let seeds_or_random = seeds.unwrap_or_else(random);
        Ok(Self {
            depth: d,
            depth_log2: d.trailing_zeros().try_into().unwrap(),
            width: w,
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
                RandomState::with_seeds(
                    seeds_or_random[8],
                    seeds_or_random[9],
                    seeds_or_random[10],
                    seeds_or_random[11],
                ),
            ],
            cardinality_estimation_method: CardinalityEstimationMethod::MaximumLikelihood,
            geometric: geometric(64, d),
        })
    }
}

impl PartialEq for Config {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth
            && self.depth_log2 == other.depth_log2
            && self.width == other.width
            && self.seeds == other.seeds
            && self.cardinality_estimation_method == other.cardinality_estimation_method
    }
}

impl Eq for Config {}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CardinalityEstimationMethod {
    /// The original cardinality estimator
    Original,
    /// An estimator that maximizes the composite (log) likelihood using a
    /// Newton-Raphson procedure
    MaximumLikelihood,
}

#[derive(Clone, Debug)]
pub enum ConfigError {
    NonPowerOfTwoDepth,
    ZeroWidth,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConfigError::NonPowerOfTwoDepth => {
                write!(f, "the depth should be a non-zero power of two")
            }
            ConfigError::ZeroWidth => write!(f, "the width should not be zero"),
        }
    }
}

impl error::Error for ConfigError {}

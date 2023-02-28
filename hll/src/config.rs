use std::{error, fmt};

use ahash::RandomState;
use rand::random;

#[derive(Clone, Debug)]
pub struct Config {
    pub(crate) num_registers: usize,
    pub(crate) alpha: f64,
    seeds: [u64; 8],
    pub(crate) hash_builders: [RandomState; 2],
}

impl Config {
    pub fn new(num_registers: usize, seeds: Option<[u64; 8]>) -> Result<Self, ConfigError> {
        if num_registers & (num_registers - 1) != 0 {
            return Err(ConfigError::NonPowerOfTwoNumRegisters);
        }
        let seeds_or_random = seeds.unwrap_or_else(random);
        Ok(Self {
            num_registers,
            alpha: alpha(num_registers),
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
        })
    }

    pub fn num_registers(&self) -> usize {
        self.num_registers
    }
}

impl PartialEq for Config {
    fn eq(&self, other: &Self) -> bool {
        self.num_registers == other.num_registers
            && self.alpha == other.alpha
            && self.seeds == other.seeds
    }
}

impl Eq for Config {}

#[derive(Clone, Debug)]
pub enum ConfigError {
    NonPowerOfTwoNumRegisters,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::NonPowerOfTwoNumRegisters => {
                write!(f, "the number of registers should be a non-zero power of 2")
            }
        }
    }
}

impl error::Error for ConfigError {}

fn alpha(num_registers: usize) -> f64 {
    debug_assert!(num_registers & (num_registers - 1) == 0); // non-zero power of 2
    match num_registers {
        1 | 2 | 4 | 8 => panic!(),
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        128 => 0.715,
        256 => 0.718,
        512 => 0.720,
        _ => 0.7213 / (1.0 + 1.079 / (num_registers as f64)),
    }
}

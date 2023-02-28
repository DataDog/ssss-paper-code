use std::{error, fmt};

use crate::counter::ResetStrategy;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Config<C> {
    /// The maximum number of counters to keep.
    pub(crate) max_num_counters: usize,
    pub(crate) reset_strategy: ResetStrategy,
    pub(crate) cardinality_sketch_config: C,
}

impl<C> Config<C> {
    pub fn new(
        size: usize,
        reset_strategy: ResetStrategy,
        cardinality_sketch_config: C,
    ) -> Result<Self, ConfigError> {
        if size == 0 {
            return Err(ConfigError::ZeroSize);
        }
        Ok(Self {
            max_num_counters: size,
            reset_strategy,
            cardinality_sketch_config,
        })
    }

    pub fn reset_strategy(&self) -> &ResetStrategy {
        &self.reset_strategy
    }

    pub fn cardinality_sketch_config(&self) -> &C {
        &self.cardinality_sketch_config
    }
}

#[derive(Clone, Debug)]
pub enum ConfigError {
    ZeroSize,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConfigError::ZeroSize => {
                write!(f, "the size should not be zero")
            }
        }
    }
}

impl error::Error for ConfigError {}

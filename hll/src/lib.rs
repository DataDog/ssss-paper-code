use std::{error, fmt, hash::Hash, iter::repeat, marker::PhantomData};

use ahash::RandomState;
use sketch_traits::{CardinalitySketch, New};

mod config;
mod linear_counting;
pub use crate::config::Config;
use crate::linear_counting::linear_counting;

#[derive(Clone, Debug)]
pub struct HyperLogLog<I> {
    config: Config,
    registers: Vec<u8>,
    num_zero_registers: usize,
    z_inv: f64,
    item_type: PhantomData<I>,
}

impl<I> New for HyperLogLog<I> {
    type Config = Config;

    fn new(config: &Self::Config) -> Self {
        Self {
            config: config.clone(),
            registers: repeat(0).take(config.num_registers).collect(),
            num_zero_registers: config.num_registers,
            z_inv: config.num_registers as f64,
            item_type: PhantomData,
        }
    }
}

impl<I> CardinalitySketch for HyperLogLog<I>
where
    I: Hash,
{
    type Item = I;
    type MergeError = MergeError;

    #[inline]
    fn insert(&mut self, item: &Self::Item) {
        let z = Self::item_hash(&self.config.hash_builders[1], item);
        self.insert_hash(item, z);
    }

    #[inline]
    fn merge(&mut self, other: &Self) -> Result<(), Self::MergeError> {
        if self.config != other.config {
            return Err(MergeError::ConfigMismatch);
        }

        self.registers
            .iter_mut()
            .zip(other.registers.iter())
            .for_each(|(s, o)| *s = *o.max(s));
        self.z_inv = self
            .registers
            .iter()
            .map(|r| 2.0_f64.powi(-i32::try_from(*r).unwrap()))
            .sum::<f64>();
        self.num_zero_registers = self.registers.iter().filter(|&n| *n == 0).count();
        Ok(())
    }

    #[inline]
    fn clear(&mut self) {
        self.registers.fill(0);
        self.z_inv = self.config.num_registers as f64;
        self.num_zero_registers = self.config.num_registers;
    }

    #[inline]
    fn cardinality(&self) -> u64 {
        let mut estimate = (((self.config.num_registers * self.config.num_registers) as f64
            * self.config.alpha)
            / self.z_inv) as u64;

        if estimate <= 5 * (self.config.num_registers as u64 >> 1) {
            // small range correction for estimate < (5/2)d
            if self.num_zero_registers > 0 {
                estimate =
                    linear_counting(self.config.num_registers, self.num_zero_registers) as u64;
            }
        }
        // TODO: large range correction

        estimate
    }
}

impl<I> HyperLogLog<I> {
    #[inline]
    pub fn config(&self) -> &Config {
        &self.config
    }

    #[inline]
    fn item_hash(hash_builder: &RandomState, item: &I) -> u8
    where
        I: Hash,
    {
        u8::try_from(hash_builder.hash_one(item).trailing_zeros()).unwrap() + 1
    }

    #[inline]
    fn insert_hash(&mut self, item: &I, z: u8)
    where
        I: Hash,
    {
        let r: usize =
            self.config.hash_builders[0].hash_one(item) as usize & (self.config.num_registers - 1);
        let register = self.registers.get_mut(r).unwrap();
        if z > *register {
            if *register == 0 {
                self.num_zero_registers -= 1;
            }
            self.z_inv -= 2.0_f64.powi(-i32::try_from(*register).unwrap());
            self.z_inv += 2.0_f64.powi(-i32::try_from(z).unwrap());
            *register = z;
        }
    }
}

#[derive(Clone, Debug)]
pub enum MergeError {
    ConfigMismatch,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MergeError::ConfigMismatch => write!(f, "sketch configs do not match"),
        }
    }
}

impl error::Error for MergeError {}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    const COUNTER_SIZE: usize = 1024;
    const SEEDS: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

    fn seeded_config() -> Config {
        Config::new(COUNTER_SIZE, Some(SEEDS)).unwrap()
    }

    #[derive(Clone, Debug)]
    struct TestCase {
        cardinality: u64,
        sketch: HyperLogLog<u64>,
    }

    fn test_cases() -> impl Strategy<Value = TestCase> {
        let test_dims = vec![(100, 9), (6_000, 3), (10_000_000, 7)];

        let sketches = test_dims
            .into_iter()
            .map(|(cardinality, num_inserts_per_item)| {
                let mut sketch = HyperLogLog::new(&seeded_config());
                (0..num_inserts_per_item)
                    .flat_map(|_| 0..cardinality)
                    .for_each(|item| sketch.insert(&item));
                sketch.clear();
                (0..num_inserts_per_item)
                    .flat_map(|_| 0..cardinality)
                    .for_each(|item| sketch.insert(&item));
                TestCase {
                    cardinality,
                    sketch,
                }
            })
            .map(Just);

        proptest::strategy::Union::new(sketches)
    }

    fn merge_same() -> impl Strategy<Value = TestCase> {
        let test_dims = vec![(100, 9), (6_000, 3), (10_000_000, 7)];

        let sketches = test_dims
            .into_iter()
            .map(|(cardinality, num_inserts_per_item)| {
                let mut sketch = HyperLogLog::new(&seeded_config());
                let mut sketch2 = HyperLogLog::new(&seeded_config());
                let mut sketch3 = HyperLogLog::new(&seeded_config());
                (0..num_inserts_per_item)
                    .flat_map(|_| 0..cardinality)
                    .for_each(|item| {
                        sketch.insert(&item);
                        sketch2.insert(&item);
                        sketch3.insert(&item);
                    });
                assert!(sketch.merge(&sketch2).is_ok());
                assert!(sketch.merge(&sketch3).is_ok());
                TestCase {
                    cardinality,
                    sketch,
                }
            })
            .map(Just);

        proptest::strategy::Union::new(sketches)
    }

    fn merge_diff() -> impl Strategy<Value = TestCase> {
        let test_dims = vec![100, 6_000, 10_000_000];

        let mut sketch = HyperLogLog::new(&seeded_config());
        let mut sketch2 = HyperLogLog::new(&seeded_config());
        let mut sketch3 = HyperLogLog::new(&seeded_config());

        (0..test_dims[0]).for_each(|item| sketch.insert(&item));
        (0..test_dims[1]).for_each(|item| sketch2.insert(&item));
        (0..test_dims[2]).for_each(|item| sketch3.insert(&item));
        assert!(sketch.merge(&sketch2).is_ok());
        assert!(sketch.merge(&sketch3).is_ok());
        let cardinality = test_dims.iter().sum::<u64>();
        let sketches = TestCase {
            cardinality,
            sketch,
        };

        proptest::strategy::Just(sketches)
    }
    #[test]
    fn it_estimates_cardinality() {
        proptest!(ProptestConfig::with_cases(16), |(test_case in test_cases())| {
            prop_assume!(test_case.cardinality > 0); // FIXME
            let cardinality = test_case.sketch.cardinality();
            prop_assert!((cardinality as f64 - test_case.cardinality as f64).abs() / test_case.cardinality as f64 <= 5e-2);
        })
    }

    #[test]
    fn it_estimates_cardinality_after_merging_same() {
        proptest!(ProptestConfig::with_cases(16), |(test_case in merge_same())| {
            prop_assume!(test_case.cardinality > 0); // FIXME
            let cardinality = test_case.sketch.cardinality();
            prop_assert!((cardinality as f64 - test_case.cardinality as f64).abs() / test_case.cardinality as f64 <= 5e-2);
        })
    }

    #[test]
    fn it_estimates_cardinality_after_merging_diff() {
        proptest!(ProptestConfig::with_cases(16), |(test_case in merge_diff())| {
            prop_assume!(test_case.cardinality > 0); // FIXME
            let cardinality = test_case.sketch.cardinality();
            prop_assert!((cardinality as f64 - test_case.cardinality as f64).abs() / test_case.cardinality as f64 <= 5e-2);
        })
    }

    #[test]
    fn merge_into_empty() {
        let mut sketch = HyperLogLog::new(&seeded_config());
        let mut sketch2 = HyperLogLog::new(&seeded_config());

        let cardinality = 100;
        for i in 0..cardinality {
            sketch2.insert(&i);
        }
        assert!(sketch.merge(&sketch2).is_ok());
        assert!(
            (cardinality as f64 - sketch.cardinality() as f64).abs() / cardinality as f64 <= 5e-2
        );
    }
}

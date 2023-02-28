use std::{error, fmt, hash::Hash, iter};

mod config;
mod dist;
use crate::dist::Distribution;

mod invertible;
use sketch_traits::New;

pub use crate::{
    config::{CardinalityEstimationMethod, Config, ConfigError},
    invertible::*,
};

#[derive(Clone, Debug)]
pub struct PointwiseSketch {
    config: Config,
    registers: Vec<u8>,
}

impl New for PointwiseSketch {
    type Config = Config;

    fn new(config: &Self::Config) -> Self {
        Self {
            config: config.clone(),
            registers: iter::repeat(0).take(config.depth * config.width).collect(),
        }
    }
}

impl PointwiseSketch {
    fn register(&self, r: usize, b: usize) -> &u8 {
        &self.registers[r + (b << self.config.depth_log2)]
    }

    fn register_mut(&mut self, index: usize) -> &mut u8 {
        &mut self.registers[index]
    }

    fn num_registers(&self) -> usize {
        self.config.depth * self.config.width
    }

    pub fn clear(&mut self) {
        self.registers.fill(0);
    }

    fn get_index<L, I>(&self, label: &L, item: &I) -> usize
    where
        L: Hash,
        I: Hash,
    {
        let digest = (item, label);
        let r: usize =
            self.config.hash_builders[0].hash_one(digest) as usize & (self.config.depth - 1);
        let b = self.config.hash_builders[2].hash_one((r, label)) as usize % self.config.width;
        r + (b << self.config.depth_log2)
    }

    fn get_z<L, I>(&self, label: &L, item: &I) -> u8
    where
        I: Hash,
        L: Hash,
    {
        u8::try_from(
            self.config.hash_builders[1]
                .hash_one((item, label))
                .trailing_zeros(),
        )
        .unwrap()
            + 1
    }

    #[inline]
    pub fn insert<L, I>(&mut self, label: &L, item: &I)
    where
        L: Hash,
        I: Hash,
    {
        let z = self.get_z(label, item);
        let index = self.get_index(label, item);
        let register = self.register_mut(index);
        *register = z.max(*register);
    }

    pub fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.config != other.config {
            return Err(MergeError::ConfigMismatch);
        }
        self.registers
            .iter_mut()
            .zip(other.registers.iter())
            .for_each(|(s, o)| *s = *o.max(s));
        Ok(())
    }

    pub fn cardinality<L: Hash>(&self, label: &L) -> u64 {
        match self.config.cardinality_estimation_method {
            CardinalityEstimationMethod::Original => {
                ((self.config.depth as f64 * alpha(self.config.depth))
                    / self
                        .signal(label)
                        .pmf_iter()
                        .map(|(c, p)| p * 2.0_f64.powi(-i32::try_from(c).unwrap()))
                        .sum::<f64>()) as u64
            }
            CardinalityEstimationMethod::MaximumLikelihood => {
                self.argmax_cl(&self.signal(label), &self.background(label))
            }
        }
    }

    fn signal<L: Hash>(&self, label: &L) -> Distribution {
        (0..self.config.depth)
            .map(|r| {
                (
                    r,
                    self.config.hash_builders[2].hash_one((r, label)) as usize % self.config.width,
                )
            })
            .map(|(r, b)| *self.register(r, b))
            .map(|r| r as usize)
            .collect()
    }

    fn background<L: Hash>(&self, label: &L) -> Distribution {
        (0..self.config.depth)
            .map(|r| {
                (
                    r,
                    self.config.hash_builders[2].hash_one((r, label)) as usize % self.config.width,
                )
            })
            .flat_map(|(r, b0)| {
                (0..self.config.width)
                    .filter(move |&b| b != b0)
                    .map(move |b| (r, b))
            })
            .map(|(r, b)| *self.register(r, b))
            .map(|r| r as usize)
            .collect()
    }

    /// Calculates the composite log likelihood.
    #[allow(dead_code)]
    fn cl(&self, signal: &Distribution, background: &Distribution, n: f64) -> f64 {
        signal
            .pmf_iter()
            .map(|(z, w)| (z as isize, w))
            .map(|(z, w)| {
                w * (self.config.geometric.cdf(z).powf(n) * background.cdf(z)
                    - self.config.geometric.cdf(z - 1).powf(n) * background.cdf(z - 1))
                .ln()
            })
            .sum()
    }

    /// Calculates the first partial derivative of the composite log likelihood
    /// with respect to n.
    fn cl_1(&self, signal: &Distribution, background: &Distribution, n: f64) -> f64 {
        let cl_1: f64 = signal
            .pmf_iter()
            .map(|(z, w)| (z as isize, w))
            .map(|(z, w)| {
                #[cfg(feature = "dbg")]
                dbg!((z, w));

                // TODO: this is used by cl_1 and cl_2, compute once only?
                (
                    w,
                    self.config.geometric.cdf(z),
                    self.config.geometric.cdf(z - 1) / self.config.geometric.cdf(z),
                    background.cdf(z - 1),
                    background.cdf(z),
                )
            })
            .map(|(w_x, g_x, r_x, phi_x_nom, phi_x_den)| {
                #[cfg(feature = "dbg")]
                dbg!((w_x, g_x, r_x, phi_x_nom, phi_x_den));

                let frac = {
                    // We need special handling of the asymptotic behavior.
                    let nom = if r_x == 0.0 && phi_x_nom == 0.0 {
                        // TODO: prove.
                        0.0
                    } else {
                        (r_x - 1.0).ln_1p() * phi_x_nom
                    };
                    let den = phi_x_den * r_x.powf(-n) - phi_x_nom;
                    if r_x == 0.00 {
                        0.0
                    } else if den == 0.0 {
                        #[cfg(feature = "dbg")]
                        dbg!(n);

                        // From the Taylor expansion, assuming that (r[x] - 1) << n(phi[x] - 1).
                        // TODO: prove above assumption.
                        -1.0 / n
                    } else {
                        #[cfg(feature = "dbg")]
                        dbg!((nom, den));

                        nom / den
                    }
                };

                #[cfg(feature = "dbg")]
                dbg!(frac);
                debug_assert!(n == 0.0 || frac <= 0.0);

                w_x * ((g_x - 1.0).ln_1p() - frac)
            })
            .sum();

        #[cfg(feature = "dbg")]
        dbg!(cl_1);
        debug_assert!(n == 0.0 || cl_1.is_finite());

        cl_1
    }

    /// Calculates the second partial derivative of the composite log likelihood
    /// with respect to n.
    fn cl_2(&self, signal: &Distribution, background: &Distribution, n: f64) -> f64 {
        let cl_2: f64 = signal
            .pmf_iter()
            .map(|(z, w)| (z as isize, w))
            .map(|(z, w)| {
                #[cfg(feature = "dbg")]
                dbg!((z, w));

                // TODO: this is used by cl_1 and cl_2, compute once only?
                (
                    w,
                    self.config.geometric.cdf(z - 1) / self.config.geometric.cdf(z),
                    background.cdf(z - 1),
                    background.cdf(z),
                )
            })
            .map(|(w_x, r_x, phi_x_nom, phi_x_den)| {
                #[cfg(feature = "dbg")]
                dbg!((w_x, r_x, phi_x_nom, phi_x_den));

                let nom = if r_x == 0.0 && phi_x_nom == 0.0 {
                    // TODO: prove.
                    0.0
                } else {
                    -r_x.powf(-n) * (r_x - 1.0).ln_1p().powi(2) * phi_x_nom * phi_x_den
                };
                let den = (phi_x_den * r_x.powf(-n) - phi_x_nom).powi(2);

                // We need special handling of the asymptotic behavior.
                let frac = if r_x == 0.0 {
                    0.0
                } else if den == 0.0 {
                    #[cfg(feature = "dbg")]
                    dbg!(n);

                    // From the Taylor expansion, assuming that (r[x] - 1) << n(phi[x] - 1).
                    // TODO: prove above assumption.
                    -1.0 / n.powi(2)
                } else {
                    #[cfg(feature = "dbg")]
                    dbg!((nom, den));

                    nom / den
                };

                #[cfg(feature = "dbg")]
                dbg!(frac);
                debug_assert!(frac <= 0.0);

                w_x * frac
            })
            .sum();

        #[cfg(feature = "dbg")]
        dbg!(cl_2);
        debug_assert!(n == 0.0 || cl_2.is_finite());
        // cl is concave.
        debug_assert!(cl_2 <= 0.0);

        cl_2
    }

    fn argmax_cl(&self, signal: &Distribution, background: &Distribution) -> u64 {
        let max_iters = 100;
        let mut iters = 0;
        let mut n = 1.0;
        loop {
            if iters > max_iters {
                break;
            }

            #[cfg(feature = "dbg")]
            let cl = self.cl(signal, background, n);

            let cl_1 = self.cl_1(signal, background, n);
            let cl_2 = self.cl_2(signal, background, n);
            let shift = -cl_1 / cl_2;

            #[cfg(feature = "dbg")]
            dbg!((n, cl, cl_1, cl_2, shift));
            n += shift;
            // TODO: refine convergence criterion.
            if shift.abs() / n < 1e-3 {
                return n.round() as u64;
            }
            iters += 1;
        }
        #[cfg(feature = "dbg")]
        dbg!("Broke after {} iters", max_iters);
        n.round() as u64
    }
}

#[derive(Clone, Debug)]
pub enum MergeError {
    ConfigMismatch,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MergeError::ConfigMismatch => write!(f, "sketch configs do not match"),
        }
    }
}

impl error::Error for MergeError {}

const fn alpha(d: usize) -> f64 {
    assert!(d & (d - 1) == 0); // non-zero power of 2
    match d {
        1 | 2 | 4 | 8 => panic!(),
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        128 => 0.715,
        256 => 0.718,
        512 => 0.720,
        _ => 0.721,
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use sketch_traits::HeavyDistinctHitterSketch;

    use super::*;

    const COUNTER_SIZE: usize = 1024;
    const SEEDS: [u64; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    fn seeded_config(d: usize, w: usize) -> Config {
        Config::new(d, w, Some(SEEDS)).unwrap()
    }

    #[derive(Clone, Debug)]
    struct TestCase {
        num_entries: u64,
        #[allow(dead_code)]
        num_labels: u64,
        sketch: PointwiseSketch,
        label: u64,
        label_cardinality: u64,
        signal: Distribution,
        background: Distribution,
    }

    fn test_cases() -> impl Strategy<Value = TestCase> {
        let test_cases = vec![(0, 1), (1, 1), (6_000, 3), (10_000_000, 7)];

        let sketches = test_cases
            .into_iter()
            .map(|(num_entries, num_labels)| {
                let mut sketch = PointwiseSketch::new(&seeded_config(COUNTER_SIZE, 1000));
                (0..num_entries).for_each(|x| sketch.insert(&(x % num_labels), &x));
                let label = num_labels - 1;
                let label_cardinality = num_entries / num_labels;
                let signal = sketch.signal(&label);
                let background = sketch.background(&label);
                TestCase {
                    num_entries,
                    num_labels,
                    sketch,
                    label,
                    label_cardinality,
                    signal,
                    background,
                }
            })
            .map(Just);

        proptest::strategy::Union::new(sketches)
    }

    #[test]
    fn it_computes_cl() {
        proptest!(ProptestConfig::with_cases(8192), |(n in 0.0..1e6, test_case in test_cases())| {
            let cl = test_case.sketch.cl(&test_case.signal, &test_case.background, n);
            prop_assert!(cl <= 0.0);
            if n == 0.0 {
                if test_case.num_entries == 0 {
                    prop_assert_eq!(cl, 0.0)
                } else {
                    prop_assert_eq!(cl, f64::NEG_INFINITY)
                }
            } else if n <= test_case.label_cardinality as f64 {
                prop_assert!(cl.is_finite());
            }
        });
    }

    #[test]
    fn it_computes_cl_1() {
        proptest!(ProptestConfig::with_cases(8192), |(n in 1.0..1e6, test_case in test_cases())| {
            // No assertions here, but debug assertions are run.
            test_case.sketch.cl_1(&test_case.signal, &test_case.background, n);
        });
    }

    #[test]
    fn it_computes_cl_2() {
        proptest!(ProptestConfig::with_cases(8192), |(n in 1.0..1e6, test_case in test_cases())| {
            // No assertions here, but debug assertions are run.
            test_case.sketch.cl_2(&test_case.signal, &test_case.background, n);
        });
    }

    #[test]
    fn it_estimates_cardinality_by_maximizing_the_composite_likelihood() {
        proptest!(ProptestConfig::with_cases(32), |(test_case in test_cases())| {
            prop_assume!(test_case.label_cardinality > 0); // FIXME
            let cardinality = test_case.sketch.cardinality(&test_case.label);
            prop_assert!((cardinality as f64 - test_case.label_cardinality as f64).abs() / test_case.label_cardinality as f64 <= 2e-2);
        });
    }

    #[test]
    #[ignore]
    fn print_cl() {
        let num_entries = 6000;
        let num_labels = 3;
        let sketch = {
            let mut sketch = PointwiseSketch::new(&seeded_config(COUNTER_SIZE, 1000));
            (0..num_entries).for_each(|x| sketch.insert(&(x % num_labels), &x));
            sketch
        };

        let signal = sketch.signal(&0);
        let background = sketch.background(&0);

        (0_u64..)
            .map(|i| i * 100)
            .take_while(|&n| n <= 10_000)
            .for_each(|n| {
                println!(
                    "{:>6} {:+.3} {:+.6} {:+.12}",
                    n,
                    sketch.cl(&signal, &background, n as f64),
                    sketch.cl_1(&signal, &background, n as f64),
                    sketch.cl_2(&signal, &background, n as f64),
                );
            });
    }

    #[test]
    fn test_top() {
        let num_labels = 8;
        let mut sketch = LabelSetCountHLL::new(&seeded_config(COUNTER_SIZE, 10));
        for l in 1..num_labels {
            let label = l.to_string();
            for i in 0..l * 10 {
                sketch.insert(label.clone(), &i);
            }
        }
        assert!(sketch.top(10).len() == num_labels - 1);

        let mut sketch = LabelArrayCountHLL::new(&seeded_config(COUNTER_SIZE, 10));
        for l in 1..num_labels {
            let label = l.to_string();
            for i in 0..l * 10 {
                sketch.insert(label.clone(), &i);
            }
        }
        assert!(sketch.top(10).len() == num_labels - 1);
    }
}
